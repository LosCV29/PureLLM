"""PureLLM built-in Chatterbox TTS platform.

Provides a native HA TTS entity that calls Chatterbox directly — no Wyoming
bridge or separate server needed. The voice pipeline calls this TTS platform
directly, and PureLLM pre-populates the cache via the same in-process dict.

This eliminates the need for:
  - The wyoming_chatterbox.py bridge
  - The HTTP pre-cache endpoint
  - The Wyoming TCP server

Instead, PureLLM writes directly to AUDIO_CACHE before returning the
conversation result. When HA's pipeline TTS step fires, this platform
serves the cached audio instantly.

Enable by setting voice_reply_tts_url to "builtin" in the PureLLM config.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import re
import struct
import time
import wave
from typing import Any

import httpx

from homeassistant.components.tts import TextToSpeechEntity, TtsAudioType
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_CHATTERBOX_URL,
    CONF_VOICE_REPLY_TTS_URL,
    CONF_VOICE_REPLY_TTS_VOICE,
    DEFAULT_CHATTERBOX_URL,
    DEFAULT_VOICE_REPLY_TTS_VOICE,
)

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared in-process audio cache — conversation.py writes, tts.py reads
# ---------------------------------------------------------------------------
# {sha256_hex: {"pcm": bytes, "ts": float}}
AUDIO_CACHE: dict[str, dict] = {}
# {sha256_hex: asyncio.Event} — set when generation completes
GENERATION_LOCKS: dict[str, asyncio.Event] = {}

CACHE_TTL = 120  # seconds


def cache_audio(key: str, pcm: bytes) -> None:
    """Store pre-generated PCM audio in the shared cache."""
    AUDIO_CACHE[key] = {"pcm": pcm, "ts": time.time()}


def _purge_expired() -> None:
    """Remove cache entries older than CACHE_TTL."""
    now = time.time()
    expired = [k for k, v in AUDIO_CACHE.items() if now - v["ts"] > CACHE_TTL]
    for k in expired:
        del AUDIO_CACHE[k]


# ---------------------------------------------------------------------------
# Chatterbox API + sentence chunking (same logic as wyoming_chatterbox.py)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2
CHANNELS = 1

SENTENCE_GAP_MS = 150
SENTENCE_GAP_BYTES = SAMPLE_RATE * SAMPLE_WIDTH * SENTENCE_GAP_MS // 1000
SENTENCE_GAP = b"\x00" * SENTENCE_GAP_BYTES

# Max sentences to generate concurrently (prevents overwhelming the GPU)
PARALLEL_SENTENCES = 2

_ABBREV = r"(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bSt)(?<!\bNo)(?<!\bvs)"
_SENTENCE_RE = re.compile(_ABBREV + r'([.!?])\s+', re.IGNORECASE)
MIN_SENTENCE_LEN = 15


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for chunked TTS generation."""
    parts = _SENTENCE_RE.split(text)
    raw_sentences = []
    i = 0
    while i < len(parts):
        sentence = parts[i].strip()
        if i + 1 < len(parts) and len(parts[i + 1]) == 1:
            sentence += parts[i + 1]
            i += 2
        else:
            i += 1
        if sentence:
            raw_sentences.append(sentence)

    merged = []
    buffer = ""
    for s in raw_sentences:
        if buffer:
            buffer = buffer + " " + s
        elif len(s) < MIN_SENTENCE_LEN:
            buffer = s
        else:
            merged.append(s)
            continue
        if len(buffer) >= MIN_SENTENCE_LEN:
            merged.append(buffer)
            buffer = ""

    if buffer:
        if merged:
            merged[-1] = merged[-1] + " " + buffer
        else:
            merged.append(buffer)

    return merged if merged else [text]


def extract_pcm(wav_bytes: bytes) -> bytes:
    """Extract raw PCM from WAV container."""
    try:
        with io.BytesIO(wav_bytes) as f:
            with wave.open(f, "rb") as w:
                return w.readframes(w.getnframes())
    except Exception:
        idx = wav_bytes.find(b"data")
        if idx != -1:
            size = struct.unpack_from("<I", wav_bytes, idx + 4)[0]
            return wav_bytes[idx + 8 : idx + 8 + size]
        return wav_bytes[44:]


def _pcm_to_wav(pcm: bytes) -> bytes:
    """Wrap raw PCM bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


async def _generate_single(chatterbox_url: str, voice: str, text: str) -> bytes:
    """Call Chatterbox TTS API and return raw PCM audio."""
    # Create SSL context in executor to avoid blocking the event loop
    # (httpx loads verify certificates synchronously on client init)
    loop = asyncio.get_running_loop()
    client = await loop.run_in_executor(
        None, lambda: httpx.AsyncClient(timeout=90.0)
    )
    async with client:
        resp = await client.post(
            f"{chatterbox_url}/v1/audio/speech",
            json={
                "input": text,
                "voice": voice,
                "model": "chatterbox",
                "speed": 1.0,
            },
        )
        resp.raise_for_status()
        return extract_pcm(resp.content)


async def generate_chunked(chatterbox_url: str, voice: str, text: str) -> bytes:
    """Split text into sentences, generate with sliding-window parallelism.

    Uses PARALLEL_SENTENCES concurrent requests to keep the GPU busy
    without overwhelming it (the GPU serializes inference anyway, so
    blasting N requests doesn't help and can hurt throughput).
    """
    sentences = split_sentences(text)

    if len(sentences) <= 1:
        return await _generate_single(chatterbox_url, voice, text)

    n = len(sentences)
    _LOGGER.info("TTS: pipelined generation of %d sentences", n)
    t0 = time.time()

    # Pre-launch first batch of tasks
    tasks: dict[int, asyncio.Task] = {}
    for i in range(min(PARALLEL_SENTENCES, n)):
        tasks[i] = asyncio.create_task(
            _generate_single(chatterbox_url, voice, sentences[i])
        )

    pcm_parts: list[bytes] = []
    for i in range(n):
        pcm = await tasks.pop(i)
        pcm_parts.append(pcm)
        if i < n - 1:
            pcm_parts.append(SENTENCE_GAP)

        # Launch next sentence in the sliding window
        next_i = i + PARALLEL_SENTENCES
        if next_i < n:
            tasks[next_i] = asyncio.create_task(
                _generate_single(chatterbox_url, voice, sentences[next_i])
            )

    total_pcm = b"".join(pcm_parts)
    _LOGGER.info("TTS: pipelined generation done in %.1fs (%d bytes)",
                 time.time() - t0, len(total_pcm))
    return total_pcm


# ---------------------------------------------------------------------------
# HA TTS platform setup
# ---------------------------------------------------------------------------
async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up PureLLM TTS entity if built-in mode is enabled."""
    config = {**config_entry.data, **config_entry.options}
    tts_url = (config.get(CONF_VOICE_REPLY_TTS_URL) or "").strip().lower()

    if tts_url != "builtin":
        return

    chatterbox_url = config.get("chatterbox_url", "http://host.docker.internal:8004")
    voice = config.get(CONF_VOICE_REPLY_TTS_VOICE, DEFAULT_VOICE_REPLY_TTS_VOICE)

    entity = PureLLMTTSEntity(config_entry, chatterbox_url, voice)
    async_add_entities([entity])

    # Store reference so conversation.py can access the cache directly
    hass.data.setdefault("purellm", {})
    hass.data["purellm"]["tts_entity"] = entity


class PureLLMTTSEntity(TextToSpeechEntity):
    """Built-in Chatterbox TTS entity with pre-cache support."""

    _attr_has_entity_name = True
    _attr_name = "PureLLM Chatterbox TTS"

    def __init__(
        self,
        config_entry: ConfigEntry,
        chatterbox_url: str,
        voice: str,
    ) -> None:
        """Initialize."""
        self._config_entry = config_entry
        self._attr_unique_id = f"{config_entry.entry_id}_tts"
        self._chatterbox_url = chatterbox_url
        self._voice = voice

    @property
    def supported_languages(self) -> list[str]:
        """Return supported languages."""
        return ["en"]

    @property
    def default_language(self) -> str:
        """Return default language."""
        return "en"

    async def async_get_tts_audio(
        self,
        message: str,
        language: str,
        options: dict[str, Any] | None = None,
    ) -> TtsAudioType:
        """Generate TTS audio — serve from cache if available."""
        _purge_expired()

        cache_key = hashlib.sha256(message.encode()).hexdigest()

        # Check cache first (pre-populated by conversation.py)
        cached = AUDIO_CACHE.pop(cache_key, None)
        if cached:
            _LOGGER.info("TTS: cache HIT (%s) — serving instantly", cache_key[:8])
            return ("wav", _pcm_to_wav(cached["pcm"]))

        # Check if generation is in progress (fire-and-forget from conversation.py)
        lock = GENERATION_LOCKS.get(cache_key)
        if lock:
            _LOGGER.info("TTS: waiting for in-progress generation (%s)", cache_key[:8])
            try:
                await asyncio.wait_for(lock.wait(), timeout=90.0)
            except asyncio.TimeoutError:
                _LOGGER.warning("TTS: generation lock timed out (%s)", cache_key[:8])
            cached = AUDIO_CACHE.pop(cache_key, None)
            if cached:
                _LOGGER.info("TTS: cache HIT after wait (%s)", cache_key[:8])
                return ("wav", _pcm_to_wav(cached["pcm"]))

        # Cache miss — generate fresh
        _LOGGER.info("TTS: cache MISS (%s) — generating", cache_key[:8])
        pcm = await generate_chunked(self._chatterbox_url, self._voice, message)
        return ("wav", _pcm_to_wav(pcm))

    async def precache(self, text: str) -> str:
        """Pre-generate TTS and store in cache. Returns the cache key."""
        cache_key = hashlib.sha256(text.encode()).hexdigest()

        if cache_key in AUDIO_CACHE:
            return cache_key

        if cache_key in GENERATION_LOCKS:
            # Already generating — just wait
            await asyncio.wait_for(GENERATION_LOCKS[cache_key].wait(), timeout=90.0)
            return cache_key

        lock = asyncio.Event()
        GENERATION_LOCKS[cache_key] = lock
        try:
            pcm = await generate_chunked(self._chatterbox_url, self._voice, text)
            AUDIO_CACHE[cache_key] = {"pcm": pcm, "ts": time.time()}
        finally:
            lock.set()
            GENERATION_LOCKS.pop(cache_key, None)

        return cache_key
