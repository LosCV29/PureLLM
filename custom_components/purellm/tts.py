"""ElevenLabs TTS platform for PureLLM (streaming).

Registers a TTS entity that calls the ElevenLabs API directly with every voice
parameter exposed in PureLLM's options ("ElevenLabs TTS" step).

Latency design (2026-09-28):
- Streaming: the reply is split into sentences; the first sentence is synthesized
  on its own so audio starts as early as possible, and each following chunk is
  requested while the previous one is still streaming (prefetch). Continuity
  across chunks uses ElevenLabs ``previous_request_ids``.
- One dedicated HTTP session with a long keep-alive plus a cheap periodic ping,
  so a reply never pays a fresh TLS handshake.
- Language defaults to "auto" (no ``language_code``) so Spanglish replies are
  read with the right phonetics per word.
"""
from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncGenerator
import logging
import re
import struct
import time
from typing import Any

import aiohttp

from homeassistant.components.tts import (
    ATTR_VOICE,
    TextToSpeechEntity,
    TTSAudioRequest,
    TTSAudioResponse,
    TtsAudioType,
    Voice,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_MODEL
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_ELEVENLABS_API_KEY,
    CONF_ELEVENLABS_KEEP_WARM,
    CONF_ELEVENLABS_LANGUAGE,
    CONF_ELEVENLABS_MODEL,
    CONF_ELEVENLABS_OUTPUT_FORMAT,
    CONF_ELEVENLABS_SEED,
    CONF_ELEVENLABS_SENTENCE_STREAMING,
    CONF_ELEVENLABS_SIMILARITY,
    CONF_ELEVENLABS_SPEAKER_BOOST,
    CONF_ELEVENLABS_SPEED,
    CONF_ELEVENLABS_STABILITY,
    CONF_ELEVENLABS_STYLE,
    CONF_ELEVENLABS_TEXT_NORMALIZATION,
    CONF_ELEVENLABS_VOICE_ID,
    DEFAULT_ELEVENLABS_API_KEY,
    DEFAULT_ELEVENLABS_KEEP_WARM,
    DEFAULT_ELEVENLABS_LANGUAGE,
    DEFAULT_ELEVENLABS_MODEL,
    DEFAULT_ELEVENLABS_OUTPUT_FORMAT,
    DEFAULT_ELEVENLABS_SEED,
    DEFAULT_ELEVENLABS_SENTENCE_STREAMING,
    DEFAULT_ELEVENLABS_SIMILARITY,
    DEFAULT_ELEVENLABS_SPEAKER_BOOST,
    DEFAULT_ELEVENLABS_SPEED,
    DEFAULT_ELEVENLABS_STABILITY,
    DEFAULT_ELEVENLABS_STYLE,
    DEFAULT_ELEVENLABS_TEXT_NORMALIZATION,
    DEFAULT_ELEVENLABS_VOICE_ID,
)

_LOGGER = logging.getLogger(__name__)

API_BASE = "https://api.elevenlabs.io"
KEEP_WARM_INTERVAL = 25  # seconds between keep-alive pings
KEEP_ALIVE_TIMEOUT = 120  # seconds an idle pooled connection is kept
MAX_PREVIOUS_REQUEST_IDS = 3
# Sentences per request: first chunk alone (earliest audio), then 2, then the rest.
CHUNK_SCHEDULE = (1, 2)

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
_DEGREE = "°"
_RX_DEGREES = re.compile(r"(\d+)\s*" + _DEGREE + r"(?:\s*F\b)?")
_RX_DEG_FAHRENHEIT = re.compile(r"\bdegrees Fahrenheit\b", re.IGNORECASE)
_RX_FAHRENHEIT = re.compile(r"\bFahrenheit\b", re.IGNORECASE)
_RX_SPACES = re.compile(r"[ \t]{2,}")

# Sentence end: . ! ? … (plus closing quotes/brackets) followed by whitespace.
_RX_SENTENCE_END = re.compile(r"[.!?…]+[\"'”’)\]]*\s+")
_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "st.", "jr.", "sr.", "vs.", "etc.", "no.",
    "a.m.", "p.m.", "e.g.", "i.e.", "approx.", "sra.", "sr.", "dra.",
}


def normalize_tts_text(text: str) -> str:
    """Never speak Fahrenheit; drop the degree glyph the voice tends to garble."""
    if not text:
        return text
    text = _RX_DEGREES.sub(r"\1 degrees", text)
    text = _RX_DEG_FAHRENHEIT.sub("degrees", text)
    text = _RX_FAHRENHEIT.sub("", text)
    return _RX_SPACES.sub(" ", text).strip()


def split_sentences(text: str) -> tuple[list[str], str]:
    """Split complete sentences off ``text``; return (sentences, remainder)."""
    sentences: list[str] = []
    start = 0
    for match in _RX_SENTENCE_END.finditer(text):
        candidate = text[start:match.end()]
        last_word = candidate.strip().rsplit(" ", 1)[-1].lower()
        if last_word in _ABBREVIATIONS:
            continue
        if candidate.strip():
            sentences.append(candidate.strip())
        start = match.end()
    return sentences, text[start:]


def schedule_chunks(sentences: list[str], schedule: list[int]) -> list[str]:
    """Group sentences into request-sized chunks, consuming ``schedule`` in place."""
    chunks: list[str] = []
    while sentences:
        take = schedule.pop(0) if schedule else len(sentences)
        chunks.append(" ".join(sentences[:take]))
        del sentences[:take]
    return chunks


# ---------------------------------------------------------------------------
# Audio container helpers
# ---------------------------------------------------------------------------
_WAV_STREAM_SIZE = 0xFFFFFFFF  # open-ended: readers play until EOF


def pcm_sample_rate(output_format: str) -> int | None:
    """Sample rate of a ``pcm_<rate>`` output format, else None."""
    fmt = str(output_format)
    if not fmt.startswith("pcm_"):
        return None
    try:
        return int(fmt[4:])
    except ValueError:
        return None


def wav_header(sample_rate: int, data_len: int | None = None) -> bytes:
    """44-byte header for 16-bit mono PCM; ``data_len=None`` = streaming (unknown length)."""
    data_size = _WAV_STREAM_SIZE if data_len is None else data_len
    riff_size = _WAV_STREAM_SIZE if data_len is None else 36 + data_len
    return (
        b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data" + struct.pack("<I", data_size)
    )


async def _prepend(head: bytes, stream: AsyncGenerator[bytes]) -> AsyncGenerator[bytes]:
    yield head
    async for data in stream:
        yield data


# ---------------------------------------------------------------------------
# Platform setup
# ---------------------------------------------------------------------------
async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up ElevenLabs TTS from a PureLLM config entry."""
    config = {**entry.data, **entry.options}
    if not config.get(CONF_ELEVENLABS_API_KEY, DEFAULT_ELEVENLABS_API_KEY):
        _LOGGER.debug("ElevenLabs TTS: no API key configured, skipping TTS entity")
        return
    async_add_entities([PureLLMElevenLabsTTS(entry)])


class PureLLMElevenLabsTTS(TextToSpeechEntity):
    """Streaming ElevenLabs TTS entity with full parameter control."""

    _attr_has_entity_name = True
    _attr_name = "ElevenLabs TTS"
    _attr_supported_options = [ATTR_VOICE, ATTR_MODEL]

    def __init__(self, entry: ConfigEntry) -> None:
        self._entry = entry
        # Same unique_id as the pre-7cd0513 entity so the old tts.elevenlabs_tts id is reused.
        self._attr_unique_id = f"{entry.entry_id}_elevenlabs_tts"
        self._session: aiohttp.ClientSession | None = None
        self._keep_warm_task: asyncio.Task | None = None
        self._voices: list[Voice] = []
        # Models that rejected previous_request_ids — stop sending them.
        self._no_request_ids: set[str] = set()

    # --- config ------------------------------------------------------------
    @property
    def _config(self) -> dict[str, Any]:
        return {**self._entry.data, **self._entry.options}

    def _get(self, key: str, default: Any) -> Any:
        return self._config.get(key, default)

    @property
    def _api_key(self) -> str:
        return str(self._get(CONF_ELEVENLABS_API_KEY, DEFAULT_ELEVENLABS_API_KEY)).strip()

    def _voice_settings(self) -> dict[str, Any]:
        speed = max(0.7, min(1.2, float(self._get(CONF_ELEVENLABS_SPEED, DEFAULT_ELEVENLABS_SPEED))))
        return {
            "stability": float(self._get(CONF_ELEVENLABS_STABILITY, DEFAULT_ELEVENLABS_STABILITY)),
            "similarity_boost": float(self._get(CONF_ELEVENLABS_SIMILARITY, DEFAULT_ELEVENLABS_SIMILARITY)),
            "style": float(self._get(CONF_ELEVENLABS_STYLE, DEFAULT_ELEVENLABS_STYLE)),
            "use_speaker_boost": bool(self._get(CONF_ELEVENLABS_SPEAKER_BOOST, DEFAULT_ELEVENLABS_SPEAKER_BOOST)),
            "speed": speed,
        }

    @property
    def default_language(self) -> str:
        return "en"

    @property
    def supported_languages(self) -> list[str]:
        return [
            "en", "es", "fr", "de", "it", "pt", "pl", "hi", "ar", "cs", "nl",
            "fi", "el", "hu", "id", "ja", "ko", "ms", "no", "ro", "ru", "sk",
            "sv", "sw", "ta", "th", "tr", "uk", "ur", "vi", "zh",
        ]

    @property
    def default_options(self) -> dict[str, Any]:
        return {
            ATTR_VOICE: self._get(CONF_ELEVENLABS_VOICE_ID, DEFAULT_ELEVENLABS_VOICE_ID),
            ATTR_MODEL: self._get(CONF_ELEVENLABS_MODEL, DEFAULT_ELEVENLABS_MODEL),
        }

    @callback
    def async_get_supported_voices(self, language: str) -> list[Voice] | None:
        return self._voices or None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {
            "voice_id": self._get(CONF_ELEVENLABS_VOICE_ID, DEFAULT_ELEVENLABS_VOICE_ID),
            "model": self._get(CONF_ELEVENLABS_MODEL, DEFAULT_ELEVENLABS_MODEL),
            **self._voice_settings(),
            "output_format": self._get(CONF_ELEVENLABS_OUTPUT_FORMAT, DEFAULT_ELEVENLABS_OUTPUT_FORMAT),
            "text_normalization": self._get(CONF_ELEVENLABS_TEXT_NORMALIZATION, DEFAULT_ELEVENLABS_TEXT_NORMALIZATION),
            "language": self._get(CONF_ELEVENLABS_LANGUAGE, DEFAULT_ELEVENLABS_LANGUAGE),
            "seed": self._get(CONF_ELEVENLABS_SEED, DEFAULT_ELEVENLABS_SEED),
            "sentence_streaming": self._get(CONF_ELEVENLABS_SENTENCE_STREAMING, DEFAULT_ELEVENLABS_SENTENCE_STREAMING),
            "keep_warm": self._get(CONF_ELEVENLABS_KEEP_WARM, DEFAULT_ELEVENLABS_KEEP_WARM),
        }

    # --- lifecycle ---------------------------------------------------------
    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(keepalive_timeout=KEEP_ALIVE_TIMEOUT),
            headers={"xi-api-key": self._api_key},
        )
        self.hass.async_create_background_task(self._load_voices(), "purellm_elevenlabs_voices")
        if self._get(CONF_ELEVENLABS_KEEP_WARM, DEFAULT_ELEVENLABS_KEEP_WARM):
            self._keep_warm_task = self.hass.async_create_background_task(
                self._keep_warm(), "purellm_elevenlabs_keep_warm"
            )

    async def async_will_remove_from_hass(self) -> None:
        if self._keep_warm_task:
            self._keep_warm_task.cancel()
            self._keep_warm_task = None
        if self._session:
            await self._session.close()
            self._session = None
        await super().async_will_remove_from_hass()

    async def _load_voices(self) -> None:
        try:
            async with self._session.get(
                f"{API_BASE}/v1/voices", timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    _LOGGER.warning("ElevenLabs voices fetch: HTTP %s", resp.status)
                    return
                data = await resp.json()
            self._voices = [
                Voice(v["voice_id"], v.get("name") or v["voice_id"])
                for v in data.get("voices", [])
                if v.get("voice_id")
            ]
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning("ElevenLabs voices fetch failed: %s", err)

    async def _keep_warm(self) -> None:
        """Ping a free endpoint so the pooled TLS connection stays open between replies."""
        while True:
            try:
                async with self._session.get(
                    f"{API_BASE}/v1/models", timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    await resp.read()
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001
                _LOGGER.debug("ElevenLabs keep-warm ping failed: %s", err)
            await asyncio.sleep(KEEP_WARM_INTERVAL)

    # --- synthesis ---------------------------------------------------------
    def _request_body(self, text: str, model: str, previous_ids: list[str]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "text": text,
            "model_id": model,
            "voice_settings": self._voice_settings(),
            "apply_text_normalization": self._get(
                CONF_ELEVENLABS_TEXT_NORMALIZATION, DEFAULT_ELEVENLABS_TEXT_NORMALIZATION
            ),
        }
        language = self._get(CONF_ELEVENLABS_LANGUAGE, DEFAULT_ELEVENLABS_LANGUAGE)
        if language and language != "auto":
            body["language_code"] = language
        seed = int(self._get(CONF_ELEVENLABS_SEED, DEFAULT_ELEVENLABS_SEED) or 0)
        if seed > 0:
            body["seed"] = seed
        if previous_ids and model not in self._no_request_ids:
            body["previous_request_ids"] = previous_ids
        return body

    async def _stream_chunk(
        self,
        text: str,
        voice_id: str,
        model: str,
        previous_ids: list[str],
        queue: asyncio.Queue,
        headers_ready: asyncio.Event,
        request_id: list[str | None],
    ) -> None:
        """Synthesize one chunk, pushing audio bytes into ``queue`` (None = done)."""
        output_format = self._output_format()
        url =f"{API_BASE}/v1/text-to-speech/{voice_id}/stream"
        try:
            for attempt in (1, 2):
                body = self._request_body(text, model, previous_ids)
                async with self._session.post(
                    url,
                    params={"output_format": output_format},
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status != 200:
                        err = (await resp.text())[:300]
                        if attempt == 1 and "previous_request_ids" in body and resp.status in (400, 422):
                            _LOGGER.info("ElevenLabs %s rejected previous_request_ids; disabling for this model", model)
                            self._no_request_ids.add(model)
                            continue
                        _LOGGER.error("ElevenLabs TTS failed (HTTP %s): %s", resp.status, err)
                        return
                    request_id[0] = resp.headers.get("request-id")
                    headers_ready.set()
                    async for data in resp.content.iter_any():
                        if data:
                            await queue.put(data)
                    return
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001
            _LOGGER.error("ElevenLabs TTS request failed: %s", err)
        finally:
            headers_ready.set()
            await queue.put(None)

    async def _synthesize_stream(
        self, message_gen: AsyncGenerator[str], voice_id: str, model: str
    ) -> AsyncGenerator[bytes]:
        """Sentence-chunked, prefetching synthesis of an incoming text stream."""
        streaming = bool(self._get(CONF_ELEVENLABS_SENTENCE_STREAMING, DEFAULT_ELEVENLABS_SENTENCE_STREAMING))
        schedule = list(CHUNK_SCHEDULE)
        pending_chunks: asyncio.Queue[str | None] = asyncio.Queue()

        async def _collect() -> None:
            buffer = ""
            try:
                async for piece in message_gen:
                    buffer += piece
                    if not streaming:
                        continue
                    sentences, buffer = split_sentences(buffer)
                    for chunk in schedule_chunks(sentences, schedule):
                        await pending_chunks.put(chunk)
                if buffer.strip():
                    await pending_chunks.put(buffer.strip())
            finally:
                await pending_chunks.put(None)

        collector = self.hass.async_create_background_task(_collect(), "purellm_elevenlabs_collect")
        previous_ids: deque[str] = deque(maxlen=MAX_PREVIOUS_REQUEST_IDS)
        t0 = time.monotonic()
        first_audio: float | None = None
        n_chunks = 0
        total_chars = 0
        # Prefetch: chunk i+1's request starts as soon as chunk i's headers (request-id) arrive.
        prev_headers: asyncio.Event | None = None
        prev_request_id: list[str | None] = [None]
        tasks: list[asyncio.Task] = []
        queues: deque[asyncio.Queue] = deque()
        chunks_done = False

        async def _start_next() -> bool:
            nonlocal prev_headers, prev_request_id, n_chunks, total_chars, chunks_done
            if chunks_done:
                return False
            chunk = await pending_chunks.get()
            if chunk is None:
                chunks_done = True
                return False
            chunk = normalize_tts_text(chunk)
            if not chunk:
                return True
            if prev_headers is not None:
                await prev_headers.wait()
                if prev_request_id[0]:
                    previous_ids.append(prev_request_id[0])
            queue: asyncio.Queue = asyncio.Queue()
            headers_ready = asyncio.Event()
            request_id: list[str | None] = [None]
            tasks.append(self.hass.async_create_background_task(
                self._stream_chunk(chunk, voice_id, model, list(previous_ids), queue, headers_ready, request_id),
                "purellm_elevenlabs_chunk",
            ))
            queues.append(queue)
            prev_headers, prev_request_id = headers_ready, request_id
            n_chunks += 1
            total_chars += len(chunk)
            return True

        try:
            while not queues and await _start_next():
                pass
            while queues:
                queue = queues[0]
                prefetch: asyncio.Task | None = None
                if not chunks_done:
                    prefetch = asyncio.ensure_future(_start_next())
                while (data := await queue.get()) is not None:
                    if first_audio is None:
                        first_audio = time.monotonic() - t0
                    yield data
                queues.popleft()
                if prefetch is not None:
                    await prefetch
                while not queues and await _start_next():
                    pass
        finally:
            collector.cancel()
            for task in tasks:
                task.cancel()
            _LOGGER.info(
                "ElevenLabs TTS: voice=%s model=%s chunks=%d chars=%d first_audio=%s total=%.2fs",
                voice_id, model, n_chunks, total_chars,
                f"{first_audio * 1000:.0f}ms" if first_audio is not None else "none",
                time.monotonic() - t0,
            )

    def _resolve(self, options: dict[str, Any] | None) -> tuple[str, str]:
        options = options or {}
        voice_id = options.get(ATTR_VOICE) or self._get(CONF_ELEVENLABS_VOICE_ID, DEFAULT_ELEVENLABS_VOICE_ID)
        model = options.get(ATTR_MODEL) or self._get(CONF_ELEVENLABS_MODEL, DEFAULT_ELEVENLABS_MODEL)
        return voice_id, model

    def _output_format(self) -> str:
        """mp3_* passes through; pcm_<rate> is wrapped in WAV (HA then only packs FLAC)."""
        output_format = str(self._get(CONF_ELEVENLABS_OUTPUT_FORMAT, DEFAULT_ELEVENLABS_OUTPUT_FORMAT))
        if output_format.startswith("mp3") or pcm_sample_rate(output_format):
            return output_format
        return DEFAULT_ELEVENLABS_OUTPUT_FORMAT

    async def async_stream_tts_audio(self, request: TTSAudioRequest) -> TTSAudioResponse:
        """Stream speech for text that may itself still be streaming in."""
        voice_id, model = self._resolve(request.options)
        audio = self._synthesize_stream(request.message_gen, voice_id, model)
        rate = pcm_sample_rate(self._output_format())
        if rate is None:
            return TTSAudioResponse("mp3", audio)
        return TTSAudioResponse("wav", _prepend(wav_header(rate), audio))

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any] | None = None
    ) -> TtsAudioType:
        """Non-streaming path (announcements, tts.speak): same pipeline, collected."""
        voice_id, model = self._resolve(options)
        if not self._api_key or not voice_id:
            _LOGGER.error("ElevenLabs TTS: API key or voice ID not configured")
            return (None, None)

        async def _one() -> AsyncGenerator[str]:
            yield message

        audio = bytearray()
        async for data in self._synthesize_stream(_one(), voice_id, model):
            audio.extend(data)
        if not audio:
            return (None, None)
        rate = pcm_sample_rate(self._output_format())
        if rate is None:
            return ("mp3", bytes(audio))
        return ("wav", wav_header(rate, len(audio)) + bytes(audio))
