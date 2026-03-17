#!/usr/bin/env python3
"""Wyoming Chatterbox TTS server with pre-cache HTTP bridge.

Runs TWO servers:
  1. Wyoming TCP server on port 10201 — speaks the Wyoming protocol for HA
  2. HTTP server on port 10202 — accepts POST /precache from PureLLM

Flow (fire-and-forget from PureLLM):
  PureLLM gets LLM response text
    → Immediately returns response to HA pipeline
    → Concurrently POST /precache {"text": "...", "key": "<sha256>"} to port 10202
    → This server splits text into sentences, generates ALL in parallel via
      Chatterbox, concatenates PCM with inter-sentence silence, stores by hash
    → HA asks Wyoming (port 10201) for TTS on same text
    → Wyoming handler: cache HIT → responds in <100ms
    → If pre-cache is still in progress, Wyoming WAITS for it (generation lock)
      instead of starting a duplicate generation
    → PE gets audio, LED stays on, no gap

Sentence chunking: Chatterbox generation scales superlinearly with text length.
A 10-word sentence generates in ~1s, but a 30-word block takes ~4s. By splitting
into sentences and generating ALL concurrently via asyncio.gather, total
generation time equals the slowest single sentence (~1-2s) instead of the sum.

Cache entries expire after 120s to prevent memory leaks from orphaned entries.
"""
import asyncio
import hashlib
import io
import logging
import re
import struct
import time
import wave

import httpx
from aiohttp import web

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHATTERBOX_URL = "http://host.docker.internal:8004"
VOICE = "Sir_David.mp3"
SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2
CHANNELS = 1
CHUNK_SIZE = 4096

WYOMING_PORT = 10201
HTTP_PORT = 10202

# Cache expiry in seconds — entries older than this are purged
CACHE_TTL = 120

# Silence between sentences: 150ms of zeros at 24kHz 16-bit mono
SENTENCE_GAP_MS = 150
SENTENCE_GAP_BYTES = SAMPLE_RATE * SAMPLE_WIDTH * SENTENCE_GAP_MS // 1000
SENTENCE_GAP = b"\x00" * SENTENCE_GAP_BYTES

# Regex to split text into sentences. Splits on period, exclamation, question
# mark followed by a space or end-of-string. Keeps the punctuation with the
# sentence. Handles common abbreviations (Mr., Mrs., Dr., St., etc.)
_ABBREV = r"(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bSt)(?<!\bNo)(?<!\bvs)"
_SENTENCE_RE = re.compile(
    _ABBREV + r'([.!?])\s+',
    re.IGNORECASE,
)

# Minimum character length for a sentence to be worth generating separately.
# Shorter fragments get merged with the next sentence.
MIN_SENTENCE_LEN = 15


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for chunked TTS generation.

    Returns a list of sentence strings. Short fragments are merged with the
    next sentence to avoid generating tiny audio clips.
    """
    # Split but keep the delimiter (punctuation) with the preceding text
    parts = _SENTENCE_RE.split(text)

    # _SENTENCE_RE.split produces: [text, punct, text, punct, ...]
    # Reassemble: join each text with its following punctuation
    raw_sentences = []
    i = 0
    while i < len(parts):
        sentence = parts[i].strip()
        # If next part is a single punctuation char, append it
        if i + 1 < len(parts) and len(parts[i + 1]) == 1:
            sentence += parts[i + 1]
            i += 2
        else:
            i += 1
        if sentence:
            raw_sentences.append(sentence)

    # Merge short fragments with the next sentence
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


# ---------------------------------------------------------------------------
# Audio cache: {sha256_hex: {"pcm": bytes, "ts": float}}
# Generation locks: {sha256_hex: asyncio.Event} — set when generation completes
# ---------------------------------------------------------------------------
AUDIO_CACHE: dict[str, dict] = {}
GENERATION_LOCKS: dict[str, asyncio.Event] = {}


def _purge_expired() -> None:
    """Remove cache entries older than CACHE_TTL."""
    now = time.time()
    expired = [k for k, v in AUDIO_CACHE.items() if now - v["ts"] > CACHE_TTL]
    for k in expired:
        del AUDIO_CACHE[k]
        _LOGGER.debug("Cache: purged expired entry %s", k[:8])


# ---------------------------------------------------------------------------
# Wyoming info
# ---------------------------------------------------------------------------
WYOMING_INFO = Info(tts=[TtsProgram(
    name="chatterbox",
    description="Chatterbox TTS",
    version=None,
    attribution=Attribution(name="Chatterbox", url=""),
    installed=True,
    voices=[TtsVoice(
        name="Sir_David",
        description="Sir David Attenborough",
        version=None,
        attribution=Attribution(name="Chatterbox", url=""),
        installed=True,
        languages=["en"],
    )],
)])


# ---------------------------------------------------------------------------
# PCM extraction from WAV bytes
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Call Chatterbox API → return PCM bytes (single text block)
# ---------------------------------------------------------------------------
async def generate_chatterbox(text: str) -> bytes:
    """Call Chatterbox TTS API and return raw PCM audio."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{CHATTERBOX_URL}/v1/audio/speech",
            json={
                "input": text,
                "voice": VOICE,
                "model": "chatterbox",
                "speed": 1.0,
            },
            timeout=90.0,
        )
        resp.raise_for_status()
        return extract_pcm(resp.content)


async def generate_chunked(text: str) -> bytes:
    """Split text into sentences, generate ALL in parallel, concatenate with gaps.

    Chatterbox generation time scales superlinearly with text length, so
    generating shorter sentences individually is faster than one big block.
    Generating them in parallel via asyncio.gather means total time equals
    the slowest sentence (~1-2s) instead of the sum of all sentences.
    A 150ms silence gap between sentences provides natural pacing.
    """
    sentences = split_sentences(text)

    if len(sentences) <= 1:
        # Single sentence — no benefit from chunking
        _LOGGER.info("Chunked: single sentence, generating directly")
        return await generate_chatterbox(text)

    _LOGGER.info("Chunked: split into %d sentences (parallel): %s",
                 len(sentences),
                 [s[:40] + "..." if len(s) > 40 else s for s in sentences])

    t0 = time.time()

    # Generate all sentences in parallel
    pcm_results = await asyncio.gather(
        *(generate_chatterbox(sentence) for sentence in sentences)
    )

    # Interleave PCM with silence gaps
    pcm_parts: list[bytes] = []
    for i, pcm in enumerate(pcm_results):
        pcm_parts.append(pcm)
        if i < len(pcm_results) - 1:
            pcm_parts.append(SENTENCE_GAP)

    total_pcm = b"".join(pcm_parts)
    total_time = time.time() - t0
    _LOGGER.info("Chunked: total generation %.1fs for %d sentences (%d bytes, parallel)",
                 total_time, len(sentences), len(total_pcm))
    return total_pcm


# ---------------------------------------------------------------------------
# Wyoming protocol handler
# ---------------------------------------------------------------------------
class ChatterboxHandler(AsyncEventHandler):
    """Handle Wyoming TTS events with cache-first lookup + generation lock awareness."""

    async def handle_event(self, event: Event) -> bool:
        _LOGGER.info("Event: %s", event.type)

        if Describe.is_type(event.type):
            await self.write_event(WYOMING_INFO.event())
            return True

        if Synthesize.is_type(event.type):
            synth = Synthesize.from_event(event)
            text = synth.text
            _LOGGER.info("Synthesize: %s", text[:80])

            # Purge expired entries on each request
            _purge_expired()

            cache_key = hashlib.sha256(text.encode()).hexdigest()

            try:
                # Check if pre-cache already completed
                cached = AUDIO_CACHE.get(cache_key)
                if cached:
                    pcm = cached["pcm"]
                    # Remove from cache (one-time use)
                    AUDIO_CACHE.pop(cache_key, None)
                    _LOGGER.info("Cache HIT (%s) — serving instantly", cache_key[:8])
                elif cache_key in GENERATION_LOCKS:
                    # Pre-cache is still generating — wait for it instead of
                    # starting a duplicate generation
                    _LOGGER.info("Cache PENDING (%s) — waiting for in-progress generation", cache_key[:8])
                    lock = GENERATION_LOCKS[cache_key]
                    await asyncio.wait_for(lock.wait(), timeout=90.0)
                    cached = AUDIO_CACHE.pop(cache_key, None)
                    if cached:
                        pcm = cached["pcm"]
                        _LOGGER.info("Cache HIT after wait (%s) — serving", cache_key[:8])
                    else:
                        _LOGGER.warning("Cache MISS after wait (%s) — generating fresh", cache_key[:8])
                        pcm = await generate_chunked(text)
                else:
                    _LOGGER.info("Cache MISS (%s) — calling Chatterbox", cache_key[:8])
                    pcm = await generate_chunked(text)

                await self.write_event(
                    AudioStart(
                        rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS
                    ).event()
                )
                offset = 0
                while offset < len(pcm):
                    chunk = pcm[offset : offset + CHUNK_SIZE]
                    await self.write_event(
                        AudioChunk(
                            rate=SAMPLE_RATE,
                            width=SAMPLE_WIDTH,
                            channels=CHANNELS,
                            audio=chunk,
                        ).event()
                    )
                    offset += CHUNK_SIZE
                await self.write_event(AudioStop().event())
                _LOGGER.info("Done")
            except Exception as e:
                _LOGGER.error("Failed: %s", e)
                await self.write_event(
                    AudioStart(
                        rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS
                    ).event()
                )
                await self.write_event(AudioStop().event())

            return True

        return True


# ---------------------------------------------------------------------------
# HTTP pre-cache endpoint
# ---------------------------------------------------------------------------
async def handle_precache(request: web.Request) -> web.Response:
    """POST /precache — pre-generate TTS and store in cache.

    Body: {"text": "...", "key": "<optional sha256>"}
    If key is omitted, it's computed from text.

    Uses parallel sentence chunking for faster generation.
    Sets a generation lock so Wyoming can wait for in-progress generation
    instead of starting a duplicate.
    """
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)

    text = data.get("text", "").strip()
    if not text:
        return web.json_response({"error": "missing text"}, status=400)

    key = data.get("key") or hashlib.sha256(text.encode()).hexdigest()

    # Check if already cached
    if key in AUDIO_CACHE:
        _LOGGER.info("Precache: already cached (%s)", key[:8])
        return web.json_response({"key": key, "cached": True, "source": "existing"})

    # Check if generation is already in progress (prevent duplicate work)
    if key in GENERATION_LOCKS:
        _LOGGER.info("Precache: generation already in progress (%s), waiting", key[:8])
        lock = GENERATION_LOCKS[key]
        await asyncio.wait_for(lock.wait(), timeout=90.0)
        return web.json_response({"key": key, "cached": True, "source": "existing"})

    # Set generation lock so Wyoming handler can wait instead of duplicating
    lock = asyncio.Event()
    GENERATION_LOCKS[key] = lock

    _LOGGER.info("Precache: generating audio for '%s' (%s)", text[:60], key[:8])
    try:
        pcm = await generate_chunked(text)
        AUDIO_CACHE[key] = {"pcm": pcm, "ts": time.time()}
        _LOGGER.info("Precache: cached %d bytes (%s)", len(pcm), key[:8])
        return web.json_response({"key": key, "cached": True, "source": "generated"})
    except Exception as e:
        _LOGGER.error("Precache: generation failed: %s", e)
        return web.json_response({"error": str(e)}, status=500)
    finally:
        # Signal any waiters and clean up the lock
        lock.set()
        GENERATION_LOCKS.pop(key, None)


async def handle_health(request: web.Request) -> web.Response:
    """GET /health — simple health check."""
    return web.json_response({
        "status": "ok",
        "cache_entries": len(AUDIO_CACHE),
        "pending_generations": len(GENERATION_LOCKS),
    })


# ---------------------------------------------------------------------------
# Main — run both servers
# ---------------------------------------------------------------------------
async def main() -> None:
    # Start HTTP server for pre-cache API
    app = web.Application()
    app.router.add_post("/precache", handle_precache)
    app.router.add_get("/health", handle_health)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", HTTP_PORT)
    await site.start()
    _LOGGER.info("HTTP pre-cache server running on port %d", HTTP_PORT)

    # Start Wyoming TCP server
    server = AsyncServer.from_uri(f"tcp://0.0.0.0:{WYOMING_PORT}")
    _LOGGER.info("Wyoming Chatterbox running on port %d", WYOMING_PORT)
    await server.run(ChatterboxHandler)


asyncio.run(main())
