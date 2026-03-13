#!/usr/bin/env python3
"""Wyoming Chatterbox TTS server with pre-cache HTTP bridge.

Runs TWO servers:
  1. Wyoming TCP server on port 10201 — speaks the Wyoming protocol for HA
  2. HTTP server on port 10202 — accepts POST /precache from PureLLM

Flow:
  PureLLM gets LLM response text
    → POST /precache {"text": "...", "key": "<sha256>"} to port 10202
    → This server calls Chatterbox, stores PCM keyed by hash
    → PureLLM returns text to HA pipeline
    → HA asks Wyoming (port 10201) for TTS on same text
    → Wyoming handler: cache HIT → responds in <100ms
    → PE gets audio instantly, LED stays on, no gap

Cache entries expire after 120s to prevent memory leaks from orphaned entries.
"""
import asyncio
import hashlib
import io
import logging
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

# ---------------------------------------------------------------------------
# Audio cache: {sha256_hex: {"pcm": bytes, "ts": float}}
# ---------------------------------------------------------------------------
AUDIO_CACHE: dict[str, dict] = {}


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
# Call Chatterbox API → return PCM bytes
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


# ---------------------------------------------------------------------------
# Wyoming protocol handler
# ---------------------------------------------------------------------------
class ChatterboxHandler(AsyncEventHandler):
    """Handle Wyoming TTS events with cache-first lookup."""

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

            # Check cache
            cache_key = hashlib.sha256(text.encode()).hexdigest()
            cached = AUDIO_CACHE.pop(cache_key, None)

            try:
                if cached:
                    pcm = cached["pcm"]
                    _LOGGER.info("Cache HIT (%s) — serving instantly", cache_key[:8])
                else:
                    _LOGGER.info("Cache MISS (%s) — calling Chatterbox", cache_key[:8])
                    pcm = await generate_chatterbox(text)

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

    _LOGGER.info("Precache: generating audio for '%s' (%s)", text[:60], key[:8])
    try:
        pcm = await generate_chatterbox(text)
        AUDIO_CACHE[key] = {"pcm": pcm, "ts": time.time()}
        _LOGGER.info("Precache: cached %d bytes (%s)", len(pcm), key[:8])
        return web.json_response({"key": key, "cached": True, "source": "generated"})
    except Exception as e:
        _LOGGER.error("Precache: generation failed: %s", e)
        return web.json_response({"error": str(e)}, status=500)


async def handle_health(request: web.Request) -> web.Response:
    """GET /health — simple health check."""
    return web.json_response({
        "status": "ok",
        "cache_entries": len(AUDIO_CACHE),
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
