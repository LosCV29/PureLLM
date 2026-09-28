"""HA-free tests for custom_components/purellm/tts.py (ElevenLabs streaming TTS).

Unit tests always run. The live test runs only when ELEVENLABS_API_KEY and
ELEVENLABS_VOICE_ID are set; it drives the real entity code against the API:

    ELEVENLABS_API_KEY=sk_... ELEVENLABS_VOICE_ID=... python -m unittest tests.test_elevenlabs_tts -v
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path
import sys
import time
import types
import unittest

PKG_DIR = Path(__file__).resolve().parents[1] / "custom_components" / "purellm"


def _load_tts():
    """Load tts.py with minimal stand-ins for the HA classes it imports."""
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    class TextToSpeechEntity:
        hass = None

        async def async_added_to_hass(self):
            pass

        async def async_will_remove_from_hass(self):
            pass

    class TTSAudioResponse:
        def __init__(self, extension, data_gen):
            self.extension, self.data_gen = extension, data_gen

    class TTSAudioRequest:
        def __init__(self, language, options, message_gen):
            self.language, self.options, self.message_gen = language, options, message_gen

    class Voice:
        def __init__(self, voice_id, name):
            self.voice_id, self.name = voice_id, name

    mod("homeassistant")
    mod("homeassistant.components")
    mod("homeassistant.components.tts", ATTR_VOICE="voice", TextToSpeechEntity=TextToSpeechEntity,
        TTSAudioRequest=TTSAudioRequest, TTSAudioResponse=TTSAudioResponse, TtsAudioType=tuple, Voice=Voice)
    mod("homeassistant.config_entries", ConfigEntry=object)
    mod("homeassistant.const", ATTR_MODEL="model")
    mod("homeassistant.core", HomeAssistant=object, callback=lambda f: f)
    mod("homeassistant.helpers")
    mod("homeassistant.helpers.entity_platform", AddEntitiesCallback=object)

    pkg = types.ModuleType("purellm_under_test")
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules["purellm_under_test"] = pkg
    for name in ("const", "tts"):
        spec = importlib.util.spec_from_file_location(f"purellm_under_test.{name}", PKG_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"purellm_under_test.{name}"] = module
        spec.loader.exec_module(module)
    return sys.modules["purellm_under_test.tts"], TTSAudioRequest


tts, TTSAudioRequest = _load_tts()


class TestTextHelpers(unittest.TestCase):
    def test_split_keeps_abbreviations(self):
        s, rest = tts.split_sentences("It is 5:52 p.m. on Thursday. Rain soon. Tail")
        self.assertEqual(s, ["It is 5:52 p.m. on Thursday.", "Rain soon."])
        self.assertEqual(rest, "Tail")

    def test_split_needs_whitespace_after_period(self):
        s, rest = tts.split_sentences("The shade is at 45.5 percent")
        self.assertEqual(s, [])
        self.assertEqual(rest, "The shade is at 45.5 percent")

    def test_fahrenheit_never_spoken(self):
        self.assertEqual(tts.normalize_tts_text("It's 85°F and 97 degrees Fahrenheit."),
                         "It's 85 degrees and 97 degrees.")

    def test_schedule_first_sentence_alone(self):
        schedule = list(tts.CHUNK_SCHEDULE)
        chunks = tts.schedule_chunks(["A.", "B.", "C.", "D.", "E."], schedule)
        self.assertEqual(chunks, ["A.", "B. C.", "D. E."])


class _FakeHass:
    def async_create_background_task(self, coro, name=None):
        return asyncio.ensure_future(coro)


class _FakeEntry:
    def __init__(self, options):
        self.entry_id = "test"
        self.data = {}
        self.options = options


@unittest.skipUnless(os.environ.get("ELEVENLABS_API_KEY") and os.environ.get("ELEVENLABS_VOICE_ID"), "live key not set")
class TestLiveStreaming(unittest.IsolatedAsyncioTestCase):
    REPLY = ("It's currently 89°F and feels like 95 with light rain in Pembroke Pines, Florida. "
             "Raining now, continuing for at least the next hour. Today's high is 91 degrees with a low of 78.")

    async def asyncSetUp(self):
        self.entity = tts.PureLLMElevenLabsTTS(_FakeEntry({
            "elevenlabs_api_key": os.environ["ELEVENLABS_API_KEY"],
            "elevenlabs_voice_id": os.environ["ELEVENLABS_VOICE_ID"],
            "elevenlabs_model": os.environ.get("ELEVENLABS_MODEL", "eleven_v4_turbo"),
        }))
        self.entity.hass = _FakeHass()
        await self.entity.async_added_to_hass()
        await asyncio.sleep(1.5)  # let keep-warm open the connection

    async def asyncTearDown(self):
        await self.entity.async_will_remove_from_hass()

    async def _run(self, streaming: bool):
        self.entity._entry.options["elevenlabs_sentence_streaming"] = streaming

        async def gen():
            yield self.REPLY

        response = await self.entity.async_stream_tts_audio(TTSAudioRequest("en", {}, gen()))
        self.assertEqual(response.extension, "mp3")
        t0 = time.monotonic()
        first, size = None, 0
        async for data in response.data_gen:
            if first is None:
                first = time.monotonic() - t0
            size += len(data)
        return first, time.monotonic() - t0, size

    async def test_stream_vs_whole(self):
        results = {}
        for streaming in (True, False, True, False):
            first, total, size = await self._run(streaming)
            results.setdefault(streaming, []).append(first)
            print(f"\n  sentence_streaming={streaming!s:5}  first_audio={first*1000:.0f}ms  total={total:.2f}s  {size} bytes")
            self.assertGreater(size, 20000)
        print(f"  best first_audio: streaming={min(results[True])*1000:.0f}ms whole={min(results[False])*1000:.0f}ms")
        self.assertEqual(self.entity._no_request_ids, set(), "model rejected previous_request_ids")


if __name__ == "__main__":
    unittest.main()
