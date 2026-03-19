"""ElevenLabs TTS platform for PureLLM.

Registers a TTS entity (tts.purellm_elevenlabs) that calls the ElevenLabs API
directly, exposing every voice parameter including speed. Select this entity
as the TTS engine in your Assist pipeline for full control.
"""
from __future__ import annotations

import logging
from typing import Any

import aiohttp

from homeassistant.components.tts import (
    TextToSpeechEntity,
    TtsAudioType,
    ATTR_VOICE,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    DOMAIN,
    CONF_ELEVENLABS_API_KEY,
    CONF_ELEVENLABS_VOICE_ID,
    CONF_ELEVENLABS_MODEL,
    CONF_ELEVENLABS_STABILITY,
    CONF_ELEVENLABS_SIMILARITY,
    CONF_ELEVENLABS_STYLE,
    CONF_ELEVENLABS_SPEAKER_BOOST,
    CONF_ELEVENLABS_SPEED,
    CONF_ELEVENLABS_OUTPUT_FORMAT,
    CONF_ELEVENLABS_TEXT_NORMALIZATION,
    DEFAULT_ELEVENLABS_API_KEY,
    DEFAULT_ELEVENLABS_VOICE_ID,
    DEFAULT_ELEVENLABS_MODEL,
    DEFAULT_ELEVENLABS_STABILITY,
    DEFAULT_ELEVENLABS_SIMILARITY,
    DEFAULT_ELEVENLABS_STYLE,
    DEFAULT_ELEVENLABS_SPEAKER_BOOST,
    DEFAULT_ELEVENLABS_SPEED,
    DEFAULT_ELEVENLABS_OUTPUT_FORMAT,
    DEFAULT_ELEVENLABS_TEXT_NORMALIZATION,
)

_LOGGER = logging.getLogger(__name__)

# Map output format prefixes to ffmpeg-compatible format names.
# HA's TTS system passes this value as the `-f` flag to ffmpeg,
# so it must be a format name (e.g. "mp3"), NOT a MIME type.
_FORMAT_TO_CONTENT_TYPE = {
    "mp3": "mp3",
    "pcm": "s16le",
    "ulaw": "mulaw",
}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up ElevenLabs TTS from a PureLLM config entry."""
    config = {**entry.data, **entry.options}
    api_key = config.get(CONF_ELEVENLABS_API_KEY, DEFAULT_ELEVENLABS_API_KEY)

    # Only add the TTS entity if an API key is configured
    if not api_key:
        _LOGGER.debug("ElevenLabs TTS: No API key configured, skipping TTS entity")
        return

    async_add_entities([PureLLMElevenLabsTTS(entry)], True)


class PureLLMElevenLabsTTS(TextToSpeechEntity):
    """ElevenLabs TTS entity with full parameter control."""

    _attr_has_entity_name = True
    _attr_name = "ElevenLabs TTS"

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the ElevenLabs TTS entity."""
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_elevenlabs_tts"

    @property
    def _config(self) -> dict[str, Any]:
        """Get merged config from entry data and options."""
        return {**self._entry.data, **self._entry.options}

    @property
    def default_language(self) -> str:
        """Return the default language."""
        return "en"

    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported languages.

        ElevenLabs multilingual models support many languages.
        We list common ones; the API handles any valid language code.
        """
        return [
            "en", "es", "fr", "de", "it", "pt", "pl", "hi", "ar",
            "cs", "nl", "fi", "el", "hu", "id", "ja", "ko", "ms",
            "no", "ro", "ru", "sk", "sv", "sw", "ta", "th", "tr",
            "uk", "ur", "vi", "zh",
        ]

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Expose current voice settings as state attributes for easy debugging."""
        config = self._config
        return {
            "voice_id": config.get(CONF_ELEVENLABS_VOICE_ID, DEFAULT_ELEVENLABS_VOICE_ID),
            "model": config.get(CONF_ELEVENLABS_MODEL, DEFAULT_ELEVENLABS_MODEL),
            "stability": config.get(CONF_ELEVENLABS_STABILITY, DEFAULT_ELEVENLABS_STABILITY),
            "similarity_boost": config.get(CONF_ELEVENLABS_SIMILARITY, DEFAULT_ELEVENLABS_SIMILARITY),
            "style": config.get(CONF_ELEVENLABS_STYLE, DEFAULT_ELEVENLABS_STYLE),
            "use_speaker_boost": config.get(CONF_ELEVENLABS_SPEAKER_BOOST, DEFAULT_ELEVENLABS_SPEAKER_BOOST),
            "speed": config.get(CONF_ELEVENLABS_SPEED, DEFAULT_ELEVENLABS_SPEED),
            "output_format": config.get(CONF_ELEVENLABS_OUTPUT_FORMAT, DEFAULT_ELEVENLABS_OUTPUT_FORMAT),
            "text_normalization": config.get(CONF_ELEVENLABS_TEXT_NORMALIZATION, DEFAULT_ELEVENLABS_TEXT_NORMALIZATION),
        }

    async def async_get_tts_audio(
        self,
        message: str,
        language: str,
        options: dict[str, Any] | None = None,
    ) -> TtsAudioType:
        """Synthesize speech via ElevenLabs API.

        Calls POST /v1/text-to-speech/{voice_id} with all configured
        voice_settings parameters including speed.
        """
        config = self._config
        api_key = config.get(CONF_ELEVENLABS_API_KEY, DEFAULT_ELEVENLABS_API_KEY)
        voice_id = config.get(CONF_ELEVENLABS_VOICE_ID, DEFAULT_ELEVENLABS_VOICE_ID)

        if not api_key or not voice_id:
            _LOGGER.error("ElevenLabs TTS: API key or voice ID not configured")
            return (None, None)

        # Allow runtime voice override via options (e.g., from service call)
        if options and ATTR_VOICE in options:
            voice_id = options[ATTR_VOICE]

        model_id = config.get(CONF_ELEVENLABS_MODEL, DEFAULT_ELEVENLABS_MODEL)
        output_format = config.get(CONF_ELEVENLABS_OUTPUT_FORMAT, DEFAULT_ELEVENLABS_OUTPUT_FORMAT)

        # Build voice_settings with every parameter
        # Clamp speed to API-enforced range (0.7–1.2) regardless of stored value
        raw_speed = float(config.get(CONF_ELEVENLABS_SPEED, DEFAULT_ELEVENLABS_SPEED))
        speed = max(0.7, min(1.2, raw_speed))

        voice_settings = {
            "stability": float(config.get(CONF_ELEVENLABS_STABILITY, DEFAULT_ELEVENLABS_STABILITY)),
            "similarity_boost": float(config.get(CONF_ELEVENLABS_SIMILARITY, DEFAULT_ELEVENLABS_SIMILARITY)),
            "style": float(config.get(CONF_ELEVENLABS_STYLE, DEFAULT_ELEVENLABS_STYLE)),
            "use_speaker_boost": bool(config.get(CONF_ELEVENLABS_SPEAKER_BOOST, DEFAULT_ELEVENLABS_SPEAKER_BOOST)),
        }

        text_normalization = config.get(
            CONF_ELEVENLABS_TEXT_NORMALIZATION, DEFAULT_ELEVENLABS_TEXT_NORMALIZATION
        )

        payload = {
            "text": message,
            "model_id": model_id,
            "voice_settings": voice_settings,
            "apply_text_normalization": text_normalization,
            "speed": speed,
        }

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format={output_format}"

        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        }

        _LOGGER.debug(
            "ElevenLabs TTS: voice=%s model=%s speed=%.2f stability=%.2f similarity=%.2f style=%.2f boost=%s format=%s",
            voice_id, model_id,
            voice_settings["speed"],
            voice_settings["stability"],
            voice_settings["similarity_boost"],
            voice_settings["style"],
            voice_settings["use_speaker_boost"],
            output_format,
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        _LOGGER.error(
                            "ElevenLabs TTS failed (HTTP %s): %s",
                            response.status, error_text,
                        )
                        return (None, None)

                    audio_data = await response.read()

                    # Determine content type from output format
                    format_prefix = output_format.split("_")[0]
                    content_type = _FORMAT_TO_CONTENT_TYPE.get(format_prefix, "audio/mpeg")

                    _LOGGER.debug(
                        "ElevenLabs TTS: received %d bytes (%s)",
                        len(audio_data), content_type,
                    )
                    return (content_type, audio_data)

        except aiohttp.ClientError as err:
            _LOGGER.error("ElevenLabs TTS request failed: %s", err)
            return (None, None)
        except TimeoutError:
            _LOGGER.error("ElevenLabs TTS request timed out")
            return (None, None)
