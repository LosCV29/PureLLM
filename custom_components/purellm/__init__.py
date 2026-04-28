"""The PureLLM integration."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

import voluptuous as vol
from aiohttp import web

from homeassistant.components.http import HomeAssistantView
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, Event, SupportsResponse, callback, ServiceCall
from homeassistant.helpers import config_validation as cv

from .const import DOMAIN
from .tools.timer import get_registered_timer, unregister_timer

_LOGGER = logging.getLogger(__name__)

# Service constants
SERVICE_ASK_AND_ACT = "ask_and_act"
ASK_AND_ACT_MARKER = "<!-- ASK_AND_ACT_DATA:"
ASK_AND_ACT_MARKER_END = " -->"

# Schema for ask_and_act service
ANSWER_SCHEMA = vol.Schema({
    vol.Required("id"): cv.string,
    vol.Required("sentences"): vol.All(cv.ensure_list, [cv.string]),
    vol.Optional("action"): vol.Schema({
        vol.Required("service"): cv.string,
        vol.Optional("target"): dict,
        vol.Optional("data"): dict,
    }),
    vol.Optional("response"): cv.string,
})

ASK_AND_ACT_SCHEMA = vol.Schema({
    vol.Required("satellite_entity_id"): cv.entity_id,
    vol.Required("question"): cv.string,
    vol.Required("answers"): vol.All(cv.ensure_list, [ANSWER_SCHEMA]),
    vol.Optional("tts_entity_id"): cv.entity_id,  # Optional, auto-detected from preferred pipeline
})

PLATFORMS: list[Platform] = [Platform.CONVERSATION, Platform.TTS, Platform.UPDATE]

# Key for storing service registration
SERVICE_REGISTERED_KEY = "ask_and_act_service_registered"

# Key for storing pending ask_and_act context (fallback when start_conversation
# fails due to pipeline issues like missing wake word provider).
# Maps satellite_entity_id -> {"extra_system_prompt": str, "timestamp": float}
PENDING_ASK_AND_ACT_KEY = "pending_ask_and_act"

# Key for storing the timer listener unsub function
TIMER_LISTENER_KEY = "timer_finished_listener"

SNAPSHOT_VIEW_KEY = "snapshot_view_registered"
SILENCE_WAV_KEY = "silence_wav_written"
SILENCE_WAV_FILENAME = "purellm_silence.wav"

# Media source URI for the silent WAV used by start_conversation.
# Resolves to <config_dir>/media/purellm_silence.wav via HA's local media source.
SILENCE_MEDIA_ID = "media-source://media_source/local/purellm_silence.wav"


# Timeout for pending ask_and_act context (seconds). If the user doesn't
# respond within this window, the stored context is discarded.
PENDING_ASK_AND_ACT_TIMEOUT = 60.0


def store_pending_ask_and_act(
    hass: HomeAssistant,
    satellite_entity_id: str,
    extra_system_prompt: str,
) -> None:
    """Store ask_and_act context so the conversation agent can pick it up.

    This is the fallback path: if start_conversation fails (e.g. due to a
    missing wake word provider), the user can still respond by pressing the
    satellite button or saying the wake word.  The PureLLM conversation agent
    checks for pending context and injects it as extra_system_prompt.
    """
    hass.data[DOMAIN].setdefault(PENDING_ASK_AND_ACT_KEY, {})[satellite_entity_id] = {
        "extra_system_prompt": extra_system_prompt,
        "timestamp": time.time(),
    }
    _LOGGER.debug(
        "ask_and_act: stored pending context for %s", satellite_entity_id,
    )


def consume_pending_ask_and_act(
    hass: HomeAssistant,
    device_id: str | None,
) -> str | None:
    """Consume and return pending ask_and_act context for a device.

    Called by the conversation agent when processing a new message.
    Maps device_id -> satellite entity, then pops the stored context.
    Returns the extra_system_prompt if found and not expired, else None.
    """
    pending = hass.data.get(DOMAIN, {}).get(PENDING_ASK_AND_ACT_KEY)
    if not pending or not device_id:
        return None

    # Map device_id to satellite_entity_id by checking the entity registry
    from homeassistant.helpers import entity_registry as er
    ent_reg = er.async_get(hass)

    # Find all assist_satellite entities belonging to this device
    for entity in er.async_entries_for_device(ent_reg, device_id):
        if entity.entity_id in pending:
            entry = pending.pop(entity.entity_id)
            elapsed = time.time() - entry["timestamp"]
            if elapsed > PENDING_ASK_AND_ACT_TIMEOUT:
                _LOGGER.debug(
                    "ask_and_act: pending context for %s expired (%.0fs)",
                    entity.entity_id, elapsed,
                )
                return None
            _LOGGER.info(
                "ask_and_act: consumed pending context for %s (waited %.1fs)",
                entity.entity_id, elapsed,
            )
            return entry["extra_system_prompt"]

    return None


def _generate_silent_wav() -> bytes:
    """Generate a minimal 0.5 second silent WAV file (16 kHz, 16-bit, mono).

    Used as start_media_id for start_conversation to skip TTS synthesis
    (which crashes with some TTS engines on empty strings) while still
    satisfying the 'must contain start_message or start_media_id' validation.
    """
    import struct
    sample_rate = 16000
    num_channels = 1
    bits_per_sample = 16
    duration_samples = sample_rate // 2  # 0.5 seconds
    data_size = duration_samples * num_channels * (bits_per_sample // 8)
    # RIFF header
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,  # file size - 8
        b"WAVE",
        b"fmt ",
        16,  # chunk size
        1,  # PCM format
        num_channels,
        sample_rate,
        sample_rate * num_channels * (bits_per_sample // 8),  # byte rate
        num_channels * (bits_per_sample // 8),  # block align
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + b"\x00" * data_size


# Cache the silent WAV so we don't regenerate it every time.
_SILENT_WAV: bytes | None = None


def _get_silent_wav() -> bytes:
    global _SILENT_WAV
    if _SILENT_WAV is None:
        _SILENT_WAV = _generate_silent_wav()
    return _SILENT_WAV


async def _ensure_silence_wav(hass: HomeAssistant) -> None:
    """Ensure the silent WAV file exists in the media directory.

    Called before each start_conversation to guard against the file being
    missing (deleted, never written, tmpfs cleared, etc.).  Skips the write
    if the file already exists on disk.
    """
    media_dir = hass.config.path("media")
    silence_path = os.path.join(media_dir, SILENCE_WAV_FILENAME)

    def _check_and_write() -> None:
        if os.path.isfile(silence_path):
            return
        os.makedirs(media_dir, exist_ok=True)
        with open(silence_path, "wb") as f:
            f.write(_get_silent_wav())

    try:
        await hass.async_add_executor_job(_check_and_write)
    except OSError as err:
        _LOGGER.warning("ask_and_act: could not ensure silent WAV at %s: %s", silence_path, err)


class PureLLMSilentWavView(HomeAssistantView):
    """Serve a minimal silent WAV file for start_conversation.

    start_conversation requires start_message or start_media_id.  When the
    question is already spoken via separate tts.speak, we use this silent WAV
    as start_media_id to skip TTS synthesis and go straight to listening.
    """

    url = "/api/purellm/silence.wav"
    name = "api:purellm:silence"
    requires_auth = False

    async def get(self, request: web.Request) -> web.Response:
        return web.Response(body=_get_silent_wav(), content_type="audio/wav")


class PureLLMSnapshotView(HomeAssistantView):
    """Serve camera snapshots saved by the camera tool.

    The companion app fetches notification images through HA's authenticated
    API, so this endpoint works reliably on-LAN and off-LAN (via Nabu Casa
    or external URL).  The /local/ static path only works if the www/
    directory existed at HA startup, which is not guaranteed.
    """

    url = "/api/purellm/camera/{camera_name}/snapshot.jpg"
    name = "api:purellm:camera:snapshot"
    requires_auth = False

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass

    async def get(self, request: web.Request, camera_name: str) -> web.Response:
        """Return the latest saved snapshot for *camera_name*."""
        # Sanitise to prevent path-traversal
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "", camera_name)
        filepath = self._hass.config.path("www", "purellm", f"camera_{safe_name}.jpg")

        if not os.path.isfile(filepath):
            return web.Response(status=404, text="Snapshot not found")

        with open(filepath, "rb") as fh:
            data = fh.read()
        return web.Response(body=data, content_type="image/jpeg")


async def _wait_for_media_player_idle(
    hass: HomeAssistant,
    media_player_entity_id: str,
    timeout: float = 15.0,
    poll_interval: float = 0.2,
) -> None:
    """Wait for a media player to finish playing and return to idle.

    After tts.speak returns, the audio is still playing on the media player.
    This polls the media player state until it is no longer 'playing' or
    until the timeout is reached.
    """
    elapsed = 0.0
    while elapsed < timeout:
        state = hass.states.get(media_player_entity_id)
        if state is None:
            _LOGGER.warning(
                "ask_and_act: media_player %s not found, proceeding",
                media_player_entity_id,
            )
            return
        if state.state not in ("playing", "buffering"):
            _LOGGER.debug(
                "ask_and_act: media_player %s is %s, proceeding",
                media_player_entity_id, state.state,
            )
            return
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    _LOGGER.warning(
        "ask_and_act: timed out waiting for media_player %s to finish (state=%s)",
        media_player_entity_id,
        hass.states.get(media_player_entity_id).state if hass.states.get(media_player_entity_id) else "unknown",
    )


async def async_setup(hass: HomeAssistant, config: dict[str, Any]) -> bool:
    """Set up the PureLLM component."""
    hass.data.setdefault(DOMAIN, {})
    return True


async def _wait_for_satellite_idle(
    hass: HomeAssistant,
    satellite_entity_id: str,
    timeout: float = 10.0,
    settle_delay: float = 0.5,
) -> None:
    """Wait for the assist satellite to return to idle before starting a new conversation.

    After TTS playback the satellite pipeline may still be winding down
    (e.g. state 'responding'). Calling start_conversation on a non-idle
    satellite is silently ignored, so we must wait.

    Uses event listeners instead of polling for reliable detection, and adds
    a settle delay after idle is reached because the satellite's internal
    pipeline teardown may still be in progress even after the entity state
    flips to 'idle'.
    """
    # Fast path: already idle
    state = hass.states.get(satellite_entity_id)
    if state is None:
        _LOGGER.warning(
            "ask_and_act: satellite %s not found, proceeding",
            satellite_entity_id,
        )
        return
    if state.state == "idle":
        _LOGGER.debug(
            "ask_and_act: satellite %s already idle, settling %.1fs",
            satellite_entity_id, settle_delay,
        )
        await asyncio.sleep(settle_delay)
        return

    # Subscribe to state_changed events for reliable detection
    idle_event = asyncio.Event()

    @callback
    def _on_state_change(event: Event) -> None:
        new_state = event.data.get("new_state")
        if (
            event.data.get("entity_id") == satellite_entity_id
            and new_state is not None
            and new_state.state == "idle"
        ):
            idle_event.set()

    unsub = hass.bus.async_listen("state_changed", _on_state_change)
    try:
        # Re-check after subscribing to avoid TOCTOU race
        state = hass.states.get(satellite_entity_id)
        if state and state.state == "idle":
            _LOGGER.debug(
                "ask_and_act: satellite %s became idle (re-check), settling %.1fs",
                satellite_entity_id, settle_delay,
            )
            await asyncio.sleep(settle_delay)
            return

        _LOGGER.debug(
            "ask_and_act: satellite %s in state '%s', waiting for idle…",
            satellite_entity_id, state.state if state else "unknown",
        )
        async with asyncio.timeout(timeout):
            await idle_event.wait()
        _LOGGER.debug(
            "ask_and_act: satellite %s reached idle, settling %.1fs",
            satellite_entity_id, settle_delay,
        )
        await asyncio.sleep(settle_delay)
    except TimeoutError:
        _LOGGER.warning(
            "ask_and_act: timed out waiting for satellite %s to become idle (state=%s)",
            satellite_entity_id,
            hass.states.get(satellite_entity_id).state if hass.states.get(satellite_entity_id) else "unknown",
        )
    finally:
        unsub()


def _resolve_tts_entity(hass: HomeAssistant) -> str | None:
    """Resolve the TTS entity from the preferred assist pipeline.

    Queries the preferred voice pipeline for its configured TTS engine
    and returns the matching entity_id (e.g. 'tts.kokoro').
    Returns None if the pipeline or TTS engine cannot be determined.
    """
    try:
        from homeassistant.components.assist_pipeline import async_get_pipeline

        # async_get_pipeline(hass, None) returns the preferred pipeline
        pipeline = async_get_pipeline(hass, pipeline_id=None)
        if pipeline is None or not pipeline.tts_engine:
            return None

        tts_engine = pipeline.tts_engine
        # The tts_engine may already be a full entity_id (tts.kokoro)
        # or just the engine name (cloud, kokoro, etc.)
        if hass.states.get(tts_engine):
            _LOGGER.debug("Resolved TTS entity from preferred pipeline: %s", tts_engine)
            return tts_engine

        candidate = f"tts.{tts_engine}" if not tts_engine.startswith("tts.") else tts_engine
        if hass.states.get(candidate):
            _LOGGER.debug("Resolved TTS entity from preferred pipeline: %s", candidate)
            return candidate

        _LOGGER.debug("Pipeline TTS engine '%s' has no matching entity", tts_engine)
    except Exception as err:  # noqa: BLE001
        _LOGGER.debug("Could not resolve TTS from pipeline: %s", err)
    return None


async def async_handle_ask_and_act(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Handle the ask_and_act service call.

    Flow (matches the original working approach):
    1. Speak the question via separate tts.speak
    2. Wait for media player to finish playing
    3. Wait for satellite to be idle
    4. start_conversation with a silent WAV as start_media_id (skips TTS
       synthesis — avoids crash with TTS engines that can't handle empty text)
       + extra_system_prompt with answer context
    5. Satellite enters listening mode, user responds, LLM processes

    Also stores context as fallback: if start_conversation fails, the user
    can respond via button/wake word and PureLLM picks up the stored context.
    """
    satellite_entity_id = call.data["satellite_entity_id"]
    question = call.data["question"]
    answers = call.data["answers"]
    tts_entity_id = call.data.get("tts_entity_id") or _resolve_tts_entity(hass) or "tts.home_assistant_cloud"

    # Derive media_player from satellite entity ID
    satellite_suffix = satellite_entity_id.split(".")[-1]
    if "_assist_satellite" in satellite_suffix:
        media_player_suffix = satellite_suffix.replace("_assist_satellite", "_media_player")
        media_player_entity_id = f"media_player.{media_player_suffix}"
    else:
        media_player_entity_id = satellite_entity_id.replace(
            "assist_satellite.", "media_player."
        ).replace("_assist_satellite", "_media_player")

    _LOGGER.info("ask_and_act: Starting - question='%s', satellite=%s, media_player=%s",
                 question, satellite_entity_id, media_player_entity_id)

    # Build the extra_system_prompt with embedded answers data.
    answers_json = json.dumps(answers)

    prompt_parts = [
        f"{ASK_AND_ACT_MARKER}{answers_json}{ASK_AND_ACT_MARKER_END}",
        "",
        f"The user was just asked: \"{question}\"",
        "",
        "Based on the user's response, reply with the appropriate short answer.",
        "",
    ]

    for answer in answers:
        sentences = answer["sentences"]
        sentences_str = ", ".join(f'"{s}"' for s in sentences)

        if "response" in answer:
            prompt_parts.append(f"If user says {sentences_str} or similar:")
            prompt_parts.append(f"  -> Say: \"{answer['response']}\"")
        else:
            prompt_parts.append(f"If user says {sentences_str} or similar:")
            prompt_parts.append(f"  -> Acknowledge briefly.")

        prompt_parts.append("")

    prompt_parts.extend([
        "RULES:",
        "- Keep your response very short (one or two words).",
        "- Do NOT call any tools. The action will be handled automatically.",
        "- If the user's reply does not clearly match one of the answers above,",
        "  respond with exactly: \"Sorry, I didn't catch that.\" so the user",
        "  knows the action did NOT run.",
    ])

    extra_system_prompt = "\n".join(prompt_parts)
    _LOGGER.debug("ask_and_act: Generated prompt:\n%s", extra_system_prompt)

    # Store context as fallback — if start_conversation's pipeline fails,
    # the user can still respond via button/wake word and PureLLM's
    # conversation agent picks up the stored context automatically.
    store_pending_ask_and_act(hass, satellite_entity_id, extra_system_prompt)

    # Step 1: Speak the question via TTS
    try:
        await hass.services.async_call(
            "tts", "speak",
            {
                "entity_id": tts_entity_id,
                "media_player_entity_id": media_player_entity_id,
                "message": question,
            },
            blocking=True,
        )
        _LOGGER.debug("ask_and_act: TTS spoke question")
    except Exception as err:
        _LOGGER.error("ask_and_act: TTS failed: %s", err)
        return {"error": f"TTS failed: {err}"}

    # Step 2: Wait for TTS audio to finish playing on the media player.
    await _wait_for_media_player_idle(hass, media_player_entity_id)

    # Step 3: Wait for the satellite pipeline to fully finish and return to idle.
    await _wait_for_satellite_idle(hass, satellite_entity_id)

    # Step 4: start_conversation to enter listening mode.
    # Use start_media_id with a silent WAV (via HA media source) instead of
    # start_message to avoid TTS synthesis — the question was already spoken
    # in step 1.  The old code used start_message="" which worked with HA Cloud
    # TTS but crashes openai-streaming (can't synthesize empty text).
    await _ensure_silence_wav(hass)
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            await hass.services.async_call(
                "assist_satellite", "start_conversation",
                {
                    "entity_id": satellite_entity_id,
                    "start_media_id": SILENCE_MEDIA_ID,
                    "extra_system_prompt": extra_system_prompt,
                    "preannounce": False,
                },
                blocking=True,
            )
        except Exception as err:
            _LOGGER.error("ask_and_act: start_conversation failed: %s", err)
            return {"error": f"start_conversation failed: {err}"}

        # Verify the satellite actually entered listening mode
        await asyncio.sleep(0.3)
        state = hass.states.get(satellite_entity_id)
        if state and state.state in ("listening", "processing"):
            _LOGGER.info(
                "ask_and_act: start_conversation succeeded (attempt %d), satellite is %s",
                attempt, state.state,
            )
            return {"success": True}

        if attempt < max_attempts:
            _LOGGER.warning(
                "ask_and_act: satellite still '%s' after start_conversation (attempt %d/%d), retrying…",
                state.state if state else "unknown", attempt, max_attempts,
            )
            await asyncio.sleep(0.5)
        else:
            _LOGGER.warning(
                "ask_and_act: satellite state is '%s' after %d attempts — may not be listening",
                state.state if state else "unknown", max_attempts,
            )

    return {"success": True}


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate old entry to current version."""
    _LOGGER.info("Migrating PureLLM from version %s", entry.version)

    if entry.version < 2:
        hass.config_entries.async_update_entry(entry, version=2)
        _LOGGER.info("Migration to version 2 successful")

    return True


@callback
def _handle_timer_finished(hass: HomeAssistant, event: Event) -> None:
    """Handle timer.finished event and announce via TTS.

    Note: This callback may be invoked from a sync worker thread,
    so we use hass.add_job for thread-safe task scheduling.
    """
    entity_id = event.data.get("entity_id")
    if not entity_id:
        return

    # Check if this timer was started by PureLLM
    timer_info = get_registered_timer(hass, entity_id)
    if not timer_info:
        _LOGGER.debug("Timer %s finished but was not started by PureLLM", entity_id)
        return

    timer_name = timer_info.get("name", "Timer")
    announce_player = timer_info.get("announce_player")

    _LOGGER.info("PureLLM timer finished: %s -> announcing on %s",
                 timer_name, announce_player or "default")

    # Unregister the timer
    unregister_timer(hass, entity_id)

    # Create announcement message
    if timer_name.lower() in ("timer", "kitchen timer", "general timer"):
        message = "Your timer is done!"
    else:
        message = f"Your {timer_name} timer is done!"

    # Use thread-safe method to schedule the async task
    hass.add_job(_announce_timer_finished(hass, message, announce_player))


async def _announce_timer_finished(
    hass: HomeAssistant,
    message: str,
    target_player: str | None
) -> None:
    """Announce timer completion via TTS or notification."""
    announced = False

    # Try TTS first
    if target_player and hass.services.has_service("tts", "speak"):
        try:
            # Get available TTS engines
            tts_entities = [
                s.entity_id for s in hass.states.async_all()
                if s.entity_id.startswith("tts.")
            ]

            if tts_entities:
                await hass.services.async_call(
                    "tts", "speak",
                    {
                        "entity_id": tts_entities[0],
                        "media_player_entity_id": target_player,
                        "message": message,
                    },
                    blocking=False
                )
                announced = True
                _LOGGER.debug("Announced timer via tts.speak on %s", target_player)
        except Exception as err:
            _LOGGER.warning("TTS speak failed: %s", err)

    # Fallback: try media_player.play_media with TTS URL
    if not announced and target_player and hass.services.has_service("tts", "google_translate_say"):
        try:
            await hass.services.async_call(
                "tts", "google_translate_say",
                {
                    "entity_id": target_player,
                    "message": message,
                },
                blocking=False
            )
            announced = True
            _LOGGER.debug("Announced timer via google_translate_say on %s", target_player)
        except Exception as err:
            _LOGGER.debug("google_translate_say failed: %s", err)

    # Fallback: try cloud say
    if not announced and target_player and hass.services.has_service("tts", "cloud_say"):
        try:
            await hass.services.async_call(
                "tts", "cloud_say",
                {
                    "entity_id": target_player,
                    "message": message,
                },
                blocking=False
            )
            announced = True
            _LOGGER.debug("Announced timer via cloud_say on %s", target_player)
        except Exception as err:
            _LOGGER.debug("cloud_say failed: %s", err)

    # Last resort: persistent notification
    if not announced:
        await hass.services.async_call(
            "persistent_notification", "create",
            {
                "title": "Timer Finished",
                "message": message,
                "notification_id": "purellm_timer",
            },
            blocking=False
        )
        _LOGGER.debug("Created persistent notification for timer (no TTS available)")


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up PureLLM from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    config = {**entry.data, **entry.options}
    hass.data[DOMAIN][entry.entry_id] = {"config": config}

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register timer.finished event listener (only once per HA instance)
    if TIMER_LISTENER_KEY not in hass.data[DOMAIN]:
        unsub = hass.bus.async_listen(
            "timer.finished",
            lambda event: _handle_timer_finished(hass, event)
        )
        hass.data[DOMAIN][TIMER_LISTENER_KEY] = unsub
        _LOGGER.debug("Registered timer.finished event listener")

    # Register HTTP views (only once per HA instance)
    if SNAPSHOT_VIEW_KEY not in hass.data[DOMAIN]:
        hass.http.register_view(PureLLMSnapshotView(hass))
        hass.http.register_view(PureLLMSilentWavView())
        hass.data[DOMAIN][SNAPSHOT_VIEW_KEY] = True

    # Write silent WAV to HA's media directory for start_conversation's
    # start_media_id.  This lets us skip TTS synthesis (which crashes with
    # some engines on empty strings) while keeping a single-flow UX.
    if SILENCE_WAV_KEY not in hass.data[DOMAIN]:
        media_dir = hass.config.path("media")
        silence_path = os.path.join(media_dir, "purellm_silence.wav")

        def _write_silence_wav() -> None:
            os.makedirs(media_dir, exist_ok=True)
            with open(silence_path, "wb") as f:
                f.write(_get_silent_wav())

        try:
            await hass.async_add_executor_job(_write_silence_wav)
            _LOGGER.debug("Wrote silent WAV to %s", silence_path)
            hass.data[DOMAIN][SILENCE_WAV_KEY] = True
        except OSError as err:
            _LOGGER.warning("Could not write silent WAV to media dir: %s", err)

    # Register ask_and_act service (only once per HA instance)
    if SERVICE_REGISTERED_KEY not in hass.data[DOMAIN]:
        async def handle_ask_and_act(call: ServiceCall) -> dict[str, Any]:
            """Service handler wrapper."""
            return await async_handle_ask_and_act(hass, call)

        hass.services.async_register(
            DOMAIN,
            SERVICE_ASK_AND_ACT,
            handle_ask_and_act,
            schema=ASK_AND_ACT_SCHEMA,
            supports_response=SupportsResponse.OPTIONAL,
        )
        hass.data[DOMAIN][SERVICE_REGISTERED_KEY] = True
        _LOGGER.info("Registered purellm.ask_and_act service")

    entry.async_on_unload(
        entry.add_update_listener(_async_update_listener)
    )

    _LOGGER.info("PureLLM setup complete")
    return True


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle config entry updates - reload the integration.

    Also clears HA's TTS audio cache. HA keys cached TTS audio by
    (message, language, options, engine_id) and does not know about the
    ElevenLabs voice_settings we read from config. Without clearing the
    cache, changes to stability/similarity/style/speaker_boost sliders
    would be silently ignored whenever a previously-spoken phrase is
    replayed, because the stale audio would be served from cache.
    """
    if hass.services.has_service("tts", "clear_cache"):
        try:
            await hass.services.async_call("tts", "clear_cache", {}, blocking=True)
            _LOGGER.debug("Cleared HA TTS cache after PureLLM options update")
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("Could not clear TTS cache on options update: %s", err)

    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)

        # Unsubscribe from timer events if this was the last entry
        remaining_entries = [
            e for e in hass.config_entries.async_entries(DOMAIN)
            if e.entry_id != entry.entry_id
        ]
        if not remaining_entries and TIMER_LISTENER_KEY in hass.data[DOMAIN]:
            unsub = hass.data[DOMAIN].pop(TIMER_LISTENER_KEY)
            unsub()
            _LOGGER.debug("Unregistered timer.finished event listener")

    return unload_ok
