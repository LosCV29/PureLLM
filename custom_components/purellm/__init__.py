"""The PureLLM integration."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
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
    vol.Optional("tts_entity_id"): cv.entity_id,  # Optional, defaults to tts.home_assistant_cloud
})

PLATFORMS: list[Platform] = [Platform.CONVERSATION, Platform.UPDATE]

# Key for storing service registration
SERVICE_REGISTERED_KEY = "ask_and_act_service_registered"

# Key for storing the timer listener unsub function
TIMER_LISTENER_KEY = "timer_finished_listener"

SNAPSHOT_VIEW_KEY = "snapshot_view_registered"


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


async def async_setup(hass: HomeAssistant, config: dict[str, Any]) -> bool:
    """Set up the PureLLM component."""
    hass.data.setdefault(DOMAIN, {})
    return True


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


async def async_handle_ask_and_act(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Handle the ask_and_act service call.

    This service leverages the LLM to:
    1. Speak question via TTS
    2. Wait for TTS audio to finish playing
    3. Listen for response via satellite
    4. LLM executes the appropriate action based on response
    5. LLM speaks confirmation

    The LLM handles action execution via its control_device tool.
    """
    satellite_entity_id = call.data["satellite_entity_id"]
    question = call.data["question"]
    answers = call.data["answers"]
    tts_entity_id = call.data.get("tts_entity_id", "tts.home_assistant_cloud")

    # Derive media_player from satellite entity ID
    # Pattern: assist_satellite.home_assistant_voice_XXXXXX_assist_satellite
    #       -> media_player.home_assistant_voice_XXXXXX_media_player
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

    # Build the extra_system_prompt with embedded answers data for reliable
    # code-based matching, plus LLM instructions as a fallback.
    # Serialize answers for the conversation handler to parse and execute directly.
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
    ])

    extra_system_prompt = "\n".join(prompt_parts)
    _LOGGER.debug("ask_and_act: Generated prompt:\n%s", extra_system_prompt)

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
    # tts.speak with blocking=True only waits for the audio to be queued,
    # not for playback to complete. The satellite cannot enter listening mode
    # while its media player is still actively playing audio.
    await _wait_for_media_player_idle(hass, media_player_entity_id)

    # Step 3: Listen for response (empty start_message = skip announcement, just listen)
    try:
        await hass.services.async_call(
            "assist_satellite", "start_conversation",
            {
                "entity_id": satellite_entity_id,
                "start_message": "",
                "extra_system_prompt": extra_system_prompt,
                "preannounce": False,
            },
            blocking=True,
        )
        _LOGGER.info("ask_and_act: start_conversation completed")
        return {"success": True}
    except Exception as err:
        _LOGGER.error("ask_and_act: start_conversation failed: %s", err)
        return {"error": f"start_conversation failed: {err}"}


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

    # Register snapshot HTTP view (only once per HA instance)
    if SNAPSHOT_VIEW_KEY not in hass.data[DOMAIN]:
        hass.http.register_view(PureLLMSnapshotView(hass))
        hass.data[DOMAIN][SNAPSHOT_VIEW_KEY] = True

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
    """Handle config entry updates - reload the integration."""
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
