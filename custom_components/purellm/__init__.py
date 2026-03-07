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


async def _wait_for_satellite_state(
    hass: HomeAssistant,
    satellite_entity_id: str,
    target_state: str,
    timeout: float = 15.0,
) -> bool:
    """Wait for the assist satellite to reach a target state using event listener.

    Returns True if the target state was reached, False on timeout.
    """
    state = hass.states.get(satellite_entity_id)
    if state and state.state == target_state:
        return True

    reached = asyncio.Event()

    @callback
    def _on_state_change(event: Event) -> None:
        new_state = event.data.get("new_state")
        if (
            event.data.get("entity_id") == satellite_entity_id
            and new_state is not None
            and new_state.state == target_state
        ):
            reached.set()

    unsub = hass.bus.async_listen("state_changed", _on_state_change)
    try:
        # Re-check after subscribing to avoid TOCTOU race
        state = hass.states.get(satellite_entity_id)
        if state and state.state == target_state:
            return True

        _LOGGER.debug(
            "ask_and_act: satellite %s in state '%s', waiting for '%s'…",
            satellite_entity_id, state.state if state else "unknown", target_state,
        )
        async with asyncio.timeout(timeout):
            await reached.wait()
        return True
    except TimeoutError:
        _LOGGER.warning(
            "ask_and_act: timed out waiting for satellite %s to reach '%s' (state=%s)",
            satellite_entity_id, target_state,
            hass.states.get(satellite_entity_id).state if hass.states.get(satellite_entity_id) else "unknown",
        )
        return False
    finally:
        unsub()


async def async_handle_ask_and_act(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Handle the ask_and_act service call.

    Two-step approach:
    1. Use assist_satellite.announce to speak the question on the satellite.
       announce handles TTS internally and blocks until audio finishes.
    2. Wait for satellite to return to idle, then call start_conversation
       to enter listening mode with the extra_system_prompt for answer matching.
    """
    satellite_entity_id = call.data["satellite_entity_id"]
    question = call.data["question"]
    answers = call.data["answers"]

    _LOGGER.info("ask_and_act: Starting - question='%s', satellite=%s",
                 question, satellite_entity_id)

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

    # Step 1: Announce the question on the satellite.
    # assist_satellite.announce handles TTS and audio playback internally,
    # so no need to manually resolve TTS entities or media players.
    try:
        await hass.services.async_call(
            "assist_satellite", "announce",
            {
                "entity_id": satellite_entity_id,
                "message": question,
            },
            blocking=True,
        )
        _LOGGER.debug("ask_and_act: announce completed")
    except Exception as err:
        _LOGGER.error("ask_and_act: announce failed: %s", err)
        return {"error": f"announce failed: {err}"}

    # Step 2: Wait for satellite to return to idle after announcement.
    # announce with blocking=True should already wait, but the satellite
    # pipeline may still be tearing down internally.
    await _wait_for_satellite_state(hass, satellite_entity_id, "idle", timeout=10.0)

    # Small settle delay - satellite internal pipeline teardown may lag
    # behind the entity state change.
    await asyncio.sleep(0.5)

    # Step 3: Start conversation to enter listening mode.
    # Retry up to 3 times because start_conversation can be silently ignored
    # if the satellite pipeline hasn't fully released yet.
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
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
        except Exception as err:
            _LOGGER.error("ask_and_act: start_conversation failed: %s", err)
            return {"error": f"start_conversation failed: {err}"}

        # Verify the satellite actually entered listening mode
        await asyncio.sleep(0.3)
        state = hass.states.get(satellite_entity_id)
        if state and state.state in ("listening", "processing", "responding"):
            _LOGGER.info(
                "ask_and_act: satellite entered '%s' (attempt %d)",
                state.state, attempt,
            )
            return {"success": True}

        if attempt < max_attempts:
            _LOGGER.warning(
                "ask_and_act: satellite still '%s' after start_conversation (attempt %d/%d), retrying…",
                state.state if state else "unknown", attempt, max_attempts,
            )
            await asyncio.sleep(0.5 * attempt)
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
