"""The PureLLM integration."""
from __future__ import annotations

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


def _get_satellite_entity(hass: HomeAssistant, entity_id: str):
    """Get the assist satellite entity object directly.

    This mirrors how HA's own ask_question service handler accesses the entity:
    it goes through the EntityComponent registered for the assist_satellite domain.
    """
    component = hass.data.get("assist_satellite")
    if component is None:
        _LOGGER.error("ask_and_act: assist_satellite component not loaded")
        return None

    entity = component.get_entity(entity_id)
    if entity is None:
        _LOGGER.error("ask_and_act: entity %s not found", entity_id)
        return None

    return entity


async def async_handle_ask_and_act(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Handle the ask_and_act service call.

    Emulates HA's own ask_question flow by calling the satellite entity's
    async_internal_ask_question method directly.  This atomically:
    1. Cancels any running pipeline
    2. Resolves TTS for the question
    3. Plays the question on the satellite (async_start_conversation)
    4. Creates a Future that gets resolved when STT finishes
    5. Matches the transcribed response against answer sentences via hassil
    Then we execute the matched action and announce the response.
    """
    satellite_entity_id = call.data["satellite_entity_id"]
    question = call.data["question"]
    answers = call.data["answers"]

    _LOGGER.info("ask_and_act: Starting - question='%s', satellite=%s",
                 question, satellite_entity_id)

    # Get the satellite entity object directly (same as HA's service handler)
    entity = _get_satellite_entity(hass, satellite_entity_id)
    if entity is None:
        return {"error": f"satellite entity {satellite_entity_id} not found"}

    # Build the answers list for ask_question (just id + sentences)
    ask_answers = [
        {"id": a["id"], "sentences": a["sentences"]}
        for a in answers
    ]

    # Call the entity's internal ask_question method directly.
    # This is the exact same method HA's own ask_question service calls.
    # It handles the full flow: cancel pipeline -> TTS -> play question ->
    # listen -> STT -> match answers.
    try:
        answer = await entity.async_internal_ask_question(
            question=question,
            preannounce=False,
            answers=ask_answers,
        )
        _LOGGER.info("ask_and_act: ask_question returned: id=%s, sentence=%s",
                     answer.id if answer else None,
                     answer.sentence if answer else None)
    except Exception as err:
        _LOGGER.error("ask_and_act: ask_question failed: %s", err)
        return {"error": f"ask_question failed: {err}"}

    # Find the matched answer and execute its action.
    matched_id = answer.id if answer else None

    if not matched_id:
        _LOGGER.info("ask_and_act: no answer matched (sentence=%s)",
                     answer.sentence if answer else None)
        return {"no_match": True}

    # Find the answer config with this ID
    matched_answer = None
    for a in answers:
        if a["id"] == matched_id:
            matched_answer = a
            break

    if not matched_answer:
        _LOGGER.warning("ask_and_act: matched id '%s' not found in answers", matched_id)
        return {"error": f"matched id '{matched_id}' not found in answers"}

    _LOGGER.info("ask_and_act: matched answer id='%s'", matched_id)

    # Execute the action if one is configured
    if "action" in matched_answer:
        action_config = matched_answer["action"]
        service_str = action_config.get("service", "")
        service_parts = service_str.split(".", 1)
        if len(service_parts) == 2:
            service_data = dict(action_config.get("data") or {})
            target = action_config.get("target")
            if target:
                service_data.update(target)

            try:
                await hass.services.async_call(
                    service_parts[0],
                    service_parts[1],
                    service_data,
                    blocking=True,
                )
                _LOGGER.info("ask_and_act: executed action %s", service_str)
            except Exception as err:
                _LOGGER.error("ask_and_act: action execution failed: %s", err)

    # Announce the response if one is configured.
    # Use the entity's own announce method for consistency.
    response_text = matched_answer.get("response")
    if response_text:
        try:
            await entity.async_internal_announce(message=response_text, preannounce=False)
            _LOGGER.debug("ask_and_act: announced response '%s'", response_text)
        except Exception as err:
            # Fall back to service call if direct method not available
            _LOGGER.debug("ask_and_act: direct announce failed (%s), trying service call", err)
            try:
                await hass.services.async_call(
                    "assist_satellite", "announce",
                    {
                        "entity_id": satellite_entity_id,
                        "message": response_text,
                        "preannounce": False,
                    },
                    blocking=True,
                )
            except Exception as err2:
                _LOGGER.error("ask_and_act: announce response failed: %s", err2)

    return {"success": True, "matched_id": matched_id}


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
