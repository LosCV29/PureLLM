"""The PolyVoice integration."""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, Event, callback, ServiceCall
from homeassistant.helpers import config_validation as cv

from .const import DOMAIN
from .tools.timer import get_registered_timer, unregister_timer

_LOGGER = logging.getLogger(__name__)

# Service constants
SERVICE_ASK_AND_ACT = "ask_and_act"

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
    vol.Required("media_player_entity_id"): cv.entity_id,
    vol.Required("tts_entity_id"): cv.entity_id,
    vol.Required("question"): cv.string,
    vol.Required("answers"): vol.All(cv.ensure_list, [ANSWER_SCHEMA]),
    vol.Optional("timeout", default=15): vol.Coerce(int),
    vol.Optional("tts_delay", default=2.5): vol.Coerce(float),
})

PLATFORMS: list[Platform] = [Platform.CONVERSATION, Platform.UPDATE]

# Key for storing service registration
SERVICE_REGISTERED_KEY = "ask_and_act_service_registered"

# Key for storing the timer listener unsub function
TIMER_LISTENER_KEY = "timer_finished_listener"


async def async_setup(hass: HomeAssistant, config: dict[str, Any]) -> bool:
    """Set up the PolyVoice component."""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_handle_ask_and_act(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Handle the ask_and_act service call.

    This service:
    1. Speaks a question via TTS
    2. Listens for user response via assist satellite
    3. Matches response against predefined answers
    4. Executes the corresponding action
    5. Speaks confirmation

    Returns a dict with the matched answer info.
    """
    satellite_entity_id = call.data["satellite_entity_id"]
    media_player_entity_id = call.data["media_player_entity_id"]
    tts_entity_id = call.data["tts_entity_id"]
    question = call.data["question"]
    answers = call.data["answers"]
    timeout = call.data.get("timeout", 15)
    tts_delay = call.data.get("tts_delay", 2.5)

    _LOGGER.info("ask_and_act: Starting - question='%s', satellite=%s", question, satellite_entity_id)

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

    # Step 2: Wait for TTS to complete
    await asyncio.sleep(tts_delay)

    # Step 3: Listen for response using assist_satellite.start_conversation
    # We use an empty start_message to just listen without speaking again
    response_text = None
    try:
        # Create a simple system prompt that just echoes the user's response
        result = await hass.services.async_call(
            "assist_satellite", "start_conversation",
            {
                "entity_id": satellite_entity_id,
                "start_message": "",
                "extra_system_prompt": "Repeat EXACTLY what the user said, word for word. Nothing else.",
                "preannounce": False,
            },
            blocking=True,
            return_response=True,
        )
        _LOGGER.debug("ask_and_act: start_conversation result: %s", result)

        # Extract the response text from the result
        if result and isinstance(result, dict):
            # Try to get the speech from the response
            for entity_result in result.values():
                if isinstance(entity_result, dict):
                    speech = entity_result.get("response", {}).get("speech", {})
                    if isinstance(speech, dict):
                        plain = speech.get("plain", {})
                        if isinstance(plain, dict):
                            response_text = plain.get("speech", "")
                        elif isinstance(plain, str):
                            response_text = plain
                    elif isinstance(speech, str):
                        response_text = speech

    except Exception as err:
        _LOGGER.error("ask_and_act: Listen failed: %s", err)
        return {"error": f"Listen failed: {err}"}

    if not response_text:
        _LOGGER.warning("ask_and_act: No response received")
        return {"error": "No response received", "matched_id": None}

    _LOGGER.info("ask_and_act: User response: '%s'", response_text)

    # Step 4: Match response against answer patterns
    matched_answer = None
    response_lower = response_text.lower().strip()

    for answer in answers:
        for sentence in answer["sentences"]:
            # Simple contains match (case-insensitive)
            sentence_lower = sentence.lower().strip()
            if sentence_lower in response_lower or response_lower in sentence_lower:
                matched_answer = answer
                _LOGGER.info("ask_and_act: Matched answer '%s' with sentence '%s'",
                           answer["id"], sentence)
                break
            # Also try word-by-word match for single words
            if len(sentence_lower.split()) == 1 and sentence_lower in response_lower.split():
                matched_answer = answer
                _LOGGER.info("ask_and_act: Matched answer '%s' with word '%s'",
                           answer["id"], sentence)
                break
        if matched_answer:
            break

    if not matched_answer:
        _LOGGER.warning("ask_and_act: No matching answer for response '%s'", response_text)
        return {"error": "No matching answer", "response": response_text, "matched_id": None}

    # Step 5: Execute the action if specified
    if "action" in matched_answer:
        action_config = matched_answer["action"]
        service_parts = action_config["service"].split(".", 1)
        if len(service_parts) == 2:
            domain, service = service_parts
            service_data = action_config.get("data", {})
            target = action_config.get("target", {})

            try:
                await hass.services.async_call(
                    domain, service,
                    {**service_data, **target} if target else service_data,
                    blocking=True,
                    target=target if target else None,
                )
                _LOGGER.info("ask_and_act: Executed action %s.%s", domain, service)
            except Exception as err:
                _LOGGER.error("ask_and_act: Action failed: %s", err)
                return {"error": f"Action failed: {err}", "matched_id": matched_answer["id"]}

    # Step 6: Speak confirmation if specified
    if "response" in matched_answer and matched_answer["response"]:
        try:
            await hass.services.async_call(
                "tts", "speak",
                {
                    "entity_id": tts_entity_id,
                    "media_player_entity_id": media_player_entity_id,
                    "message": matched_answer["response"],
                },
                blocking=False,
            )
            _LOGGER.debug("ask_and_act: Spoke confirmation")
        except Exception as err:
            _LOGGER.warning("ask_and_act: Confirmation TTS failed: %s", err)

    return {
        "matched_id": matched_answer["id"],
        "response": response_text,
        "action_executed": "action" in matched_answer,
    }


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate old entry to current version."""
    _LOGGER.info("Migrating PolyVoice from version %s", entry.version)

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

    # Check if this timer was started by PolyVoice
    timer_info = get_registered_timer(hass, entity_id)
    if not timer_info:
        _LOGGER.debug("Timer %s finished but was not started by PolyVoice", entity_id)
        return

    timer_name = timer_info.get("name", "Timer")
    announce_player = timer_info.get("announce_player")

    _LOGGER.info("PolyVoice timer finished: %s -> announcing on %s",
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
                "notification_id": "polyvoice_timer",
            },
            blocking=False
        )
        _LOGGER.debug("Created persistent notification for timer (no TTS available)")


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up PolyVoice from a config entry."""
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
            supports_response=True,
        )
        hass.data[DOMAIN][SERVICE_REGISTERED_KEY] = True
        _LOGGER.info("Registered purellm.ask_and_act service")

    entry.async_on_unload(
        entry.add_update_listener(_async_update_listener)
    )

    _LOGGER.info("PolyVoice setup complete")
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
