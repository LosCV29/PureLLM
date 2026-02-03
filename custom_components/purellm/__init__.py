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
# Simplified: LLM handles action execution via its tools
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

    This service leverages the LLM to:
    1. Speak a question via TTS
    2. Listen for user response via satellite
    3. LLM executes the appropriate action based on response
    4. LLM speaks confirmation

    The LLM handles action execution via its control_device tool.
    """
    satellite_entity_id = call.data["satellite_entity_id"]
    media_player_entity_id = call.data["media_player_entity_id"]
    tts_entity_id = call.data["tts_entity_id"]
    question = call.data["question"]
    answers = call.data["answers"]

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

    # Step 2: Wait for TTS to finish playing
    await asyncio.sleep(2.5)

    # Build the extra_system_prompt that instructs the LLM what to do
    prompt_parts = [
        f"The user was just asked: \"{question}\"",
        "",
        "YOU MUST FOLLOW THESE INSTRUCTIONS EXACTLY:",
        "",
    ]

    for answer in answers:
        sentences = answer["sentences"]
        sentences_str = ", ".join(f'"{s}"' for s in sentences)

        if "action" in answer:
            action = answer["action"]
            service = action["service"]
            target = action.get("target", {})
            data = action.get("data", {})
            entity_id = target.get("entity_id", "")

            if service.startswith("light.turn_on"):
                brightness = data.get("brightness_pct", 100)
                color_temp = data.get("color_temp_kelvin", 4000)
                prompt_parts.append(f"If user says {sentences_str} or similar affirmative:")
                prompt_parts.append(f"  1. FIRST call control_device(entity_id=\"{entity_id}\", action=\"turn_on\", brightness={brightness}, color_temp={color_temp})")
                if "response" in answer:
                    prompt_parts.append(f"  2. AFTER the tool call succeeds, say: \"{answer['response']}\"")
            elif service.startswith("light.turn_off"):
                prompt_parts.append(f"If user says {sentences_str} or similar:")
                prompt_parts.append(f"  1. FIRST call control_device(entity_id=\"{entity_id}\", action=\"turn_off\")")
                if "response" in answer:
                    prompt_parts.append(f"  2. AFTER the tool call succeeds, say: \"{answer['response']}\"")
            elif service.startswith("switch."):
                action_type = "turn_on" if "turn_on" in service else "turn_off"
                prompt_parts.append(f"If user says {sentences_str} or similar:")
                prompt_parts.append(f"  1. FIRST call control_device(entity_id=\"{entity_id}\", action=\"{action_type}\")")
                if "response" in answer:
                    prompt_parts.append(f"  2. AFTER the tool call succeeds, say: \"{answer['response']}\"")
        else:
            # No action, just respond
            prompt_parts.append(f"If user says {sentences_str} or similar:")
            if "response" in answer:
                prompt_parts.append(f"  -> Just say: \"{answer['response']}\"")

        prompt_parts.append("")

    prompt_parts.extend([
        "CRITICAL RULES:",
        "- You MUST call control_device BEFORE saying anything",
        "- Use entity_id parameter, NOT device parameter",
        "- Do NOT say 'Done' or any response until AFTER the tool call completes",
        "- If you don't call the tool first, the action will NOT happen",
    ])

    extra_system_prompt = "\n".join(prompt_parts)
    _LOGGER.debug("ask_and_act: Generated prompt:\n%s", extra_system_prompt)

    # Step 3: Use start_conversation to listen for response
    # Empty start_message = just listen, don't speak
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
