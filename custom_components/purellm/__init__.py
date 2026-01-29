"""The PolyVoice integration."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, Event, callback, ServiceCall
from homeassistant.helpers import config_validation as cv

from .const import DOMAIN
from .tools.timer import get_registered_timer, unregister_timer

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.CONVERSATION, Platform.UPDATE]

# Key for storing the timer listener unsub function
TIMER_LISTENER_KEY = "timer_finished_listener"

# Key for storing pending questions
PENDING_QUESTIONS_KEY = "pending_questions"

# Schema for ask_question service
ANSWER_SCHEMA = vol.Schema({
    vol.Required("id"): cv.string,
    vol.Required("sentences"): vol.All(cv.ensure_list, [cv.string]),
})

ASK_QUESTION_SCHEMA = vol.Schema({
    vol.Required("entity_id"): cv.entity_id,
    vol.Required("question"): cv.string,
    vol.Required("answers"): vol.All(cv.ensure_list, [ANSWER_SCHEMA]),
    vol.Optional("preannounce", default=True): cv.boolean,
    vol.Optional("timeout", default=30): vol.Coerce(int),
})


async def async_setup(hass: HomeAssistant, config: dict[str, Any]) -> bool:
    """Set up the PolyVoice component."""
    hass.data.setdefault(DOMAIN, {})
    return True


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


def _normalize_text(text: str) -> str:
    """Normalize text for matching - lowercase and remove punctuation."""
    return re.sub(r'[^\w\s]', '', text.lower().strip())


def _match_answer(text: str, answers: list[dict]) -> dict | None:
    """Match text against expected answer sentences.

    Returns the matched answer dict with 'id' or None if no match.
    """
    normalized_text = _normalize_text(text)

    for answer in answers:
        for sentence in answer.get("sentences", []):
            if _normalize_text(sentence) == normalized_text:
                return {"id": answer["id"], "matched_sentence": sentence}

    # Try partial matching (text contains or is contained by a sentence)
    for answer in answers:
        for sentence in answer.get("sentences", []):
            norm_sentence = _normalize_text(sentence)
            if norm_sentence in normalized_text or normalized_text in norm_sentence:
                return {"id": answer["id"], "matched_sentence": sentence}

    return None


def get_pending_question(hass: HomeAssistant, device_id: str | None = None) -> dict | None:
    """Get pending question for a device (or any device if device_id is None).

    Returns the pending question dict or None.
    """
    pending = hass.data.get(DOMAIN, {}).get(PENDING_QUESTIONS_KEY, {})

    if device_id and device_id in pending:
        question = pending[device_id]
        # Check if not expired
        if time.time() < question.get("expires_at", 0):
            return question
        # Expired - clean up
        pending.pop(device_id, None)
        return None

    # If no device_id specified, return any non-expired pending question
    for dev_id, question in list(pending.items()):
        if time.time() < question.get("expires_at", 0):
            return question
        # Expired - clean up
        pending.pop(dev_id, None)

    return None


def clear_pending_question(hass: HomeAssistant, device_id: str) -> None:
    """Clear a pending question for a device."""
    pending = hass.data.get(DOMAIN, {}).get(PENDING_QUESTIONS_KEY, {})
    pending.pop(device_id, None)


def match_pending_question(hass: HomeAssistant, text: str, device_id: str | None = None) -> dict | None:
    """Check if text matches a pending question's answers.

    Returns {"id": answer_id, "matched": True} or None if no match.
    """
    question = get_pending_question(hass, device_id)
    if not question:
        return None

    match = _match_answer(text, question.get("answers", []))
    if match:
        # Clear the pending question after a match
        actual_device_id = device_id or question.get("device_id")
        if actual_device_id:
            clear_pending_question(hass, actual_device_id)
        return match

    return None


async def _async_ask_question_service(hass: HomeAssistant, call: ServiceCall) -> dict:
    """Handle purellm.ask_question service call.

    This service:
    1. Stores the expected answers in memory
    2. Announces the question via assist_satellite.announce
    3. Triggers the satellite to start listening
    4. Waits for a matching response
    5. Returns the matched answer ID
    """
    entity_id = call.data["entity_id"]
    question = call.data["question"]
    answers = call.data["answers"]
    preannounce = call.data.get("preannounce", True)
    timeout = call.data.get("timeout", 30)

    _LOGGER.info(
        "purellm.ask_question called: entity=%s, question='%s', answers=%s, timeout=%d",
        entity_id, question, [a["id"] for a in answers], timeout
    )

    # Extract device identifier from entity_id
    # e.g., assist_satellite.home_assistant_voice_099fca_assist_satellite -> home_assistant_voice_099fca
    device_id = entity_id.replace("assist_satellite.", "").replace("_assist_satellite", "")

    # Initialize pending questions storage
    hass.data[DOMAIN].setdefault(PENDING_QUESTIONS_KEY, {})

    # Create an event to signal when we get an answer
    answer_event = asyncio.Event()
    result_holder = {"answer": None}

    # Store the pending question with callback
    pending_question = {
        "entity_id": entity_id,
        "device_id": device_id,
        "question": question,
        "answers": answers,
        "expires_at": time.time() + timeout,
        "answer_event": answer_event,
        "result_holder": result_holder,
    }
    hass.data[DOMAIN][PENDING_QUESTIONS_KEY][device_id] = pending_question

    try:
        # Step 1: Announce the question using assist_satellite.announce
        _LOGGER.debug("Announcing question via assist_satellite.announce")
        await hass.services.async_call(
            "assist_satellite",
            "announce",
            {
                "entity_id": entity_id,
                "message": question,
                "preannounce": preannounce,
            },
            blocking=True,
        )

        # Step 2: Wait a moment for TTS to complete, then trigger listening
        # Try to find and call the ESPHome service to start voice assistant
        await asyncio.sleep(0.5)  # Brief pause after TTS

        # Look for ESPHome voice assistant start service
        # Format: esphome.<device_name>_start_voice_assistant
        esphome_service = f"{device_id}_start_voice_assistant"
        if hass.services.has_service("esphome", esphome_service):
            _LOGGER.debug("Triggering listening via esphome.%s", esphome_service)
            await hass.services.async_call(
                "esphome",
                esphome_service,
                {},
                blocking=False,
            )
        else:
            # Try alternate naming convention
            alt_service = device_id.replace("home_assistant_voice_", "ha_voice_pe_") + "_start_voice_assistant"
            if hass.services.has_service("esphome", alt_service):
                _LOGGER.debug("Triggering listening via esphome.%s", alt_service)
                await hass.services.async_call("esphome", alt_service, {}, blocking=False)
            else:
                _LOGGER.warning(
                    "Could not find ESPHome service to trigger listening. "
                    "User will need to use wake word. Tried: esphome.%s",
                    esphome_service
                )

        # Step 3: Wait for the answer with timeout
        _LOGGER.debug("Waiting for answer (timeout=%ds)", timeout)
        try:
            await asyncio.wait_for(answer_event.wait(), timeout=timeout)
            answer = result_holder.get("answer")
            if answer:
                _LOGGER.info("Got answer: %s", answer)
                return {"id": answer.get("id"), "matched_sentence": answer.get("matched_sentence", "")}
        except asyncio.TimeoutError:
            _LOGGER.warning("ask_question timed out waiting for answer")

        # No answer received
        return {"id": None, "error": "no_answer"}

    finally:
        # Clean up pending question
        clear_pending_question(hass, device_id)


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
