"""Lyrics display tool — shows/hides the SyncLyrics page on the living room TV.

The heavy lifting lives in two HA scripts (script.show_living_room_lyrics /
script.hide_living_room_lyrics) which drive TV Bro on the Shield over ADB.
This tool just fires the right script so the LLM stays decoupled from the
ADB/browser mechanics.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

SHOW_SCRIPT = "script.show_living_room_lyrics"
HIDE_SCRIPT = "script.hide_living_room_lyrics"


async def display_lyrics(arguments: dict[str, Any], hass: "HomeAssistant") -> dict[str, Any]:
    """Show or hide scrolling lyrics on the living room TV."""
    action = (arguments or {}).get("action", "show")
    script_id = SHOW_SCRIPT if action == "show" else HIDE_SCRIPT

    if hass.states.get(script_id) is None:
        return {"error": f"Lyrics script {script_id} is not configured in Home Assistant."}

    try:
        await hass.services.async_call(
            "script", "turn_on", {"entity_id": script_id}, blocking=False
        )
    except Exception as err:  # noqa: BLE001
        _LOGGER.error("display_lyrics failed to run %s: %s", script_id, err)
        return {"error": f"Failed to run {script_id}: {err}"}

    _LOGGER.info("display_lyrics: action=%s -> %s", action, script_id)
    if action == "hide":
        return {"status": "ok", "message": "Lyrics dismissed from the living room TV."}
    return {
        "status": "ok",
        "message": "Lyrics are coming up on the living room TV. "
        "If no music is playing, start some first.",
    }
