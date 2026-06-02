"""Dedicated speaker/voice-volume tool for PureLLM.

Sets the volume of the satellite that heard the request, resolved from the
request's `device_id` to that device's own `media_player` entity (the same
resolution `control_device` uses for device="speaker").

Kept separate from `control_device` on purpose: local LLMs reliably call a
small, single-purpose tool but often fail to pick the volume action out of
`control_device`'s large action enum — which is exactly why "set your volume"
was unreliable. No regex short-circuit; this is just a normal tool.

Note: these ESPHome satellite media_players support VOLUME_SET but NOT
VOLUME_STEP, so up/down is implemented by reading volume_level and re-setting
it (media_player.volume_up/volume_down would silently no-op).
"""
from __future__ import annotations

import logging

_LOGGER = logging.getLogger(__name__)

STEP = 0.1  # 10% per up/down


async def set_speaker_volume(
    arguments: dict,
    hass,
    device_id: str | None = None,
    room_player_mapping: dict | None = None,
) -> dict:
    """Set/raise/lower the requesting satellite's own speaker volume."""
    action = (arguments.get("action") or "set").strip().lower()
    level = arguments.get("level")
    if level is None:
        level = arguments.get("volume")  # tolerate the control_device-style arg name

    from .timer import get_player_for_device
    player = get_player_for_device(hass, device_id, room_player_mapping)
    if not player:
        _LOGGER.warning(
            "set_speaker_volume: no media_player resolved (device_id=%s)", device_id
        )
        return {"error": "I couldn't find a speaker to adjust."}

    state = hass.states.get(player)
    current = (state.attributes.get("volume_level") if state else None)

    if action in ("set", "set_volume"):
        if level is None:
            return {"error": "Tell me what level to set, e.g. 'set your volume to 40'."}
        lvl = max(0, min(100, int(level)))
        await hass.services.async_call(
            "media_player", "volume_set",
            {"entity_id": player, "volume_level": lvl / 100.0},
            blocking=True,
        )
        _LOGGER.info("set_speaker_volume: %s -> %d%%", player, lvl)
        return {"status": "ok", "response_text": f"Volume set to {lvl} percent."}

    if action in ("up", "raise", "increase", "louder", "volume_up"):
        new = min(1.0, (current if current is not None else 0.5) + STEP)
        await hass.services.async_call(
            "media_player", "volume_set",
            {"entity_id": player, "volume_level": new},
            blocking=True,
        )
        _LOGGER.info("set_speaker_volume: %s up -> %d%%", player, round(new * 100))
        return {"status": "ok", "response_text": f"Volume up, now {round(new * 100)} percent."}

    if action in ("down", "lower", "decrease", "quieter", "volume_down"):
        new = max(0.0, (current if current is not None else 0.5) - STEP)
        await hass.services.async_call(
            "media_player", "volume_set",
            {"entity_id": player, "volume_level": new},
            blocking=True,
        )
        _LOGGER.info("set_speaker_volume: %s down -> %d%%", player, round(new * 100))
        return {"status": "ok", "response_text": f"Volume down, now {round(new * 100)} percent."}

    return {"error": f"Unknown volume action: {action}"}
