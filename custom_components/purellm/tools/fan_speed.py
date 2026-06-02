"""Dedicated fan speed / preset tool for PureLLM.

Sets a fan's speed (exact percentage or low/medium/high), steps it up/down, or
selects a named preset mode (e.g. "fresh", "nature").

Kept separate from `control_device` on purpose: weak local LLMs reliably call a
small, single-purpose tool but fail to drive fan speed through `control_device`'s
large action enum — which never exposed a numeric percentage or preset_mode at
all, so "set the fan to 20 percent" / "set the fan to fresh" were impossible to
express and always came back as "I couldn't complete that request." This tool
calls fan.set_percentage / fan.set_preset_mode directly.
"""
from __future__ import annotations

import logging

from ..utils.fuzzy_matching import find_entity_by_name

_LOGGER = logging.getLogger(__name__)

# Named speeds -> percentage, mirroring the legacy mapping in device.py so
# "low/medium/high" behave identically to how fans worked before.
LEVEL_MAP = {"low": 33, "medium": 66, "high": 100, "auto": 50}

# Fallback step for up/down when the fan reports no percentage_step.
DEFAULT_STEP = 20

# FanEntityFeature bits (homeassistant.components.fan)
_FEATURE_SET_SPEED = 1
_FEATURE_PRESET_MODE = 8


def _fan_entities(hass) -> list[tuple[str, str]]:
    """Return (entity_id, friendly_name) for every fan.* entity."""
    out = []
    for state in hass.states.async_all("fan"):
        out.append((state.entity_id, state.attributes.get("friendly_name", state.entity_id)))
    return out


def _resolve_fan(hass, arguments: dict) -> tuple[str | None, str | None, str | None]:
    """Resolve the target fan entity.

    Returns (entity_id, friendly_name, error_text). On success error_text is None.
    """
    fans = _fan_entities(hass)

    # 1) Explicit entity_id
    eid = (arguments.get("entity_id") or "").strip()
    if eid:
        if not eid.startswith("fan."):
            return None, None, f"'{eid}' is not a fan."
        state = hass.states.get(eid)
        if not state:
            return None, None, f"Fan '{eid}' not found."
        return eid, state.attributes.get("friendly_name", eid), None

    # 2) Named device -> shared fuzzy matcher, but only accept a fan result
    device = (arguments.get("device") or "").strip()
    if device:
        match_id, match_name = find_entity_by_name(hass, device)
        if match_id and match_id.startswith("fan."):
            return match_id, match_name, None
        # Fall back to a direct substring match against fan names only, so a
        # non-fan fuzzy hit (e.g. a "fan" switch) doesn't shadow a real fan.
        dl = device.lower()
        for fid, fname in fans:
            if dl in fname.lower() or fname.lower() in dl:
                return fid, fname, None
        return None, None, (
            f"I couldn't find a fan called '{device}'."
            + (f" Fans I know: {', '.join(n for _, n in fans)}." if fans else "")
        )

    # 3) No name given — fine if there's exactly one fan
    if len(fans) == 1:
        return fans[0][0], fans[0][1], None
    if not fans:
        return None, None, "There are no fans set up in Home Assistant."
    return None, None, (
        "Which fan? I have: " + ", ".join(n for _, n in fans) + "."
    )


async def set_fan_speed(arguments: dict, hass) -> dict:
    """Set a fan's speed/percentage, step it up/down, or set a preset mode."""
    entity_id, friendly_name, err = _resolve_fan(hass, arguments)
    if err:
        return {"error": err}

    state = hass.states.get(entity_id)
    attrs = state.attributes if state else {}
    features = attrs.get("supported_features", 0) or 0
    current_pct = attrs.get("percentage")

    action = (arguments.get("action") or "").strip().lower()
    percentage = arguments.get("percentage")
    level = (arguments.get("level") or "").strip().lower()
    preset_mode = (arguments.get("preset_mode") or "").strip()

    # --- Preset mode (e.g. "fresh", "nature") ---
    if preset_mode:
        if not (features & _FEATURE_PRESET_MODE):
            return {"error": f"The {friendly_name} doesn't support preset modes."}
        valid = attrs.get("preset_modes") or []
        chosen = next((p for p in valid if p.lower() == preset_mode.lower()), None)
        if not chosen:
            return {
                "error": (
                    f"'{preset_mode}' isn't a mode for the {friendly_name}."
                    + (f" Available: {', '.join(valid)}." if valid else "")
                )
            }
        await hass.services.async_call(
            "fan", "set_preset_mode",
            {"entity_id": entity_id, "preset_mode": chosen}, blocking=True,
        )
        _LOGGER.info("set_fan_speed: %s preset -> %s", entity_id, chosen)
        return {"status": "ok", "response_text": f"The {friendly_name} is now set to {chosen}."}

    # Resolve a target percentage from the various ways speed can be expressed.
    target = None
    if percentage is not None:
        try:
            target = int(percentage)
        except (TypeError, ValueError):
            return {"error": "Speed must be a number between 0 and 100."}
    elif level:
        if level not in LEVEL_MAP:
            return {"error": f"Unknown speed '{level}'. Use low, medium, high, or auto."}
        target = LEVEL_MAP[level]
    elif action in ("up", "raise", "increase", "faster"):
        step = int(attrs.get("percentage_step") or DEFAULT_STEP) or DEFAULT_STEP
        step = max(step, DEFAULT_STEP)  # avoid 1%-step fans crawling up
        target = (current_pct if current_pct is not None else 0) + step
    elif action in ("down", "lower", "decrease", "slower"):
        step = int(attrs.get("percentage_step") or DEFAULT_STEP) or DEFAULT_STEP
        step = max(step, DEFAULT_STEP)
        target = (current_pct if current_pct is not None else 0) - step

    if target is None:
        return {"error": "Tell me a speed, e.g. 'set the fan to 40 percent', 'high', or 'up'."}

    if not (features & _FEATURE_SET_SPEED):
        return {"error": f"The {friendly_name} doesn't support speed control."}

    target = max(0, min(100, target))
    await hass.services.async_call(
        "fan", "set_percentage",
        {"entity_id": entity_id, "percentage": target}, blocking=True,
    )
    _LOGGER.info("set_fan_speed: %s -> %d%%", entity_id, target)
    if target == 0:
        return {"status": "ok", "response_text": f"The {friendly_name} is now off."}
    return {"status": "ok", "response_text": f"The {friendly_name} is now set to {target} percent."}
