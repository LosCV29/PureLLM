"""SofaBaton X2 remote tool handler using Home Assistant switch entities."""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


async def control_sofabaton(
    hass: "HomeAssistant",
    arguments: dict[str, Any],
    sofabaton_activities: list[dict[str, str]],
) -> dict[str, Any]:
    """Control SofaBaton X2 activities via Home Assistant switch entities.

    Args:
        hass: Home Assistant instance
        arguments: Tool arguments (activity, action)
        sofabaton_activities: List of configured activities with name, entity_id

    Returns:
        Control result dict
    """
    activity_name = arguments.get("activity", "").strip().lower()
    action = arguments.get("action", "").strip().lower()

    if not activity_name:
        return {"error": "No activity specified"}

    if action not in ("start", "stop"):
        return {"error": f"Invalid action '{action}'. Must be 'start' or 'stop'."}

    # Find matching activity (case-insensitive)
    matched_activity = None
    for activity in sofabaton_activities:
        if activity.get("name", "").lower() == activity_name:
            matched_activity = activity
            break

    # Try fuzzy matching if exact match not found
    if not matched_activity:
        for activity in sofabaton_activities:
            if activity_name in activity.get("name", "").lower():
                matched_activity = activity
                break

    if not matched_activity:
        available = [a.get("name", "") for a in sofabaton_activities]
        return {
            "error": f"Activity '{activity_name}' not found. Available activities: {', '.join(available) or 'none configured'}"
        }

    # Get the entity ID
    entity_id = matched_activity.get("entity_id", "")
    if not entity_id:
        return {"error": f"No entity configured for activity '{matched_activity.get('name')}'"}

    # Verify entity exists
    state = hass.states.get(entity_id)
    if state is None:
        return {"error": f"Entity '{entity_id}' not found in Home Assistant"}

    # Determine service based on action
    service = "turn_on" if action == "start" else "turn_off"

    # Call the switch service
    try:
        await hass.services.async_call(
            "switch",
            service,
            {"entity_id": entity_id},
            blocking=True,
        )

        friendly_name = matched_activity.get("name", activity_name)
        action_past = "started" if action == "start" else "stopped"
        return {
            "status": "success",
            "response_text": f"{friendly_name} {action_past}."
        }

    except Exception as err:
        _LOGGER.error("SofaBaton switch control failed: %s", err)
        return {"error": f"Failed to control SofaBaton activity: {str(err)}"}
