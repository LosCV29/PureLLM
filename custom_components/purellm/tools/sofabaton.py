"""SofaBaton X2 remote tool handler."""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from ..const import API_TIMEOUT

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)


async def control_sofabaton(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    sofabaton_activities: list[dict[str, str]],
) -> dict[str, Any]:
    """Control SofaBaton X2 activities.

    Args:
        arguments: Tool arguments (activity, action)
        session: aiohttp client session
        sofabaton_activities: List of configured activities with name, start_url, stop_url

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

    # Get the appropriate URL (supports both old key format and new full URL format)
    if action == "start":
        url = matched_activity.get("start_key", "") or matched_activity.get("start_url", "")
        if not url:
            return {"error": f"No start URL configured for activity '{matched_activity.get('name')}'"}
    else:  # stop
        url = matched_activity.get("stop_key", "") or matched_activity.get("stop_url", "")
        if not url:
            return {"error": f"No stop URL configured for activity '{matched_activity.get('name')}'"}

    # If it's not a full URL, it might be an old-style key - try the legacy format
    if not url.startswith("http"):
        # Legacy format - append to old API endpoint (likely won't work anymore)
        url = f"https://rc.sofa.ai/api/open/activity/{url}"
        _LOGGER.warning(
            "Using legacy SofaBaton API key format. Please update to full URL format "
            "by copying the URL from the Sofabaton app."
        )

    # Make the API call
    try:
        async with session.get(url, timeout=API_TIMEOUT) as response:
            if response.status == 200:
                friendly_name = matched_activity.get("name", activity_name)
                action_past = "started" if action == "start" else "stopped"
                return {
                    "status": "success",
                    "response_text": f"{friendly_name} {action_past}."
                }
            else:
                text = await response.text()
                _LOGGER.error("SofaBaton API error: %s - %s", response.status, text)
                return {"error": f"SofaBaton API returned status {response.status}"}

    except Exception as err:
        _LOGGER.error("SofaBaton API call failed: %s", err)
        return {"error": f"Failed to control SofaBaton: {str(err)}"}
