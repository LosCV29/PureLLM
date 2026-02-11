"""Camera tool handlers - Frigate integration.

Uses Frigate's HTTP API for camera checks with GenAI descriptions and snapshots.
"""
from __future__ import annotations

import logging
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)

# Default camera friendly names
DEFAULT_CAMERA_FRIENDLY_NAMES = {
    "porch": "Front Porch",
    "driveway": "Driveway",
    "garage": "Garage",
    "backyard": "Backyard",
    "kitchen": "Kitchen",
    "living_room": "Living Room",
    "front_door": "Front Door",
}


def _resolve_camera(
    location: str,
    frigate_camera_names: dict[str, str] | None,
    camera_friendly_names: dict[str, str] | None,
) -> tuple[str | None, str]:
    """Resolve a location key to Frigate camera name and friendly name.

    Args:
        location: Location key from LLM (e.g., "porch")
        frigate_camera_names: Mapping of location_key -> frigate_camera_name
        camera_friendly_names: Mapping of location_key -> Friendly Name

    Returns:
        Tuple of (frigate_camera_name, friendly_display_name)
    """
    friendly_names = camera_friendly_names or DEFAULT_CAMERA_FRIENDLY_NAMES
    friendly_name = friendly_names.get(location, location.replace("_", " ").title())

    frigate_names = frigate_camera_names or {}
    frigate_name = frigate_names.get(location)

    if not frigate_name:
        # Fall back: try the location key directly as the Frigate camera name
        frigate_name = location

    return frigate_name, friendly_name


async def check_camera(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    frigate_url: str,
    frigate_camera_names: dict[str, str] | None = None,
    camera_friendly_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Check a camera with AI analysis via Frigate.

    Gets recent events with GenAI descriptions and a snapshot from Frigate.

    Args:
        arguments: Tool arguments (location, query)
        session: aiohttp client session
        frigate_url: Frigate API base URL (e.g., http://192.168.1.100:5000)
        frigate_camera_names: Mapping of location_key -> frigate_camera_name
        camera_friendly_names: Custom camera name mappings

    Returns:
        Camera analysis dict
    """
    location = arguments.get("location", "").lower().strip()
    query = arguments.get("query", "")

    if not location:
        return {"error": "No camera location specified"}

    if not frigate_url:
        return {"error": "Frigate URL not configured"}

    frigate_name, friendly_name = _resolve_camera(
        location, frigate_camera_names, camera_friendly_names
    )

    base_url = frigate_url.rstrip("/")

    try:
        # Get recent events for this camera (last 5 minutes)
        after_time = time.time() - 300
        events_url = (
            f"{base_url}/api/events"
            f"?cameras={frigate_name}"
            f"&limit=5"
            f"&after={after_time:.0f}"
            f"&include_thumbnails=0"
        )

        import asyncio
        async with asyncio.timeout(15):
            async with session.get(events_url) as resp:
                if resp.status != 200:
                    _LOGGER.error(
                        "Frigate events API returned %s for camera %s",
                        resp.status, frigate_name
                    )
                    return {
                        "location": friendly_name,
                        "status": "unavailable",
                        "error": f"Could not access {friendly_name} camera (HTTP {resp.status})"
                    }
                events = await resp.json()

        # Build description from event GenAI descriptions
        descriptions = []
        detected_objects = []
        for event in events:
            label = event.get("label", "")
            if label:
                detected_objects.append(label)

            # GenAI description is in data.description
            data = event.get("data") or {}
            desc = data.get("description", "")
            if desc and desc not in descriptions:
                descriptions.append(desc)

        if descriptions:
            analysis = " ".join(descriptions)
        elif detected_objects:
            obj_summary = ", ".join(set(detected_objects))
            analysis = f"Detected: {obj_summary}. No detailed AI descriptions available for recent events."
        else:
            analysis = "No recent activity detected."

        snapshot_url = f"{base_url}/api/{frigate_name}/latest.jpg"

        response = {
            "location": friendly_name,
            "status": "checked",
            "description": analysis,
            "snapshot_url": snapshot_url,
        }

        # Include detected object labels
        if detected_objects:
            response["detected_objects"] = list(set(detected_objects))

        return response

    except Exception as err:
        _LOGGER.error("Error checking camera %s: %s", location, err, exc_info=True)
        return {
            "location": friendly_name,
            "status": "error",
            "error": f"Failed to check {friendly_name} camera: {str(err)}"
        }


async def quick_camera_check(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    frigate_url: str,
    frigate_camera_names: dict[str, str] | None = None,
    camera_friendly_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Fast camera check - recent detections + one sentence via Frigate.

    Args:
        arguments: Tool arguments (location)
        session: aiohttp client session
        frigate_url: Frigate API base URL
        frigate_camera_names: Mapping of location_key -> frigate_camera_name
        camera_friendly_names: Custom camera name mappings

    Returns:
        Brief camera check dict
    """
    location = arguments.get("location", "").lower().strip()

    if not location:
        return {"error": "No camera location specified"}

    if not frigate_url:
        return {"error": "Frigate URL not configured"}

    frigate_name, friendly_name = _resolve_camera(
        location, frigate_camera_names, camera_friendly_names
    )

    base_url = frigate_url.rstrip("/")

    try:
        # Get most recent event for this camera (last 2 minutes)
        after_time = time.time() - 120
        events_url = (
            f"{base_url}/api/events"
            f"?cameras={frigate_name}"
            f"&limit=1"
            f"&after={after_time:.0f}"
            f"&include_thumbnails=0"
        )

        import asyncio
        async with asyncio.timeout(10):
            async with session.get(events_url) as resp:
                if resp.status != 200:
                    return {"location": friendly_name, "error": "Camera unavailable"}
                events = await resp.json()

        snapshot_url = f"{base_url}/api/{frigate_name}/latest.jpg"

        if events:
            event = events[0]
            label = event.get("label", "unknown")
            data = event.get("data") or {}
            desc = data.get("description", "")

            if desc:
                brief = desc.split('.')[0] + '.' if '.' in desc else desc
            else:
                brief = f"{label.title()} detected."
        else:
            brief = "No recent activity."

        response = {
            "location": friendly_name,
            "brief": brief,
            "snapshot_url": snapshot_url,
        }

        return response

    except Exception as err:
        _LOGGER.error("Error quick-checking camera %s: %s", location, err)
        return {"location": friendly_name, "error": "Check failed"}
