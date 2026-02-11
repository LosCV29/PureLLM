"""Camera tool handlers - Frigate integration with Gemini vision analysis.

Captures snapshots from Frigate and uses Google Gemini to analyze
the scene, providing real visual descriptions instead of just event logs.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)

# Gemini vision defaults
GEMINI_VISION_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_VISION_MODEL = "gemini-2.0-flash"

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


async def _fetch_snapshot(
    session: "aiohttp.ClientSession",
    base_url: str,
    frigate_name: str,
) -> bytes | None:
    """Download a snapshot JPEG from Frigate's latest.jpg endpoint."""
    url = f"{base_url}/api/{frigate_name}/latest.jpg"
    try:
        async with asyncio.timeout(10):
            async with session.get(url) as resp:
                if resp.status != 200:
                    _LOGGER.error("Frigate snapshot returned %s for %s", resp.status, frigate_name)
                    return None
                return await resp.read()
    except Exception as err:
        _LOGGER.error("Failed to fetch snapshot for %s: %s", frigate_name, err)
        return None


async def _capture_frames(
    session: "aiohttp.ClientSession",
    base_url: str,
    frigate_name: str,
    count: int = 3,
    interval: float = 2.0,
) -> list[bytes]:
    """Capture multiple snapshot frames over time for scene analysis."""
    frames = []
    for i in range(count):
        frame = await _fetch_snapshot(session, base_url, frigate_name)
        if frame:
            frames.append(frame)
        if i < count - 1:
            await asyncio.sleep(interval)
    return frames


async def _analyze_with_gemini(
    session: "aiohttp.ClientSession",
    frames: list[bytes],
    google_api_key: str,
    location: str,
    query: str = "",
) -> str | None:
    """Send camera frames to Gemini for vision analysis."""
    parts = []

    prompt = (
        f"You are analyzing live security camera frames from the {location} camera. "
        f"There are {len(frames)} frame(s) captured over a few seconds. "
    )
    if query:
        prompt += f"The user specifically wants to know: {query}. "
    prompt += (
        "Describe what you see: people, vehicles, animals, activity, weather/lighting "
        "conditions, and anything notable. Be concise but descriptive (2-3 sentences)."
    )
    parts.append({"text": prompt})

    for frame in frames:
        b64 = base64.b64encode(frame).decode("utf-8")
        parts.append({
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": b64,
            }
        })

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"maxOutputTokens": 300, "temperature": 0.3},
    }

    url = f"{GEMINI_VISION_URL}/models/{GEMINI_VISION_MODEL}:generateContent"
    headers = {"x-goog-api-key": google_api_key}

    try:
        async with asyncio.timeout(30):
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    _LOGGER.error("Gemini vision API error %s: %s", resp.status, error)
                    return None
                data = await resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            _LOGGER.warning("Gemini vision returned no candidates")
            return None

        response_parts = candidates[0].get("content", {}).get("parts", [])
        text = " ".join(p["text"] for p in response_parts if "text" in p).strip()
        return text if text else None

    except Exception as err:
        _LOGGER.error("Gemini vision analysis failed: %s", err, exc_info=True)
        return None


def _save_snapshot(config_dir: str, frigate_name: str, image_bytes: bytes) -> str:
    """Save snapshot to HA's www directory and return the local URL.

    Saves to /config/www/purellm/camera_{name}.jpg so it's accessible
    via /local/purellm/camera_{name}.jpg through HA's web server.
    """
    www_dir = os.path.join(config_dir, "www", "purellm")
    os.makedirs(www_dir, exist_ok=True)

    # Use timestamp to bust cache
    filename = f"camera_{frigate_name}.jpg"
    filepath = os.path.join(www_dir, filename)

    with open(filepath, "wb") as f:
        f.write(image_bytes)

    return f"/local/purellm/{filename}?t={int(time.time())}"


async def _get_frigate_events_description(
    session: "aiohttp.ClientSession",
    base_url: str,
    frigate_name: str,
) -> str:
    """Fallback: get description from Frigate event log (original behavior)."""
    after_time = time.time() - 300
    events_url = (
        f"{base_url}/api/events"
        f"?cameras={frigate_name}"
        f"&limit=5"
        f"&after={after_time:.0f}"
        f"&include_thumbnails=0"
    )

    try:
        async with asyncio.timeout(15):
            async with session.get(events_url) as resp:
                if resp.status != 200:
                    return "Could not access Frigate events."
                events = await resp.json()

        descriptions = []
        detected_objects = []
        for event in events:
            label = event.get("label", "")
            if label:
                detected_objects.append(label)
            data = event.get("data") or {}
            desc = data.get("description", "")
            if desc and desc not in descriptions:
                descriptions.append(desc)

        if descriptions:
            return " ".join(descriptions)
        elif detected_objects:
            obj_summary = ", ".join(set(detected_objects))
            return f"Detected: {obj_summary}."
        else:
            return "No recent activity detected."

    except Exception as err:
        _LOGGER.error("Frigate events query failed: %s", err)
        return "Could not retrieve camera events."


async def check_camera(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    frigate_url: str,
    frigate_camera_names: dict[str, str] | None = None,
    camera_friendly_names: dict[str, str] | None = None,
    google_api_key: str = "",
    config_dir: str = "",
) -> dict[str, Any]:
    """Check a camera with visual scene analysis.

    Captures multiple snapshots from Frigate over ~4 seconds, sends them
    to Google Gemini for vision analysis, and returns a scene description.
    Falls back to Frigate event descriptions if Gemini is not available.

    Args:
        arguments: Tool arguments (location, query)
        session: aiohttp client session
        frigate_url: Frigate API base URL
        frigate_camera_names: Mapping of location_key -> frigate_camera_name
        camera_friendly_names: Custom camera name mappings
        google_api_key: Google API key for Gemini vision analysis
        config_dir: HA config directory path for saving snapshots

    Returns:
        Camera analysis dict with description and snapshot URL
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
        # Capture frames for analysis
        if google_api_key:
            _LOGGER.info("Capturing %d frames from %s for vision analysis", 3, frigate_name)
            frames = await _capture_frames(session, base_url, frigate_name, count=3, interval=2.0)
        else:
            _LOGGER.info("No vision API key - capturing single frame from %s", frigate_name)
            frames = await _capture_frames(session, base_url, frigate_name, count=1, interval=0)

        if not frames:
            return {
                "location": friendly_name,
                "status": "unavailable",
                "error": f"Could not get snapshot from {friendly_name} camera",
            }

        # Save latest snapshot for notification
        snapshot_url = f"{base_url}/api/{frigate_name}/latest.jpg"
        if config_dir:
            try:
                snapshot_url = _save_snapshot(config_dir, frigate_name, frames[-1])
                _LOGGER.debug("Saved snapshot to %s", snapshot_url)
            except Exception as err:
                _LOGGER.warning("Failed to save snapshot locally: %s, using Frigate URL", err)

        # Vision analysis with Gemini
        if google_api_key and frames:
            _LOGGER.info("Sending %d frame(s) to Gemini for vision analysis", len(frames))
            analysis = await _analyze_with_gemini(
                session, frames, google_api_key, friendly_name, query
            )
            if not analysis:
                _LOGGER.warning("Gemini vision failed, falling back to Frigate events")
                analysis = await _get_frigate_events_description(session, base_url, frigate_name)
        else:
            # Fallback to Frigate event descriptions
            analysis = await _get_frigate_events_description(session, base_url, frigate_name)

        return {
            "location": friendly_name,
            "status": "checked",
            "description": analysis,
            "snapshot_url": snapshot_url,
        }

    except Exception as err:
        _LOGGER.error("Error checking camera %s: %s", location, err, exc_info=True)
        return {
            "location": friendly_name,
            "status": "error",
            "error": f"Failed to check {friendly_name} camera: {str(err)}",
        }


async def quick_camera_check(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    frigate_url: str,
    frigate_camera_names: dict[str, str] | None = None,
    camera_friendly_names: dict[str, str] | None = None,
    google_api_key: str = "",
    config_dir: str = "",
) -> dict[str, Any]:
    """Fast camera check - single snapshot with brief vision analysis.

    Args:
        arguments: Tool arguments (location)
        session: aiohttp client session
        frigate_url: Frigate API base URL
        frigate_camera_names: Mapping of location_key -> frigate_camera_name
        camera_friendly_names: Custom camera name mappings
        google_api_key: Google API key for Gemini vision analysis
        config_dir: HA config directory path for saving snapshots

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
        # Single snapshot for quick check
        frame = await _fetch_snapshot(session, base_url, frigate_name)

        if not frame:
            return {"location": friendly_name, "error": "Camera unavailable"}

        # Save snapshot for notification
        snapshot_url = f"{base_url}/api/{frigate_name}/latest.jpg"
        if config_dir:
            try:
                snapshot_url = _save_snapshot(config_dir, frigate_name, frame)
            except Exception:
                pass

        # Quick vision analysis with Gemini (single frame)
        if google_api_key:
            analysis = await _analyze_with_gemini(
                session, [frame], google_api_key, friendly_name
            )
            if analysis:
                # Truncate to first sentence for quick check
                brief = analysis.split('.')[0] + '.' if '.' in analysis else analysis
            else:
                brief = await _get_quick_frigate_description(session, base_url, frigate_name)
        else:
            brief = await _get_quick_frigate_description(session, base_url, frigate_name)

        return {
            "location": friendly_name,
            "brief": brief,
            "snapshot_url": snapshot_url,
        }

    except Exception as err:
        _LOGGER.error("Error quick-checking camera %s: %s", location, err)
        return {"location": friendly_name, "error": "Check failed"}


async def _get_quick_frigate_description(
    session: "aiohttp.ClientSession",
    base_url: str,
    frigate_name: str,
) -> str:
    """Fallback quick description from Frigate events."""
    after_time = time.time() - 120
    events_url = (
        f"{base_url}/api/events"
        f"?cameras={frigate_name}"
        f"&limit=1"
        f"&after={after_time:.0f}"
        f"&include_thumbnails=0"
    )

    try:
        async with asyncio.timeout(10):
            async with session.get(events_url) as resp:
                if resp.status != 200:
                    return "Camera unavailable."
                events = await resp.json()

        if events:
            event = events[0]
            label = event.get("label", "unknown")
            data = event.get("data") or {}
            desc = data.get("description", "")
            if desc:
                return desc.split('.')[0] + '.' if '.' in desc else desc
            return f"{label.title()} detected."
        return "No recent activity."

    except Exception:
        return "Could not check camera."
