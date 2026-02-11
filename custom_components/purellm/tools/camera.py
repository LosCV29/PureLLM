"""Camera tool handlers - Frigate integration with local LLM video analysis.

Captures video clips from Frigate and sends them to Qwen3-VL (via vLLM)
for real-time scene analysis using the OpenAI-compatible video_url format.
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

# Video clip settings
VIDEO_CLIP_DURATION = 5  # seconds of video to capture

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
    """Resolve a location key to Frigate camera name and friendly name."""
    friendly_names = camera_friendly_names or DEFAULT_CAMERA_FRIENDLY_NAMES
    friendly_name = friendly_names.get(location, location.replace("_", " ").title())

    frigate_names = frigate_camera_names or {}
    frigate_name = frigate_names.get(location)

    if not frigate_name:
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


async def _capture_video_clip(
    base_url: str,
    frigate_name: str,
    duration: int = VIDEO_CLIP_DURATION,
) -> bytes | None:
    """Capture a video clip from Frigate's MJPEG stream using ffmpeg.

    Uses ffmpeg to read the live MJPEG stream and transcode to a compact
    MP4 suitable for sending to a vision LLM. The MP4 uses fragmented
    format so it can be piped to stdout without seeking.
    """
    mjpeg_url = f"{base_url}/api/{frigate_name}"

    cmd = [
        "ffmpeg", "-y",
        "-t", str(duration),
        "-i", mjpeg_url,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-vf", "scale='min(1280,iw)':'-2'",
        "-movflags", "frag_keyframe+empty_moov",
        "-f", "mp4",
        "pipe:1",
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=duration + 20
        )

        if proc.returncode != 0:
            err_msg = stderr.decode(errors="replace")[-500:] if stderr else "unknown"
            _LOGGER.error("ffmpeg clip capture failed (rc=%s): %s", proc.returncode, err_msg)
            return None

        if not stdout or len(stdout) < 1000:
            _LOGGER.warning("ffmpeg produced too-small output (%s bytes)", len(stdout) if stdout else 0)
            return None

        _LOGGER.info(
            "Captured %ds video clip from %s (%s bytes)",
            duration, frigate_name, len(stdout),
        )
        return stdout

    except asyncio.TimeoutError:
        _LOGGER.error("ffmpeg clip capture timed out after %ds for %s", duration + 20, frigate_name)
        try:
            proc.kill()
        except Exception:
            pass
        return None
    except FileNotFoundError:
        _LOGGER.error("ffmpeg not found - cannot capture video clips")
        return None
    except Exception as err:
        _LOGGER.error("Video clip capture failed for %s: %s", frigate_name, err, exc_info=True)
        return None


async def _analyze_video_with_llm(
    session: "aiohttp.ClientSession",
    video_bytes: bytes,
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
    location: str,
    query: str = "",
) -> str | None:
    """Send an MP4 video clip to Qwen3-VL via vLLM's video_url content type."""
    prompt = (
        f"You are analyzing a live video clip from the {location} security camera. "
        f"This is {VIDEO_CLIP_DURATION} seconds of real-time footage. "
    )
    if query:
        prompt += f"The user specifically wants to know: {query}. "
    prompt += (
        "Describe what you see: people, vehicles, animals, movement, activity, "
        "weather/lighting conditions, and anything notable or unusual. "
        "Be concise but descriptive (2-3 sentences)."
    )

    b64_video = base64.b64encode(video_bytes).decode("utf-8")

    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{b64_video}"},
        },
    ]

    payload = {
        "model": llm_model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 300,
        "temperature": 0.3,
    }

    url = f"{llm_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {llm_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with asyncio.timeout(90):
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    _LOGGER.error("Vision LLM video API error %s: %s", resp.status, error)
                    return None
                data = await resp.json()

        choices = data.get("choices", [])
        if not choices:
            _LOGGER.warning("Vision LLM returned no choices for video")
            return None

        text = choices[0].get("message", {}).get("content", "").strip()
        return text if text else None

    except Exception as err:
        _LOGGER.error("Vision LLM video analysis failed: %s", err, exc_info=True)
        return None


async def _analyze_snapshot_with_llm(
    session: "aiohttp.ClientSession",
    frame: bytes,
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
    location: str,
    query: str = "",
) -> str | None:
    """Fallback: send a single snapshot image to the vision LLM."""
    prompt = (
        f"You are analyzing a snapshot from the {location} security camera. "
    )
    if query:
        prompt += f"The user specifically wants to know: {query}. "
    prompt += (
        "Describe what you see: people, vehicles, animals, activity, weather/lighting "
        "conditions, and anything notable. Be concise (1-2 sentences)."
    )

    b64 = base64.b64encode(frame).decode("utf-8")

    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        },
    ]

    payload = {
        "model": llm_model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 300,
        "temperature": 0.3,
    }

    url = f"{llm_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {llm_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with asyncio.timeout(60):
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    _LOGGER.error("Vision LLM snapshot API error %s: %s", resp.status, error)
                    return None
                data = await resp.json()

        choices = data.get("choices", [])
        if not choices:
            return None

        text = choices[0].get("message", {}).get("content", "").strip()
        return text if text else None

    except Exception as err:
        _LOGGER.error("Vision LLM snapshot analysis failed: %s", err, exc_info=True)
        return None


def _save_snapshot(config_dir: str, frigate_name: str, image_bytes: bytes) -> str:
    """Save snapshot to HA's www directory and return the local URL."""
    www_dir = os.path.join(config_dir, "www", "purellm")
    os.makedirs(www_dir, exist_ok=True)

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
    """Fallback: get description from Frigate event log."""
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
    llm_base_url: str = "",
    llm_api_key: str = "",
    llm_model: str = "",
    config_dir: str = "",
) -> dict[str, Any]:
    """Check a camera with video scene analysis.

    Captures a ~10s video clip from Frigate via ffmpeg, sends it to
    Qwen3-VL for analysis. Falls back to snapshot analysis if ffmpeg
    is unavailable, then to Frigate events if no LLM is configured.
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
    has_vision = bool(llm_base_url and llm_model)

    try:
        # Always grab a snapshot for notifications
        snapshot = await _fetch_snapshot(session, base_url, frigate_name)

        snapshot_url = f"{base_url}/api/{frigate_name}/latest.jpg"
        if snapshot and config_dir:
            try:
                snapshot_url = _save_snapshot(config_dir, frigate_name, snapshot)
            except Exception as err:
                _LOGGER.warning("Failed to save snapshot: %s", err)

        if not has_vision:
            if not snapshot:
                return {
                    "location": friendly_name,
                    "status": "unavailable",
                    "error": f"Could not get snapshot from {friendly_name} camera",
                }
            analysis = await _get_frigate_events_description(session, base_url, frigate_name)
            return {
                "location": friendly_name,
                "status": "checked",
                "description": analysis,
                "snapshot_url": snapshot_url,
            }

        # --- Vision analysis path ---
        analysis = None

        # Try video clip first (primary)
        _LOGGER.info("Capturing %ds video clip from %s", VIDEO_CLIP_DURATION, frigate_name)
        video_clip = await _capture_video_clip(base_url, frigate_name, VIDEO_CLIP_DURATION)

        if video_clip:
            _LOGGER.info(
                "Sending video clip (%s bytes) to %s for analysis",
                len(video_clip), llm_model,
            )
            analysis = await _analyze_video_with_llm(
                session, video_clip, llm_base_url, llm_api_key, llm_model,
                friendly_name, query,
            )

        # Fallback to snapshot if video failed
        if not analysis and snapshot:
            _LOGGER.warning("Video analysis unavailable, falling back to snapshot")
            analysis = await _analyze_snapshot_with_llm(
                session, snapshot, llm_base_url, llm_api_key, llm_model,
                friendly_name, query,
            )

        # Final fallback to Frigate events
        if not analysis:
            _LOGGER.warning("Vision LLM failed, falling back to Frigate events")
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


