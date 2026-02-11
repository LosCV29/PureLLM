"""Camera tool handlers - Frigate integration with local LLM video analysis.

Fetches camera names and RTSP URLs directly from Frigate's ``/api/config``
endpoint so that camera resolution uses the actual names configured in the
Frigate integration rather than requiring manual mapping in PureLLM.

Captures video clips from camera RTSP streams and sends them to Qwen3-VL
(via vLLM) for real-time scene analysis.  Live snapshots are fetched from
Frigate's ``/api/<camera>/latest.jpg`` endpoint for still-image display.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from typing import Any, TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)

# Video clip settings
VIDEO_CLIP_DURATION = 3  # seconds of video to capture


async def fetch_frigate_cameras(
    session: "aiohttp.ClientSession",
    frigate_url: str,
) -> dict[str, str]:
    """Fetch camera names and RTSP input URLs from Frigate's /api/config.

    Returns a dict mapping ``camera_name -> rtsp_url``.  The RTSP URL is
    taken from the first ``ffmpeg`` input that has the ``record`` role,
    falling back to the first input with any ``detect`` role, then the
    first input found.
    """
    if not frigate_url:
        return {}

    url = f"{frigate_url.rstrip('/')}/api/config"
    try:
        async with asyncio.timeout(10):
            async with session.get(url) as resp:
                if resp.status != 200:
                    _LOGGER.warning("Frigate config request failed (%s)", resp.status)
                    return {}
                config = await resp.json()
    except Exception as err:
        _LOGGER.warning("Failed to fetch Frigate config: %s", err)
        return {}

    cameras: dict[str, str] = {}
    for cam_name, cam_config in config.get("cameras", {}).items():
        inputs = cam_config.get("ffmpeg", {}).get("inputs", [])
        if not inputs:
            cameras[cam_name] = ""
            continue

        # Prefer input with "record" role, then "detect", then first available
        best_url = ""
        for inp in inputs:
            path = inp.get("path", "")
            if not path:
                continue
            roles = inp.get("roles", [])
            if "record" in roles:
                best_url = path
                break
            if "detect" in roles and not best_url:
                best_url = path
            if not best_url:
                best_url = path

        cameras[cam_name] = best_url

    for cam, url in cameras.items():
        _LOGGER.debug("Frigate camera: %s -> %s", cam, url)
    _LOGGER.debug("Fetched %d cameras from Frigate: %s", len(cameras), list(cameras.keys()))
    return cameras


def _match_camera(
    location: str,
    frigate_cameras: dict[str, str],
) -> str | None:
    """Match a spoken location to an actual Frigate camera name.

    Matching strategy (first match wins):
      1. Exact match against Frigate camera names.
      2. Normalised match (underscores ↔ spaces, strip common suffixes
         like "camera", "cam").
      3. Substring: a camera name is contained in the location or vice-versa.
    """
    if not frigate_cameras:
        return None

    cam_names = list(frigate_cameras.keys())

    # Strip common suffixes from user's spoken location
    loc = location.lower().strip()
    for suffix in (" camera", " cam"):
        if loc.endswith(suffix):
            loc = loc[: -len(suffix)].strip()

    # Build normalised lookup mapping multiple forms to the real camera name.
    # For "back_yard" this produces: "back_yard", "back yard", "backyard"
    norm_to_real: dict[str, str] = {}
    for name in cam_names:
        norm_to_real[name] = name
        norm_to_real[name.replace("_", " ")] = name
        norm_to_real[name.replace("_", "")] = name  # collapsed form

    # Also try collapsed form of user input ("back yard" -> "backyard")
    loc_under = loc.replace(" ", "_")
    loc_collapsed = loc.replace(" ", "").replace("_", "")

    # 1. Exact / normalised match
    for variant in (loc, loc_under, loc_collapsed):
        if variant in norm_to_real:
            return norm_to_real[variant]

    # 2. Substring: camera name is in location or location is in camera name
    #    Longest match wins for specificity.
    candidates = [(n, real) for n, real in norm_to_real.items() if n in loc]
    if candidates:
        best = max(candidates, key=lambda x: len(x[0]))
        return best[1]

    candidates = [(n, real) for n, real in norm_to_real.items() if loc in n]
    if candidates:
        best = min(candidates, key=lambda x: len(x[0]))
        return best[1]

    return None


async def _capture_video_clip(
    rtsp_url: str,
    duration: int = VIDEO_CLIP_DURATION,
) -> bytes | None:
    """Capture a video clip from a direct RTSP stream using ffmpeg.

    Connects to the camera's native RTSP stream and transcodes to a compact
    MP4 suitable for sending to a vision LLM.  The MP4 uses fragmented
    format so it can be piped to stdout without seeking.
    """
    cmd = [
        "ffmpeg", "-y",
        "-rtsp_transport", "tcp",
        "-t", str(duration),
        "-i", rtsp_url,
        "-vf", "scale=640:-2",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "30",
        "-an",
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
            proc.communicate(), timeout=duration + 15
        )

        if proc.returncode != 0:
            err_msg = stderr.decode(errors="replace")[-500:] if stderr else "unknown"
            _LOGGER.error("ffmpeg clip capture failed (rc=%s): %s", proc.returncode, err_msg)
            return None

        if not stdout or len(stdout) < 1000:
            _LOGGER.warning("ffmpeg produced too-small output (%s bytes)", len(stdout) if stdout else 0)
            return None

        _LOGGER.info(
            "Captured %ds video clip (%s bytes)",
            duration, len(stdout),
        )
        return stdout

    except asyncio.TimeoutError:
        _LOGGER.error("ffmpeg clip capture timed out after %ds", duration + 15)
        try:
            proc.kill()
        except Exception:
            pass
        return None
    except FileNotFoundError:
        _LOGGER.error("ffmpeg not found - cannot capture video clips")
        return None
    except Exception as err:
        _LOGGER.error("Video clip capture failed: %s", err, exc_info=True)
        return None


async def _fetch_frigate_snapshot(
    session: "aiohttp.ClientSession",
    frigate_url: str,
    camera_name: str,
    height: int = 720,
) -> bytes | None:
    """Fetch a live snapshot from Frigate's latest-frame API.

    Uses ``GET /api/<camera>/latest.jpg?h=<height>`` which returns the
    current frame from the detect stream as a JPEG.  This is far more
    reliable than trying to extract a frame from a captured RTSP clip.
    """
    url = f"{frigate_url.rstrip('/')}/api/{camera_name}/latest.jpg?h={height}"
    try:
        async with asyncio.timeout(10):
            async with session.get(url) as resp:
                if resp.status != 200:
                    _LOGGER.warning(
                        "Frigate snapshot request failed (%s) for %s",
                        resp.status, camera_name,
                    )
                    return None
                data = await resp.read()
                if not data or len(data) < 100:
                    _LOGGER.warning("Frigate returned empty snapshot for %s", camera_name)
                    return None
                _LOGGER.info(
                    "Fetched live snapshot from Frigate for %s (%s bytes)",
                    camera_name, len(data),
                )
                return data
    except Exception as err:
        _LOGGER.warning("Failed to fetch Frigate snapshot for %s: %s", camera_name, err)
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



def _save_snapshot_sync(config_dir: str, frigate_name: str, image_bytes: bytes) -> str:
    """Save snapshot to HA's www directory and return an API URL.

    Uses the /api/purellm/camera/<name>/snapshot endpoint (registered
    in __init__.py) instead of /local/ because HA only registers the
    /local static route if the www/ directory existed at startup.
    """
    www_dir = os.path.join(config_dir, "www", "purellm")
    os.makedirs(www_dir, exist_ok=True)

    filename = f"camera_{frigate_name}.jpg"
    filepath = os.path.join(www_dir, filename)

    with open(filepath, "wb") as f:
        f.write(image_bytes)

    return f"/api/purellm/camera/{frigate_name}/snapshot.jpg?t={int(time.time())}"


async def _save_snapshot(config_dir: str, frigate_name: str, image_bytes: bytes) -> str:
    """Save snapshot to HA's www directory without blocking the event loop."""
    return await asyncio.get_running_loop().run_in_executor(
        None, _save_snapshot_sync, config_dir, frigate_name, image_bytes
    )



async def check_camera(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    frigate_url: str,
    llm_base_url: str = "",
    llm_api_key: str = "",
    llm_model: str = "",
    config_dir: str = "",
    camera_rtsp_urls: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Check a camera with video scene analysis.

    Fetches camera names from Frigate's ``/api/config`` endpoint and
    captures video via Frigate's go2rtc restream (``rtsp://<host>:8554/<cam>``)
    to avoid competing RTSP connections to the NVR.

    A live snapshot for display is fetched from Frigate's
    ``/api/<camera>/latest.jpg`` endpoint.
    """
    location = arguments.get("location", "").lower().strip()
    query = arguments.get("query", "")

    if not location:
        return {"error": "No camera location specified"}

    if not llm_base_url or not llm_model:
        return {"error": "Vision LLM not configured"}

    if not frigate_url:
        return {"error": "Frigate URL not configured"}

    # Fetch actual cameras from Frigate - this is the source of truth
    frigate_cameras = await fetch_frigate_cameras(session, frigate_url)

    if not frigate_cameras:
        return {
            "status": "error",
            "error": "Could not fetch cameras from Frigate. Check that the Frigate URL is correct and Frigate is running.",
        }

    # Match spoken location to a Frigate camera name
    camera_name = _match_camera(location, frigate_cameras)

    if not camera_name:
        available = ", ".join(frigate_cameras.keys())
        return {
            "status": "error",
            "error": f"No camera matching '{location}' found in Frigate. Available cameras: {available}.",
        }

    # Display name derived from Frigate camera name
    friendly_name = camera_name.replace("_", " ").title()

    # Use Frigate's go2rtc restream instead of the NVR's direct RTSP URL.
    # This avoids opening a competing RTSP connection to the NVR which
    # causes intermittent failures when the NVR's connection limit is hit.
    frigate_host = urlparse(frigate_url).hostname
    rtsp_url = f"rtsp://{frigate_host}:8554/{camera_name}"

    try:
        _LOGGER.info("Capturing %ds video clip from %s via Frigate restream (url=%s)", VIDEO_CLIP_DURATION, camera_name, rtsp_url)

        # Run video capture and snapshot fetch in parallel
        video_task = _capture_video_clip(rtsp_url, VIDEO_CLIP_DURATION)
        snapshot_task = _fetch_frigate_snapshot(session, frigate_url, camera_name)
        video_clip, snapshot = await asyncio.gather(video_task, snapshot_task)

        if not video_clip:
            _LOGGER.error("Video clip capture failed for %s (url=%s)", camera_name, rtsp_url)
            return {
                "location": friendly_name,
                "status": "error",
                "source": "none",
                "error": f"Failed to capture video clip from {friendly_name} camera. "
                         f"ffmpeg could not connect to the RTSP stream. URL: {rtsp_url}",
            }

        snapshot_url = None
        if snapshot and config_dir:
            try:
                snapshot_url = await _save_snapshot(config_dir, camera_name, snapshot)
            except Exception as err:
                _LOGGER.warning("Failed to save snapshot: %s", err)

        # Analyze with vision LLM
        _LOGGER.info(
            "Sending video clip (%s bytes) to %s for analysis",
            len(video_clip), llm_model,
        )
        analysis = await _analyze_video_with_llm(
            session, video_clip, llm_base_url, llm_api_key, llm_model,
            friendly_name, query,
        )

        if not analysis:
            _LOGGER.error("Vision LLM returned no analysis for %s", camera_name)
            return {
                "location": friendly_name,
                "status": "error",
                "source": "none",
                "error": f"Vision LLM failed to analyze video from {friendly_name} camera. "
                         "The video was captured but the LLM did not return a response.",
                "snapshot_url": snapshot_url,
            }

        return {
            "location": friendly_name,
            "camera_name": camera_name,
            "status": "checked",
            "source": "video_clip",
            "description": analysis,
            "snapshot_url": snapshot_url,
        }

    except Exception as err:
        _LOGGER.error("Error checking camera %s: %s", camera_name, err, exc_info=True)
        return {
            "location": friendly_name,
            "status": "error",
            "error": f"Failed to check {friendly_name} camera: {str(err)}",
        }
