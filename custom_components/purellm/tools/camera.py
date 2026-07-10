"""Camera tool handlers - Frigate integration with local LLM video analysis.

Fetches camera names and RTSP URLs directly from Frigate's ``/api/config``
endpoint so that camera resolution uses the actual names configured in the
Frigate integration rather than requiring manual mapping in PureLLM.

Captures a short sequence of still frames from camera RTSP streams and sends
them as multiple images to a vision LLM for real-time scene analysis.  Sending
images (rather than an MP4) keeps this compatible with image-only inference
backends such as llama.cpp, which cannot decode video.  Live snapshots are
fetched from Frigate's ``/api/<camera>/latest.jpg`` endpoint for still-image
display.
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

# Frame-sequence capture settings.
# Image-only vision backends (e.g. llama.cpp) can't decode video, so instead of
# an MP4 we sample a handful of still frames spread evenly across a short span
# and send them as multiple images.  This gives the LLM temporal context
# (motion, direction, changes) while staying within image-only model limits.
FRAME_COUNT = 12              # number of still frames to capture
FRAME_CAPTURE_DURATION = 4    # seconds spanned by the frames (=> 3 fps)
FRAME_SCALE_HEIGHT = 720      # downscale to 720p (height); width auto by aspect

# Retry settings for go2rtc cold-start latency
MAX_RETRIES_PER_SOURCE = 3
RETRY_DELAY_SECONDS = 3


async def fetch_frigate_cameras(
    session: "aiohttp.ClientSession",
    frigate_url: str,
) -> dict[str, str]:
    """Fetch camera names and RTSP input URLs from Frigate's /api/config.

    Returns a dict mapping ``camera_name -> rtsp_url``.  The RTSP URL is
    taken from the first ``ffmpeg`` input that has the ``detect`` role
    (sub stream — lighter weight, connects faster), falling back to the
    first input with a ``record`` role (main stream), then the first
    input found.
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

        # Prefer input with "detect" role (sub/lighter stream), then "record"
        # (main stream), then first available
        best_url = ""
        for inp in inputs:
            path = inp.get("path", "")
            if not path:
                continue
            roles = inp.get("roles", [])
            if "detect" in roles:
                best_url = path
                break
            if "record" in roles and not best_url:
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


async def _capture_frames(
    rtsp_url: str,
    count: int = FRAME_COUNT,
    duration: int = FRAME_CAPTURE_DURATION,
    scale_height: int = FRAME_SCALE_HEIGHT,
) -> list[bytes] | None:
    """Capture ``count`` JPEG frames evenly spaced over ``duration`` seconds.

    Connects to the camera's RTSP stream and samples frames at ``count/duration``
    fps, returning them as a list of JPEG byte blobs in chronological order so
    they can be sent to the vision LLM as multiple ``image_url`` items.  Frames
    are produced via ffmpeg's ``image2pipe`` muxer (concatenated MJPEG on stdout)
    and split on the JPEG start-of-image marker, avoiding any temp files.
    """
    cmd = [
        "ffmpeg", "-y",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-vf", f"fps={count}/{duration},scale=-2:{scale_height}",
        "-frames:v", str(count),
        "-q:v", "5",
        "-f", "image2pipe",
        "-c:v", "mjpeg",
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
            _LOGGER.error("ffmpeg frame capture failed (rc=%s): %s", proc.returncode, err_msg)
            return None

        if not stdout or len(stdout) < 1000:
            _LOGGER.warning("ffmpeg produced too-small output (%s bytes)", len(stdout) if stdout else 0)
            return None

        # Split the concatenated MJPEG stream into individual JPEGs on the
        # start-of-image marker (0xFFD8).  SOI only appears at the start of each
        # frame (FF bytes inside scan data are byte-stuffed), so this is safe.
        soi = b"\xff\xd8"
        frames = [soi + part for part in stdout.split(soi)[1:] if part]

        if not frames:
            _LOGGER.warning("No JPEG frames parsed from ffmpeg output (%s bytes)", len(stdout))
            return None

        _LOGGER.info(
            "Captured %d frames over %ds (%s bytes total)",
            len(frames), duration, len(stdout),
        )
        return frames

    except asyncio.TimeoutError:
        _LOGGER.error("ffmpeg frame capture timed out after %ds", duration + 20)
        try:
            proc.kill()
        except Exception:
            pass
        return None
    except FileNotFoundError:
        _LOGGER.error("ffmpeg not found - cannot capture frames")
        return None
    except Exception as err:
        _LOGGER.error("Frame capture failed: %s", err, exc_info=True)
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


async def _analyze_images_with_llm(
    session: "aiohttp.ClientSession",
    frames: list[bytes],
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
    location: str,
    query: str = "",
    duration: int = FRAME_CAPTURE_DURATION,
) -> str | None:
    """Send a sequence of JPEG frames to a vision LLM as multiple images.

    Uses the OpenAI-compatible ``image_url`` content type (one entry per frame),
    which image-only backends such as llama.cpp support.  The prompt tells the
    model the frames are chronological so it can infer motion over time.
    """
    prompt = (
        f"You are analyzing {len(frames)} still frames captured in chronological "
        f"order over about {duration} seconds from the {location} security camera. "
        "Treat them as a time sequence so you can perceive motion, direction of "
        "travel, and changes between frames. "
    )
    if query:
        prompt += f"The user specifically wants to know: {query}. "
    prompt += (
        "Describe what you see across the sequence: people, vehicles, animals, "
        "movement and direction, activity, weather/lighting conditions, and "
        "anything notable or unusual. Be concise but descriptive (2-3 sentences)."
    )

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame_bytes in frames:
        b64_image = base64.b64encode(frame_bytes).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
            }
        )

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
                    _LOGGER.error("Vision LLM image API error %s: %s", resp.status, error)
                    return None
                data = await resp.json()

        choices = data.get("choices", [])
        if not choices:
            _LOGGER.warning("Vision LLM returned no choices for image sequence")
            return None

        text = choices[0].get("message", {}).get("content", "").strip()
        return text if text else None

    except Exception as err:
        _LOGGER.error("Vision LLM image analysis failed: %s", err, exc_info=True)
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
) -> dict[str, Any]:
    """Check a camera with image-sequence scene analysis.

    Fetches camera names from Frigate's ``/api/config`` endpoint and captures a
    sequence of still frames via Frigate's go2rtc restream
    (``rtsp://<host>:8554/<cam>``) to avoid competing RTSP connections to the
    NVR.  The frames are sent to the vision LLM as multiple images.

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

    # Build ordered RTSP candidates: sub stream first (lighter, connects faster),
    # then main stream as fallback.  go2rtc exposes sub streams as <cam>_sub
    # when Frigate is configured with a separate detect input.
    frigate_host = urlparse(frigate_url).hostname
    rtsp_candidates = [
        f"rtsp://{frigate_host}:8554/{camera_name}",
        f"rtsp://{frigate_host}:8554/{camera_name}_sub",
    ]

    # Try each RTSP source with retries (handles go2rtc cold-start latency)
    frames = None
    snapshot = None
    last_url = rtsp_candidates[0]

    for rtsp_url in rtsp_candidates:
        last_url = rtsp_url
        for attempt in range(1, MAX_RETRIES_PER_SOURCE + 1):
            _LOGGER.info(
                "Capturing %d frames over %ds from %s (attempt %d/%d, url=%s)",
                FRAME_COUNT, FRAME_CAPTURE_DURATION, camera_name, attempt,
                MAX_RETRIES_PER_SOURCE, rtsp_url,
            )

            # Run frame capture and snapshot fetch in parallel on first attempt
            if snapshot is None:
                frames_task = _capture_frames(rtsp_url)
                snapshot_task = _fetch_frigate_snapshot(session, frigate_url, camera_name)
                frames, snapshot = await asyncio.gather(frames_task, snapshot_task)
            else:
                frames = await _capture_frames(rtsp_url)

            if frames:
                break

            if attempt < MAX_RETRIES_PER_SOURCE:
                _LOGGER.info(
                    "Retry %d/%d for %s in %ds...",
                    attempt, MAX_RETRIES_PER_SOURCE, rtsp_url,
                    RETRY_DELAY_SECONDS,
                )
                await asyncio.sleep(RETRY_DELAY_SECONDS)

        if frames:
            break

    try:
        if not frames:
            _LOGGER.error("All frame capture attempts failed for %s", camera_name)
            return {
                "location": friendly_name,
                "status": "error",
                "source": "none",
                "error": f"Failed to capture frames from {friendly_name} camera after "
                         f"{MAX_RETRIES_PER_SOURCE} retries per source. Last URL: {last_url}",
            }

        snapshot_url = None
        if snapshot and config_dir:
            try:
                snapshot_url = await _save_snapshot(config_dir, camera_name, snapshot)
            except Exception as err:
                _LOGGER.warning("Failed to save snapshot: %s", err)

        # Analyze the frame sequence with the vision LLM
        _LOGGER.info(
            "Sending %d frames to %s for analysis",
            len(frames), llm_model,
        )
        analysis = await _analyze_images_with_llm(
            session, frames, llm_base_url, llm_api_key, llm_model,
            friendly_name, query,
        )

        if not analysis:
            _LOGGER.error("Vision LLM returned no analysis for %s", camera_name)
            return {
                "location": friendly_name,
                "status": "error",
                "source": "none",
                "error": f"Vision LLM failed to analyze frames from {friendly_name} camera. "
                         "The frames were captured but the LLM did not return a response.",
                "snapshot_url": snapshot_url,
            }

        return {
            "location": friendly_name,
            "camera_name": camera_name,
            "status": "checked",
            "source": "image_sequence",
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
