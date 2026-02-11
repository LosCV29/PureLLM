"""Camera tool handlers - Frigate integration with local LLM video analysis.

Captures video clips directly from camera RTSP streams and sends them to
Qwen3-VL (via vLLM) for real-time scene analysis using the OpenAI-compatible
video_url format.  Snapshots are extracted from the captured clip so they are
guaranteed fresh (not Frigate's cached latest.jpg).
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



def _best_key_match(lookup: dict[str, str], location: str) -> str | None:
    """Find the best matching key in *lookup* for the given location string.

    Tries, in order:
      1. Exact match.
      2. A config key that is a substring of *location* (e.g. key ``backyard``
         matches location ``backyard camera``).  Longest match wins so that
         ``front porch`` beats ``front``.
      3. *location* is a substring of a config key.
    """
    if location in lookup:
        return location

    # Normalise underscores so "back_yard" can match "back yard camera"
    norm = {k.replace("_", " "): k for k in lookup}

    # key is a substring of location  (longest first → most specific)
    candidates = [n for n in norm if n in location]
    if candidates:
        best = max(candidates, key=len)
        return norm[best]

    # location is a substring of a key
    candidates = [n for n in norm if location in n]
    if candidates:
        best = min(candidates, key=len)
        return norm[best]

    return None


def _resolve_camera(
    location: str,
    frigate_camera_names: dict[str, str] | None,
    camera_friendly_names: dict[str, str] | None,
) -> tuple[str | None, str]:
    """Resolve a location key to Frigate camera name and friendly name.

    Uses flexible substring matching so that e.g. a user saying
    "backyard camera" correctly resolves to the ``backyard`` config key.
    """
    friendly_names = camera_friendly_names or {}
    frigate_names = frigate_camera_names or {}

    matched_key = _best_key_match(frigate_names, location)
    frigate_name = frigate_names[matched_key] if matched_key else location

    friendly_name = friendly_names.get(
        frigate_name,
        friendly_names.get(
            matched_key or location,
            frigate_name.replace("_", " ").title(),
        ),
    )

    return frigate_name, friendly_name


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
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
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
            "Captured %ds video clip (%s bytes)",
            duration, len(stdout),
        )
        return stdout

    except asyncio.TimeoutError:
        _LOGGER.error("ffmpeg clip capture timed out after %ds", duration + 20)
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


async def _extract_snapshot_from_clip(
    video_bytes: bytes,
    clip_duration: int = VIDEO_CLIP_DURATION,
) -> bytes | None:
    """Extract the last frame of a captured MP4 clip as a JPEG.

    Pipes the MP4 through ffmpeg and grabs a frame near the end.
    Uses -ss with a known offset instead of -sseof because piped
    fragmented MP4s are not seekable from the end.
    """
    seek_pos = max(clip_duration - 1, 0)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(seek_pos),   # seek to near end (known duration)
        "-i", "pipe:0",         # read MP4 from stdin
        "-frames:v", "1",       # grab one frame
        "-q:v", "2",            # JPEG quality (2 = high quality)
        "-f", "image2",
        "pipe:1",
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=video_bytes), timeout=15
        )

        if proc.returncode != 0:
            err_msg = stderr.decode(errors="replace")[-300:] if stderr else "unknown"
            _LOGGER.warning("Snapshot extraction failed (rc=%s): %s", proc.returncode, err_msg)
            return None

        if not stdout or len(stdout) < 100:
            _LOGGER.warning("Snapshot extraction produced too-small output (%s bytes)", len(stdout) if stdout else 0)
            return None

        _LOGGER.info("Extracted snapshot from clip (%s bytes)", len(stdout))
        return stdout

    except asyncio.TimeoutError:
        _LOGGER.warning("Snapshot extraction timed out")
        try:
            proc.kill()
        except Exception:
            pass
        return None
    except Exception as err:
        _LOGGER.warning("Snapshot extraction failed: %s", err)
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
    """Save snapshot to HA's www directory and return the local URL (sync)."""
    www_dir = os.path.join(config_dir, "www", "purellm")
    os.makedirs(www_dir, exist_ok=True)

    filename = f"camera_{frigate_name}.jpg"
    filepath = os.path.join(www_dir, filename)

    with open(filepath, "wb") as f:
        f.write(image_bytes)

    return f"/local/purellm/{filename}?t={int(time.time())}"


async def _save_snapshot(config_dir: str, frigate_name: str, image_bytes: bytes) -> str:
    """Save snapshot to HA's www directory without blocking the event loop."""
    return await asyncio.get_running_loop().run_in_executor(
        None, _save_snapshot_sync, config_dir, frigate_name, image_bytes
    )



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
    camera_rtsp_urls: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Check a camera with video scene analysis.

    Captures a video clip from the camera's direct RTSP stream using ffmpeg,
    extracts a fresh snapshot from the clip, and sends the video to Qwen3-VL
    for analysis.  No fallbacks — if capture or analysis fails, the failure
    is reported directly.
    """
    location = arguments.get("location", "").lower().strip()
    query = arguments.get("query", "")

    if not location:
        return {"error": "No camera location specified"}

    if not llm_base_url or not llm_model:
        return {"error": "Vision LLM not configured"}

    frigate_name, friendly_name = _resolve_camera(
        location, frigate_camera_names, camera_friendly_names
    )

    # Look up the direct RTSP URL for this camera
    rtsp_urls = camera_rtsp_urls or {}
    rtsp_url = rtsp_urls.get(frigate_name)

    if not rtsp_url:
        # Fuzzy-match against RTSP URL keys using frigate name and raw location
        rtsp_key = _best_key_match(rtsp_urls, frigate_name) or _best_key_match(rtsp_urls, location)
        if rtsp_key:
            rtsp_url = rtsp_urls[rtsp_key]

    if not rtsp_url:
        _LOGGER.error("No RTSP URL configured for camera %s", frigate_name)
        return {
            "location": friendly_name,
            "status": "error",
            "source": "none",
            "error": f"No RTSP URL configured for {friendly_name} camera. "
                     "Add the camera's RTSP URL in PureLLM settings → Camera Friendly Names.",
        }

    try:
        # Capture video clip from direct RTSP stream
        _LOGGER.info("Capturing %ds video clip from %s", VIDEO_CLIP_DURATION, frigate_name)
        video_clip = await _capture_video_clip(rtsp_url, VIDEO_CLIP_DURATION)

        if not video_clip:
            _LOGGER.error("Video clip capture failed for %s", frigate_name)
            return {
                "location": friendly_name,
                "status": "error",
                "source": "none",
                "error": f"Failed to capture video clip from {friendly_name} camera. "
                         "ffmpeg could not connect to the RTSP stream.",
            }

        # Extract a fresh snapshot from the captured clip (not cached)
        snapshot_url = None
        snapshot = await _extract_snapshot_from_clip(video_clip)
        if snapshot and config_dir:
            try:
                snapshot_url = await _save_snapshot(config_dir, frigate_name, snapshot)
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
            _LOGGER.error("Vision LLM returned no analysis for %s", frigate_name)
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
            "status": "checked",
            "source": "video_clip",
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
