"""Shared HTTP client utilities for API calls."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any, TYPE_CHECKING

from ..const import API_TIMEOUT

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)

# Simple TTL cache for API responses
# Structure: {cache_key: (data, expiry_timestamp)}
_api_cache: dict[str, tuple[Any, float]] = {}
_cache_lock = asyncio.Lock()

# Default TTL values in seconds
CACHE_TTL_SHORT = 300  # 5 minutes - for sports scores, weather
CACHE_TTL_MEDIUM = 900  # 15 minutes - for places, search
CACHE_TTL_LONG = 3600  # 1 hour - for Wikipedia, static data


def _make_cache_key(url: str, method: str = "GET", body: dict | None = None) -> str:
    """Generate a cache key from URL and optional body."""
    key_data = f"{method}:{url}"
    if body:
        key_data += f":{hashlib.md5(str(sorted(body.items())).encode()).hexdigest()}"
    return hashlib.md5(key_data.encode()).hexdigest()


async def _get_cached(cache_key: str) -> tuple[Any, bool]:
    """Get cached response if valid. Returns (data, hit)."""
    async with _cache_lock:
        if cache_key in _api_cache:
            data, expiry = _api_cache[cache_key]
            if time.time() < expiry:
                return data, True
            # Expired, remove it
            del _api_cache[cache_key]
    return None, False


async def _set_cached(cache_key: str, data: Any, ttl: float) -> None:
    """Store data in cache with TTL."""
    async with _cache_lock:
        _api_cache[cache_key] = (data, time.time() + ttl)
        # Clean up expired entries if cache is getting large
        if len(_api_cache) > 100:
            now = time.time()
            expired = [k for k, (_, exp) in _api_cache.items() if now >= exp]
            for k in expired:
                del _api_cache[k]


def log_and_error(message: str, err: Exception | None = None, exc_info: bool = True) -> dict[str, str]:
    """Log an error and return a standardized error response.

    Args:
        message: Error message to log and return
        err: Optional exception that caused the error
        exc_info: Whether to include traceback in log

    Returns:
        Dict with "error" key
    """
    if err:
        _LOGGER.error("%s: %s", message, err, exc_info=exc_info)
        return {"error": f"{message}: {str(err)}"}
    else:
        _LOGGER.error(message)
        return {"error": message}


async def fetch_json(
    session: "aiohttp.ClientSession",
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = API_TIMEOUT,
    cache_ttl: float | None = None,
) -> tuple[dict[str, Any] | None, int]:
    """Fetch JSON from a URL with standardized timeout and error handling.

    Args:
        session: aiohttp session
        url: URL to fetch
        headers: Optional request headers
        timeout: Request timeout in seconds
        cache_ttl: Optional TTL in seconds to cache the response

    Returns:
        Tuple of (json_data or None, status_code)
    """
    # Check cache first if caching is enabled
    cache_key = None
    if cache_ttl is not None:
        cache_key = _make_cache_key(url)
        cached_data, cache_hit = await _get_cached(cache_key)
        if cache_hit:
            _LOGGER.debug("Cache hit for: %s", url)
            return cached_data, 200

    try:
        async with asyncio.timeout(timeout):
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    # Cache the successful response
                    if cache_key is not None:
                        await _set_cached(cache_key, data, cache_ttl)
                    return data, response.status
                return None, response.status
    except asyncio.TimeoutError:
        _LOGGER.warning("Request timed out: %s", url)
        return None, 408
    except Exception as err:
        _LOGGER.error("Request failed for %s: %s", url, err)
        return None, 500


async def post_json(
    session: "aiohttp.ClientSession",
    url: str,
    body: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = API_TIMEOUT,
) -> tuple[dict[str, Any] | None, int]:
    """POST JSON to a URL with standardized timeout and error handling.

    Args:
        session: aiohttp session
        url: URL to post to
        body: JSON body to send
        headers: Optional request headers
        timeout: Request timeout in seconds

    Returns:
        Tuple of (json_data or None, status_code)
    """
    try:
        async with asyncio.timeout(timeout):
            async with session.post(url, json=body, headers=headers) as response:
                if response.status == 200:
                    return await response.json(), response.status
                return None, response.status
    except asyncio.TimeoutError:
        _LOGGER.warning("Request timed out: %s", url)
        return None, 408
    except Exception as err:
        _LOGGER.error("Request failed for %s: %s", url, err)
        return None, 500


