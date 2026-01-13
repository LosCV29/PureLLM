"""Shared HTTP client utilities for API calls."""
from __future__ import annotations

import asyncio
import logging
from functools import wraps
from typing import Any, Callable, TYPE_CHECKING

from ..const import API_TIMEOUT

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)


def tool_error(message: str) -> dict[str, str]:
    """Create a standardized error response dict.

    Args:
        message: Error message to include

    Returns:
        Dict with "error" key
    """
    return {"error": message}


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
) -> tuple[dict[str, Any] | None, int]:
    """Fetch JSON from a URL with standardized timeout and error handling.

    Args:
        session: aiohttp session
        url: URL to fetch
        headers: Optional request headers
        timeout: Request timeout in seconds

    Returns:
        Tuple of (json_data or None, status_code)
    """
    try:
        async with asyncio.timeout(timeout):
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json(), response.status
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


class APIError(Exception):
    """Exception for API errors with status code."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


async def require_json(
    session: "aiohttp.ClientSession",
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = API_TIMEOUT,
    error_message: str = "API request failed",
) -> dict[str, Any]:
    """Fetch JSON or raise APIError.

    Args:
        session: aiohttp session
        url: URL to fetch
        headers: Optional request headers
        timeout: Request timeout in seconds
        error_message: Error message prefix for failures

    Returns:
        JSON response data

    Raises:
        APIError: If request fails or returns non-200 status
    """
    data, status = await fetch_json(session, url, headers=headers, timeout=timeout)
    if data is None:
        raise APIError(f"{error_message}: HTTP {status}", status)
    return data


async def require_post_json(
    session: "aiohttp.ClientSession",
    url: str,
    body: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = API_TIMEOUT,
    error_message: str = "API request failed",
) -> dict[str, Any]:
    """POST JSON or raise APIError.

    Args:
        session: aiohttp session
        url: URL to post to
        body: JSON body to send
        headers: Optional request headers
        timeout: Request timeout in seconds
        error_message: Error message prefix for failures

    Returns:
        JSON response data

    Raises:
        APIError: If request fails or returns non-200 status
    """
    data, status = await post_json(session, url, body, headers=headers, timeout=timeout)
    if data is None:
        raise APIError(f"{error_message}: HTTP {status}", status)
    return data
