"""Web search tool handler using Tavily API."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING

from ..const import API_TIMEOUT
from ..utils.http_client import log_and_error

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)

TAVILY_API_URL = "https://api.tavily.com/search"

# Keywords that indicate a news/current events query
NEWS_KEYWORDS = {
    "news", "latest", "recent", "today", "yesterday", "this week",
    "breaking", "update", "announced", "released", "happened",
    "election", "stock", "market", "price", "score", "weather",
}

# Keywords that indicate the user wants very fresh results
FRESHNESS_KEYWORDS = {
    "today", "yesterday", "this week", "latest", "recent", "now",
    "current", "right now", "just", "breaking",
}


def _detect_topic(query: str) -> str:
    """Detect if query is news-related or general.

    Args:
        query: The search query

    Returns:
        "news" or "general"
    """
    query_lower = query.lower()
    for keyword in NEWS_KEYWORDS:
        if keyword in query_lower:
            return "news"
    return "general"


def _needs_fresh_results(query: str) -> bool:
    """Detect if query needs very fresh results.

    Args:
        query: The search query

    Returns:
        True if query implies need for recent data
    """
    query_lower = query.lower()
    for keyword in FRESHNESS_KEYWORDS:
        if keyword in query_lower:
            return True
    return False


async def web_search(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    api_key: str,
    track_api_call: callable,
) -> dict[str, Any]:
    """Search the web using Tavily API.

    Tavily is optimized for LLM applications and returns structured,
    relevant results with optional AI-generated answers.

    Args:
        arguments: Tool arguments containing:
            - query: Search query (required)
            - topic: "general" or "news" (auto-detected if not provided)
            - max_results: Number of results (default: 5)
            - include_answer: Whether to include AI summary (default: True)
            - days: Limit results to last N days (optional, for freshness)
        session: aiohttp session
        api_key: Tavily API key
        track_api_call: Callback to track API usage

    Returns:
        Search results dict with answer, results array, and metadata
    """
    query = arguments.get("query", "").strip()

    if not query:
        return {"error": "No search query provided"}

    if not api_key:
        return {"error": "Tavily API key not configured. Add it in Settings → PureLLM → API Keys."}

    # Auto-detect topic if not specified
    topic = arguments.get("topic") or _detect_topic(query)

    # Determine search depth - use advanced for better results
    search_depth = arguments.get("search_depth", "advanced")

    # Number of results
    max_results = min(arguments.get("max_results", 5), 10)

    # Include AI-generated answer (highly recommended)
    include_answer = arguments.get("include_answer", True)

    # Include raw content for more context
    include_raw_content = arguments.get("include_raw_content", False)

    # Days filter for freshness (None = no filter)
    days = arguments.get("days")

    # Auto-detect if we need fresh results
    if days is None and _needs_fresh_results(query):
        days = 7  # Default to last week for "fresh" queries

    # Build request payload
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": search_depth,
        "topic": topic,
        "max_results": max_results,
        "include_answer": include_answer,
        "include_raw_content": include_raw_content,
        "include_images": False,  # Skip images for voice assistant
    }

    # Add days filter if specified
    if days:
        payload["days"] = days

    _LOGGER.info(
        "Tavily search: query='%s', topic=%s, depth=%s, max=%d, days=%s",
        query, topic, search_depth, max_results, days
    )

    try:
        track_api_call("tavily_search")

        async with asyncio.timeout(API_TIMEOUT + 5):  # Tavily can be slower
            async with session.post(
                TAVILY_API_URL,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    _LOGGER.error("Tavily API error %d: %s", response.status, error_text)

                    if response.status == 401:
                        return {"error": "Invalid Tavily API key"}
                    elif response.status == 429:
                        return {"error": "Tavily rate limit exceeded. Try again later."}
                    else:
                        return {"error": f"Search failed: HTTP {response.status}"}

                data = await response.json()

        # Extract and format results
        results = []
        for item in data.get("results", []):
            result = {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),  # Tavily calls it "content"
                "score": round(item.get("score", 0), 3),
            }

            # Include published date if available
            if item.get("published_date"):
                result["published"] = item["published_date"]

            # Include raw content if requested and available
            if include_raw_content and item.get("raw_content"):
                # Truncate raw content to avoid huge responses
                raw = item["raw_content"][:2000]
                if len(item["raw_content"]) > 2000:
                    raw += "..."
                result["raw_content"] = raw

            results.append(result)

        response_data = {
            "query": query,
            "topic": topic,
            "result_count": len(results),
            "results": results,
        }

        # Include AI-generated answer if available (this is the gold!)
        if data.get("answer"):
            response_data["answer"] = data["answer"]
            _LOGGER.info("Tavily returned answer: %s...", data["answer"][:100])

        # Include response time for debugging
        if data.get("response_time"):
            response_data["response_time_ms"] = round(data["response_time"] * 1000)

        _LOGGER.info(
            "Tavily search complete: %d results, answer=%s",
            len(results),
            "yes" if data.get("answer") else "no"
        )

        return response_data

    except asyncio.TimeoutError:
        return log_and_error("Search timed out", exc_info=False)
    except Exception as err:
        return log_and_error("Search failed", err)
