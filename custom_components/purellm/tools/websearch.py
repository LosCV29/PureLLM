"""Web search tool handler using Google Custom Search."""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)

# Google Custom Search API endpoint
GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"


async def web_search(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    api_key: str,
    search_engine_id: str,
    track_api_call: callable,
) -> dict[str, Any]:
    """Search the web for current information.

    Args:
        arguments: Dict with 'query' (required) and optional 'num_results' (1-10)
        session: aiohttp client session
        api_key: Google API key (same as Places API key)
        search_engine_id: Google Programmable Search Engine ID
        track_api_call: Callback to track API usage

    Returns:
        Dict with search results or error
    """
    query = arguments.get("query", "").strip()
    num_results = min(max(arguments.get("num_results", 3), 1), 10)

    if not query:
        return {"error": "No search query provided"}

    if not api_key:
        return {"error": "Google API key not configured. Add it in API Keys settings."}

    if not search_engine_id:
        return {"error": "Google Search Engine ID not configured. Create one at programmablesearchengine.google.com and add it in API Keys settings."}

    _LOGGER.info("Web search: '%s'", query)
    track_api_call("web_search")

    try:
        params = {
            "key": api_key,
            "cx": search_engine_id,
            "q": query,
            "num": num_results,
        }

        async with session.get(GOOGLE_SEARCH_URL, params=params, timeout=10) as response:
            if response.status != 200:
                error_text = await response.text()
                _LOGGER.error("Google Search API error: %s", error_text)

                # Common error handling
                if response.status == 403:
                    return {"error": "Google Search API access denied. Enable Custom Search API in Google Cloud Console."}
                elif response.status == 429:
                    return {"error": "Search rate limit exceeded. Try again later."}

                return {"error": f"Search failed: {response.status}"}

            data = await response.json()

        # Parse search results
        items = data.get("items", [])
        if not items:
            return {"message": f"No results found for '{query}'"}

        results = []
        for item in items:
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", ""),
            })

        return {
            "query": query,
            "results": results,
            "total_results": data.get("searchInformation", {}).get("totalResults", "0"),
        }

    except Exception as e:
        _LOGGER.error("Web search error: %s", e, exc_info=True)
        return {"error": f"Search failed: {str(e)}"}
