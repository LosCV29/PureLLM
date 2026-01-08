"""Web search tool handler using Tavily API."""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)

# Tavily API endpoint
TAVILY_SEARCH_URL = "https://api.tavily.com/search"


async def web_search(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    tavily_api_key: str,
    track_api_call: callable,
) -> dict[str, Any]:
    """Search the web for current information using Tavily.

    Args:
        arguments: Dict with 'query' (required) and optional 'num_results' (1-10)
        session: aiohttp client session
        tavily_api_key: Tavily API key
        track_api_call: Callback to track API usage

    Returns:
        Dict with search results or error
    """
    query = arguments.get("query", "").strip()
    num_results = min(max(arguments.get("num_results", 5), 1), 10)

    if not query:
        return {"error": "No search query provided"}

    if not tavily_api_key:
        return {"error": "Tavily API key not configured. Get one at tavily.com and add it in API Keys settings."}

    _LOGGER.info("Tavily web search: '%s'", query)
    track_api_call("web_search")

    try:
        payload = {
            "api_key": tavily_api_key,
            "query": query,
            "search_depth": "basic",  # "basic" or "advanced" (advanced costs more)
            "include_answer": True,  # Get AI-generated answer
            "include_raw_content": False,
            "max_results": num_results,
        }

        async with session.post(TAVILY_SEARCH_URL, json=payload, timeout=15) as response:
            if response.status != 200:
                error_text = await response.text()
                _LOGGER.error("Tavily API error: %s", error_text)

                if response.status == 401:
                    return {"error": "Invalid Tavily API key. Check your key at tavily.com"}
                elif response.status == 429:
                    return {"error": "Tavily rate limit exceeded. Try again later."}

                return {"error": f"Search failed: {response.status}"}

            data = await response.json()

        # Tavily returns an AI-generated answer plus sources
        answer = data.get("answer", "")
        results = data.get("results", [])

        if not results and not answer:
            return {"message": f"No results found for '{query}'"}

        # Format results for the LLM
        formatted_results = []
        for item in results:
            formatted_results.append({
                "title": item.get("title", ""),
                "content": item.get("content", ""),  # Tavily gives full content, not just snippets
                "url": item.get("url", ""),
                "score": item.get("score", 0),  # Relevance score
            })

        response_data = {
            "query": query,
            "results": formatted_results,
        }

        # Include Tavily's AI answer if available (very useful for voice)
        if answer:
            response_data["answer"] = answer

        return response_data

    except Exception as e:
        _LOGGER.error("Tavily search error: %s", e, exc_info=True)
        return {"error": f"Search failed: {str(e)}"}
