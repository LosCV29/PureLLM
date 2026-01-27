"""Web search tool handler using Tavily API."""
from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse
from typing import Any, TYPE_CHECKING

from ..const import API_TIMEOUT
from ..utils.http_client import log_and_error

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)

TAVILY_API_URL = "https://api.tavily.com/search"

# Map common domains to clean source names
SOURCE_NAME_MAP = {
    "cnn.com": "CNN",
    "bbc.com": "BBC",
    "bbc.co.uk": "BBC",
    "nytimes.com": "The New York Times",
    "washingtonpost.com": "The Washington Post",
    "theguardian.com": "The Guardian",
    "reuters.com": "Reuters",
    "apnews.com": "Associated Press",
    "foxnews.com": "Fox News",
    "nbcnews.com": "NBC News",
    "cbsnews.com": "CBS News",
    "abcnews.go.com": "ABC News",
    "usatoday.com": "USA Today",
    "wsj.com": "Wall Street Journal",
    "bloomberg.com": "Bloomberg",
    "forbes.com": "Forbes",
    "techcrunch.com": "TechCrunch",
    "theverge.com": "The Verge",
    "wired.com": "Wired",
    "arstechnica.com": "Ars Technica",
    "engadget.com": "Engadget",
    "cnet.com": "CNET",
    "zdnet.com": "ZDNet",
    "wikipedia.org": "Wikipedia",
    "en.wikipedia.org": "Wikipedia",
    "reddit.com": "Reddit",
    "quora.com": "Quora",
    "stackoverflow.com": "Stack Overflow",
    "medium.com": "Medium",
    "healthline.com": "Healthline",
    "webmd.com": "WebMD",
    "mayoclinic.org": "Mayo Clinic",
    "espn.com": "ESPN",
    "sports.yahoo.com": "Yahoo Sports",
    "bleacherreport.com": "Bleacher Report",
    "imdb.com": "IMDb",
    "rottentomatoes.com": "Rotten Tomatoes",
    "themoviedb.org": "TMDB",
    "variety.com": "Variety",
    "hollywoodreporter.com": "Hollywood Reporter",
    "allrecipes.com": "Allrecipes",
    "foodnetwork.com": "Food Network",
    "epicurious.com": "Epicurious",
    "yelp.com": "Yelp",
    "tripadvisor.com": "TripAdvisor",
    # Music sources
    "allmusic.com": "AllMusic",
    "discogs.com": "Discogs",
    "genius.com": "Genius",
    "billboard.com": "Billboard",
    "pitchfork.com": "Pitchfork",
    "rollingstone.com": "Rolling Stone",
    "nme.com": "NME",
    "stereogum.com": "Stereogum",
    "consequenceofsound.net": "Consequence of Sound",
    "hiphopdx.com": "HipHopDX",
    "complex.com": "Complex",
    "xxlmag.com": "XXL",
}


def _extract_source_name(url: str) -> str:
    """Extract a clean source name from a URL.

    Args:
        url: The full URL

    Returns:
        Clean source name (e.g., "CNN", "The New York Times")
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]

        # Check our map first
        if domain in SOURCE_NAME_MAP:
            return SOURCE_NAME_MAP[domain]

        # For subdomains, try the main domain
        parts = domain.split(".")
        if len(parts) > 2:
            main_domain = ".".join(parts[-2:])
            if main_domain in SOURCE_NAME_MAP:
                return SOURCE_NAME_MAP[main_domain]

        # Fall back to capitalizing the domain name
        # e.g., "example.com" -> "Example"
        site_name = parts[-2] if len(parts) >= 2 else parts[0]
        return site_name.capitalize()

    except Exception:
        return "Web"


# Keywords that indicate a news/current events query
NEWS_KEYWORDS = {
    "news", "latest", "recent", "today", "yesterday", "this week",
    "breaking", "update", "announced", "released", "happened",
    "election", "stock", "market", "price", "weather",
}

# Keywords that indicate the user wants very fresh results
FRESHNESS_KEYWORDS = {
    "today", "yesterday", "this week", "latest", "recent", "now",
    "current", "right now", "just", "breaking",
}

# Smart domain patterns - maps keywords to target domains for better results
# Format: (keywords_tuple, domains_list, enable_raw_content)
DOMAIN_PATTERNS = [
    # Movie/TV synopsis and plot information
    (
        ("synopsis", "plot", "storyline", "what is it about", "what's it about",
         "summary of", "plot summary", "movie plot", "film plot"),
        ["wikipedia.org", "imdb.com", "rottentomatoes.com", "themoviedb.org"],
        True  # Need raw content to get full synopsis/plot text
    ),
    # Movie/TV ratings and reviews
    (
        ("rotten tomatoes", "rt score", "tomatometer", "tomato score", "critics score",
         "audience score", "certified fresh", "rotten score"),
        ["rottentomatoes.com"],
        True  # Need raw content for scores
    ),
    (
        ("imdb rating", "imdb score", "imdb review"),
        ["imdb.com"],
        True
    ),
    (
        ("metacritic", "metascore"),
        ["metacritic.com"],
        True
    ),
    # Restaurant/business reviews
    (
        ("yelp review", "yelp rating", "yelp score"),
        ["yelp.com"],
        True
    ),
    # Recipes
    (
        ("recipe for", "how to make", "how to cook", "how to bake"),
        ["allrecipes.com", "foodnetwork.com", "bonappetit.com", "seriouseats.com", "epicurious.com"],
        False
    ),
    # Tech reviews
    (
        ("review of", "reviews for", "best rated", "top rated", "comparison"),
        ["cnet.com", "theverge.com", "techradar.com", "tomsguide.com", "wirecutter.com"],
        False
    ),
    # Health information
    (
        ("symptoms of", "treatment for", "side effects", "medication"),
        ["mayoclinic.org", "webmd.com", "healthline.com", "nih.gov"],
        False
    ),
]


def _detect_target_domains(query: str) -> tuple[list[str] | None, bool]:
    """Detect if query should target specific domains.

    Args:
        query: The search query

    Returns:
        Tuple of (domains_list or None, enable_raw_content)
    """
    query_lower = query.lower()

    for keywords, domains, raw_content in DOMAIN_PATTERNS:
        for keyword in keywords:
            if keyword in query_lower:
                _LOGGER.debug("Detected domain pattern '%s' -> %s", keyword, domains)
                return domains, raw_content

    return None, False


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
            - topic: "general", "news", or "finance" (auto-detected if not provided)
            - max_results: Number of results (default: 5)
            - include_answer: Whether to include AI summary (default: True)
            - days: Limit results to last N days (optional, for freshness)
            - include_domains: List of domains to search (e.g., ["rottentomatoes.com"])
            - exclude_domains: List of domains to exclude
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

    # Smart domain detection - auto-target specific sites based on query
    auto_domains, auto_raw_content = _detect_target_domains(query)

    # Use explicit domains if provided, otherwise use auto-detected
    include_domains = arguments.get("include_domains") or auto_domains
    exclude_domains = arguments.get("exclude_domains")

    # Include raw content if explicitly requested OR if auto-detected for domain targeting
    include_raw_content = arguments.get("include_raw_content", auto_raw_content)

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

    # Add domain filters if specified
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains

    _LOGGER.info(
        "Tavily search: query='%s', topic=%s, depth=%s, max=%d, days=%s, domains=%s, raw=%s",
        query, topic, search_depth, max_results, days, include_domains, include_raw_content
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

        # Extract and format results with source attribution
        results = []
        sources_used = []  # Track unique sources for attribution

        for item in data.get("results", []):
            url = item.get("url", "")
            source_name = _extract_source_name(url)

            result = {
                "source": source_name,  # Clean source name for attribution
                "title": item.get("title", ""),
                "url": url,
                "snippet": item.get("content", ""),  # Tavily calls it "content"
            }

            # Track unique sources
            if source_name not in sources_used:
                sources_used.append(source_name)

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

        # Build response with strict instructions
        response_data = {
            "query": query,
            "result_count": len(results),
            "results": results,
        }

        # Include AI-generated answer if available - this should be used as the primary response
        if data.get("answer"):
            response_data["answer"] = data["answer"]
            response_data["instruction"] = f"CRITICAL: Use ONLY this answer - do not make up information. Say: 'According to {sources_used[0] if sources_used else 'web search'}, {data['answer']}'"
            if sources_used:
                response_data["source"] = sources_used[0]
        else:
            response_data["instruction"] = "CRITICAL: Use ONLY the information in the results below. Do NOT make up or guess any information. If the answer isn't in the results, say you couldn't find it."
            response_data["sources"] = sources_used

        _LOGGER.info(
            "Tavily search complete: %d results, sources=%s",
            len(results),
            ", ".join(sources_used[:3])
        )

        return response_data

    except asyncio.TimeoutError:
        return log_and_error("Search timed out", exc_info=False)
    except Exception as err:
        return log_and_error("Search failed", err)
