"""Wikipedia and age calculation tool handlers."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, TYPE_CHECKING
from urllib.parse import quote

from ..const import VERSION
from ..utils.helpers import get_nested
from ..utils.http_client import CACHE_TTL_LONG, fetch_json, log_and_error

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)


async def calculate_age(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
) -> dict[str, Any]:
    """Calculate a person's age from Wikidata birthdate."""
    person_name = arguments.get("person_name", "").strip()

    if not person_name:
        return {"error": "No person name provided"}

    try:
        # Step 1: Search Wikidata for the person
        search_url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={person_name}&language=en&format=json&limit=1"

        search_data, status = await fetch_json(session, search_url, headers=_WIKI_HEADERS)
        if search_data is None:
            return {"error": "Failed to search Wikidata"}

        search_results = search_data.get("search", [])
        if not search_results:
            return {"error": f"Could not find '{person_name}' on Wikidata"}

        entity_id = search_results[0].get("id")
        entity_label = search_results[0].get("label", person_name)
        entity_description = search_results[0].get("description", "")

        # Step 2: Get entity details including birthdate
        entity_url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={entity_id}&props=claims&format=json"

        entity_data, status = await fetch_json(session, entity_url, headers=_WIKI_HEADERS)
        if entity_data is None:
            return {"error": "Failed to get entity details"}

        entities = entity_data.get("entities", {})
        entity = entities.get(entity_id, {})
        claims = entity.get("claims", {})

        # P569 = date of birth in Wikidata
        birth_claims = claims.get("P569", [])
        if not birth_claims:
            return {"error": f"No birthdate found for {entity_label}"}

        birth_claim = birth_claims[0]
        time_value = get_nested(birth_claim, "mainsnak", "datavalue", "value", "time", default="")

        if not time_value:
            return {"error": f"Could not parse birthdate for {entity_label}"}

        # Parse Wikidata time format: "+1984-12-30T00:00:00Z"
        try:
            birth_date_str = time_value.lstrip("+").split("T")[0]
            birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d")
        except ValueError:
            return {"error": f"Invalid birthdate format for {entity_label}"}

        # Calculate age
        today = datetime.now()
        age = today.year - birth_date.year

        # Adjust if birthday hasn't occurred yet this year
        if (today.month, today.day) < (birth_date.month, birth_date.day):
            age -= 1

        # Check for death date (P570)
        death_claims = claims.get("P570", [])
        is_deceased = bool(death_claims)

        result = {
            "name": entity_label,
            "age": age,
            "birth_date": birth_date.strftime("%B %d, %Y"),
            "description": entity_description,
        }

        if is_deceased:
            death_time = get_nested(death_claims[0], "mainsnak", "datavalue", "value", "time", default="")
            if death_time:
                try:
                    death_date_str = death_time.lstrip("+").split("T")[0]
                    death_date = datetime.strptime(death_date_str, "%Y-%m-%d")
                    result["death_date"] = death_date.strftime("%B %d, %Y")
                    result["age_at_death"] = age
                    result["is_deceased"] = True
                except ValueError:
                    pass

        _LOGGER.info("Age lookup: %s is %d years old (born %s)", entity_label, age, result["birth_date"])
        return result

    except Exception as err:
        return log_and_error("Failed to calculate age", err)


# Wikimedia's robot policy returns HTTP 403 for generic client User-Agents;
# a descriptive UA with a contact URL is required for reliable access.
_WIKI_HEADERS = {
    "User-Agent": f"PureLLM/{VERSION} (https://github.com/LosCV29/PureLLM) aiohttp",
}

# Instruction appended to every failure so the LLM never falls back to
# stale training data when Wikipedia can't answer.
_WIKI_FAIL_INSTRUCTION = (
    "LOOKUP FAILED. Tell the user you couldn't look that up. "
    "Do NOT answer from memory or guess. You may try web_search instead."
)


async def _fetch_summary(
    session: "aiohttp.ClientSession", title: str
) -> dict[str, Any] | None:
    """Fetch a Wikipedia REST summary for a title, with one retry on
    transient failures. Returns None on 404 or persistent failure."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title.replace(' ', '_'))}"
    for attempt in range(2):
        data, status = await fetch_json(session, url, headers=_WIKI_HEADERS, cache_ttl=CACHE_TTL_LONG)
        if data is not None:
            return data
        if status == 404:
            return None
        # Transient (timeout / 5xx / 429): brief backoff then retry once
        _LOGGER.warning("Wikipedia summary fetch failed (HTTP %s), attempt %d", status, attempt + 1)
        if attempt == 0:
            await asyncio.sleep(0.5)
    return None


async def _search_titles(
    session: "aiohttp.ClientSession", topic: str, limit: int = 3
) -> list[str]:
    """Search Wikipedia for candidate page titles."""
    search_url = (
        "https://en.wikipedia.org/w/api.php?action=query&list=search"
        f"&srsearch={quote(topic)}&format=json&srlimit={limit}"
    )
    for attempt in range(2):
        data, status = await fetch_json(session, search_url, headers=_WIKI_HEADERS, cache_ttl=CACHE_TTL_LONG)
        if data is not None:
            results = data.get("query", {}).get("search", [])
            return [r["title"] for r in results if r.get("title")]
        _LOGGER.warning("Wikipedia search failed (HTTP %s), attempt %d", status, attempt + 1)
        if attempt == 0:
            await asyncio.sleep(0.5)
    return []


async def get_wikipedia_summary(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
) -> dict[str, Any]:
    """Get Wikipedia summary for a topic.

    Robust flow: direct title lookup first; on miss or a disambiguation
    page, fall back to full-text search and take the first concrete page.
    Every failure path carries an instruction telling the LLM not to
    answer from memory.
    """
    topic = arguments.get("topic", "").strip()

    if not topic:
        return {"error": "No topic provided", "instruction": _WIKI_FAIL_INSTRUCTION}

    try:
        data = await _fetch_summary(session, topic)

        # Direct miss or a disambiguation page → search for concrete pages.
        if data is None or data.get("type") == "disambiguation":
            for title in await _search_titles(session, topic):
                candidate = await _fetch_summary(session, title)
                if candidate and candidate.get("type") != "disambiguation":
                    data = candidate
                    break
            else:
                if data is None or data.get("type") == "disambiguation":
                    return {
                        "error": f"Could not find a Wikipedia article about '{topic}'",
                        "instruction": _WIKI_FAIL_INSTRUCTION,
                    }

        title = data.get("title", topic)
        extract = data.get("extract", "No summary available")
        page_url = get_nested(data, "content_urls", "desktop", "page", default="")
        description = data.get("description", "")

        result = {
            "title": title,
            "summary": extract,
            "description": description,
            "instruction": (
                "Answer using ONLY this summary. If the answer is not in it, "
                "use web_search or say you couldn't find it — never answer from memory."
            ),
        }

        if page_url:
            result["url"] = page_url

        _LOGGER.info("Wikipedia lookup: %s", title)
        return result

    except Exception as err:
        error = log_and_error("Failed to get Wikipedia summary", err)
        error["instruction"] = _WIKI_FAIL_INSTRUCTION
        return error
