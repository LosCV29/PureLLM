"""Intent-based tool routing for PureLLM.

Classifies user utterances by intent using fast keyword matching,
then returns only the relevant tool subset. This dramatically reduces
token usage per LLM request (~3500 → ~200 tokens for tool definitions).
"""
from __future__ import annotations

import logging
from typing import Any

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent keyword patterns
# ---------------------------------------------------------------------------
# Each pattern is checked as a substring of the lowercased user text.
# Multi-word patterns naturally avoid false matches (e.g. "play " won't match
# "display").  If ANY pattern in a category matches, that intent is selected.
#
# Design principle: be INCLUSIVE. A false positive (extra tool sent) costs
# ~100 tokens. A false negative (missing the right tool) breaks the response.
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: dict[str, list[str]] = {
    "music": [
        "play ", "shuffle", "pause", "resume", "skip",
        "next song", "previous song", "restart track",
        "what's playing", "whats playing", "now playing",
        " album", " artist", " song ", " track ",
        "music", "transfer to the", "transfer to my",
    ],
    "weather": [
        "weather", " rain", "raining", "forecast",
        "sunrise", "sunset", "sun time",
        "outside temp", "temperature outside",
        "hot outside", "cold outside",
        "sunny", "cloudy", "snow", "storm", "humid",
    ],
    "thermostat": [
        "thermostat", " ac", "a.c.", "a/c",
        "air condition", "hvac",
        "raise the temp", "lower the temp",
        "turn up the", "turn down the",
        "set it to", "degrees",
        "heat mode", "cool mode",
    ],
    "camera": [
        "camera",
        "anyone outside", "anyone on the", "someone on the",
        "someone outside", "is there anyone", "is anyone",
        "check the back yard", "check the backyard",
        "check the driveway", "check the front door",
        "check front door", "check the side gate",
        "check the nursery", "check the sala", "check the kitchen",
        "what's on the driveway", "whats on the driveway",
        "what's happening on", "whats happening on",
        "what's going on", "whats going on outside",
        "nursery cam", "sala cam", "kitchen cam",
    ],
    "sports": [
        " game", "score", " nfl", " nba", " mlb", " nhl", " mls",
        "ufc", "fight card", "premier league", "champions league",
        "la liga", "standings", "league games",
        "heat play", "dolphins", "marlins", "panthers",
        "inter miami", "hurricanes",
    ],
    "timer": [
        "timer", "set a timer", "countdown",
        "minute timer", "second timer", "hour timer",
    ],
    "list": [
        " list", "shopping", "grocery", "to-do", "todo",
        "add to my", "add it to",
    ],
    "calendar": [
        "calendar", " events", "schedule", "appointment", "birthday",
        "what's on my", "whats on my", "what do i have",
    ],
    "places": [
        "nearest ", "closest ", "find a ",
        "near me", "nearby",
        "gas station", "pharmacy", "cvs", "walgreens",
    ],
    "knowledge": [
        "who is ", "who was ", "what is a ", "what is the ",
        "how old is", "wikipedia", "tell me about ",
    ],
    "datetime": [
        "what time", "what day", "what date",
        "today's date", "todays date",
        "what's today", "whats today",
    ],
    "device": [
        "turn on", "turn off", "toggle",
        "lights", "light on", "light off",
        "lock", "unlock",
        " fan", "switch",
        "vacuum", "blinds", "shades",
        " dim", "brightness",
        "open the", "close the",
        "porch light", "backyard light", "kitchen light",
        "garage door", "mailbox",
        "front door", "back door", "side gate",
        "purifier", "diffuser",
    ],
    "device_status": [
        "is the ", "is my ", "status of",
        "are the ", "check the lock", "check the door",
        "check the gate", "check the garage",
        "check the mailbox",
        "locked", "unlocked", "open or closed",
    ],
    "search": [
        "search for", "search the web", "look up",
        "latest news", "current news",
    ],
    "sofabaton": [
        "sofabaton", " tv", "television",
        "apple tv", "roku", "fire stick",
        "watch ", "start watching",
    ],
}

# Map intent categories → tool function names
_INTENT_TO_TOOLS: dict[str, list[str]] = {
    "music": ["control_music"],
    "weather": ["get_weather_forecast"],
    "thermostat": ["control_thermostat"],
    "camera": ["check_camera"],
    "sports": [
        "get_sports_info", "get_ufc_info",
        "check_league_games", "list_league_games",
    ],
    "timer": ["control_timer"],
    "list": ["manage_list"],
    "calendar": ["get_calendar_events"],
    "places": ["find_nearby_places"],
    "knowledge": ["calculate_age", "get_wikipedia_summary"],
    "datetime": ["get_current_datetime"],
    "device": ["control_device"],
    "device_status": ["check_device_status"],
    "search": ["web_search"],
    "sofabaton": ["control_sofabaton"],
}

# Tiny tools that are cheap to always include (~30 tokens each)
_ALWAYS_INCLUDE = {"get_current_datetime"}


def classify_intent(user_text: str) -> set[str]:
    """Classify user text into intent categories using keyword matching.

    Returns a set of matched intent category names.
    Empty set = no match → caller should fall back to all tools.
    """
    text = f" {user_text.lower().strip()} "
    matched: set[str] = set()

    for intent, patterns in _INTENT_PATTERNS.items():
        for pattern in patterns:
            if pattern in text:
                matched.add(intent)
                break

    return matched


def filter_tools_by_intent(
    all_tools: list[dict[str, Any]],
    intents: set[str],
) -> list[dict[str, Any]]:
    """Filter tool definitions to only those matching classified intents.

    Returns all_tools unchanged if intents is empty (no match = fallback).
    """
    if not intents:
        _LOGGER.debug("No intent matched — sending all %d tools", len(all_tools))
        return all_tools

    # Collect tool names for matched intents
    needed: set[str] = set(_ALWAYS_INCLUDE)
    for intent in intents:
        needed.update(_INTENT_TO_TOOLS.get(intent, []))

    filtered = [
        tool for tool in all_tools
        if tool.get("function", {}).get("name") in needed
    ]

    _LOGGER.info(
        "Intent routing: %s → %d/%d tools (%s)",
        intents, len(filtered), len(all_tools),
        ", ".join(sorted(needed)),
    )
    return filtered
