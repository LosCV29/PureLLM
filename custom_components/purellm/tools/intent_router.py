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
    # Verbatim phone relay between household members (v7.67.0). Patterns are
    # addressing forms, NOT content words, because the message body is
    # arbitrary: "tell Elise I want to play later" also matches the music
    # intent on "play ", and that is fine — classify_intent returns a SET, so
    # both tool groups are offered and the model picks. What matters is that
    # send_partner_message is never the missing option.
    "partner_message": [
        "tell elise", "tell carlos", "tell my wife", "tell my husband",
        "text elise", "text carlos", "text my wife", "text my husband",
        "message elise", "message carlos", "message my wife",
        "message my husband", "send elise", "send carlos",
        "send a message to elise", "send a message to carlos",
        "let elise know", "let carlos know",
        "ask elise", "ask carlos",
    ],
    # Spoken relay — say it out loud in a room (v8.5.0). Patterns are the
    # LOCATION/out-loud forms, since the addressing forms above already pull in
    # the messaging group; "tell Carlos ... on the kitchen speaker" matches
    # both and the model chooses between two clearly-described tools. The bare
    # room words are intentionally NOT patterns here — "turn on the kitchen
    # lights" must not offer an announce tool.
    "announce": [
        "on the kitchen speaker", "on the living room speaker",
        "on the bedroom speaker", "on the nursery speaker",
        "on the bathroom speaker", "on the office speaker",
        "on the speaker", "on the speakers", "over the speaker",
        "on speaker", "through the speaker", "on the intercom", "intercom",
        "announce", "announcement", "out loud", "aloud", "say it in the",
        "say out loud", "broadcast", "over the house", "in the house",
        "everywhere", "all the speakers", "every speaker", "whole house",
        "page the", "anuncia", "en voz alta", "por el altavoz",
    ],
    "music": [
        "play ", "shuffle", "pause", "resume", "skip",
        "next song", "previous song", "restart track",
        # Bare transport commands. These were falling through to NO-MATCH →
        # core bundle (which has no control_music) → the model improvised with
        # control_device({'device':'media_player'}) and skipped on a random
        # media_player entity in another room (2026-07-26). Note the old
        # " track " / " song " patterns require a TRAILING space, so the very
        # common "next track." / "skip this song." never matched.
        "next track", "next tracks", "previous track", "prev track",
        "last song", "last track", "go back a song", "go back a track",
        "skip back", "skip ahead", "skip this", "skip it", "next one",
        "restart the song", "restart the track", "play it again",
        "play that again", "replay", "start over", "from the top",
        "what song", "what's this song", "whats this song", "who is this",
        "what's playing", "whats playing", "now playing",
        " album", " artist", " song", " track",
        "music", "transfer to the", "transfer to my",
        "music volume", "the music volume",
        "raise the music", "lower the music",
        "turn up the music", "turn down the music",
        # Spanish
        "pon ", "reproduce", "pausa", "reanuda",
        "música", "musica", "canción", "cancion",
        "siguiente", "anterior",
    ],
    # Be generous here: a weather miss falls through to the core bundle, which
    # has NO weather tool, so the brain answers from training data — the exact
    # ungrounded failure the grounding rules exist to prevent. Patterns that
    # could collide with a device are anchored with spaces (" wind " must not
    # match "window"; "showers" must not match the bathroom shower).
    "weather": [
        "weather", " rain", "raining", "rainy", "forecast",
        "drizzl", "showers", "downpour", "pouring",
        "sunrise", "sunset", "sun time", "sun set", "sun rise",
        "outside temp", "temperature outside", "degrees outside",
        "hot outside", "cold outside", "hot out", "cold out",
        "warm out", "nice out", "chilly", "muggy",
        "how cold", "how hot", "how warm", "how humid",
        "sunny", "cloudy", "overcast", "snow", "sleet", "hail",
        "storm", "thunder", "lightning", "hurricane", "tropical",
        "humid", "dew point", "feels like out",
        "windy", " wind ", " winds", "wind speed", "gust",
        "uv index", "sunscreen", "sunburn",
        "umbrella", "need a jacket", "need a coat", "wear a jacket",
        "weather alert", "weather warning", "storm warning",
        "freeze", "freezing", "frost",
    ],
    "thermostat": [
        "thermostat", " ac", "a.c.", "a/c",
        "air condition", "hvac",
        "raise the temp", "lower the temp",
        "turn up the", "turn down the",
        "set it to", "degrees",
        "heat mode", "cool mode",
        # Spanish
        "termostato", "aire acondicionado",
        "grados", "temperatura",
        "sube la temperatura", "baja la temperatura",
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
        # Postseason terms — without these, "what is the X playoff series at?"
        # only matches the greedy "what is the " in knowledge intent and the
        # sports tool never gets exposed.
        " playoff", " series", " seed", "semifinal",
        "conference final", " finals",
    ],
    "timer": [
        "timer", "set a timer", "countdown",
        "minute timer", "second timer", "hour timer",
    ],
    "list": [
        " list", "shopping", "grocery", "to-do", "todo",
        "add to my", "add it to",
        # Named household lists (todo.costco / todo.amazon, added 2026-08-11)
        # so phrasing that omits the word "list" ("add batteries to amazon")
        # still routes to manage_list.
        "costco", "amazon",
    ],
    "calendar": [
        "calendar", " events", "schedule", "appointment", "birthday",
        "what's on my", "whats on my", "what do i have",
        "holiday", "holidays", "next holiday",
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
        # Spanish
        "qué hora", "que hora", "qué día", "que dia",
        "qué fecha", "que fecha",
    ],
    "device": [
        "turn on", "turn off", "toggle",
        "launch ",
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
        "white noise", "sound machine", "noise machine",
        "ruido blanco", "máquina de ruido", "maquina de ruido",
        " tv", "television", "apple tv", "roku", "fire stick",
        "pause the", "resume the", "mute the", "unmute the",
        "volume up", "volume down",
        "set volume", "set the volume", "speaker volume", "voice volume",
        "your volume", "raise your", "lower your",
        "louder", "quieter", "speak up",
        # Spanish (household speaks Spanish to HA; STT emits accented text)
        "enciende", "prende ", "apaga", "apagas",
        " luz", "luces",
        "ventilador", "aspiradora",
        "persiana", "cortina",
        "abre la", "abre el", "cierra la", "cierra el",
        "cierra con llave", "pon el seguro",
        "sube el volumen", "baja el volumen", "silencia",
        "más fuerte", "mas fuerte", "más alto", "mas alto",
        "televisión",
    ],
    "device_status": [
        "is the ", "is my ", "status of",
        "are the ", "check the lock", "check the door",
        "check the gate", "check the garage",
        "check the mailbox",
        "locked", "unlocked", "open or closed",
        " status", "what's the ", "whats the ",
    ],
    "search": [
        "search for", "search the web", "look up",
        "latest news", "current news",
    ],
    "sofabaton": [
        "sofabaton",
        "watch ", "start watching",
    ],
    "plants": [
        " plant", " plants",
        "moisture", " soil ",
        "watering", "watered",
        "need water", "needs water", "need watering",
        "thirsty", " dry", " wet",
        "hydrated", "hydration",
        "overwater", "underwater", "drown",
        " dli",
    ],
}

# Map intent categories → tool function names
_INTENT_TO_TOOLS: dict[str, list[str]] = {
    "music": ["control_music", "search_music", "set_white_noise_sound"],
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
    "device": ["control_device", "check_device_status", "set_speaker_volume", "set_fan_speed", "set_white_noise_sound"],
    "device_status": ["check_device_status"],
    "search": ["web_search"],
    "sofabaton": ["control_sofabaton", "control_device"],
    "plants": ["check_plant_status"],
    # The two relay channels always travel together: whichever intent matched,
    # the model must be able to see both the phone one and the out-loud one, or
    # it will deliver on the only channel it was given (2026-08-20).
    "partner_message": ["send_partner_message", "announce_on_speaker"],
    "announce": ["announce_on_speaker", "send_partner_message"],
}

# Tools that are always included regardless of matched intent.
# - get_current_datetime: tiny (~30 tokens).
# - web_search: the universal escape hatch (~100 tokens). Grounding rules
#   forbid answering from training data, so when a dedicated tool errors or
#   comes up empty (e.g. an unmapped league), the model MUST be able to
#   escalate to web_search — even when no "search" keyword matched.
_ALWAYS_INCLUDE = {"get_current_datetime", "web_search", "report_garbled_speech"}

# Imperative openers that mark an utterance as a complete command. When one of
# these is the FIRST word, report_garbled_speech is withheld from the toolset:
# a sentence that opens with a clear action verb is by definition not a
# truncated fragment, so the escape hatch has no legitimate use — yet the
# model would still sometimes take it when a proper noun later in the sentence
# looked misheard ("Play is it my love by oh he did", band "Oh He Dead",
# 2026-07-29). Withholding the tool here keeps the decision entirely with the
# model among its REAL tools — the same exposure-shaping this router already
# does for every intent — rather than second-guessing its output after the
# fact. Fragments keep the tool: "the master bedroom shade", "itchen lights"
# and bare "play the"-style stubs (fewer than 3 words) open with none of these
# or are too short.
_COMMAND_OPENERS = frozenset({
    "play", "put", "turn", "set", "open", "close", "stop", "pause", "resume",
    "skip", "shuffle", "dim", "brighten", "lock", "unlock", "start", "cancel",
    "add", "remove", "delete", "lower", "raise", "mute", "unmute",
    "tell", "show", "check", "search", "find", "remind", "wake", "arm",
    "disarm", "enable", "disable", "restart", "switch",
})


def _garble_tool_offered(user_text: str | None) -> bool:
    """Should report_garbled_speech be in the toolset for this utterance?"""
    if not user_text:
        return True
    words = user_text.strip().lower().split()
    if len(words) < 3:
        return True  # short stubs ("play the") are exactly what the tool is for
    return words[0].strip(".,!?'\"") not in _COMMAND_OPENERS

# Fallback bundle used when NO intent matches.
#
# Previously a no-match sent the ENTIRE catalog (~3500 tokens, 24 tools). That
# is the exact condition that produced the v7.61.0 "brain loops": with every
# tool in front of it and no clear intent, the model would either monologue or
# pick a wrong tool and burn a full extra round-trip.
#
# Instead send a small core: the two things an unclassified utterance is most
# likely to actually want (device control / device state), plus the always-on
# escape hatches. ~1400 tokens instead of ~3500.
#
# Losing niche tools on a no-match is acceptable because web_search is always
# present — the model can still ground an off-topic question rather than
# hallucinate. If this proves too tight, widen this set, do NOT go back to
# sending everything.
_CORE_FALLBACK_TOOLS = {
    "control_device",
    "check_device_status",
    # control_music MUST be here. Without it, any music utterance the keyword
    # patterns miss ("next track.") reaches the model with control_device as
    # the only plausible tool — and the model dutifully calls
    # control_device({'device': 'media_player', 'action': 'next'}), which
    # fuzzy-matches some unrelated media_player entity and skips a track in
    # the wrong room (2026-07-26: kitchen "next track" skipped on the master
    # bathroom Voice satellite). The router must never make a wrong tool the
    # only option for a whole intent class.
    "control_music",
} | _ALWAYS_INCLUDE

# If the full catalog is already at or under this size, filtering buys nothing
# and only costs KV prefix-cache reuse — send everything instead.
_MIN_CATALOG_FOR_FALLBACK_FILTER = 8

# Running match-rate counters. Purely observational: they tell us how often the
# fallback path actually fires in real use, which is the input to deciding
# whether the keyword patterns need widening. Reset on HA restart.
_STATS = {"matched": 0, "unmatched": 0}


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


def _match_rate() -> str:
    """Format the running intent match rate for log output."""
    total = _STATS["matched"] + _STATS["unmatched"]
    if not total:
        return "n/a"
    return f"{_STATS['matched']}/{total} ({100 * _STATS['matched'] // total}%)"


def filter_tools_by_intent(
    all_tools: list[dict[str, Any]],
    intents: set[str],
    user_text: str | None = None,
) -> list[dict[str, Any]]:
    """Filter tool definitions to only those matching classified intents.

    On no-match, falls back to a small core bundle rather than the full
    catalog. Pass user_text to have unmatched utterances logged so the
    keyword patterns can be grown from real usage.
    """
    # Complete commands never see the garble escape hatch (see _COMMAND_OPENERS).
    # Filtering the source list here covers every return path below.
    if not _garble_tool_offered(user_text):
        _LOGGER.info(
            "PureLLM intent-router: command opener %r → report_garbled_speech withheld",
            (user_text or "").strip().split()[0].lower(),
        )
        all_tools = [
            tool for tool in all_tools
            if tool.get("function", {}).get("name") != "report_garbled_speech"
        ]

    if not intents:
        _STATS["unmatched"] += 1

        # Catalog already small — filtering costs prefix-cache reuse for no gain.
        if len(all_tools) <= _MIN_CATALOG_FOR_FALLBACK_FILTER:
            _LOGGER.info(
                "PureLLM intent-router: NO-MATCH utterance=%r → sending all %d "
                "tools (catalog at/below %d, not worth filtering) [match rate %s]",
                user_text, len(all_tools), _MIN_CATALOG_FOR_FALLBACK_FILTER,
                _match_rate(),
            )
            return all_tools

        filtered = [
            tool for tool in all_tools
            if tool.get("function", {}).get("name") in _CORE_FALLBACK_TOOLS
        ]

        # Defensive: if feature gating stripped the core bundle down to nothing,
        # a toolless request would make the model answer ungrounded. Send the
        # full catalog rather than no tools at all.
        if not filtered:
            _LOGGER.warning(
                "PureLLM intent-router: NO-MATCH utterance=%r → core bundle "
                "resolved to 0 tools, falling back to all %d",
                user_text, len(all_tools),
            )
            return all_tools

        _LOGGER.info(
            "PureLLM intent-router: NO-MATCH utterance=%r → core bundle "
            "%d/%d tools (%s) [match rate %s]",
            user_text, len(filtered), len(all_tools),
            ", ".join(sorted(t["function"]["name"] for t in filtered)),
            _match_rate(),
        )
        return filtered

    _STATS["matched"] += 1

    # Collect tool names for matched intents
    needed: set[str] = set(_ALWAYS_INCLUDE)
    for intent in intents:
        needed.update(_INTENT_TO_TOOLS.get(intent, []))

    filtered = [
        tool for tool in all_tools
        if tool.get("function", {}).get("name") in needed
    ]

    _LOGGER.info(
        "PureLLM intent-router: MATCH %s → %d/%d tools (%s) [match rate %s]",
        sorted(intents), len(filtered), len(all_tools),
        ", ".join(sorted(needed)),
        _match_rate(),
    )
    return filtered
