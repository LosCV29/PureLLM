"""Tool definitions builder for PureLLM.

This module provides a helper function to build tool definitions
based on enabled features. Uses a cleaner factory pattern instead
of 375 lines of repetitive boilerplate.
"""
from __future__ import annotations

from typing import Any


def _tool(name: str, description: str, properties: dict = None, required: list = None) -> dict:
    """Create a tool definition in OpenAI format.

    Args:
        name: Tool function name
        description: Tool description for the LLM
        properties: Parameter properties dict
        required: List of required parameter names

    Returns:
        Tool definition dict
    """
    params = {"type": "object", "properties": properties or {}}
    if required:
        params["required"] = required

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params
        }
    }


def build_tools(config: "ToolConfig") -> list[dict]:
    """Build the tools list based on enabled features."""
    tools = []

    # ===== CORE TOOLS (always enabled) =====
    tools.append(_tool("get_current_datetime", "Get current date/time."))

    # ===== WEATHER =====
    if config.enable_weather and config.openweathermap_api_key:
        tools.append(_tool(
            "get_weather_forecast",
            "Get weather. Omit location for local.",
            {
                "location": {"type": "string", "description": "City, State/Country"},
                "forecast_type": {"type": "string", "enum": ["current", "tomorrow", "weekly", "sun_times"]}
            }
        ))

    # ===== PLACES =====
    if config.enable_places and config.google_places_api_key:
        tools.append(_tool(
            "find_nearby_places",
            "Find nearby places (NOT restaurants).",
            {"query": {"type": "string"}, "max_results": {"type": "integer"}},
            ["query"]
        ))

    # ===== THERMOSTAT =====
    if config.enable_thermostat and config.thermostat_entity:
        temp_unit = config.temp_unit
        step = config.thermostat_temp_step
        tools.append(_tool(
            "control_thermostat",
            f"Control thermostat. raise/lower=±{step}{temp_unit}, set=exact temp, set_mode, check=status.",
            {
                "action": {"type": "string", "enum": ["raise", "lower", "set", "check", "set_mode"]},
                "temperature": {"type": "number", "description": f"Temp in {temp_unit}"},
                "hvac_mode": {"type": "string", "enum": ["heat", "cool", "heat_cool", "auto", "off"]}
            },
            ["action"]
        ))

    # ===== WIKIPEDIA/AGE =====
    if config.enable_wikipedia:
        tools.append(_tool(
            "calculate_age", "Get person's age.",
            {"person_name": {"type": "string"}},
            ["person_name"]
        ))
        tools.append(_tool(
            "get_wikipedia_summary", "Get Wikipedia info.",
            {"topic": {"type": "string"}},
            ["topic"]
        ))

    # ===== SPORTS =====
    if config.enable_sports:
        tools.append(_tool(
            "get_sports_info",
            "Get team schedule/scores. Include sport for ambiguous names. Include 'Champions League' for European games.",
            {
                "team_name": {"type": "string", "description": "Team + sport/competition"},
                "query_type": {"type": "string", "enum": ["last_game", "next_game", "standings", "schedule", "both"]}
            },
            ["team_name"]
        ))
        tools.append(_tool(
            "get_ufc_info", "Get UFC event info.",
            {"query_type": {"type": "string", "enum": ["next_event", "upcoming"]}}
        ))
        tools.append(_tool(
            "check_league_games", "Count league games today/tomorrow.",
            {"league": {"type": "string"}, "date": {"type": "string", "enum": ["today", "tomorrow"]}},
            ["league"]
        ))
        tools.append(_tool(
            "list_league_games", "List league matchups/times.",
            {"league": {"type": "string"}, "date": {"type": "string", "enum": ["today", "tomorrow"]}},
            ["league"]
        ))

    # ===== CALENDAR =====
    if config.enable_calendar and config.calendar_entities:
        tools.append(_tool(
            "get_calendar_events", "Get calendar events.",
            {"days_ahead": {"type": "integer"}}
        ))

    # ===== CAMERAS =====
    if config.enable_cameras:
        camera_desc = "Check camera with AI vision."
        if config.frigate_camera_names:
            camera_desc += f" Cameras: {', '.join(config.frigate_camera_names.keys())}."
        tools.append(_tool(
            "check_camera", camera_desc,
            {"location": {"type": "string", "description": "Camera name"}, "query": {"type": "string", "description": "What to look for"}},
            ["location"]
        ))

    # ===== DEVICE STATUS =====
    if config.enable_device_status:
        status_desc = "Check device status."
        if config.device_aliases:
            alias_names = ", ".join(config.device_aliases.keys())
            status_desc += f" Known devices: {alias_names}."
        tools.append(_tool(
            "check_device_status", status_desc,
            {"device": {"type": "string"}},
            ["device"]
        ))

    # ===== MUSIC =====
    if config.enable_music and config.room_player_mapping:
        rooms_list = ", ".join(config.room_player_mapping.keys())
        tools.append(_tool(
            "control_music",
            f"Room music (NOT specific devices like TVs). Rooms: {rooms_list}. Room required for play/shuffle. Stop/pause/skip auto-detects player. Play REQUIRES media_type.",
            {
                "action": {"type": "string", "enum": ["play", "pause", "resume", "stop", "skip_next", "skip_previous", "restart_track", "what_playing", "transfer", "shuffle"]},
                "media_type": {"type": "string", "enum": ["artist", "album", "track"], "description": "Required for play"},
                "query": {"type": "string", "description": "Search query or modifier phrase"},
                "album": {"type": "string", "description": "Album name or genre tag"},
                "artist": {"type": "string"},
                "song_on_album": {"type": "string", "description": "Find album containing this song"},
                "room": {"type": "string"},
            },
            ["action"]
        ))

    # ===== TIMERS (always enabled) =====
    tools.append(_tool(
        "control_timer", "Control timers.",
        {
            "action": {"type": "string", "enum": ["start", "cancel", "pause", "resume", "status", "add", "restart", "finish"]},
            "duration": {"type": "string", "description": "e.g. '10 minutes', 'half an hour'"},
            "name": {"type": "string"},
            "add_time": {"type": "string"}
        },
        ["action"]
    ))

    # ===== LISTS (always enabled) =====
    tools.append(_tool(
        "manage_list",
        "Manage shopping/to-do lists. clear/show/sort REQUIRE status param.",
        {
            "action": {"type": "string", "enum": ["add", "complete", "remove", "remove_all", "show", "clear", "sort", "list_all"]},
            "item": {"type": "string"},
            "list_name": {"type": "string"},
            "status": {"type": "string", "enum": ["active", "completed"], "description": "Required for clear/show/sort"}
        },
        ["action"]
    ))

    # ===== DEVICE CONTROL (always enabled) =====
    device_desc = "Control devices (lights, switches, locks, fans, blinds, covers, media_player). For specific device commands (TV pause, etc). Only include params user explicitly requested."
    if config.device_aliases:
        alias_names = ", ".join(config.device_aliases.keys())
        device_desc += f" Known devices: {alias_names}."
    tools.append(_tool(
        "control_device",
        device_desc,
        {
            "device": {"type": "string", "description": "Device name (fuzzy matched)"},
            "entity_id": {"type": "string"},
            "entity_ids": {"type": "array", "items": {"type": "string"}},
            "area": {"type": "string", "description": "Area name (instead of device)"},
            "domain": {"type": "string", "enum": ["light", "switch", "lock", "cover", "fan", "media_player", "climate", "vacuum", "scene", "script", "all"]},
            "action": {"type": "string", "enum": ["turn_on", "turn_off", "toggle", "dim", "lock", "unlock", "open", "close", "stop", "preset", "set_position", "play", "pause", "resume", "next", "previous", "volume_up", "volume_down", "set_volume", "mute", "unmute", "set_temperature", "set_hvac_mode", "start", "dock", "locate", "return_home", "activate"]},
            "brightness": {"type": "integer"},
            "color": {"type": "string"},
            "color_temp": {"type": "integer"},
            "position": {"type": "integer"},
            "volume": {"type": "integer"},
            "temperature": {"type": "number"},
            "hvac_mode": {"type": "string", "enum": ["heat", "cool", "heat_cool", "auto", "off"]},
            "fan_speed": {"type": "string", "enum": ["low", "medium", "high", "auto"]}
        },
        ["action", "device"]
    ))

    # ===== SOFABATON ACTIVITIES =====
    if config.sofabaton_activities:
        activity_names = [a.get("name", "") for a in config.sofabaton_activities if a.get("name")]
        tools.append(_tool(
            "control_sofabaton",
            f"SofaBaton: {', '.join(activity_names)}.",
            {"activity": {"type": "string"}, "action": {"type": "string", "enum": ["start", "stop"]}},
            ["activity", "action"]
        ))

    # ===== WEB SEARCH =====
    if config.enable_search and config.tavily_api_key:
        tools.append(_tool(
            "web_search", "Web search for news, reviews, current events.",
            {
                "query": {"type": "string"},
                "days": {"type": "integer", "description": "Limit to last N days"},
                "include_domains": {"type": "array", "items": {"type": "string"}},
                "exclude_domains": {"type": "array", "items": {"type": "string"}}
            },
            ["query"]
        ))

    return tools


class ToolConfig:
    """Configuration for tool building.

    This class wraps the entity configuration to provide a clean interface
    for the build_tools function.
    """

    def __init__(self, entity):
        """Initialize from entity configuration."""
        self.enable_weather = entity.enable_weather
        self.enable_calendar = entity.enable_calendar
        self.enable_cameras = entity.enable_cameras
        self.enable_sports = entity.enable_sports
        self.enable_places = entity.enable_places
        self.enable_thermostat = entity.enable_thermostat
        self.enable_device_status = entity.enable_device_status
        self.enable_wikipedia = entity.enable_wikipedia
        self.enable_music = entity.enable_music
        self.enable_search = entity.enable_search

        self.openweathermap_api_key = entity.openweathermap_api_key
        self.google_places_api_key = entity.google_places_api_key
        self.tavily_api_key = entity.tavily_api_key

        self.thermostat_entity = entity.thermostat_entity
        self.thermostat_temp_step = entity.thermostat_temp_step
        self.temp_unit = entity.temp_unit

        self.calendar_entities = entity.calendar_entities
        self.room_player_mapping = entity.room_player_mapping
        self.sofabaton_activities = getattr(entity, 'sofabaton_activities', [])
        self.frigate_camera_names = getattr(entity, 'frigate_camera_names', {})
        self.device_aliases = getattr(entity, 'device_aliases', {})
