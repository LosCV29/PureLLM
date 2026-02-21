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
    """Build the tools list based on enabled features.

    Args:
        config: Configuration object with feature flags and settings

    Returns:
        List of tool definitions in OpenAI format
    """
    tools = []

    # ===== CORE TOOLS (always enabled) =====
    tools.append(_tool(
        "get_current_datetime",
        "Get current date/time.",
    ))

    # ===== WEATHER =====
    if config.enable_weather and config.openweathermap_api_key:
        tools.append(_tool(
            "get_weather_forecast",
            "Get weather. Omit location for local weather. Include state for US cities (Austin, TX), country for international (Paris, France).",
            {
                "location": {"type": "string", "description": "City with state/country. Omit for local."},
                "forecast_type": {"type": "string", "enum": ["current", "tomorrow", "weekly", "sun_times"], "description": "current=today, tomorrow, weekly=5-day, sun_times=sunrise/sunset"}
            }
        ))

    # ===== PLACES =====
    if config.enable_places and config.google_places_api_key:
        tools.append(_tool(
            "find_nearby_places",
            "Find nearby places (NOT restaurants). Use for: salons, gas, pharmacy, gym, etc.",
            {"query": {"type": "string", "description": "What to search"}, "max_results": {"type": "integer", "description": "Max results (default 5)"}},
            ["query"]
        ))

    # ===== THERMOSTAT =====
    if config.enable_thermostat and config.thermostat_entity:
        temp_unit = config.temp_unit
        step = config.thermostat_temp_step
        tools.append(_tool(
            "control_thermostat",
            f"Control thermostat. raise/lower=±{step}{temp_unit}, set=specific temp, set_mode=heat/cool/off, check=status.",
            {
                "action": {"type": "string", "enum": ["raise", "lower", "set", "check", "set_mode"], "description": "Action"},
                "temperature": {"type": "number", "description": f"Temp in {temp_unit} (for 'set')"},
                "hvac_mode": {"type": "string", "enum": ["heat", "heating", "cool", "cooling", "heat_cool", "auto", "off"], "description": "Mode (for 'set_mode')"}
            },
            ["action"]
        ))

    # ===== WIKIPEDIA/AGE =====
    if config.enable_wikipedia:
        tools.append(_tool(
            "calculate_age",
            "Get person's age. Never guess - always use this.",
            {"person_name": {"type": "string", "description": "Person's name"}},
            ["person_name"]
        ))

        tools.append(_tool(
            "get_wikipedia_summary",
            "Get Wikipedia info about people, places, events.",
            {"topic": {"type": "string", "description": "Topic to look up"}},
            ["topic"]
        ))

    # ===== SPORTS =====
    if config.enable_sports:
        tools.append(_tool(
            "get_sports_info",
            "Get team info (schedule, scores). Include sport for ambiguous names: 'Panthers hockey' for NHL, 'Panthers football' for NFL. Include 'Champions League' or 'UCL' for European competition, 'football'/'basketball' for college.",
            {
                "team_name": {"type": "string", "description": "Team + sport/competition. Add sport keyword for ambiguous names (e.g. 'Panthers hockey'). MUST include 'Champions League' or 'UCL' for European competition games."},
                "query_type": {"type": "string", "enum": ["last_game", "next_game", "standings", "schedule", "both"], "description": "last_game, next_game, standings, schedule, or both (default)"}
            },
            ["team_name"]
        ))

        tools.append(_tool(
            "get_ufc_info",
            "Get UFC event info.",
            {"query_type": {"type": "string", "enum": ["next_event", "upcoming"], "description": "next_event or upcoming"}}
        ))

        tools.append(_tool(
            "check_league_games",
            "Check if league has games (count only). For specific teams use get_sports_info.",
            {"league": {"type": "string", "description": "NFL, NBA, MLB, NHL, MLS, college basketball, college football, etc."}, "date": {"type": "string", "enum": ["today", "tomorrow"], "description": "Day (default: today)"}},
            ["league"]
        ))

        tools.append(_tool(
            "list_league_games",
            "List all games in league with matchups/times.",
            {"league": {"type": "string", "description": "NFL, NBA, MLB, NHL, MLS, college basketball, college football, etc."}, "date": {"type": "string", "enum": ["today", "tomorrow"], "description": "Day (default: today)"}},
            ["league"]
        ))

    # ===== CALENDAR =====
    if config.enable_calendar and config.calendar_entities:
        tools.append(_tool(
            "get_calendar_events",
            "Get calendar events.",
            {"days_ahead": {"type": "integer", "description": "Days ahead (default 7, max 30)"}}
        ))

    # ===== RESTAURANTS =====
    if config.enable_restaurants and config.google_places_api_key:
        tools.append(_tool(
            "get_restaurant_recommendations",
            "Find restaurants. For non-food places use find_nearby_places.",
            {
                "query": {"type": "string", "description": "Food/restaurant type"},
                "sort_by": {"type": "string", "enum": ["rating", "review_count", "distance"], "description": "Sort by (default: rating)"},
                "price": {"type": "string", "description": "1=$, 2=$$, 3=$$$, 4=$$$$. Combine: '1,2'"},
                "max_results": {"type": "integer", "description": "Results (default 3)"}
            },
            ["query"]
        ))

        tools.append(_tool(
            "book_restaurant",
            "Get reservation link for specific restaurant.",
            {
                "restaurant_name": {"type": "string", "description": "Restaurant name"},
                "location": {"type": "string", "description": "City (optional)"},
                "party_size": {"type": "integer", "description": "Guests (default 2)"},
                "date": {"type": "string", "description": "Date YYYY-MM-DD"},
                "time": {"type": "string", "description": "Time (7pm, 19:00)"}
            },
            ["restaurant_name"]
        ))

    # ===== CAMERAS (Frigate + Vision LLM) =====
    if config.enable_cameras:
        camera_desc = "Check camera with live video analysis. Captures a short video clip and describes the scene using AI vision."
        if config.frigate_camera_names:
            cams_list = ", ".join(config.frigate_camera_names.keys())
            camera_desc += f" Available cameras: {cams_list}."

        tools.append(_tool(
            "check_camera",
            camera_desc,
            {"location": {"type": "string", "description": "Camera name (e.g. backyard, front_porch)"}, "query": {"type": "string", "description": "What to look for (optional)"}},
            ["location"]
        ))

    # ===== DEVICE STATUS =====
    if config.enable_device_status:
        tools.append(_tool(
            "check_device_status",
            "Check device status.",
            {"device": {"type": "string", "description": "Device name as user said it"}},
            ["device"]
        ))

        tools.append(_tool(
            "get_device_history",
            "Get device state history.",
            {
                "device": {"type": "string", "description": "Device name"},
                "days_back": {"type": "integer", "description": "Days of history (default 1)"},
                "date": {"type": "string", "description": "Specific date YYYY-MM-DD"}
            },
            ["device"]
        ))

    # ===== MUSIC =====
    if config.enable_music and config.room_player_mapping:
        rooms_list = ", ".join(config.room_player_mapping.keys())
        tools.append(_tool(
            "control_music",
            f"Control room-based music only (NOT for specific devices like TVs). To pause/resume/play a specific device, use control_device. Rooms: {rooms_list}. Room required for play/shuffle. For play action you MUST set media_type.",
            {
                "action": {"type": "string", "enum": ["play", "pause", "resume", "stop", "skip_next", "skip_previous", "restart_track", "what_playing", "transfer", "shuffle"], "description": "Action"},
                "media_type": {"type": "string", "enum": ["artist", "album", "track"], "description": "REQUIRED for play action. Set 'album' for albums, 'track' for songs, 'artist' for artist radio"},
                "query": {"type": "string", "description": "Search query. For tracks: song name. For artists: artist name. For ordinal/tagged album requests (e.g. 'first christmas album'), put the full modifier phrase here (e.g. 'first christmas album')"},
                "album": {"type": "string", "description": "Album name - REQUIRED when media_type is 'album'. For ordinal/tagged requests (first/second/latest + genre), set this to the genre/tag word only (e.g. 'christmas', 'holiday', 'live')"},
                "artist": {"type": "string", "description": "Artist name"},
                "song_on_album": {"type": "string", "description": "Song name to find the album containing it"},
                "room": {"type": "string", "description": "Target room"},
            },
            ["action"]
        ))

    # ===== TIMERS (always enabled) =====
    tools.append(_tool(
        "control_timer",
        "Control timers. Natural language: 'half an hour', '2 and a half hours'.",
        {
            "action": {"type": "string", "enum": ["start", "cancel", "pause", "resume", "status", "add", "restart", "finish"], "description": "Action"},
            "duration": {"type": "string", "description": "Duration: '10 minutes', 'half an hour'"},
            "name": {"type": "string", "description": "Timer name (optional)"},
            "add_time": {"type": "string", "description": "Time to add (for 'add' action)"}
        },
        ["action"]
    ))

    # ===== LISTS (always enabled) =====
    tools.append(_tool(
        "manage_list",
        "Manage shopping/to-do lists. clear/show/sort REQUIRE status param: 'active' for unchecked items, 'completed' for checked/done items.",
        {
            "action": {"type": "string", "enum": ["add", "complete", "remove", "remove_all", "show", "clear", "sort", "list_all"], "description": "Action"},
            "item": {"type": "string", "description": "Item name"},
            "list_name": {"type": "string", "description": "List name (optional)"},
            "status": {"type": "string", "enum": ["active", "completed"], "description": "REQUIRED for clear/show/sort. 'active'=unchecked items, 'completed'=checked/done items"}
        },
        ["action"]
    ))

    # ===== REMINDERS (always enabled) =====
    tools.append(_tool(
        "create_reminder",
        "Create reminder.",
        {"reminder": {"type": "string", "description": "What to remind"}, "time": {"type": "string", "description": "When (e.g., 'at 5pm', 'tomorrow')"}},
        ["reminder"]
    ))

    tools.append(_tool(
        "get_reminders",
        "Get upcoming reminders.",
    ))

    # ===== DEVICE CONTROL (always enabled - LLM fallback) =====
    tools.append(_tool(
        "control_device",
        "Control devices (lights, switches, locks, fans, blinds, covers, media_player). Use device name for fuzzy matching. Blinds: open/close/stop. Media: pause/resume/play/mute/unmute. ALWAYS use this for specific device commands (e.g. 'resume the TV', 'pause the TV').",
        {
            "device": {"type": "string", "description": "Device name (fuzzy matched)"},
            "entity_id": {"type": "string", "description": "Exact entity ID (optional)"},
            "entity_ids": {"type": "array", "items": {"type": "string"}, "description": "Multiple entity IDs"},
            "area": {"type": "string", "description": "Control all in area"},
            "domain": {"type": "string", "enum": ["light", "switch", "lock", "cover", "fan", "media_player", "climate", "vacuum", "scene", "script", "all"], "description": "Device type filter"},
            "action": {"type": "string", "enum": ["turn_on", "turn_off", "toggle", "dim", "lock", "unlock", "open", "close", "stop", "preset", "favorite", "set_position", "play", "pause", "resume", "next", "previous", "volume_up", "volume_down", "set_volume", "mute", "unmute", "set_temperature", "set_hvac_mode", "start", "dock", "locate", "return_home", "activate"], "description": "Action"},
            "brightness": {"type": "integer", "description": "0-100"},
            "color": {"type": "string", "description": "Color name"},
            "color_temp": {"type": "integer", "description": "Kelvin (2700-6500)"},
            "position": {"type": "integer", "description": "Cover 0-100"},
            "volume": {"type": "integer", "description": "0-100"},
            "temperature": {"type": "number", "description": "Target temp"},
            "hvac_mode": {"type": "string", "enum": ["heat", "heating", "cool", "cooling", "heat_cool", "auto", "off"], "description": "HVAC mode"},
            "fan_speed": {"type": "string", "enum": ["low", "medium", "high", "auto"], "description": "Fan speed"}
        },
        ["action"]
    ))

    # ===== SOFABATON ACTIVITIES =====
    if config.sofabaton_activities:
        activity_names = [a.get("name", "") for a in config.sofabaton_activities if a.get("name")]
        tools.append(_tool(
            "control_sofabaton",
            f"Control SofaBaton activities: {', '.join(activity_names)}.",
            {
                "activity": {"type": "string", "description": "Activity name"},
                "action": {"type": "string", "enum": ["start", "stop"], "description": "start or stop"}
            },
            ["activity", "action"]
        ))

    # ===== WEB SEARCH =====
    if config.enable_search and config.tavily_api_key:
        tools.append(_tool(
            "web_search",
            "Web search. Use for 'search for', 'google', news, reviews, current events.",
            {
                "query": {"type": "string", "description": "Search query"},
                "days": {"type": "integer", "description": "Limit to last N days (optional)"},
                "include_domains": {"type": "array", "items": {"type": "string"}, "description": "Target domains (optional)"},
                "exclude_domains": {"type": "array", "items": {"type": "string"}, "description": "Exclude domains (optional)"}
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
        self.enable_restaurants = entity.enable_restaurants
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
