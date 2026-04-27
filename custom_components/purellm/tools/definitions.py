"""Tool definitions builder for PureLLM.

This module provides a helper function to build tool definitions
based on enabled features. Uses a cleaner factory pattern instead
of 375 lines of repetitive boilerplate.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


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


def build_tools(config: "ToolConfig", hass: "HomeAssistant | None" = None) -> list[dict]:
    """Build the tools list based on enabled features."""
    tools = []

    # Plant names are still needed to decide whether to expose check_plant_status,
    # since the tool is gated on at least one plant existing.
    _has_plants = False
    if hass:
        from .plants import list_plant_names
        _has_plants = bool(list_plant_names(hass))

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
            f"Control thermostat. raise/lower=±{step} degrees, set=exact temp, set_mode, check=status.",
            {
                "action": {"type": "string", "enum": ["raise", "lower", "set", "check", "set_mode"]},
                "temperature": {"type": "number", "description": "Temp in degrees"},
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
            "Team schedule/scores. Include sport for ambiguous names; include 'Champions League' for European games.",
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
            "get_calendar_events", "Get calendar events, birthdays, or holidays.",
            {
                "query_type": {
                    "type": "string",
                    "enum": ["upcoming", "today", "tomorrow", "week", "month", "birthday", "holiday"],
                    "description": "Type of calendar query. Use 'holiday' for holiday-related questions."
                }
            }
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
        tools.append(_tool(
            "check_device_status", "Check device status. Fuzzy-matches the user's name.",
            {"device": {"type": "string"}},
            ["device"]
        ))

    # ===== PLANT STATUS (Olen homeassistant-plant integration) =====
    # Only register if plants are actually present — keeps the tool list lean
    # for users who don't use the plant integration.
    if config.enable_plants and _has_plants:
        tools.append(_tool(
            "check_plant_status",
            "Plant sensor readings. Omit plant for all plants. problems_only=true to scan for issues.",
            {
                "plant": {
                    "type": "string",
                    "description": "Plant name (no 'the plant'/'my'). Omit for all.",
                },
                "metric": {
                    "type": "string",
                    "enum": ["moisture", "temperature", "conductivity", "illuminance",
                             "humidity", "dli", "battery", "status", "thresholds"],
                    "description": "Use 'moisture' for water/dry/thirsty.",
                },
                "problems_only": {"type": "boolean"},
            },
        ))

    # ===== MUSIC =====
    if config.enable_music and config.room_player_mapping:
        rooms_list = ", ".join(config.room_player_mapping.keys())
        tools.append(_tool(
            "control_music",
            f"Room-based music. Rooms: {rooms_list}. Room required for play/shuffle; auto-detected for stop/pause/resume/skip/volume. media_type required for play. For specific devices use control_device.",
            {
                "action": {"type": "string", "enum": ["play", "pause", "resume", "stop", "skip_next", "skip_previous", "restart_track", "what_playing", "transfer", "shuffle", "volume_up", "volume_down", "set_volume"]},
                "media_type": {"type": "string", "enum": ["artist", "album", "track"]},
                "query": {"type": "string", "description": "Track/artist/album name. For ordinal/themed (e.g. 'first christmas album') use the full phrase."},
                "album": {"type": "string"},
                "artist": {"type": "string"},
                "song_on_album": {"type": "string", "description": "Find the album containing this song."},
                "room": {"type": "string"},
                "volume": {"type": "integer", "description": "0-100"},
            },
            ["action"]
        ))

    # ===== WHITE NOISE / AMBIENT SOUNDS =====
    if config.room_player_mapping:
        rooms_list = ", ".join(config.room_player_mapping.keys())
        tools.append(_tool(
            "control_white_noise",
            f"Ambient sounds (white/pink/brown noise, rain, ocean, fan, thunder, shushing) for sleep/focus. Rooms: {rooms_list}. Room required for play; optional for stop/volume.",
            {
                "action": {"type": "string", "enum": ["play", "stop", "volume_up", "volume_down", "set_volume"]},
                "sound": {"type": "string", "enum": ["white", "pink", "brown", "rain", "ocean", "fan", "thunder", "shushing"]},
                "room": {"type": "string"},
                "volume": {"type": "integer", "description": "0-100"},
            },
            ["action"],
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
    tools.append(_tool(
        "control_device",
        "Control lights, switches, locks, fans, covers, media_player, scripts, automations. For 'launch X' pass device=X as-is (app/script name). Only include params the user requested.",
        {
            "device": {"type": "string", "description": "Fuzzy-matched name."},
            "entity_id": {"type": "string"},
            "entity_ids": {"type": "array", "items": {"type": "string"}},
            "area": {"type": "string"},
            "domain": {"type": "string", "enum": ["light", "switch", "lock", "cover", "fan", "media_player", "climate", "vacuum", "scene", "script", "automation", "all"]},
            "action": {"type": "string", "enum": ["turn_on", "turn_off", "toggle", "dim", "lock", "unlock", "open", "close", "stop", "preset", "set_position", "play", "pause", "resume", "next", "previous", "volume_up", "volume_down", "set_volume", "mute", "unmute", "set_temperature", "set_hvac_mode", "start", "dock", "locate", "return_home", "activate", "launch"]},
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
            f"SofaBaton multi-device activities. Activities: {', '.join(activity_names)}. Use control_device for individual devices.",
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
        self.enable_plants = entity.enable_plants

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
