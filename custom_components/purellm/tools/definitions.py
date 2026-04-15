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

    # Get exposed entity names for tool descriptions
    _exposed_names: list[str] = []
    _status_names: list[str] = []  # Includes sensors/binary_sensors for status checks
    _plant_names: list[str] = []
    if hass:
        from ..utils.fuzzy_matching import get_exposed_entity_names
        from .plants import list_plant_names
        _exposed_names = get_exposed_entity_names(hass)
        _status_names = get_exposed_entity_names(hass, include_sensors=True)
        _plant_names = list_plant_names(hass)

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
        status_desc = "Check device status."
        if hass and _status_names:
            status_desc += f" Known devices: {', '.join(_status_names[:80])}."
        tools.append(_tool(
            "check_device_status", status_desc,
            {"device": {"type": "string"}},
            ["device"]
        ))

    # ===== PLANT STATUS (Olen homeassistant-plant integration) =====
    # Only register if plants are actually present — keeps the tool list lean
    # for users who don't use the plant integration.
    if config.enable_plants and _plant_names:
        plant_desc = (
            "Read-only plant sensor queries. ALWAYS use this (never check_device_status) "
            "for ANY plant-related question — moisture, water, watering, soil, "
            "temperature, conductivity, illuminance, humidity, dli, battery, health.\n"
            "EXAMPLES (copy the args exactly):\n"
            "  'what is the soil moisture for X the plant' / 'soil moisture for X'\n"
            "      -> plant='X', metric='moisture'\n"
            "  'does any plant need water' / 'do any plants need watering' / 'any plant dry'\n"
            "      -> metric='moisture', problems_only=true\n"
            "  'is any plant in trouble' / 'any plants with problems'\n"
            "      -> problems_only=true\n"
            "  'how is X the plant' / 'check on X'\n"
            "      -> plant='X'\n"
            "  'what's X's moisture threshold' / 'minimum moisture for X'\n"
            "      -> plant='X', metric='thresholds'\n"
            "'Water', 'watering', 'dry', 'thirsty' ALWAYS map to metric='moisture'.\n"
            f"Known plants: {', '.join(_plant_names)}."
        )
        tools.append(_tool(
            "check_plant_status", plant_desc,
            {
                "plant": {
                    "type": "string",
                    "description": "Plant name only (e.g. 'boogie'). Do NOT include 'the plant' or 'my'. Omit for all plants.",
                },
                "metric": {
                    "type": "string",
                    "enum": ["moisture", "temperature", "conductivity", "illuminance",
                             "humidity", "dli", "battery", "status", "thresholds"],
                    "description": "Specific metric. 'moisture' for water/soil/dry/thirsty questions. Omit for full readout."
                },
                "problems_only": {
                    "type": "boolean",
                    "description": "True for 'needs water' / 'in trouble' / 'has problems' sweeps across all plants."
                },
            },
        ))

    # ===== MUSIC =====
    if config.enable_music and config.room_player_mapping:
        rooms_list = ", ".join(config.room_player_mapping.keys())
        tools.append(_tool(
            "control_music",
            f"Control room-based music only (NOT for specific devices like TVs). To pause/resume/play a specific device, use control_device. Rooms: {rooms_list}. Room required for play/shuffle. Room NOT needed for stop/pause/resume/skip/volume — just pass the action and the tool auto-detects the active player. For volume control while music is playing: use volume_up, volume_down, or set_volume with volume param (0-100). For play action you MUST set media_type. For ordinal/themed requests like 'first christmas album by X', set media_type='album', artist=X, and query to the full phrase (e.g. 'first christmas album').",
            {
                "action": {"type": "string", "enum": ["play", "pause", "resume", "stop", "skip_next", "skip_previous", "restart_track", "what_playing", "transfer", "shuffle", "volume_up", "volume_down", "set_volume"], "description": "Action"},
                "media_type": {"type": "string", "enum": ["artist", "album", "track"], "description": "REQUIRED for play action. Set 'album' for albums, 'track' for songs, 'artist' for artist radio"},
                "query": {"type": "string", "description": "Search query. For tracks: song name. For artists: artist name. For albums: album name. For ordinal/themed requests (e.g. 'first christmas album'), include the full phrase"},
                "album": {"type": "string", "description": "Album name - REQUIRED when media_type is 'album'"},
                "artist": {"type": "string", "description": "Artist name - REQUIRED for ordinal/themed album requests (e.g. 'first christmas album by Kelly Clarkson')"},
                "song_on_album": {"type": "string", "description": "Song name to find the album containing it"},
                "room": {"type": "string", "description": "Target room"},
                "volume": {"type": "integer", "description": "Volume level 0-100 for set_volume action"},
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
    device_desc = "Control devices (lights, switches, locks, fans, blinds, covers, media_player, scripts, automations). For specific device commands (TV pause, etc) or launching/running exposed scripts and automations. When user says 'launch X', use action=launch with device=X (the app/script name the user said, e.g. 'YouTube', 'Netflix') — do NOT substitute a physical device name like a TV. The tool will find the correct launch script or streaming device automatically. Only include params user explicitly requested."
    if _exposed_names:
        device_desc += f" Known devices (user aliases — use ONLY when the user's words clearly match one of these; pass the user's original words as-is if no close match): {', '.join(_exposed_names[:50])}."
    tools.append(_tool(
        "control_device",
        device_desc,
        {
            "device": {"type": "string", "description": "Device name (fuzzy matched)"},
            "entity_id": {"type": "string"},
            "entity_ids": {"type": "array", "items": {"type": "string"}},
            "area": {"type": "string", "description": "Area name (instead of device)"},
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
            f"SofaBaton multi-device activities ONLY (NOT for controlling individual devices like TVs/media players — use control_device for those). Activities: {', '.join(activity_names)}.",
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
