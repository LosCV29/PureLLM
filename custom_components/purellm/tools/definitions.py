"""Tool definitions builder for PolyVoice.

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
        "Get the current date and time. Use for 'what day is it', 'what's the date', 'what time is it', or any time/date questions.",
    ))

    # ===== WEATHER =====
    if config.enable_weather and config.openweathermap_api_key:
        tools.append(_tool(
            "get_weather_forecast",
            "Get current weather AND forecast. LOCATION RULES: 1) 'what's the weather' with NO city mentioned = DO NOT pass location (uses user's default). 2) US cities = ALWAYS include state (e.g., 'Austin, TX', 'Miami, FL', 'Portland, OR'). 3) International = ALWAYS include country (e.g., 'Paris, France', 'London, UK', 'Tokyo, Japan').",
            {
                "location": {
                    "type": "string",
                    "description": "DO NOT PASS if user says 'what's the weather' without naming a city. For US cities: MUST include state abbreviation (e.g., 'Chicago, IL', 'Seattle, WA'). For international: MUST include country (e.g., 'Berlin, Germany'). This prevents ambiguous matches like Paris, TX vs Paris, France."
                },
                "forecast_type": {
                    "type": "string",
                    "enum": ["current", "weekly"],
                    "description": "Use 'current' for today's weather (default). Use 'weekly' ONLY when user asks for 'weekly forecast', 'this week', or '5-day forecast'."
                }
            }
        ))

    # ===== PLACES =====
    if config.enable_places and config.google_places_api_key:
        tools.append(_tool(
            "find_nearby_places",
            "Find nearby places for DIRECTIONS. Use for 'nearest', 'closest', 'where is', 'find a', 'directions to'. NOT for food recommendations (use get_restaurant_recommendations instead).",
            {
                "query": {"type": "string", "description": "What to search for (e.g., 'Publix', 'gas station', 'pharmacy')"},
                "max_results": {"type": "integer", "description": "Max results (default: 5, max: 20)"}
            },
            ["query"]
        ))

    # ===== THERMOSTAT =====
    if config.enable_thermostat and config.thermostat_entity:
        temp_unit = config.temp_unit
        step = config.thermostat_temp_step
        tools.append(_tool(
            "control_thermostat",
            f"Control or check the thermostat/AC/air. Use for: 'raise/lower the AC' (±{step}{temp_unit}), 'set AC to 72', 'what is the AC set to', 'what's the temp inside'.",
            {
                "action": {
                    "type": "string",
                    "enum": ["raise", "lower", "set", "check"],
                    "description": f"'raise' = +{step}{temp_unit}, 'lower' = -{step}{temp_unit}, 'set' = specific temp, 'check' = get current status"
                },
                "temperature": {
                    "type": "number",
                    "description": f"Target temperature in {temp_unit} (only for 'set' action)"
                }
            },
            ["action"]
        ))

    # ===== WIKIPEDIA/AGE =====
    if config.enable_wikipedia:
        tools.append(_tool(
            "calculate_age",
            "REQUIRED for 'how old is [person]' questions. Looks up birthdate from Wikidata and calculates current age. NEVER guess ages - ALWAYS use this tool.",
            {"person_name": {"type": "string", "description": "The person's name (e.g., 'LeBron James', 'Taylor Swift')"}},
            ["person_name"]
        ))

        tools.append(_tool(
            "get_wikipedia_summary",
            "Get information from Wikipedia. Use for 'who is', 'what is', 'tell me about' questions.",
            {"topic": {"type": "string", "description": "The topic to look up (e.g., 'Albert Einstein', 'World War II')"}},
            ["topic"]
        ))

    # ===== SPORTS =====
    if config.enable_sports:
        tools.append(_tool(
            "get_sports_info",
            "MANDATORY: You MUST call this tool for ANY sports question. NEVER answer sports questions from memory - scores and schedules change constantly. Use for: 'did [team] win', 'when is the next [team] game', '[team] score'.",
            {
                "team_name": {
                    "type": "string",
                    "description": "Team name. If user mentions a specific league (Champions League, UCL), include it! Examples: 'Liverpool Champions League', 'Miami Heat'"
                },
                "query_type": {
                    "type": "string",
                    "enum": ["last_game", "next_game", "standings", "both"],
                    "description": "What info to get: 'last_game' for recent result, 'next_game' for upcoming, 'standings' for league position, 'both' for last and next games (default)"
                }
            },
            ["team_name"]
        ))

        tools.append(_tool(
            "get_ufc_info",
            "Get UFC/MMA fight information. Use for: 'next UFC event', 'when is UFC', 'upcoming UFC fights'.",
            {
                "query_type": {
                    "type": "string",
                    "enum": ["next_event", "upcoming"],
                    "description": "What info to get: 'next_event' for the next UFC event, 'upcoming' for list of upcoming events"
                }
            }
        ))

    # ===== STOCKS =====
    if config.enable_stocks:
        tools.append(_tool(
            "get_stock_price",
            "Get current stock price and daily change. Use for: 'Apple stock', 'what's Tesla at', 'AAPL price'. Works with symbols (AAPL, TSLA) or company names.",
            {"symbol": {"type": "string", "description": "Stock symbol (e.g., 'AAPL', 'TSLA') or company name"}},
            ["symbol"]
        ))

    # ===== NEWS =====
    if config.enable_news and config.newsapi_key:
        tools.append(_tool(
            "get_news",
            "Get latest news headlines. Use for 'what's in the news', 'latest headlines', 'news about X', 'tech news'.",
            {
                "category": {
                    "type": "string",
                    "enum": ["general", "science", "sports", "business", "health", "entertainment", "tech", "politics", "food", "travel"],
                    "description": "News category (optional)"
                },
                "topic": {"type": "string", "description": "Specific topic to search for (e.g., 'Tesla', 'AI')"},
                "max_results": {"type": "integer", "description": "Number of articles to return (default: 5, max: 10)"}
            }
        ))

    # ===== CALENDAR =====
    if config.enable_calendar and config.calendar_entities:
        tools.append(_tool(
            "get_calendar_events",
            "Get upcoming calendar events. Use for 'what's on my calendar', 'any events today', 'my schedule'.",
            {"days_ahead": {"type": "integer", "description": "Number of days to look ahead (default: 7, max: 30)"}}
        ))

    # ===== RESTAURANTS =====
    if config.enable_restaurants and config.yelp_api_key:
        tools.append(_tool(
            "get_restaurant_recommendations",
            "Get restaurant recommendations from Yelp. SORTING: 'best'/'top rated' = rating, 'popular'/'most reviewed' = review_count, 'fancy'/'expensive'/'upscale' = use price filter $$$$, 'cheap'/'budget' = price $. Default sort is rating.",
            {
                "query": {"type": "string", "description": "What type of food or restaurant to search for"},
                "sort_by": {
                    "type": "string",
                    "enum": ["rating", "review_count", "distance", "best_match"],
                    "description": "How to sort: 'rating' for highest rated (default), 'review_count' for most popular/reviewed, 'distance' for closest, 'best_match' for Yelp's algorithm"
                },
                "price": {
                    "type": "string",
                    "description": "Price filter: '1' for $, '2' for $$, '3' for $$$, '4' for $$$$. Can combine like '1,2' for $ and $$. Use '3,4' for expensive/upscale, '1,2' for cheap/budget."
                },
                "max_results": {"type": "integer", "description": "Number of results to return (default: 3, max: 10)"}
            },
            ["query"]
        ))

        tools.append(_tool(
            "book_restaurant",
            "Get a reservation link for a specific restaurant. Use AFTER get_restaurant_recommendations when user wants to book. Returns Yelp reservation URL if available, or Google search fallback.",
            {
                "restaurant_name": {"type": "string", "description": "The exact restaurant name to book (e.g., 'Uchi', 'Olive Garden')"},
                "party_size": {"type": "integer", "description": "Number of guests (default: 2)"},
                "date": {"type": "string", "description": "Reservation date in YYYY-MM-DD format (e.g., '2024-01-20')"},
                "time": {"type": "string", "description": "Reservation time - can be natural ('7pm', '7:30 PM') or 24hr ('19:00')"}
            },
            ["restaurant_name"]
        ))

    # ===== CAMERAS =====
    if config.enable_cameras:
        tools.append(_tool(
            "check_camera",
            "Check a camera with AI vision analysis. Returns scene description and activity detection. Use for: 'check the [location] camera', 'what's happening in [location]'.",
            {
                "location": {"type": "string", "description": "The camera location to check (e.g., 'garage', 'kitchen', 'driveway')"},
                "query": {"type": "string", "description": "Optional specific question about what to look for"}
            },
            ["location"]
        ))

        tools.append(_tool(
            "quick_camera_check",
            "FAST camera check - quickly confirms if anyone is present + one sentence description. Use for: 'is there anyone in [location]'.",
            {"location": {"type": "string", "description": "The camera location to check"}},
            ["location"]
        ))

    # ===== DEVICE STATUS =====
    if config.enable_device_status:
        tools.append(_tool(
            "check_device_status",
            "Check the current status of any device, sensor, door, lock, light, switch, or cover. IMPORTANT: Pass the COMPLETE device name exactly as the user said it.",
            {"device": {"type": "string", "description": "The COMPLETE device name exactly as spoken by the user"}},
            ["device"]
        ))

    # ===== MUSIC =====
    if config.enable_music and config.room_player_mapping:
        rooms_list = ", ".join(config.room_player_mapping.keys())
        tools.append(_tool(
            "control_music",
            f"Control ALL music playback via Music Assistant. Available rooms: {rooms_list}. For pause/stop/resume/skip - NO room needed. For play/shuffle - specify room.",
            {
                "action": {
                    "type": "string",
                    "enum": ["play", "pause", "resume", "stop", "skip_next", "skip_previous", "restart_track", "what_playing", "transfer", "shuffle"],
                    "description": "'play' for songs/albums/artists. 'shuffle' for shuffled playlists by artist/genre."
                },
                "query": {"type": "string", "description": "Song title, album name, or smart modifier. For albums: can include 'latest', 'last', 'newest', 'first', 'debut'. Example: 'bad bunny's latest album' → query='latest', artist='Bad Bunny', media_type='album'."},
                "artist": {"type": "string", "description": "Artist name. REQUIRED for tracks and albums."},
                "album": {"type": "string", "description": "Album name. Use when playing a specific track FROM an album."},
                "song_on_album": {"type": "string", "description": "Use when user wants an album that contains a specific song. Example: 'play the jay-z album with big pimpin' → song_on_album='Big Pimpin', artist='Jay-Z', media_type='album'. The system will find which album contains that song and play it."},
                "room": {"type": "string", "description": f"Target room: {rooms_list}"},
                "media_type": {
                    "type": "string",
                    "enum": ["artist", "album", "track"],
                    "description": "'track' = specific song, 'album' = album, 'artist' = play all music by artist."
                }
            },
            ["action"]
        ))

    # ===== TIMERS (always enabled) =====
    tools.append(_tool(
        "control_timer",
        "Control timers. Understands natural language like 'half an hour', 'one minute', '2 and a half hours'. Use for: 'set a timer', 'cancel timer', 'how much time left', 'pause', 'add 5 minutes', 'restart the timer'.",
        {
            "action": {
                "type": "string",
                "enum": ["start", "cancel", "pause", "resume", "status", "add", "restart", "finish"],
                "description": "'start' to create, 'cancel' to stop, 'pause'/'resume' to control, 'status' to check, 'add' to extend, 'restart' same duration, 'finish' to complete early"
            },
            "duration": {
                "type": "string",
                "description": "Natural language duration: '10 minutes', 'half an hour', 'one hour', '90 seconds', '2 and a half hours', or just '15' for 15 minutes"
            },
            "name": {
                "type": "string",
                "description": "Optional timer name for multi-timer support (e.g., 'pizza', 'laundry', 'eggs')"
            },
            "add_time": {
                "type": "string",
                "description": "Time to add when action='add' (e.g., '5 minutes', 'another 10')"
            }
        },
        ["action"]
    ))

    # ===== LISTS (always enabled) =====
    tools.append(_tool(
        "manage_list",
        "Manage shopping lists and to-do lists. Use for: 'add milk to shopping list', 'what's on my list', 'complete eggs', 'clear the list', 'sort the list', 'show completed items', 'sort my completed items'.",
        {
            "action": {
                "type": "string",
                "enum": ["add", "complete", "remove", "show", "clear", "sort", "list_all"],
                "description": "'add' item, 'complete' (check off), 'remove' (delete), 'show' items, 'clear' all, 'sort' to alphabetize, 'list_all' available lists"
            },
            "item": {
                "type": "string",
                "description": "Item to add/complete/remove"
            },
            "list_name": {
                "type": "string",
                "description": "Optional list name (defaults to shopping list)"
            },
            "status": {
                "type": "string",
                "enum": ["active", "completed"],
                "description": "For 'show' or 'sort': 'active' (default) or 'completed' to view/sort checked-off items"
            }
        },
        ["action"]
    ))

    # ===== REMINDERS (always enabled) =====
    tools.append(_tool(
        "create_reminder",
        "Create reminders. Use for: 'remind me to call mom at 5pm', 'set a reminder for tomorrow'.",
        {
            "reminder": {
                "type": "string",
                "description": "What to remind about"
            },
            "time": {
                "type": "string",
                "description": "When to remind (e.g., 'in 30 minutes', 'at 5pm', 'tomorrow at noon')"
            }
        },
        ["reminder"]
    ))

    tools.append(_tool(
        "get_reminders",
        "Get upcoming reminders. Use for: 'what reminders do I have', 'show my reminders'.",
    ))

    # ===== DEVICE CONTROL (always enabled - LLM fallback) =====
    tools.append(_tool(
        "control_device",
        "Control smart home devices (lights, switches, locks, fans, blinds, shades, covers, media_player, receivers, TVs). Use this when you need to control a device. IMPORTANT: Use the 'device' parameter with the user's spoken name - it does fuzzy matching! For blinds/shades: 'raise/up'=open, 'lower/down'=close, 'stop'=halt. For media players/receivers/TVs: 'mute'=mute audio, 'unmute'=unmute audio, 'play'=resume, 'pause'=pause, 'volume_up'/'volume_down'=adjust volume.",
        {
            "device": {"type": "string", "description": "PREFERRED: Use the device name the user said - fuzzy matching finds the right entity."},
            "entity_id": {"type": "string", "description": "Only if you know the exact entity ID. Prefer 'device' for fuzzy matching."},
            "entity_ids": {"type": "array", "items": {"type": "string"}, "description": "Multiple exact entity IDs"},
            "area": {"type": "string", "description": "Control all devices in area"},
            "domain": {
                "type": "string",
                "enum": ["light", "switch", "lock", "cover", "fan", "media_player", "climate", "vacuum", "scene", "script", "all"],
                "description": "Device type filter for area"
            },
            "action": {
                "type": "string",
                "enum": ["turn_on", "turn_off", "toggle", "lock", "unlock", "open", "close", "stop", "preset", "favorite", "set_position", "play", "pause", "next", "previous", "volume_up", "volume_down", "set_volume", "mute", "unmute", "set_temperature", "start", "dock", "locate", "return_home", "activate"],
                "description": "Action to perform. For BLINDS/SHADES: 'open'=raise, 'close'=lower, 'stop'=halt movement, 'favorite' or 'preset'=go to saved position"
            },
            "brightness": {"type": "integer", "description": "Light brightness 0-100"},
            "color": {"type": "string", "description": "Light color name (red, blue, warm, cool, etc.)"},
            "color_temp": {"type": "integer", "description": "Color temperature in Kelvin (2700=warm, 6500=cool)"},
            "position": {"type": "integer", "description": "Cover position 0-100 (0=closed)"},
            "volume": {"type": "integer", "description": "Volume level 0-100"},
            "temperature": {"type": "number", "description": "Target temperature for climate"},
            "hvac_mode": {"type": "string", "enum": ["heat", "cool", "auto", "off", "fan_only", "dry"], "description": "HVAC mode for climate"},
            "fan_speed": {"type": "string", "enum": ["low", "medium", "high", "auto"], "description": "Fan speed"}
        },
        ["action"]
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
        self.enable_stocks = entity.enable_stocks
        self.enable_news = entity.enable_news
        self.enable_places = entity.enable_places
        self.enable_restaurants = entity.enable_restaurants
        self.enable_thermostat = entity.enable_thermostat
        self.enable_device_status = entity.enable_device_status
        self.enable_wikipedia = entity.enable_wikipedia
        self.enable_music = entity.enable_music

        self.openweathermap_api_key = entity.openweathermap_api_key
        self.google_places_api_key = entity.google_places_api_key
        self.yelp_api_key = entity.yelp_api_key
        self.newsapi_key = entity.newsapi_key

        self.thermostat_entity = entity.thermostat_entity
        self.thermostat_temp_step = entity.thermostat_temp_step
        self.temp_unit = entity.temp_unit

        self.calendar_entities = entity.calendar_entities
        self.room_player_mapping = entity.room_player_mapping
