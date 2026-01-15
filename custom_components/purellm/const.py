"""Constants for PureLLM - Pure LLM Voice Assistant."""
import json
from pathlib import Path
from typing import Final

DOMAIN: Final = "purellm"

# Cache version at module load time to avoid blocking calls in async context
def _load_version() -> str:
    """Load version from manifest.json once at startup."""
    try:
        manifest_path = Path(__file__).parent / "manifest.json"
        with open(manifest_path) as f:
            return json.load(f).get("version", "unknown")
    except Exception:
        return "unknown"

VERSION: Final = _load_version()


def get_version() -> str:
    """Get cached version."""
    return VERSION

# =============================================================================
# LLM PROVIDER SETTINGS
# =============================================================================
CONF_PROVIDER: Final = "provider"
CONF_BASE_URL: Final = "base_url"
CONF_API_KEY: Final = "api_key"
CONF_MODEL: Final = "model"
CONF_TEMPERATURE: Final = "temperature"
CONF_MAX_TOKENS: Final = "max_tokens"
CONF_TOP_P: Final = "top_p"

# Provider choices
PROVIDER_LM_STUDIO: Final = "lm_studio"
PROVIDER_GOOGLE: Final = "google"

ALL_PROVIDERS: Final = [
    PROVIDER_LM_STUDIO,
    PROVIDER_GOOGLE,
]

PROVIDER_NAMES: Final = {
    PROVIDER_LM_STUDIO: "LM Studio / vLLM (Local)",
    PROVIDER_GOOGLE: "Google Gemini",
}

# Default base URLs per provider
PROVIDER_BASE_URLS: Final = {
    PROVIDER_LM_STUDIO: "http://localhost:1234/v1",
    PROVIDER_GOOGLE: "https://generativelanguage.googleapis.com/v1beta",
}

# Default models per provider
PROVIDER_DEFAULT_MODELS: Final = {
    PROVIDER_LM_STUDIO: "local-model",
    PROVIDER_GOOGLE: "gemini-2.0-flash",
}

# Suggested models per provider (for UI hints)
PROVIDER_MODELS: Final = {
    PROVIDER_LM_STUDIO: ["local-model", "qwen2.5-7b-instruct", "llama-3.2-3b"],
    PROVIDER_GOOGLE: ["gemini-2.0-flash", "gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
}

DEFAULT_PROVIDER: Final = PROVIDER_LM_STUDIO
DEFAULT_BASE_URL: Final = "http://localhost:1234/v1"
DEFAULT_API_KEY: Final = "lm-studio"
DEFAULT_MODEL: Final = "local-model"
DEFAULT_TEMPERATURE: Final = 0.7
DEFAULT_MAX_TOKENS: Final = 2000
DEFAULT_TOP_P: Final = 0.95

# =============================================================================
# FEATURE TOGGLES - Enable/disable function categories
# =============================================================================
CONF_ENABLE_WEATHER: Final = "enable_weather"
CONF_ENABLE_CALENDAR: Final = "enable_calendar"
CONF_ENABLE_CAMERAS: Final = "enable_cameras"
CONF_ENABLE_SPORTS: Final = "enable_sports"
CONF_ENABLE_STOCKS: Final = "enable_stocks"
CONF_ENABLE_NEWS: Final = "enable_news"
CONF_ENABLE_PLACES: Final = "enable_places"
CONF_ENABLE_RESTAURANTS: Final = "enable_restaurants"
CONF_ENABLE_THERMOSTAT: Final = "enable_thermostat"
CONF_ENABLE_DEVICE_STATUS: Final = "enable_device_status"
CONF_ENABLE_WIKIPEDIA: Final = "enable_wikipedia"
CONF_ENABLE_MUSIC: Final = "enable_music"

DEFAULT_ENABLE_WEATHER: Final = True
DEFAULT_ENABLE_CALENDAR: Final = True
DEFAULT_ENABLE_CAMERAS: Final = False  # Requires vllm_video integration
DEFAULT_ENABLE_SPORTS: Final = True
DEFAULT_ENABLE_STOCKS: Final = True
DEFAULT_ENABLE_NEWS: Final = True
DEFAULT_ENABLE_PLACES: Final = True
DEFAULT_ENABLE_RESTAURANTS: Final = True
DEFAULT_ENABLE_THERMOSTAT: Final = True
DEFAULT_ENABLE_DEVICE_STATUS: Final = True
DEFAULT_ENABLE_WIKIPEDIA: Final = True
DEFAULT_ENABLE_MUSIC: Final = False  # Requires Music Assistant + player config

# =============================================================================
# ENTITY CONFIGURATION - User-defined entities
# =============================================================================
CONF_THERMOSTAT_ENTITY: Final = "thermostat_entity"
CONF_CALENDAR_ENTITIES: Final = "calendar_entities"
CONF_ROOM_PLAYER_MAPPING: Final = "room_player_mapping"
CONF_DEVICE_ALIASES: Final = "device_aliases"
CONF_CAMERA_ENTITIES: Final = "camera_entities"

# Thermostat settings - user-configurable temperature range and step
CONF_THERMOSTAT_MIN_TEMP: Final = "thermostat_min_temp"
CONF_THERMOSTAT_MAX_TEMP: Final = "thermostat_max_temp"
CONF_THERMOSTAT_TEMP_STEP: Final = "thermostat_temp_step"
CONF_THERMOSTAT_USE_CELSIUS: Final = "thermostat_use_celsius"

DEFAULT_THERMOSTAT_ENTITY: Final = ""
DEFAULT_CALENDAR_ENTITIES: Final = ""
DEFAULT_ROOM_PLAYER_MAPPING: Final = ""  # room:entity_id, one per line
DEFAULT_DEVICE_ALIASES: Final = ""
DEFAULT_CAMERA_ENTITIES: Final = ""

# Thermostat defaults (Fahrenheit by default)
DEFAULT_THERMOSTAT_MIN_TEMP: Final = 60
DEFAULT_THERMOSTAT_MAX_TEMP: Final = 85
DEFAULT_THERMOSTAT_TEMP_STEP: Final = 2
DEFAULT_THERMOSTAT_USE_CELSIUS: Final = False

# Thermostat defaults for Celsius mode
DEFAULT_THERMOSTAT_MIN_TEMP_CELSIUS: Final = 15
DEFAULT_THERMOSTAT_MAX_TEMP_CELSIUS: Final = 30
DEFAULT_THERMOSTAT_TEMP_STEP_CELSIUS: Final = 1

# =============================================================================
# SYSTEM PROMPT
# =============================================================================
CONF_SYSTEM_PROMPT: Final = "system_prompt"

DEFAULT_SYSTEM_PROMPT: Final = """You are a smart home assistant. Be concise (1-2 sentences for voice responses).
NEVER reveal your internal thinking or reasoning. Do NOT say things like "I need to check", "Let me look this up", "I'll check the latest score", or similar phrases. Just give the answer directly.

STATELESS: This is a stateless voice assistant - there is NO conversation memory. NEVER ask follow-up questions like "which room?", "what artist?", or "could you clarify?". If information is missing, make a reasonable assumption or say you couldn't complete the request. Each request must be handled completely in one response.

CRITICAL: You MUST call a tool function before responding about ANY device. NEVER say a device "is already" at a position or state without calling a tool first. If you respond about device state without calling a tool, you are LYING.

DEVICE CONFIRMATIONS: After executing a device control command, respond with ONLY 2-3 words. Examples: "Done.", "Light on.", "Shade opened.", "Track skipped.", "Volume set.", NEVER add room names, locations, or extra details unless the user specifically asked about a room. NEVER hallucinate or guess which room a device is in.

[CURRENT_DATE_WILL_BE_INJECTED_HERE]

GENERAL GUIDELINES:
- For weather questions, call get_weather_forecast
- For camera checks: use check_camera for detailed view, quick_camera_check for fast "is anyone there" queries
- For thermostat control, use control_thermostat
- For device status, use check_device_status
- For BLINDS/SHADES/COVERS: ALWAYS call control_device with device name and action. Actions: open, close, favorite, preset, set_position. DO NOT assume state - EXECUTE the command by calling control_device.
- For sports questions, ALWAYS call get_sports_info (never answer from memory). CRITICAL: Your response MUST be the response_text field VERBATIM - copy it exactly, do NOT rephrase, do NOT change "yesterday" to a date, do NOT restructure the sentence
- For Wikipedia/knowledge questions, use get_wikipedia_summary
- For age questions, use calculate_age (never guess ages)
- For places/directions, use find_nearby_places
- For restaurant recommendations, use get_restaurant_recommendations
- For news, use get_news
- For calendar events, use get_calendar_events
- For music control (play, skip, pause, etc.), use control_music
  TWO MUSIC PATTERNS - MEMORIZE THESE:
  1. PLAY = "play [SONG] by [ARTIST] in the [ROOM]"
     → action="play", query="[SONG]", artist="[ARTIST]", room="[ROOM]", media_type="track"
     Example: "play Humble by Kendrick Lamar in the living room"
     → query="Humble", artist="Kendrick Lamar", room="living room", media_type="track"
  2. SHUFFLE = "shuffle [ARTIST/GENRE] in the [ROOM]"
     → action="shuffle", query="[ARTIST/GENRE]", room="[ROOM]" (NO artist param!)
     Example: "shuffle Young Dolph in the kitchen"
     → query="Young Dolph", room="kitchen"
  CRITICAL: "in the living room/kitchen/bedroom/office" is ALWAYS the room - NEVER part of query!
  WRONG: query="Young Dolph in the living room" ← NEVER do this!
- For ALL device control (lights, locks, switches, fans, etc.), use control_device - ALL commands go through the LLM pipeline
"""

# =============================================================================
# LOCATION
# =============================================================================
CONF_CUSTOM_LATITUDE: Final = "custom_latitude"
CONF_CUSTOM_LONGITUDE: Final = "custom_longitude"

# Use 0.0 as default to indicate "use Home Assistant's configured location"
DEFAULT_CUSTOM_LATITUDE: Final = 0.0
DEFAULT_CUSTOM_LONGITUDE: Final = 0.0

# =============================================================================
# API KEYS
# =============================================================================
CONF_OPENWEATHERMAP_API_KEY: Final = "openweathermap_api_key"
CONF_GOOGLE_PLACES_API_KEY: Final = "google_places_api_key"
CONF_YELP_API_KEY: Final = "yelp_api_key"
CONF_NEWSAPI_KEY: Final = "newsapi_key"

DEFAULT_OPENWEATHERMAP_API_KEY: Final = ""
DEFAULT_GOOGLE_PLACES_API_KEY: Final = ""
DEFAULT_YELP_API_KEY: Final = ""
DEFAULT_NEWSAPI_KEY: Final = ""

# =============================================================================
# NOTIFICATIONS
# =============================================================================
CONF_NOTIFICATION_ENTITIES: Final = "notification_entities"
CONF_NOTIFY_ON_PLACES: Final = "notify_on_places"
CONF_NOTIFY_ON_RESTAURANTS: Final = "notify_on_restaurants"
CONF_NOTIFY_ON_CAMERA: Final = "notify_on_camera"

DEFAULT_NOTIFICATION_ENTITIES: Final = ""  # Newline-separated list of notify service names
DEFAULT_NOTIFY_ON_PLACES: Final = True
DEFAULT_NOTIFY_ON_RESTAURANTS: Final = True
DEFAULT_NOTIFY_ON_CAMERA: Final = True

# =============================================================================
# VOICE SCRIPTS - User-configurable trigger phrases mapped to scripts
# =============================================================================
CONF_VOICE_SCRIPTS: Final = "voice_scripts"

# Default voice scripts with trigger phrases and script mappings
# Format: JSON list of objects with trigger, open_script, close_script, sensor fields
DEFAULT_VOICE_SCRIPTS: Final = "[]"

# =============================================================================
# CAMERA FRIENDLY NAMES - User-configurable camera location names
# =============================================================================
CONF_CAMERA_FRIENDLY_NAMES: Final = "camera_friendly_names"

# Default camera friendly names (empty - will use camera.py defaults if not configured)
# Format: location_key: Friendly Name (one per line)
DEFAULT_CAMERA_FRIENDLY_NAMES: Final = ""

# =============================================================================
# SOFABATON ACTIVITIES - API keys for SofaBaton X2 remote activities
# =============================================================================
CONF_SOFABATON_ACTIVITIES: Final = "sofabaton_activities"

# Default SofaBaton activities (empty JSON list)
# Format: JSON list of objects with name, start_key, stop_key fields
DEFAULT_SOFABATON_ACTIVITIES: Final = "[]"

# =============================================================================
# API TIMEOUT - Shared timeout for external API calls
# =============================================================================
API_TIMEOUT: Final = 15  # seconds
