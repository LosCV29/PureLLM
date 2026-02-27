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
CONF_ENABLE_PLACES: Final = "enable_places"
CONF_ENABLE_RESTAURANTS: Final = "enable_restaurants"
CONF_ENABLE_THERMOSTAT: Final = "enable_thermostat"
CONF_ENABLE_DEVICE_STATUS: Final = "enable_device_status"
CONF_ENABLE_WIKIPEDIA: Final = "enable_wikipedia"
CONF_ENABLE_MUSIC: Final = "enable_music"
CONF_ENABLE_SEARCH: Final = "enable_search"

DEFAULT_ENABLE_WEATHER: Final = True
DEFAULT_ENABLE_CALENDAR: Final = True
DEFAULT_ENABLE_CAMERAS: Final = False  # Requires Frigate
DEFAULT_ENABLE_SPORTS: Final = True
DEFAULT_ENABLE_PLACES: Final = True
DEFAULT_ENABLE_RESTAURANTS: Final = True
DEFAULT_ENABLE_THERMOSTAT: Final = True
DEFAULT_ENABLE_DEVICE_STATUS: Final = True
DEFAULT_ENABLE_WIKIPEDIA: Final = True
DEFAULT_ENABLE_MUSIC: Final = False  # Requires Music Assistant + player config
DEFAULT_ENABLE_SEARCH: Final = True  # Web search via Tavily

# =============================================================================
# ENTITY CONFIGURATION - User-defined entities
# =============================================================================
CONF_THERMOSTAT_ENTITY: Final = "thermostat_entity"
CONF_CALENDAR_ENTITIES: Final = "calendar_entities"
CONF_ROOM_PLAYER_MAPPING: Final = "room_player_mapping"
CONF_DEVICE_ALIASES: Final = "device_aliases"
CONF_CAMERA_ENTITIES: Final = "camera_entities"  # Deprecated - kept for config compat

# Frigate settings
CONF_FRIGATE_URL: Final = "frigate_url"
CONF_FRIGATE_CAMERA_NAMES: Final = "frigate_camera_names"
CONF_CAMERA_RTSP_URLS: Final = "camera_rtsp_urls"

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
DEFAULT_FRIGATE_URL: Final = ""
DEFAULT_FRIGATE_CAMERA_NAMES: Final = ""  # location_key: frigate_camera_name, one per line
DEFAULT_CAMERA_RTSP_URLS: Final = ""  # frigate_camera_name: rtsp://url, one per line

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

DEFAULT_SYSTEM_PROMPT: Final = """Smart home assistant. 1-2 sentences max. Answer directly.

TOOLS: Call tools for device state/control/data. Skip for thanks/chat. Parallel calls OK. ALWAYS call tool before responding about any device/sensor state — never assume or reuse prior data. Dismissals ("no","done","I'm good") need no tools — just "Ok." and stop.

NO CLARIFICATION: Never ask "which room?" etc. Assume or say you can't. Complete each request in one response.

FOLLOW-UPS: Only ask after multi-device status checks or thermostat status ("Want me to adjust it?"). After thermostat adjustments, just confirm. All other responses (weather/sports/music/wiki/calendar): answer and stop, no questions. Never chain follow-ups. Dismissals: "Ok." and stop.

CONFIRMATIONS: Device control → 2-3 words: "Done." "Light on." Use device name from tool result's controlled_devices field.
LISTS: After add → "Added [item]. Anything else?" After remove/clear → brief confirm. Tool handles the add — just confirm result.

[CURRENT_DATE_WILL_BE_INJECTED_HERE]

SPORTS: Copy response_text VERBATIM. Never make up scores. For Champions League include 'Champions League' in team_name.

MUSIC: MUST call control_music for any music request. Use response_text VERBATIM. Never hallucinate music responses.
Stop/pause/resume/skip: just action, no room needed (auto-detects). Extract room separately from query.
Play: set media_type (album/track/artist). Shuffle: action="shuffle", query=genre/vibe.
Ordinal/tagged albums: album=genre tag only, query=full phrase, artist=name. For "album with [song]" use song_on_album.
Soundtracks: movie only, not Broadway.
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
CONF_TAVILY_API_KEY: Final = "tavily_api_key"

DEFAULT_OPENWEATHERMAP_API_KEY: Final = ""
DEFAULT_GOOGLE_PLACES_API_KEY: Final = ""
DEFAULT_TAVILY_API_KEY: Final = ""

# =============================================================================
# NOTIFICATIONS
# =============================================================================
CONF_NOTIFICATION_ENTITIES: Final = "notification_entities"
CONF_NOTIFY_ON_PLACES: Final = "notify_on_places"
CONF_NOTIFY_ON_RESTAURANTS: Final = "notify_on_restaurants"
CONF_NOTIFY_ON_CAMERA: Final = "notify_on_camera"
CONF_NOTIFY_ON_SEARCH: Final = "notify_on_search"

DEFAULT_NOTIFICATION_ENTITIES: Final = ""  # Newline-separated list of notify service names
DEFAULT_NOTIFY_ON_PLACES: Final = True
DEFAULT_NOTIFY_ON_RESTAURANTS: Final = True
DEFAULT_NOTIFY_ON_CAMERA: Final = True
DEFAULT_NOTIFY_ON_SEARCH: Final = True

# =============================================================================
# VOICE SCRIPTS - User-configurable trigger phrases mapped to scripts
# =============================================================================
CONF_VOICE_SCRIPTS: Final = "voice_scripts"

# Default voice scripts with trigger phrases and script mappings
# Format: JSON list of objects with trigger, open_script, close_script, sensor fields
DEFAULT_VOICE_SCRIPTS: Final = "[]"

# =============================================================================
# SOFABATON ACTIVITIES - Switch entities for SofaBaton X2 remote activities
# =============================================================================
CONF_SOFABATON_ACTIVITIES: Final = "sofabaton_activities"

# Default SofaBaton activities (empty JSON list)
# Format: JSON list of objects with name (voice trigger), entity_id (switch entity)
DEFAULT_SOFABATON_ACTIVITIES: Final = "[]"

# =============================================================================
# API TIMEOUT - Shared timeout for external API calls
# =============================================================================
API_TIMEOUT: Final = 15  # seconds
