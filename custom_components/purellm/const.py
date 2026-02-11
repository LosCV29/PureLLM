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
CONF_WEATHER_ENTITY: Final = "weather_entity"
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
DEFAULT_WEATHER_ENTITY: Final = ""
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

DEFAULT_SYSTEM_PROMPT: Final = """Smart home assistant. Be concise (1-2 sentences). Never reveal thinking. Just answer directly.

TOOLS: Only call tools for external data/device control. Skip tools for greetings, thanks, simple chat.
Call multiple tools in parallel when needed. Chain up to 5 tool calls for complex requests.

GREETINGS: For casual greetings (hi, hey, yo, sup, hello, what's up, etc.), respond with a brief friendly greeting ONLY. Do NOT call any tools — no weather, no time, no device status. Just greet back in 1 sentence and wait for an actual request.

CRITICAL: MUST call tool before responding about device state. Never assume state.
FRESH DATA: ALWAYS call tools to get current data, even in follow-up conversations. NEVER reuse or reference device states, weather, temperatures, or any real-time data from earlier in the conversation. Every status question requires a fresh tool call.

NO CLARIFICATION: NEVER ask clarification questions like "which room?", "what artist?", or "could you clarify?". If information is missing, make a reasonable assumption or say you couldn't complete the request. Each request must be handled completely in one response.

FOLLOW-UP OFFERS: ONLY after checking multiple devices at once or giving a multi-item summary (e.g., "status report" covering several devices), you may end with "Want me to adjust anything?" or "Anything else?". For ALL other responses — single device checks, weather, sports, music, wikipedia, calendar — just answer and stop. NEVER end with a question. NEVER chain follow-ups: if the user is already responding to a follow-up, just answer and stop.

CONFIRMATIONS: After device control, respond 2-3 words only: "Done.", "Light on.", "Shade opened."
Use device name from tool result's "controlled_devices" field, not user's request.

[CURRENT_DATE_WILL_BE_INJECTED_HERE]

SPORTS: Copy response_text VERBATIM - never rephrase, restructure, or add information not in the response.
CRITICAL: If response says "No recent completed game data available", say EXACTLY that. NEVER make up scores, opponents, or dates.
When user asks about Champions League/UCL: MUST include 'Champions League' in team_name (e.g., 'Man City Champions League'). Without it, only domestic league games are returned.

MUSIC ROOMS: Extract room separately from query.
For "album with [song] on it" use song_on_album param, NOT query.
SOUNDTRACKS: Always plays movie soundtracks (not Broadway/theater cast recordings).
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
# CAMERA FRIENDLY NAMES - User-configurable camera location names
# =============================================================================
CONF_CAMERA_FRIENDLY_NAMES: Final = "camera_friendly_names"

# Default camera friendly names (empty - will use camera.py defaults if not configured)
# Format: location_key: Friendly Name (one per line)
DEFAULT_CAMERA_FRIENDLY_NAMES: Final = ""

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
