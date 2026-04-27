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
PROVIDER_ANTHROPIC: Final = "anthropic"

ALL_PROVIDERS: Final = [
    PROVIDER_LM_STUDIO,
    PROVIDER_ANTHROPIC,
]

PROVIDER_NAMES: Final = {
    PROVIDER_LM_STUDIO: "LM Studio / vLLM (Local)",
    PROVIDER_ANTHROPIC: "Anthropic Claude",
}

# Default base URLs per provider
PROVIDER_BASE_URLS: Final = {
    PROVIDER_LM_STUDIO: "http://localhost:1234/v1",
    PROVIDER_ANTHROPIC: "https://api.anthropic.com/v1",
}

# Default models per provider
PROVIDER_DEFAULT_MODELS: Final = {
    PROVIDER_LM_STUDIO: "local-model",
    PROVIDER_ANTHROPIC: "claude-haiku-4-5",
}

# Suggested models per provider (for UI hints)
PROVIDER_MODELS: Final = {
    PROVIDER_LM_STUDIO: ["local-model", "qwen2.5-7b-instruct", "llama-3.2-3b"],
    PROVIDER_ANTHROPIC: ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"],
}

# Anthropic API version header (sent with every request)
ANTHROPIC_API_VERSION: Final = "2023-06-01"

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
CONF_ENABLE_THERMOSTAT: Final = "enable_thermostat"
CONF_ENABLE_DEVICE_STATUS: Final = "enable_device_status"
CONF_ENABLE_WIKIPEDIA: Final = "enable_wikipedia"
CONF_ENABLE_MUSIC: Final = "enable_music"
CONF_ENABLE_SEARCH: Final = "enable_search"
CONF_ENABLE_PLANTS: Final = "enable_plants"

DEFAULT_ENABLE_WEATHER: Final = True
DEFAULT_ENABLE_CALENDAR: Final = True
DEFAULT_ENABLE_CAMERAS: Final = False  # Requires Frigate
DEFAULT_ENABLE_SPORTS: Final = True
DEFAULT_ENABLE_PLACES: Final = True
DEFAULT_ENABLE_THERMOSTAT: Final = True
DEFAULT_ENABLE_DEVICE_STATUS: Final = True
DEFAULT_ENABLE_WIKIPEDIA: Final = True
DEFAULT_ENABLE_MUSIC: Final = False  # Requires Music Assistant + player config
DEFAULT_ENABLE_SEARCH: Final = True  # Web search via Tavily
DEFAULT_ENABLE_PLANTS: Final = True  # Auto-discovers plant.* entities; no-op if none present

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

DEFAULT_SYSTEM_PROMPT: Final = """Smart home assistant. 1-2 sentences. Answer directly.

RULES:
- Always call a tool before reporting device/sensor state or controlling anything. Never assume success.
- Dismissals ("no", "done", "I'm good"): reply "Ok." and stop. No tool, no follow-up.
- Never ask "which room?". If ROOM CONTEXT is set, that's "here". Otherwise assume.
- Confirmations: 2-3 words. Use the name from the tool result.
- Follow-ups only after multi-device status or thermostat status. Never chain them.

SPORTS: Use response_text from the tool. Keep venue/home-away/TV channel. Never invent scores. Include "Champions League" in team_name for those games.

MUSIC: Always call control_music. Use response_text verbatim. Volume: "raise/lower the music" → control_music; "raise/lower your volume" → control_device(device="speaker"). Extract room from "in the X" — never put it in query/album/artist. media_type required for play: "album" / "track" / "artist". Shuffle uses query, no media_type.

PLANTS: Plant questions go to check_plant_status (NOT check_device_status). Strip "the plant"/"my" from the name. water/dry/thirsty/wet → metric="moisture". "any plants need water/in trouble" → problems_only=true. Repeat response_text verbatim.

[CURRENT_DATE_WILL_BE_INJECTED_HERE]
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
CONF_NOTIFY_ON_CAMERA: Final = "notify_on_camera"
CONF_NOTIFY_ON_SEARCH: Final = "notify_on_search"

DEFAULT_NOTIFICATION_ENTITIES: Final = ""  # Newline-separated list of notify service names
DEFAULT_NOTIFY_ON_PLACES: Final = True
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
# ELEVENLABS TTS SETTINGS
# =============================================================================
CONF_ELEVENLABS_API_KEY: Final = "elevenlabs_api_key"
CONF_ELEVENLABS_VOICE_ID: Final = "elevenlabs_voice_id"
CONF_ELEVENLABS_MODEL: Final = "elevenlabs_model"
CONF_ELEVENLABS_STABILITY: Final = "elevenlabs_stability"
CONF_ELEVENLABS_SIMILARITY: Final = "elevenlabs_similarity"
CONF_ELEVENLABS_STYLE: Final = "elevenlabs_style"
CONF_ELEVENLABS_SPEAKER_BOOST: Final = "elevenlabs_speaker_boost"
CONF_ELEVENLABS_SPEED: Final = "elevenlabs_speed"
CONF_ELEVENLABS_OUTPUT_FORMAT: Final = "elevenlabs_output_format"
CONF_ELEVENLABS_TEXT_NORMALIZATION: Final = "elevenlabs_text_normalization"

DEFAULT_ELEVENLABS_API_KEY: Final = ""
DEFAULT_ELEVENLABS_VOICE_ID: Final = ""
DEFAULT_ELEVENLABS_MODEL: Final = "eleven_turbo_v2_5"
DEFAULT_ELEVENLABS_STABILITY: Final = 0.5
DEFAULT_ELEVENLABS_SIMILARITY: Final = 0.75
DEFAULT_ELEVENLABS_STYLE: Final = 0.0
DEFAULT_ELEVENLABS_SPEAKER_BOOST: Final = True
DEFAULT_ELEVENLABS_SPEED: Final = 1.0
DEFAULT_ELEVENLABS_OUTPUT_FORMAT: Final = "mp3_44100_128"
DEFAULT_ELEVENLABS_TEXT_NORMALIZATION: Final = "auto"

ELEVENLABS_TEXT_NORMALIZATION_MODES: Final = [
    "auto",
    "on",
    "off",
]

ELEVENLABS_MODELS: Final = [
    "eleven_turbo_v2_5",
    "eleven_multilingual_v2",
    "eleven_turbo_v2",
    "eleven_monolingual_v1",
    "eleven_flash_v2_5",
    "eleven_flash_v2",
]

ELEVENLABS_OUTPUT_FORMATS: Final = [
    "mp3_44100_128",
    "mp3_44100_64",
    "mp3_44100_32",
    "mp3_22050_32",
    "pcm_16000",
    "pcm_22050",
    "pcm_24000",
    "pcm_44100",
    "ulaw_8000",
]

# =============================================================================
# API TIMEOUT - Shared timeout for external API calls
# =============================================================================
API_TIMEOUT: Final = 15  # seconds
