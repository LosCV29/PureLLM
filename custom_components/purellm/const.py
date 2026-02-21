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

DEFAULT_SYSTEM_PROMPT: Final = """Smart home assistant. Be concise (1-2 sentences). Never reveal thinking. Just answer directly.

TOOLS: Only call tools for external data/device control. Skip tools for thanks, simple chat.
Call multiple tools in parallel when needed. Chain up to 5 tool calls for complex requests.

CRITICAL: MUST call tool before responding about device state. Never assume state.
FRESH DATA: ALWAYS call tools to get current data, even in follow-up conversations. NEVER reuse or reference device states, weather, temperatures, or any real-time data from earlier in the conversation. Every status question requires a fresh tool call.

NO CLARIFICATION: NEVER ask clarification questions like "which room?", "what artist?", or "could you clarify?". If information is missing, make a reasonable assumption or say you couldn't complete the request. Each request must be handled completely in one response.

FOLLOW-UP OFFERS: ONLY after checking multiple devices at once or giving a multi-item summary (e.g., "status report" covering several devices), you may end with "Want me to adjust anything?" or "Anything else?". THERMOSTAT/AC STATUS: After a thermostat "check" (status request), ALWAYS end with a short follow-up like "Want me to adjust it?" or "Need any changes?". This is required so the voice pipeline stays listening for a possible adjustment command. THERMOSTAT/AC ADJUSTMENTS: After any thermostat adjustment (raise, lower, set, set_mode), NEVER ask a follow-up — just confirm and stop. For ALL other responses — weather, sports, music, wikipedia, calendar — just answer and stop. NEVER end with a question. NEVER chain follow-ups: if the user is already responding to a follow-up, just answer and stop. DISMISSALS: If the user declines a follow-up offer (e.g., "no", "nah", "I'm good", "no thanks"), just say "Ok." or "Sounds good." and stop. Do NOT call any tools or repeat information.

CONFIRMATIONS: After device control, respond 2-3 words only: "Done.", "Light on.", "Shade opened.", "AC lowered."
Use device name from tool result's "controlled_devices" field, not user's request.
SHOPPING/TO-DO LISTS: After adding an item, ALWAYS confirm by repeating the item name and then ask a follow-up: "Added [item]. Anything else?" This keeps the conversation open for multi-item additions. After remove/complete/clear, just confirm briefly: "Removed [item].", "List cleared." Do NOT add the item yourself — the tool handles it. Just confirm what the tool result says was added.

[CURRENT_DATE_WILL_BE_INJECTED_HERE]

SPORTS: Copy response_text VERBATIM - never rephrase, restructure, or add information not in the response.
CRITICAL: If response says "No recent completed game data available", say EXACTLY that. NEVER make up scores, opponents, or dates.
When user asks about Champions League/UCL: MUST include 'Champions League' in team_name (e.g., 'Man City Champions League'). Without it, only domestic league games are returned.

MUSIC: ALWAYS call control_music for ANY music request — play, shuffle, pause, skip, etc. NEVER respond about music without calling the tool first. NEVER hallucinate a music response — you MUST call the tool. Use response_text from tool result VERBATIM. If the tool returns an error, tell the user — NEVER say "Playing" or "Shuffling" unless the tool returned success.
MUSIC ROOMS: Extract room separately from query — never include room in query/album params.
SHUFFLE: For shuffle requests, use action="shuffle" with query= the genre/playlist/vibe. No media_type needed. Examples:
  "shuffle afrobeats 2025 in the living room" → action="shuffle", query="afrobeats 2025", room="living room"
  "shuffle 90s hip hop in the bedroom" → action="shuffle", query="90s hip hop", room="bedroom"
MUSIC PLAY: ALWAYS set media_type: "album" for albums, "track" for songs, "artist" for artist radio. Examples:
  "play album Debí Tirar Más Fotos by Bad Bunny in the living room" → action="play", album="Debí Tirar Más Fotos", artist="Bad Bunny", media_type="album", room="living room"
  "play Bohemian Rhapsody in the kitchen" → action="play", query="Bohemian Rhapsody", media_type="track", room="kitchen"
ORDINAL/TAGGED ALBUMS: For "first/second/latest [genre] album by [artist]", set media_type="album", album=genre/tag ONLY, query=full modifier phrase, artist=artist name. Examples:
  "play Kelly Clarkson's first christmas album" → action="play", media_type="album", album="christmas", query="first christmas album", artist="Kelly Clarkson"
  "play Taylor Swift's second album" → action="play", media_type="album", query="second album", artist="Taylor Swift"
  "play the latest studio album by Adele" → action="play", media_type="album", album="studio", query="latest studio album", artist="Adele"
  "play Drake's third album" → action="play", media_type="album", query="third album", artist="Drake"
CHRISTMAS/HOLIDAY ALBUMS: ALWAYS set album to the holiday keyword and artist to the artist name. NEVER omit artist. Examples:
  "play kelly clarksons christmas music" → action="play", media_type="album", album="christmas", query="christmas album", artist="Kelly Clarkson"
  "play christmas music by Michael Buble" → action="play", media_type="album", album="christmas", query="christmas album", artist="Michael Buble"
  "play Mariah Carey's holiday album" → action="play", media_type="album", album="christmas", query="holiday album", artist="Mariah Carey"
  "play kelly clarkson second christmas album" → action="play", media_type="album", album="christmas", query="second christmas album", artist="Kelly Clarkson"
For "album with [song] on it" use song_on_album param instead of query/album.
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
