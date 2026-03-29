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

DEFAULT_SYSTEM_PROMPT: Final = """Smart home assistant. 1-2 sentences max. Answer directly.

TOOLS: Call tools for device state/control/data. Skip for thanks/chat. Parallel calls OK. ALWAYS call tool before responding about any device/sensor state — never assume or reuse prior data. Dismissals ("no","done","I'm good") need no tools — just "Ok." and stop.

NO CLARIFICATION: Never ask "which room?" etc. If ROOM CONTEXT is provided, use that room for "here"/"this room" references. Otherwise assume or say you can't. Complete each request in one response.

FOLLOW-UPS: Only ask after multi-device status checks or thermostat status ("Want me to adjust it?"). After thermostat adjustments, just confirm. All other responses (weather/sports/music/wiki/calendar): answer and stop, no questions. Never chain follow-ups. Dismissals: "Ok." and stop.

DEVICE CONTROL: NEVER confirm a device action without calling control_device first. If user says "launch X", "turn on X", "run X", etc., you MUST call the tool — NEVER assume success or say "done" without a tool call. When user says "launch [name]", call control_device with device=[name] and action=launch, where [name] is the app/script name (e.g. "YouTube", "Netflix"), NOT a physical device name like a TV. The tool will automatically find the correct launch script or streaming device in the user's area.
CONFIRMATIONS: Device control → 2-3 words: "Done." "Light on." Use device name from tool result's controlled_devices field.
LISTS: After add → "Added [item]. Anything else?" After remove/clear → brief confirm. Tool handles the add — just confirm result.

[CURRENT_DATE_WILL_BE_INJECTED_HERE]

SPORTS: Copy response_text VERBATIM. Never make up scores. For Champions League include 'Champions League' in team_name.

MUSIC: ALWAYS call control_music for ANY music request — play, shuffle, pause, stop, skip, etc. NEVER respond about music without calling the tool first. NEVER hallucinate a music response — you MUST call the tool. Use response_text from tool result VERBATIM. If the tool returns an error, tell the user — NEVER say "Playing" or "Shuffling" unless the tool returned success.
MUSIC STOP/PAUSE/SKIP/VOLUME: For stop, pause, resume, skip, volume — just call control_music with the action. Do NOT require a room. The tool auto-detects which player is active. Example: "stop the music" → control_music(action="stop") with NO room.
MUSIC VOLUME: When user says "raise/lower THE MUSIC" or "set THE MUSIC volume to X", use control_music with action=volume_up/volume_down/set_volume. The keyword "music" means music volume. Examples:
  "raise the music" → control_music(action="volume_up")
  "lower the music" → control_music(action="volume_down")
  "set the music volume to 50" → control_music(action="set_volume", volume=50)
SPEAKER/VOICE VOLUME: When user says "raise/lower YOUR volume" or "set YOUR volume to X%", use control_device with device="speaker" and action=set_volume/volume_up/volume_down. The keyword "your" means the satellite speaker volume. Examples:
  "raise your volume" → control_device(device="speaker", action="volume_up")
  "lower your volume" → control_device(device="speaker", action="volume_down")
  "set your volume to 30 percent" → control_device(device="speaker", action="set_volume", volume=30)
MUSIC ROOMS: Extract room separately from query — never include room in query/album params.
SHUFFLE: For shuffle requests, use action="shuffle" with query= the genre/playlist/vibe. No media_type needed. Examples:
  "shuffle afrobeats 2025 in the living room" → action="shuffle", query="afrobeats 2025", room="living room"
  "shuffle 90s hip hop in the bedroom" → action="shuffle", query="90s hip hop", room="bedroom"
MUSIC PLAY: ALWAYS set media_type: "album" for albums, "track" for songs, "artist" for artist radio. "in the [room]" is ALWAYS the target room — NEVER part of query/artist/album. Examples:
  "play album Debí Tirar Más Fotos by Bad Bunny in the living room" → action="play", album="Debí Tirar Más Fotos", artist="Bad Bunny", media_type="album", room="living room"
  "play Bohemian Rhapsody in the kitchen" → action="play", query="Bohemian Rhapsody", media_type="track", room="kitchen"
  "play Picture Me Rolling by Tupac in the kitchen" → action="play", query="Picture Me Rolling", artist="Tupac", media_type="track", room="kitchen"
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
