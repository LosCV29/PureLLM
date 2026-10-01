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
CONF_CAMERA_ENTITIES: Final = "camera_entities"  # Deprecated - kept for config compat

# Frigate settings
CONF_FRIGATE_URL: Final = "frigate_url"
# Optional separate vision LLM for check_camera (blank = use the main LLM).
# Lets a text-only voice brain hand camera frames to a multimodal model.
CONF_VISION_BASE_URL: Final = "vision_base_url"
CONF_VISION_MODEL: Final = "vision_model"

# Thermostat settings - user-configurable temperature range and step
CONF_THERMOSTAT_MIN_TEMP: Final = "thermostat_min_temp"
CONF_THERMOSTAT_MAX_TEMP: Final = "thermostat_max_temp"
CONF_THERMOSTAT_TEMP_STEP: Final = "thermostat_temp_step"
CONF_THERMOSTAT_USE_CELSIUS: Final = "thermostat_use_celsius"

DEFAULT_THERMOSTAT_ENTITY: Final = ""
DEFAULT_CALENDAR_ENTITIES: Final = ""
DEFAULT_ROOM_PLAYER_MAPPING: Final = ""  # room:entity_id, one per line
DEFAULT_CAMERA_ENTITIES: Final = ""
DEFAULT_FRIGATE_URL: Final = ""
DEFAULT_VISION_BASE_URL: Final = ""
DEFAULT_VISION_MODEL: Final = ""

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

SPORTS: Use response_text from the tool. Keep venue/home-away/TV channel/betting odds. Never invent scores or odds. Include "Champions League" in team_name for those games.

MUSIC: Always call control_music. Use response_text verbatim. Volume: "raise/lower the music" → control_music. Speaker/voice volume ("set/raise/lower your volume", "set the speaker volume to N", "your volume up/down", "louder", "quieter") ALWAYS → set_speaker_volume(action=set|up|down, level=N). Never route speaker volume to control_device, a TV, or a PS5. Extract room ONLY from "in the X" / "on the X" — otherwise OMIT room (defaults to the speaker that heard the request). Never put room in query/album/artist. media_type required for play: "album" / "track" / "artist". Shuffle uses query, no media_type. When a play/shuffle request is vague, misheard, or an artist name that could also be a song title, call search_music FIRST, then control_music with the chosen candidate's media_uri. Transport actions (pause/resume/skip/stop/volume) never need search_music. EXCEPTION: pause/resume/stop naming a specific DEVICE ("pause the Shield", "pause the TV", "resume the Apple TV") means pause whatever VIDEO that device is playing → control_tv(action=pause/resume/stop, device=<name>), NOT control_music. Bare "pause"/"pause the music" → control_music.

PLANTS: Plant questions go to check_plant_status (NOT check_device_status). Strip "the plant"/"my" from the name. water/dry/thirsty/wet → metric="moisture". "any plants need water/in trouble" → problems_only=true. Repeat response_text verbatim.

Today's date: {current_date}
"""

# Used INSTEAD of the smart-home prompt when the intent router classifies
# nothing — i.e. the user is talking, not commanding (v7.66.0, 2026-07-27).
#
# Why this exists: the smart-home prompt bled into chitchat. "Talk dirty to me."
# came back as "Since there's no action to take, I'll leave the dirty talk to
# your imagination!" — the model narrating its own tool-lessness, because the
# prompt told it that it was a device controller and to answer in 1-2 sentences.
#
# The routine-domain guard (v7.65.0) already makes these turns unable to fire
# automations/scripts/scenes, so a permissive persona here cannot move the house.
# Verbatim phone relay between the two paired adults in this household
# (v7.67.0, 2026-07-27). Keys are lowercase spoken forms; values are notify
# services. Scope is deliberately CLOSED — an unlisted name is refused, never
# fuzzy-matched, so a misheard word cannot text private content to a contact
# who never opted into this channel. Add a name here to add a participant.
PARTNER_MESSAGE_CONTACTS: Final = {
    "elise": "notify.mobile_app_elise_munoz",
    "wife": "notify.mobile_app_elise_munoz",
    "my wife": "notify.mobile_app_elise_munoz",
    "carlos": "notify.mobile_app_pixel_9",
    "husband": "notify.mobile_app_pixel_9",
    "my husband": "notify.mobile_app_pixel_9",
}

CONVERSATION_MODE_PROMPT: Final = """You are a relaxed, witty companion having a spoken conversation. Nothing in this turn matched a smart-home intent, so it is PROBABLY chat rather than a command.

- Talk like a person, not an assistant. No "How can I help you?", no listing your capabilities, no narrating what you can or can't do.
- Never mention tools, actions, devices, or that you are an assistant, unless the user is actually asking for one.
- Answer what was actually asked. Be direct, warm, and blunt when bluntness fits.
- Swearing, dark humour, and adult topics are fine. Do not lecture, moralise, or add disclaimers.
- Keep it to a few sentences — this is spoken aloud, so it must stay listenable.

"Probably" is not "certainly" — the keyword router misses phrasings, and the
tools you can see in this turn are real ones. If this turn IS a request to
change something (a shade, a light, music, a thermostat), call the tool that
fits it. Replying "done" for an action you never took is the one thing you may
never do: that is a lie, not conversation. If you cannot do it, say so plainly.

Today's date: {current_date}
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
# API TIMEOUT - Shared timeout for external API calls
# =============================================================================
API_TIMEOUT: Final = 15  # seconds

# =============================================================================
# ELEVENLABS TTS SETTINGS
# =============================================================================
# Restored 2026-09-28 (removed in 7cd0513 when the subscription lapsed), now with
# streaming: sentence-by-sentence synthesis on a kept-warm connection.
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
CONF_ELEVENLABS_LANGUAGE: Final = "elevenlabs_language"
CONF_ELEVENLABS_SEED: Final = "elevenlabs_seed"
CONF_ELEVENLABS_SENTENCE_STREAMING: Final = "elevenlabs_sentence_streaming"
CONF_ELEVENLABS_KEEP_WARM: Final = "elevenlabs_keep_warm"

DEFAULT_ELEVENLABS_API_KEY: Final = ""
DEFAULT_ELEVENLABS_VOICE_ID: Final = ""
DEFAULT_ELEVENLABS_MODEL: Final = "eleven_v4_turbo"
DEFAULT_ELEVENLABS_STABILITY: Final = 0.5
DEFAULT_ELEVENLABS_SIMILARITY: Final = 0.75
DEFAULT_ELEVENLABS_STYLE: Final = 0.0
DEFAULT_ELEVENLABS_SPEAKER_BOOST: Final = True
DEFAULT_ELEVENLABS_SPEED: Final = 1.0
DEFAULT_ELEVENLABS_OUTPUT_FORMAT: Final = "mp3_44100_128"
DEFAULT_ELEVENLABS_TEXT_NORMALIZATION: Final = "auto"
# "auto" = send no language_code, so the model follows the text (Spanglish code-switching).
# Forcing the pipeline language ("en") would read Spanish words with English phonetics.
DEFAULT_ELEVENLABS_LANGUAGE: Final = "auto"
DEFAULT_ELEVENLABS_SEED: Final = 0  # 0 = random
DEFAULT_ELEVENLABS_SENTENCE_STREAMING: Final = True
DEFAULT_ELEVENLABS_KEEP_WARM: Final = True

ELEVENLABS_TEXT_NORMALIZATION_MODES: Final = ["auto", "on", "off"]

ELEVENLABS_LANGUAGES: Final = ["auto", "en", "es", "pt", "fr", "de", "it"]

# Static fallback when the models API can't be reached from the options screen.
ELEVENLABS_MODELS: Final = [
    "eleven_v4_turbo",
    "eleven_v4",
    "eleven_flash_v2_5",
    "eleven_turbo_v2_5",
    "eleven_v3",
    "eleven_multilingual_v2",
]

# Streaming always needs a container HA can pass through or transcode: mp3 only.
ELEVENLABS_OUTPUT_FORMATS: Final = [
    "mp3_44100_128",
    "mp3_44100_192",
    "mp3_44100_64",
    "mp3_22050_32",
    "pcm_48000",  # raw PCM wrapped in WAV: satellites' 48 kHz FLAC needs no decode/resample
]
