"""PureLLM Conversation Entity - Pure LLM Voice Assistant v5.0.

This is the main conversation entity that handles ALL voice commands
through the LLM pipeline with tool calling and STREAMING TTS support.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, TYPE_CHECKING

# Conversation history settings
CONVERSATION_TIMEOUT_SECONDS = 300  # 5 minutes - conversations expire after this
MAX_CONVERSATION_HISTORY = 4  # Max message pairs to keep per conversation (reduced for memory)

from homeassistant.components import conversation
from homeassistant.components.conversation import ChatLog, ConversationEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util

from openai import AsyncOpenAI

from .const import (
    DOMAIN,
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_CALENDAR_ENTITIES,
    CONF_CAMERA_ENTITIES,
    CONF_CUSTOM_LATITUDE,
    CONF_CUSTOM_LONGITUDE,
    CONF_DEVICE_ALIASES,
    CONF_ENABLE_CALENDAR,
    CONF_ENABLE_CAMERAS,
    CONF_ENABLE_DEVICE_STATUS,
    CONF_ENABLE_MUSIC,
    CONF_ENABLE_PLACES,
    CONF_ENABLE_RESTAURANTS,
    CONF_ENABLE_SPORTS,
    CONF_ENABLE_THERMOSTAT,
    CONF_ENABLE_WEATHER,
    CONF_ENABLE_WIKIPEDIA,
    CONF_ENABLE_SEARCH,
    CONF_GOOGLE_PLACES_API_KEY,
    CONF_TAVILY_API_KEY,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_OPENWEATHERMAP_API_KEY,
    CONF_PROVIDER,
    CONF_ROOM_PLAYER_MAPPING,
    CONF_SYSTEM_PROMPT,
    CONF_TEMPERATURE,
    CONF_THERMOSTAT_ENTITY,
    CONF_THERMOSTAT_MAX_TEMP,
    CONF_THERMOSTAT_MIN_TEMP,
    CONF_THERMOSTAT_TEMP_STEP,
    CONF_THERMOSTAT_USE_CELSIUS,
    CONF_TOP_P,
    CONF_NOTIFICATION_ENTITIES,
    CONF_NOTIFY_ON_PLACES,
    CONF_NOTIFY_ON_RESTAURANTS,
    CONF_NOTIFY_ON_CAMERA,
    CONF_NOTIFY_ON_SEARCH,
    CONF_VOICE_SCRIPTS,
    DEFAULT_VOICE_SCRIPTS,
    CONF_CAMERA_FRIENDLY_NAMES,
    DEFAULT_CAMERA_FRIENDLY_NAMES,
    CONF_SOFABATON_ACTIVITIES,
    DEFAULT_SOFABATON_ACTIVITIES,
    DEFAULT_API_KEY,
    DEFAULT_NOTIFICATION_ENTITIES,
    DEFAULT_NOTIFY_ON_PLACES,
    DEFAULT_NOTIFY_ON_RESTAURANTS,
    DEFAULT_NOTIFY_ON_CAMERA,
    DEFAULT_NOTIFY_ON_SEARCH,
    DEFAULT_ENABLE_CALENDAR,
    DEFAULT_ENABLE_CAMERAS,
    DEFAULT_ENABLE_DEVICE_STATUS,
    DEFAULT_ENABLE_MUSIC,
    DEFAULT_ENABLE_PLACES,
    DEFAULT_ENABLE_RESTAURANTS,
    DEFAULT_ENABLE_SPORTS,
    DEFAULT_ENABLE_THERMOSTAT,
    DEFAULT_ENABLE_WEATHER,
    DEFAULT_ENABLE_WIKIPEDIA,
    DEFAULT_ENABLE_SEARCH,
    DEFAULT_TAVILY_API_KEY,
    DEFAULT_PROVIDER,
    DEFAULT_ROOM_PLAYER_MAPPING,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_THERMOSTAT_MAX_TEMP,
    DEFAULT_THERMOSTAT_MAX_TEMP_CELSIUS,
    DEFAULT_THERMOSTAT_MIN_TEMP,
    DEFAULT_THERMOSTAT_MIN_TEMP_CELSIUS,
    DEFAULT_THERMOSTAT_TEMP_STEP,
    DEFAULT_THERMOSTAT_TEMP_STEP_CELSIUS,
    DEFAULT_THERMOSTAT_USE_CELSIUS,
    PROVIDER_BASE_URLS,
    PROVIDER_GOOGLE,
    PROVIDER_LM_STUDIO,
    get_version,
)

# Import from new modules
from .utils.parsing import parse_entity_config, parse_list_config

from .tools.definitions import build_tools, ToolConfig

# Tool handlers
from .tools import weather as weather_tool
from .tools import sports as sports_tool
from .tools import places as places_tool
from .tools import wikipedia as wikipedia_tool
from .tools import calendar as calendar_tool
from .tools import camera as camera_tool
from .tools import thermostat as thermostat_tool
from .tools import device as device_tool
from .tools.music import MusicController
from .tools import timer as timer_tool
from .tools import lists as lists_tool
from .tools import sofabaton as sofabaton_tool
from .tools import reminders as reminders_tool
from .tools import search as search_tool

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)

# Type for delta dictionaries used in streaming
ContentDelta = dict[str, Any]


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entity."""
    agent = PureLLMConversationEntity(config_entry)
    async_add_entities([agent])

    # Store agent reference for service calls
    hass.data.setdefault("purellm", {})
    hass.data["purellm"][config_entry.entry_id] = agent


class PureLLMConversationEntity(ConversationEntity):
    """PureLLM conversation agent entity - Pure LLM pipeline with streaming."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = False  # Disabled: prevents voice pipeline timing issues with micro wake words

    @property
    def supported_languages(self) -> list[str] | str:
        """Return supported languages - use MATCH_ALL for all languages."""
        return conversation.MATCH_ALL

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = config_entry
        self._attr_unique_id = config_entry.entry_id
        self._session: aiohttp.ClientSession | None = None

        # Usage tracking
        self._api_calls = {
            "weather": 0, "places": 0, "restaurants": 0,
            "sports": 0, "wikipedia": 0, "llm": 0,
        }

        # Caches
        self._tools: list[dict] | None = None
        self._cached_system_prompt: str | None = None
        self._cached_system_prompt_date: str | None = None

        # Music controller (initialized after config)
        self._music_controller: MusicController | None = None

        # Current query for tool context
        self._current_user_query: str = ""

        # Current user_input for tool execution
        self._current_user_input: conversation.ConversationInput | None = None

        # Conversation history storage: {conversation_id: {"messages": [...], "last_access": timestamp}}
        self._conversation_history: dict[str, dict[str, Any]] = {}

        # Initialize config
        self._update_from_config({**config_entry.data, **config_entry.options})

    @property
    def device_info(self):
        """Return device info."""
        return {
            "identifiers": {(DOMAIN, self.entry.entry_id)},
            "name": self.entry.title,
            "manufacturer": "LosCV29",
            "model": "Voice Assistant",
            "entry_type": "service",
            "sw_version": get_version(),
        }

    def _update_from_config(self, config: dict[str, Any]) -> None:
        """Update configuration."""
        # Provider settings
        self.provider = config.get(CONF_PROVIDER, DEFAULT_PROVIDER)
        self.api_key = config.get(CONF_API_KEY, DEFAULT_API_KEY)
        self.model = config.get(CONF_MODEL, "")
        self.temperature = config.get(CONF_TEMPERATURE, 0.7)
        self.max_tokens = config.get(CONF_MAX_TOKENS, 2000)
        self.top_p = config.get(CONF_TOP_P, 0.95)

        # Base URL
        base_url = config.get(CONF_BASE_URL)
        if not base_url:
            base_url = PROVIDER_BASE_URLS.get(self.provider, "http://localhost:1234/v1")
        self.base_url = base_url

        # Mark client for deferred initialization (avoid blocking SSL on event loop)
        self.client = None
        self._client_init_needed = True

        # Conversation features
        self._attr_supported_features = conversation.ConversationEntityFeature.CONTROL
        self.system_prompt = config.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT)

        # Custom location
        def _parse_coord(key):
            try:
                v = float(config.get(key) or 0)
                return v if v != 0 else None
            except (ValueError, TypeError):
                return None
        self.custom_latitude = _parse_coord(CONF_CUSTOM_LATITUDE)
        self.custom_longitude = _parse_coord(CONF_CUSTOM_LONGITUDE)

        # API keys
        self.openweathermap_api_key = config.get(CONF_OPENWEATHERMAP_API_KEY, "")
        self.google_places_api_key = config.get(CONF_GOOGLE_PLACES_API_KEY, "")
        self.tavily_api_key = config.get(CONF_TAVILY_API_KEY, DEFAULT_TAVILY_API_KEY)

        # Feature toggles
        self.enable_weather = config.get(CONF_ENABLE_WEATHER, DEFAULT_ENABLE_WEATHER)
        self.enable_calendar = config.get(CONF_ENABLE_CALENDAR, DEFAULT_ENABLE_CALENDAR)
        self.enable_cameras = config.get(CONF_ENABLE_CAMERAS, DEFAULT_ENABLE_CAMERAS)
        self.enable_sports = config.get(CONF_ENABLE_SPORTS, DEFAULT_ENABLE_SPORTS)
        self.enable_places = config.get(CONF_ENABLE_PLACES, DEFAULT_ENABLE_PLACES)
        self.enable_restaurants = config.get(CONF_ENABLE_RESTAURANTS, DEFAULT_ENABLE_RESTAURANTS)
        self.enable_thermostat = config.get(CONF_ENABLE_THERMOSTAT, DEFAULT_ENABLE_THERMOSTAT)
        self.enable_device_status = config.get(CONF_ENABLE_DEVICE_STATUS, DEFAULT_ENABLE_DEVICE_STATUS)
        self.enable_wikipedia = config.get(CONF_ENABLE_WIKIPEDIA, DEFAULT_ENABLE_WIKIPEDIA)
        self.enable_music = config.get(CONF_ENABLE_MUSIC, DEFAULT_ENABLE_MUSIC)
        self.enable_search = config.get(CONF_ENABLE_SEARCH, DEFAULT_ENABLE_SEARCH)

        # Entity configuration
        self.room_player_mapping = parse_entity_config(config.get(CONF_ROOM_PLAYER_MAPPING, DEFAULT_ROOM_PLAYER_MAPPING))
        self.thermostat_entity = config.get(CONF_THERMOSTAT_ENTITY, "")
        self.calendar_entities = parse_list_config(config.get(CONF_CALENDAR_ENTITIES, ""))
        self.camera_entities = parse_list_config(config.get(CONF_CAMERA_ENTITIES, ""))
        self.device_aliases = self._parse_device_aliases(config.get(CONF_DEVICE_ALIASES, ""))

        # Thermostat settings
        self.thermostat_use_celsius = config.get(CONF_THERMOSTAT_USE_CELSIUS, DEFAULT_THERMOSTAT_USE_CELSIUS)
        if self.thermostat_use_celsius:
            default_min, default_max = DEFAULT_THERMOSTAT_MIN_TEMP_CELSIUS, DEFAULT_THERMOSTAT_MAX_TEMP_CELSIUS
            default_step = DEFAULT_THERMOSTAT_TEMP_STEP_CELSIUS
        else:
            default_min, default_max = DEFAULT_THERMOSTAT_MIN_TEMP, DEFAULT_THERMOSTAT_MAX_TEMP
            default_step = DEFAULT_THERMOSTAT_TEMP_STEP

        self.thermostat_min_temp = config.get(CONF_THERMOSTAT_MIN_TEMP) or default_min
        self.thermostat_max_temp = config.get(CONF_THERMOSTAT_MAX_TEMP) or default_max
        self.thermostat_temp_step = config.get(CONF_THERMOSTAT_TEMP_STEP) or default_step

        # Notification settings
        notify_entities_str = config.get(CONF_NOTIFICATION_ENTITIES, DEFAULT_NOTIFICATION_ENTITIES)
        if isinstance(notify_entities_str, str) and notify_entities_str:
            # Parse newline-separated service names
            self.notification_entities = [e.strip() for e in notify_entities_str.split("\n") if e.strip()]
        else:
            self.notification_entities = []
        self.notify_on_places = config.get(CONF_NOTIFY_ON_PLACES, DEFAULT_NOTIFY_ON_PLACES)
        self.notify_on_restaurants = config.get(CONF_NOTIFY_ON_RESTAURANTS, DEFAULT_NOTIFY_ON_RESTAURANTS)
        self.notify_on_camera = config.get(CONF_NOTIFY_ON_CAMERA, DEFAULT_NOTIFY_ON_CAMERA)
        self.notify_on_search = config.get(CONF_NOTIFY_ON_SEARCH, DEFAULT_NOTIFY_ON_SEARCH)

        # JSON list parser helper
        def _parse_json_list(key, default):
            raw = config.get(key, default)
            try:
                return json.loads(raw) if raw else []
            except (json.JSONDecodeError, TypeError):
                return []

        # Voice scripts configuration
        self.voice_scripts = _parse_json_list(CONF_VOICE_SCRIPTS, DEFAULT_VOICE_SCRIPTS)

        # Camera friendly names configuration
        camera_names_str = config.get(CONF_CAMERA_FRIENDLY_NAMES, DEFAULT_CAMERA_FRIENDLY_NAMES)
        raw_camera_names = parse_entity_config(camera_names_str) if camera_names_str else {}
        self.camera_friendly_names = {}
        for entity_id, friendly_name in raw_camera_names.items():
            if entity_id.startswith("camera."):
                location_key = entity_id[7:]
            else:
                location_key = entity_id
            self.camera_friendly_names[location_key] = friendly_name

        # SofaBaton activities configuration
        self.sofabaton_activities = _parse_json_list(CONF_SOFABATON_ACTIVITIES, DEFAULT_SOFABATON_ACTIVITIES)

        # Clear caches on config update
        self._tools = None
        self._cached_system_prompt = None

    @property
    def temp_unit(self) -> str:
        """Get temperature unit string."""
        return "°C" if self.thermostat_use_celsius else "°F"

    def format_temp(self, temp: float | int | None) -> str:
        """Format temperature with unit."""
        if temp is None:
            return "unknown"
        return f"{int(temp)}{self.temp_unit}"

    def _parse_device_aliases(self, config_value: str) -> dict[str, str]:
        """Parse device aliases from config - supports both old text and new JSON format.

        Old format (text): "alias:entity_id" per line
        New format (JSON): [{"alias": "name", "entity": "entity_id"}, ...]

        Returns: dict mapping alias -> entity_id
        """
        result: dict[str, str] = {}
        if not config_value:
            return result

        # Try JSON format first (new format)
        try:
            aliases_list = json.loads(config_value)
            if isinstance(aliases_list, list):
                for item in aliases_list:
                    if isinstance(item, dict):
                        alias = item.get("alias", "").lower().strip()
                        entity = item.get("entity", "").strip()
                        if alias and entity:
                            result[alias] = entity
                return result
        except (json.JSONDecodeError, TypeError):
            pass

        # Fall back to old text format: "alias:entity_id" per line
        for line in config_value.strip().split("\n"):
            line = line.strip()
            if ":" in line:
                alias, entity_id = line.split(":", 1)
                result[alias.strip().lower()] = entity_id.strip()

        return result

    def _create_openai_client(self):
        """Create OpenAI client (runs in executor to avoid blocking SSL)."""
        if self.provider == PROVIDER_LM_STUDIO:
            return AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key if self.api_key else "lm-studio",
                timeout=180.0,  # 3 minutes - VLM operations need more time
                max_retries=2,
            )
        return None

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()

        # Initialize OpenAI client in executor (SSL cert loading is blocking)
        if self._client_init_needed:
            self.client = await self.hass.async_add_executor_job(self._create_openai_client)
            self._client_init_needed = False

        # Initialize shared session
        self._session = async_get_clientsession(self.hass)

        # Initialize music controller
        if self.enable_music and self.room_player_mapping:
            self._music_controller = MusicController(
                self.hass,
                self.room_player_mapping
            )

        # Warm up provider connection in background (pre-establish SSL handshake)
        # This saves 100-300ms on the first real query
        asyncio.create_task(self._warmup_provider_connection())

        # Listen for config updates
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_config_updated)
        )

    async def _warmup_provider_connection(self) -> None:
        """Pre-establish connection to LLM provider.

        Makes a lightweight request to complete SSL handshake before the first
        real query, reducing latency by 100-300ms on first interaction.
        """
        import aiohttp

        try:
            if self.provider == PROVIDER_LM_STUDIO and self.client:
                # For OpenAI SDK clients, list models to warm up connection
                try:
                    async with asyncio.timeout(5):
                        await self.client.models.list()
                    _LOGGER.debug("Warmed up %s connection via models list", self.provider)
                except Exception:
                    # Fallback: just establish TCP/SSL connection
                    pass

            elif self.provider == PROVIDER_GOOGLE:
                # Google: lightweight models list
                if self._session:
                    try:
                        async with asyncio.timeout(5):
                            async with self._session.get(
                                f"{self.base_url}/models?key={self.api_key}&pageSize=1",
                                timeout=aiohttp.ClientTimeout(total=5),
                            ) as resp:
                                await resp.read()
                        _LOGGER.debug("Warmed up Google Gemini connection")
                    except Exception:
                        pass

        except Exception as e:
            # Connection warmup is best-effort, never fail startup
            _LOGGER.debug("Connection warmup skipped: %s", e)

    @staticmethod
    async def _async_config_updated(hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Handle config entry update."""
        await hass.config_entries.async_reload(entry.entry_id)

    def _track_api_call(self, api_name: str) -> None:
        """Track API usage."""
        if api_name in self._api_calls:
            self._api_calls[api_name] += 1

    def _get_effective_system_prompt(self) -> str:
        """Get system prompt with current date."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._cached_system_prompt and self._cached_system_prompt_date == today:
            return self._cached_system_prompt

        prompt = self.system_prompt.replace("{current_date}", today)
        self._cached_system_prompt = prompt
        self._cached_system_prompt_date = today
        return prompt

    def _build_tools(self) -> list[dict]:
        """Build tools list based on enabled features."""
        if self._tools is not None:
            return self._tools

        self._tools = build_tools(ToolConfig(self))
        return self._tools

    def _cleanup_expired_conversations(self) -> None:
        """Remove expired conversations from history."""
        now = time.time()
        expired = [
            conv_id for conv_id, data in self._conversation_history.items()
            if now - data.get("last_access", 0) > CONVERSATION_TIMEOUT_SECONDS
        ]
        for conv_id in expired:
            del self._conversation_history[conv_id]
            _LOGGER.debug("Expired conversation %s", conv_id)

    def _get_conversation_history(self, conversation_id: str | None) -> list[dict]:
        """Get message history for a conversation, or empty list if none/expired."""
        if not conversation_id:
            return []

        self._cleanup_expired_conversations()

        if conversation_id in self._conversation_history:
            data = self._conversation_history[conversation_id]
            data["last_access"] = time.time()
            return data.get("messages", [])

        return []

    def _save_conversation_turn(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        extra_system_prompt: str | None = None,
    ) -> None:
        """Save a conversation turn to history."""
        if conversation_id not in self._conversation_history:
            self._conversation_history[conversation_id] = {
                "messages": [],
                "last_access": time.time(),
                "extra_system_prompt": extra_system_prompt,
            }

        data = self._conversation_history[conversation_id]
        data["last_access"] = time.time()

        # Store extra_system_prompt if provided (for continuing conversations)
        if extra_system_prompt:
            data["extra_system_prompt"] = extra_system_prompt

        messages = data["messages"]
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})

        # Trim to max history (keep most recent)
        max_messages = MAX_CONVERSATION_HISTORY * 2  # pairs of user/assistant
        if len(messages) > max_messages:
            data["messages"] = messages[-max_messages:]

        _LOGGER.debug(
            "Saved conversation turn for %s (history: %d messages)",
            conversation_id, len(data["messages"])
        )

    def _get_extra_system_prompt(
        self,
        user_input: conversation.ConversationInput,
        chat_log: ChatLog | None,
    ) -> str | None:
        """Get extra system prompt from user_input, chat_log, or stored conversation.

        Priority:
        1. user_input.extra_system_prompt (from ask_question/start_conversation)
        2. chat_log.extra_system_prompt (from conversation framework)
        3. stored conversation history
        """
        # First check user_input for extra_system_prompt (ask_question passes it here)
        if hasattr(user_input, 'extra_system_prompt') and user_input.extra_system_prompt:
            return user_input.extra_system_prompt

        # Then check chat_log for extra_system_prompt
        if hasattr(chat_log, 'extra_system_prompt') and chat_log.extra_system_prompt:
            return chat_log.extra_system_prompt

        # Check if we have a stored extra_system_prompt for this conversation
        if user_input.conversation_id and user_input.conversation_id in self._conversation_history:
            return self._conversation_history[user_input.conversation_id].get("extra_system_prompt")

        return None

    async def async_process(
        self,
        user_input: conversation.ConversationInput,
    ) -> conversation.ConversationResult:
        """Process a conversation input from Home Assistant.

        This is the main entry point called by Home Assistant's conversation
        framework, including voice pipelines and assist_satellite services.
        """
        return await self._async_handle_message(user_input, None)

    async def _try_ask_and_act_match(
        self,
        extra_system_prompt: str | None,
        user_text: str,
        user_input: conversation.ConversationInput,
        conversation_id: str,
    ) -> conversation.ConversationResult | None:
        """Try to match user text against ask_and_act answers and execute directly.

        Returns a ConversationResult if matched, or None to fall through to LLM.
        """
        from . import ASK_AND_ACT_MARKER, ASK_AND_ACT_MARKER_END

        if not extra_system_prompt or ASK_AND_ACT_MARKER not in extra_system_prompt:
            return None

        # Parse embedded answers JSON from the extra_system_prompt
        try:
            start = extra_system_prompt.index(ASK_AND_ACT_MARKER) + len(ASK_AND_ACT_MARKER)
            end = extra_system_prompt.index(ASK_AND_ACT_MARKER_END, start)
            answers = json.loads(extra_system_prompt[start:end])
        except (ValueError, json.JSONDecodeError) as err:
            _LOGGER.warning("ask_and_act: failed to parse embedded answers: %s", err)
            return None

        # Match user text against answer sentences (case-insensitive, punctuation-stripped)
        # STT often adds trailing punctuation (e.g. "Yes." instead of "yes")
        user_clean = user_text.lower().strip().rstrip(".,!?;:")
        matched_answer = None
        for answer in answers:
            for sentence in answer.get("sentences", []):
                if sentence.lower().strip().rstrip(".,!?;:") == user_clean:
                    matched_answer = answer
                    break
            if matched_answer:
                break

        if not matched_answer:
            _LOGGER.debug("ask_and_act: no exact match for '%s', falling through to LLM", user_text)
            return None

        _LOGGER.info("ask_and_act: matched answer id='%s' for user text '%s'",
                      matched_answer.get("id", "?"), user_text)

        # Execute the action if one is configured
        if "action" in matched_answer:
            action_config = matched_answer["action"]
            service_str = action_config.get("service", "")
            service_parts = service_str.split(".", 1)
            if len(service_parts) == 2:
                service_data = action_config.get("data") or {}
                target = action_config.get("target")

                try:
                    _LOGGER.info(
                        "ask_and_act: executing %s target=%s data=%s",
                        service_str, target, service_data,
                    )
                    await self.hass.services.async_call(
                        service_parts[0],
                        service_parts[1],
                        service_data,
                        target=target,
                        blocking=True,
                    )
                except Exception as err:
                    _LOGGER.error("ask_and_act: action execution failed: %s", err)

        # Build response
        response_text = matched_answer.get("response", "OK")

        # Save conversation turn for multi-turn continuity
        self._save_conversation_turn(
            conversation_id, user_text, response_text, extra_system_prompt
        )

        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(response_text)
        return conversation.ConversationResult(
            response=response,
            conversation_id=conversation_id,
        )

    # =========================================================================
    # Follow-up Conversation (Continuing Conversation)
    # =========================================================================

    @staticmethod
    def _response_has_follow_up(response: str | None) -> bool:
        """Check if the LLM response ends with a follow-up question.

        HA's native continuing conversation (2025.4+) uses the same heuristic:
        if the response ends with '?' the voice pipeline keeps the satellite
        listening without requiring the wake word again.

        We add a length gate so trivial responses don't trigger it.
        """
        if not response:
            return False
        stripped = response.rstrip()
        # Must end with '?' and be longer than a trivial response
        return stripped.endswith("?") and len(stripped) > 20

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: ChatLog | None,
    ) -> conversation.ConversationResult:
        """Handle an incoming chat message.

        Uses streaming for local models (LM Studio/vLLM) for faster first-token response.
        Uses simple non-streaming calls for cloud providers (more reliable).
        Supports continuing conversations with conversation_id tracking.
        """
        user_text = user_input.text.strip()
        self._current_user_query = user_text
        self._current_user_input = user_input

        # Get or create conversation_id for tracking
        conversation_id = user_input.conversation_id or str(uuid.uuid4())

        # Get extra_system_prompt (from ask_question, start_conversation, or stored)
        extra_system_prompt = self._get_extra_system_prompt(user_input, chat_log)

        # Only get conversation history if this is a start_conversation call (has extra_system_prompt)
        history = self._get_conversation_history(user_input.conversation_id) if extra_system_prompt else []

        _LOGGER.info(
            "PureLLM processing: '%s' provider=%s conversation_id=%s history_turns=%d extra_prompt=%s",
            user_text, self.provider, conversation_id[:8] if conversation_id else "new",
            len(history) // 2, bool(extra_system_prompt)
        )

        if extra_system_prompt:
            _LOGGER.debug("extra_system_prompt received (%d chars)", len(extra_system_prompt))

        # --- ask_and_act: code-based answer matching and direct action execution ---
        ask_act_result = await self._try_ask_and_act_match(
            extra_system_prompt, user_text, user_input, conversation_id
        )
        if ask_act_result is not None:
            return ask_act_result

        tools = self._build_tools()
        system_prompt = self._get_effective_system_prompt()

        # Append extra_system_prompt if provided (from start_conversation)
        if extra_system_prompt:
            system_prompt = f"{system_prompt}\n\nAdditional context:\n{extra_system_prompt}"

        max_tokens = self._calculate_max_tokens(user_text)

        try:
            if self.provider == PROVIDER_LM_STUDIO:
                # Use streaming for local models - faster first-token response
                _LOGGER.debug("Using streaming for local provider: %s", self.provider)
                stream = self._stream_openai_compatible(
                    user_text, tools, system_prompt, max_tokens, history
                )

                # Collect streaming response
                final_response = ""
                async for delta in stream:
                    if "content" in delta:
                        final_response += delta["content"]

            elif self.provider == PROVIDER_GOOGLE:
                final_response = await self._call_google(
                    user_text, tools, system_prompt, max_tokens, history
                )
            else:
                final_response = "Unknown provider."

            _LOGGER.info("PureLLM response (%d chars): %s", len(final_response) if final_response else 0, (final_response or "")[:100])

            # Only save conversation history for start_conversation calls
            # (when extra_system_prompt exists). Never save status responses
            # to history — it gives the LLM stale data to parrot.
            if extra_system_prompt:
                self._save_conversation_turn(
                    conversation_id, user_text, final_response or "", extra_system_prompt
                )

        except Exception as err:
            _LOGGER.error("PureLLM error: %s", err, exc_info=True)
            final_response = "Sorry, there was an error."

        # --- Continuing conversation (HA 2025.4+ native support) ---
        # If the response ends with a question, tell the voice pipeline to
        # keep the satellite listening after TTS finishes — no wake word needed.
        keep_listening = self._response_has_follow_up(final_response)
        if keep_listening:
            _LOGGER.info("Continuing conversation: response ends with follow-up question")

        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(final_response or "No response.")

        return conversation.ConversationResult(
            response=response,
            conversation_id=conversation_id,
            continue_conversation=keep_listening,
        )

    async def _call_google(
        self,
        user_text: str,
        tools: list,
        system_prompt: str,
        max_tokens: int,
        history: list[dict] | None = None,
    ) -> str:
        """Simple non-streaming Google Gemini call with conversation history support."""
        function_declarations = []
        for tool in tools:
            func = tool.get("function", {})
            function_declarations.append({
                "name": func.get("name"),
                "description": func.get("description"),
                "parameters": func.get("parameters", {"type": "object", "properties": {}})
            })

        contents = [
            {"role": "user", "parts": [{"text": f"System: {system_prompt}"}]},
            {"role": "model", "parts": [{"text": "Understood."}]},
        ]

        # Add conversation history if present
        if history:
            for msg in history:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        # Add current user message
        contents.append({"role": "user", "parts": [{"text": user_text}]})

        for iteration in range(5):
            payload = {
                "contents": contents,
                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": self.temperature},
            }
            if function_declarations:
                payload["tools"] = [{"functionDeclarations": function_declarations}]

            url = f"{self.base_url}/models/{self.model}:generateContent"
            headers = {"x-goog-api-key": self.api_key}

            self._track_api_call("llm")
            _LOGGER.debug("Google API call to %s", url)

            async with self._session.post(url, json=payload, headers=headers, timeout=120) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    _LOGGER.error("Google API error %d: %s", resp.status, error)
                    return "Error calling Google API."

                data = await resp.json()

            candidates = data.get("candidates", [])
            if not candidates:
                return "No response from Google."

            parts = candidates[0].get("content", {}).get("parts", [])
            text_response = ""
            function_calls = []

            for part in parts:
                if "text" in part:
                    text_response += part["text"]
                elif "functionCall" in part:
                    function_calls.append(part["functionCall"])

            if function_calls:
                contents.append({"role": "model", "parts": parts})
                function_responses = []

                for fc in function_calls:
                    result = await self._execute_tool(fc["name"], fc.get("args", {}))
                    if isinstance(result, dict) and "response_text" in result:
                        resp_content = result["response_text"]
                    else:
                        resp_content = json.dumps(result, ensure_ascii=False)

                    function_responses.append({
                        "functionResponse": {"name": fc["name"], "response": {"result": resp_content}}
                    })

                contents.append({"role": "user", "parts": function_responses})
                continue

            if text_response:
                return text_response

        return "Could not get response."

    def _calculate_max_tokens(self, user_text: str) -> int:
        """Return configured max_tokens - no caps for local GPU."""
        return self.max_tokens

    # =========================================================================
    # Streaming LLM Provider Methods
    # =========================================================================

    async def _stream_openai_compatible(
        self,
        user_text: str,
        tools: list[dict],
        system_prompt: str,
        max_tokens: int,
        history: list[dict] | None = None,
    ) -> AsyncGenerator[ContentDelta, None]:
        """Stream from OpenAI-compatible API with tool support and conversation history."""
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Add conversation history if present
        if history:
            messages.extend(history)

        # Add current user message
        messages.append({"role": "user", "content": user_text})

        called_tools: set[str] = set()
        last_tool_response_text: str = ""  # Track last tool response for fallback

        for iteration in range(5):  # Max 5 tool iterations
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": max_tokens,
                "top_p": self.top_p,
                "stream": True,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            accumulated_content = ""
            tool_calls_buffer: list[dict] = []

            self._track_api_call("llm")

            try:
                stream = await self.client.chat.completions.create(**kwargs)

                try:
                    async for chunk in stream:
                        if not chunk.choices:
                            continue

                        delta = chunk.choices[0].delta

                        # Yield content deltas immediately for streaming TTS
                        if delta.content:
                            accumulated_content += delta.content
                            yield {"content": delta.content}

                        # Accumulate tool calls (new format)
                        if delta.tool_calls:
                            for tc_delta in delta.tool_calls:
                                # Default index to 0 if missing (common with local LLMs)
                                idx = tc_delta.index if tc_delta.index is not None else 0
                                while len(tool_calls_buffer) <= idx:
                                    tool_calls_buffer.append({
                                        "id": None,
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    })

                                current = tool_calls_buffer[idx]
                                if tc_delta.id:
                                    current["id"] = tc_delta.id
                                if tc_delta.function:
                                    if tc_delta.function.name:
                                        current["function"]["name"] += tc_delta.function.name
                                    if tc_delta.function.arguments:
                                        current["function"]["arguments"] += tc_delta.function.arguments

                        # Handle legacy function_call format (some local LLMs use this)
                        if hasattr(delta, "function_call") and delta.function_call and not delta.tool_calls:
                            if not tool_calls_buffer:
                                tool_calls_buffer.append({
                                    "id": "call_legacy_0",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                })
                            current = tool_calls_buffer[0]
                            if delta.function_call.name:
                                current["function"]["name"] += delta.function_call.name
                            if delta.function_call.arguments:
                                current["function"]["arguments"] += delta.function_call.arguments
                finally:
                    await stream.close()

                # Debug: log what the LLM produced
                if not accumulated_content and not tool_calls_buffer:
                    _LOGGER.debug("LLM produced no content and no tool calls on iteration %d", iteration)
                elif tool_calls_buffer:
                    _LOGGER.debug("LLM produced %d tool call(s): %s",
                                len(tool_calls_buffer),
                                [tc.get("function", {}).get("name", "?") for tc in tool_calls_buffer])

                # Process tool calls if any
                # Generate synthetic IDs for tool calls missing them (common with local LLMs)
                for i, tc in enumerate(tool_calls_buffer):
                    if not tc.get("id") and tc.get("function", {}).get("name"):
                        tc["id"] = f"call_{iteration}_{i}"

                valid_tool_calls = [
                    tc for tc in tool_calls_buffer
                    if tc.get("id") and tc.get("function", {}).get("name")
                ]

                unique_tool_calls = []
                for tc in valid_tool_calls:
                    tool_key = f"{tc['function']['name']}:{tc['function']['arguments']}"
                    if tool_key not in called_tools:
                        called_tools.add(tool_key)
                        unique_tool_calls.append(tc)
                    else:
                        _LOGGER.debug("LLM repeated tool call (deduped): %s", tc['function']['name'])

                if unique_tool_calls:
                    _LOGGER.info("Executing %d tool call(s)", len(unique_tool_calls))

                    # Add assistant message with tool calls to conversation
                    messages.append({
                        "role": "assistant",
                        "content": accumulated_content if accumulated_content else None,
                        "tool_calls": unique_tool_calls
                    })

                    # Execute tools in parallel
                    tool_tasks = []
                    for tool_call in unique_tool_calls:
                        tool_name = tool_call["function"]["name"]
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"])
                        except json.JSONDecodeError:
                            arguments = {}

                        _LOGGER.info("Tool call: %s(%s)", tool_name, arguments)
                        tool_tasks.append(self._execute_tool(tool_name, arguments))

                    tool_results = await asyncio.gather(*tool_tasks, return_exceptions=True)

                    # Yield tool calls and results as deltas
                    for tool_call, result in zip(unique_tool_calls, tool_results):
                        if isinstance(result, Exception):
                            _LOGGER.error("Tool %s failed: %s", tool_call["function"]["name"], result)
                            result = {"error": str(result)}

                        # Get content for the message
                        if isinstance(result, dict) and "response_text" in result:
                            content = result["response_text"]
                            last_tool_response_text = content
                        else:
                            content = json.dumps(result, ensure_ascii=False)

                        _LOGGER.debug("Tool result for %s: %s", tool_call["function"]["name"], content[:200])

                        # Add tool result to messages for next iteration
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": content,
                        })

                    # Continue to next iteration to get LLM's response after tools
                    continue

                # No tool calls - we're done
                if accumulated_content:
                    return

                _LOGGER.debug("LLM iteration %d: no content and no tool calls, breaking", iteration)
                break

            except Exception as e:
                _LOGGER.error("OpenAI API error: %s", e, exc_info=True)
                yield {"content": "Sorry, there was an error processing your request."}
                return

        # If we get here with no content, use tool response_text as fallback
        if last_tool_response_text:
            _LOGGER.info("LLM failed to respond after tool call, using tool response_text directly")
            yield {"content": last_tool_response_text}
        else:
            _LOGGER.error("LLM fallback triggered after %d iterations - no response produced", iteration + 1)
            yield {"content": "I apologize, but I couldn't complete that request."}

    # =========================================================================
    # Notification Helpers
    # =========================================================================

    async def _send_notification(
        self,
        notification_data: dict[str, Any],
        notification_type: str = "notification",
    ) -> None:
        """Send notification to all configured notification entities."""
        _LOGGER.info("%s notification data: %s", notification_type.capitalize(), notification_data)

        for entity_id in self.notification_entities:
            service_name = entity_id.replace("notify.", "") if entity_id.startswith("notify.") else entity_id
            _LOGGER.info("Calling notify.%s for %s", service_name, notification_type)
            try:
                await self.hass.services.async_call(
                    "notify",
                    service_name,
                    notification_data,
                    blocking=False,
                )
                _LOGGER.info("Successfully sent %s notification to %s", notification_type, service_name)
            except Exception as notify_err:
                _LOGGER.error("Failed to send %s notification to %s: %s", notification_type, entity_id, notify_err)

    def _build_notification_data(
        self,
        title: str,
        message: str,
        actions: list[dict] | None = None,
        click_url: str = "",
        image_url: str = "",
    ) -> dict[str, Any]:
        """Build standard notification data structure."""
        data: dict[str, Any] = {
            "push": {"interruption-level": "time-sensitive"},
        }

        if click_url:
            data["url"] = click_url
            data["clickAction"] = click_url

        if actions:
            data["actions"] = actions

        if image_url:
            data["image"] = image_url
            data["attachment"] = {"url": image_url}

        return {"title": title, "message": message, "data": data}

    async def _send_places_notification(self, places_result: dict[str, Any]) -> None:
        """Send notification with places/directions info to configured devices."""
        try:
            places = places_result.get("places", [])
            query = places_result.get("query", "location")

            _LOGGER.info("Sending places notification for query: %s, places count: %d", query, len(places))

            if not places:
                return

            top_place = places[0]
            place_name = top_place.get("name", "Unknown")
            address = top_place.get("short_address") or top_place.get("address", "")
            distance = top_place.get("distance_miles")
            directions_url = top_place.get("directions_url", "")
            website = top_place.get("website", "")
            phone = top_place.get("phone", "")
            coordinates = top_place.get("coordinates", {})

            apple_maps_url = ""
            if coordinates and coordinates.get("lat") and coordinates.get("lng"):
                lat, lng = coordinates["lat"], coordinates["lng"]
                apple_maps_url = f"https://maps.apple.com/?daddr={lat},{lng}&dirflg=d"

            title = f"📍 {place_name}"
            message_parts = []
            if address:
                message_parts.append(address)
            if distance:
                message_parts.append(f"{distance:.1f} miles away")
            message = "\n".join(message_parts) if message_parts else place_name

            actions = []
            if directions_url:
                actions.append({"action": "URI", "title": "🗺️ Google Maps", "uri": directions_url})
            if apple_maps_url:
                actions.append({"action": "URI", "title": "🍎 Apple Maps", "uri": apple_maps_url})
            if website:
                actions.append({"action": "URI", "title": "🌐 Website", "uri": website})
            if phone:
                phone_clean = "".join(c for c in phone if c.isdigit() or c == "+")
                actions.append({"action": "URI", "title": "📞 Call", "uri": f"tel:{phone_clean}"})

            notification_data = self._build_notification_data(
                title, message, actions, directions_url or apple_maps_url
            )
            await self._send_notification(notification_data, "places")

        except Exception as err:
            _LOGGER.error("Error sending places notification: %s", err, exc_info=True)

    async def _send_restaurant_notification(self, restaurant_result: dict[str, Any]) -> None:
        """Send notification with restaurant info to configured devices."""
        try:
            restaurants = restaurant_result.get("restaurants", [])
            query = restaurant_result.get("query", "restaurant")

            _LOGGER.info("Sending restaurant notification for query: %s, count: %d", query, len(restaurants))

            if not restaurants:
                return

            top_restaurants = restaurants[:3]

            title = f"🍽️ Top {len(top_restaurants)} for '{query}'"
            message_lines = []

            for i, restaurant in enumerate(top_restaurants, 1):
                name = restaurant.get("name", "Unknown")
                rating = restaurant.get("rating")
                review_count = restaurant.get("review_count", 0)
                price = restaurant.get("price", "")
                distance = restaurant.get("distance", "")

                line_parts = [f"{i}. {name}"]
                if rating:
                    line_parts.append(f"★{rating}")
                if review_count:
                    line_parts.append(f"({review_count:,})")
                if price:
                    line_parts.append(price)
                if distance:
                    line_parts.append(distance)
                message_lines.append(" ".join(line_parts))

            message = "\n".join(message_lines)

            top_pick = top_restaurants[0]
            maps_url = top_pick.get("directions_url", "")
            coordinates = top_pick.get("coordinates", {})

            if not maps_url and coordinates and coordinates.get("lat") and coordinates.get("lng"):
                lat, lng = coordinates["lat"], coordinates["lng"]
                maps_url = f"https://www.google.com/maps/dir/?api=1&destination={lat},{lng}"

            actions = []
            if maps_url:
                actions.append({"action": "URI", "title": "🗺️ #1 Directions", "uri": maps_url})
            if len(top_restaurants) > 1 and top_restaurants[1].get("directions_url"):
                actions.append({"action": "URI", "title": "🗺️ #2 Directions", "uri": top_restaurants[1]["directions_url"]})
            if len(top_restaurants) > 2 and top_restaurants[2].get("directions_url"):
                actions.append({"action": "URI", "title": "🗺️ #3 Directions", "uri": top_restaurants[2]["directions_url"]})

            notification_data = self._build_notification_data(
                title, message, actions, maps_url
            )
            await self._send_notification(notification_data, "restaurant")

        except Exception as err:
            _LOGGER.error("Error sending restaurant notification: %s", err, exc_info=True)

    async def _send_reservation_notification(self, reservation_result: dict[str, Any]) -> None:
        """Send notification with reservation link to configured devices."""
        try:
            restaurant_name = reservation_result.get("restaurant_name", "Restaurant")
            reservation_url = reservation_result.get("reservation_url", "")
            party_size = reservation_result.get("party_size", 2)
            date = reservation_result.get("date", "")
            time = reservation_result.get("time", "")
            phone = reservation_result.get("phone", "")
            address = reservation_result.get("address", "")

            _LOGGER.info("Sending reservation notification for: %s", restaurant_name)

            title = f"🍽️ Book {restaurant_name}"

            message_parts = []
            if date and time:
                message_parts.append(f"📅 {date} at {time}")
            elif date:
                message_parts.append(f"📅 {date}")
            elif time:
                message_parts.append(f"🕐 {time}")
            if party_size:
                message_parts.append(f"👥 Party of {party_size}")
            if address:
                message_parts.append(f"📍 {address}")
            if phone:
                message_parts.append(f"📞 {phone}")
            message = "\n".join(message_parts) if message_parts else f"Book a table at {restaurant_name}"

            actions = []
            if reservation_url:
                actions.append({"action": "URI", "title": "🔍 Search Reservations", "uri": reservation_url})
            if phone:
                clean_phone = "".join(c for c in phone if c.isdigit() or c == "+")
                actions.append({"action": "URI", "title": "📞 Call", "uri": f"tel:{clean_phone}"})

            notification_data = self._build_notification_data(title, message, actions, reservation_url)
            await self._send_notification(notification_data, "reservation")

        except Exception as err:
            _LOGGER.error("Error sending reservation notification: %s", err, exc_info=True)

    async def _send_camera_notification(self, camera_result: dict[str, Any]) -> None:
        """Send notification with camera snapshot to configured devices."""
        try:
            location = camera_result.get("location", "Camera")
            description = camera_result.get("description", "")
            snapshot_url = camera_result.get("snapshot_url", "")
            identified_people = camera_result.get("identified_people", [])

            _LOGGER.info("Sending camera notification for: %s", location)

            title = f"📷 {location}"

            if description:
                message = description.split('.')[0] + '.' if '.' in description else description
            else:
                message = "Camera check completed."

            if identified_people:
                people_names = [p.get("name", "Unknown") for p in identified_people if p.get("name")]
                if people_names:
                    message += f"\n👤 Identified: {', '.join(people_names)}"

            notification_data = self._build_notification_data(title, message, image_url=snapshot_url)
            await self._send_notification(notification_data, "camera")

        except Exception as err:
            _LOGGER.error("Error sending camera notification: %s", err, exc_info=True)

    async def _send_search_notification(self, search_result: dict[str, Any]) -> None:
        """Send notification with search result link to configured devices.

        Uses the source_url that best matches the AI-generated answer,
        not just the first search result.
        """
        try:
            query = search_result.get("query", "search")
            source = search_result.get("source", "Web")
            source_url = search_result.get("source_url", "")
            answer = search_result.get("answer", "")

            if not source_url:
                _LOGGER.debug("No source_url in search result, skipping notification")
                return

            _LOGGER.info("Sending search notification: source=%s, url=%s", source, source_url)

            title = f"🔍 {source}"
            message = answer[:150] + "..." if len(answer) > 150 else answer if answer else f"Search: {query}"

            actions = [
                {"action": "URI", "title": f"📖 Read on {source}", "uri": source_url}
            ]

            notification_data = self._build_notification_data(
                title, message, actions, source_url
            )
            await self._send_notification(notification_data, "search")

        except Exception as err:
            _LOGGER.error("Error sending search notification: %s", err, exc_info=True)

    # =========================================================================
    # Tool Execution
    # =========================================================================

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a tool call."""
        try:
            # Get location and timezone context
            latitude = self.custom_latitude or self.hass.config.latitude
            longitude = self.custom_longitude or self.hass.config.longitude
            hass_tz = dt_util.get_time_zone(self.hass.config.time_zone)

            # Built-in datetime tool
            if tool_name == "get_current_datetime":
                now = datetime.now(hass_tz)
                return {
                    "date": now.strftime("%A, %B %d, %Y"),
                    "time": now.strftime("%I:%M %p"),
                    "timezone": self.hass.config.time_zone,
                }

            # Sports tools (all use same pattern)
            if tool_name in ("get_sports_info", "get_ufc_info", "check_league_games", "list_league_games"):
                handler = getattr(sports_tool, tool_name)
                return await handler(arguments, self._session, hass_tz, self._track_api_call, self.tavily_api_key)

            # Wikipedia tools
            if tool_name in ("calculate_age", "get_wikipedia_summary"):
                handler = getattr(wikipedia_tool, tool_name)
                return await handler(arguments, self._session, self._track_api_call)

            # Reminder tools
            if tool_name in ("create_reminder", "get_reminders"):
                handler = getattr(reminders_tool, tool_name)
                return await handler(arguments, self.hass, hass_tz)

            # Tool handlers - maps tool name to async callable
            tool_handlers = {
                # Weather & Info
                "get_weather_forecast": lambda: weather_tool.get_weather_forecast(
                    arguments, self._session, self.openweathermap_api_key,
                    latitude, longitude, self._track_api_call, self._current_user_query
                ),
                "get_calendar_events": lambda: calendar_tool.get_calendar_events(
                    arguments, self.hass, self.calendar_entities, hass_tz
                ),
                # Device control
                "control_thermostat": lambda: thermostat_tool.control_thermostat(
                    arguments, self.hass, self.thermostat_entity,
                    self.thermostat_temp_step, self.thermostat_min_temp,
                    self.thermostat_max_temp, self.format_temp
                ),
                "check_device_status": lambda: device_tool.check_device_status(
                    arguments, self.hass, self.device_aliases,
                    self._current_user_query, self.format_temp
                ),
                "get_device_history": lambda: device_tool.get_device_history(
                    arguments, self.hass, self.device_aliases,
                    hass_tz, self._current_user_query
                ),
                "control_device": lambda: device_tool.control_device(
                    arguments, self.hass, self.device_aliases, self.voice_scripts
                ),
                "control_timer": lambda: timer_tool.control_timer(
                    arguments, self.hass,
                    device_id=self._current_user_input.device_id if self._current_user_input else None,
                    room_player_mapping=self.room_player_mapping
                ),
                "manage_list": lambda: lists_tool.manage_list(arguments, self.hass),
                # Places (with notification post-processing)
                "find_nearby_places": lambda: places_tool.find_nearby_places(
                    arguments, self._session, self.google_places_api_key,
                    latitude, longitude, self._track_api_call
                ),
                "get_restaurant_recommendations": lambda: places_tool.get_restaurant_recommendations(
                    arguments, self._session, self.google_places_api_key,
                    latitude, longitude, self._track_api_call
                ),
                "book_restaurant": lambda: places_tool.book_restaurant(
                    arguments, self._session, self.google_places_api_key,
                    latitude, longitude, self._track_api_call
                ),
                # Camera (with notification post-processing)
                "check_camera": lambda: camera_tool.check_camera(
                    arguments, self.hass, self.camera_friendly_names or None
                ),
                "quick_camera_check": lambda: camera_tool.quick_camera_check(
                    arguments, self.hass, self.camera_friendly_names or None
                ),
                # Web search
                "web_search": lambda: search_tool.web_search(
                    arguments, self._session, self.tavily_api_key, self._track_api_call
                ),
            }

            # Execute tool if it's in our handlers
            if tool_name in tool_handlers:
                result = await tool_handlers[tool_name]()

                # Apply notification post-processing if configured
                if self.notification_entities:
                    if tool_name == "find_nearby_places" and self.notify_on_places and result.get("places"):
                        await self._send_places_notification(result)
                    elif tool_name == "get_restaurant_recommendations" and self.notify_on_restaurants and result.get("restaurants"):
                        await self._send_restaurant_notification(result)
                    elif tool_name == "book_restaurant" and result.get("reservation_url"):
                        await self._send_reservation_notification(result)
                    elif tool_name in ("check_camera", "quick_camera_check") and self.notify_on_camera and result.get("snapshot_url"):
                        await self._send_camera_notification(result)
                    elif tool_name == "web_search" and self.notify_on_search and result.get("source_url"):
                        await self._send_search_notification(result)

                return result

            # Conditional tools (require configuration)
            if tool_name == "control_sofabaton":
                if not self.sofabaton_activities:
                    return {"error": "No SofaBaton activities configured"}
                return await sofabaton_tool.control_sofabaton(
                    self.hass, arguments, self.sofabaton_activities
                )

            if tool_name == "control_music":
                if not self._music_controller:
                    return {"error": "Music control not configured"}
                return await self._music_controller.control_music(arguments)

            # Fall back to script execution
            if self.hass.services.has_service("script", tool_name):
                response = await self.hass.services.async_call(
                    "script", tool_name, arguments, blocking=True, return_response=True
                )
                if response:
                    script_entity = f"script.{tool_name}"
                    if isinstance(response, dict) and script_entity in response:
                        return response[script_entity]
                    return response
                return {"status": "success", "script": tool_name}

            _LOGGER.warning("Unknown tool '%s' called", tool_name)
            return {"success": True, "message": f"Custom function {tool_name} called"}

        except Exception as err:
            _LOGGER.error("Error executing tool %s: %s", tool_name, err, exc_info=True)
            return {"error": str(err)}
