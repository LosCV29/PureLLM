"""PureLLM Conversation Entity - Pure LLM Voice Assistant v5.0.

This is the main conversation entity that handles ALL voice commands
through the LLM pipeline with tool calling and STREAMING TTS support.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, TYPE_CHECKING

from homeassistant.components import conversation
from homeassistant.components.conversation import ChatLog, ConversationEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util

from openai import AsyncOpenAI, AsyncAzureOpenAI

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
    CONF_ENABLE_NEWS,
    CONF_ENABLE_PLACES,
    CONF_ENABLE_RESTAURANTS,
    CONF_ENABLE_SPORTS,
    CONF_ENABLE_STOCKS,
    CONF_ENABLE_THERMOSTAT,
    CONF_ENABLE_WEATHER,
    CONF_ENABLE_WIKIPEDIA,
    CONF_GOOGLE_PLACES_API_KEY,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_NEWSAPI_KEY,
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
    CONF_YELP_API_KEY,
    CONF_NOTIFICATION_ENTITIES,
    CONF_NOTIFY_ON_PLACES,
    CONF_NOTIFY_ON_RESTAURANTS,
    CONF_NOTIFY_ON_CAMERA,
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
    DEFAULT_ENABLE_CALENDAR,
    DEFAULT_ENABLE_CAMERAS,
    DEFAULT_ENABLE_DEVICE_STATUS,
    DEFAULT_ENABLE_MUSIC,
    DEFAULT_ENABLE_NEWS,
    DEFAULT_ENABLE_PLACES,
    DEFAULT_ENABLE_RESTAURANTS,
    DEFAULT_ENABLE_SPORTS,
    DEFAULT_ENABLE_STOCKS,
    DEFAULT_ENABLE_THERMOSTAT,
    DEFAULT_ENABLE_WEATHER,
    DEFAULT_ENABLE_WIKIPEDIA,
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
    OPENAI_COMPATIBLE_PROVIDERS,
    PROVIDER_ANTHROPIC,
    PROVIDER_AZURE,
    PROVIDER_BASE_URLS,
    PROVIDER_GOOGLE,
    PROVIDER_LM_STUDIO,
    PROVIDER_OLLAMA,
    get_version,
)

# Import from new modules
from .utils.parsing import parse_entity_config, parse_list_config

from .tools.definitions import build_tools, ToolConfig

# Tool handlers
from .tools import weather as weather_tool
from .tools import sports as sports_tool
from .tools import stocks as stocks_tool
from .tools import news as news_tool
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
    _attr_supports_streaming = True  # Enable streaming TTS support!

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
            "weather": 0, "places": 0, "restaurants": 0, "news": 0,
            "sports": 0, "wikipedia": 0, "llm": 0, "stocks": 0,
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
        try:
            lat = float(config.get(CONF_CUSTOM_LATITUDE) or 0)
            self.custom_latitude = lat if lat != 0 else None
        except (ValueError, TypeError):
            self.custom_latitude = None

        try:
            lon = float(config.get(CONF_CUSTOM_LONGITUDE) or 0)
            self.custom_longitude = lon if lon != 0 else None
        except (ValueError, TypeError):
            self.custom_longitude = None

        # API keys
        self.openweathermap_api_key = config.get(CONF_OPENWEATHERMAP_API_KEY, "")
        self.google_places_api_key = config.get(CONF_GOOGLE_PLACES_API_KEY, "")
        self.yelp_api_key = config.get(CONF_YELP_API_KEY, "")
        self.newsapi_key = config.get(CONF_NEWSAPI_KEY, "")

        # Feature toggles
        self.enable_weather = config.get(CONF_ENABLE_WEATHER, DEFAULT_ENABLE_WEATHER)
        self.enable_calendar = config.get(CONF_ENABLE_CALENDAR, DEFAULT_ENABLE_CALENDAR)
        self.enable_cameras = config.get(CONF_ENABLE_CAMERAS, DEFAULT_ENABLE_CAMERAS)
        self.enable_sports = config.get(CONF_ENABLE_SPORTS, DEFAULT_ENABLE_SPORTS)
        self.enable_stocks = config.get(CONF_ENABLE_STOCKS, DEFAULT_ENABLE_STOCKS)
        self.enable_news = config.get(CONF_ENABLE_NEWS, DEFAULT_ENABLE_NEWS)
        self.enable_places = config.get(CONF_ENABLE_PLACES, DEFAULT_ENABLE_PLACES)
        self.enable_restaurants = config.get(CONF_ENABLE_RESTAURANTS, DEFAULT_ENABLE_RESTAURANTS)
        self.enable_thermostat = config.get(CONF_ENABLE_THERMOSTAT, DEFAULT_ENABLE_THERMOSTAT)
        self.enable_device_status = config.get(CONF_ENABLE_DEVICE_STATUS, DEFAULT_ENABLE_DEVICE_STATUS)
        self.enable_wikipedia = config.get(CONF_ENABLE_WIKIPEDIA, DEFAULT_ENABLE_WIKIPEDIA)
        self.enable_music = config.get(CONF_ENABLE_MUSIC, DEFAULT_ENABLE_MUSIC)

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

        # Voice scripts configuration
        voice_scripts_json = config.get(CONF_VOICE_SCRIPTS, DEFAULT_VOICE_SCRIPTS)
        try:
            self.voice_scripts = json.loads(voice_scripts_json) if voice_scripts_json else []
        except (json.JSONDecodeError, TypeError):
            self.voice_scripts = []

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
        sofabaton_json = config.get(CONF_SOFABATON_ACTIVITIES, DEFAULT_SOFABATON_ACTIVITIES)
        try:
            self.sofabaton_activities = json.loads(sofabaton_json) if sofabaton_json else []
        except (json.JSONDecodeError, TypeError):
            self.sofabaton_activities = []

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
        if self.provider == PROVIDER_AZURE:
            azure_endpoint = self.base_url
            if "/openai/deployments/" in azure_endpoint:
                azure_endpoint = azure_endpoint.split("/openai/deployments/")[0]
            return AsyncAzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=self.api_key,
                api_version="2024-02-01",
                timeout=180.0,  # 3 minutes - VLM operations need more time
            )
        elif self.provider in OPENAI_COMPATIBLE_PROVIDERS:
            return AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key if self.api_key else "ollama",
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
            self._music_controller = MusicController(self.hass, self.room_player_mapping)

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
            if self.provider in OPENAI_COMPATIBLE_PROVIDERS and self.client:
                # For OpenAI SDK clients, list models to warm up connection
                try:
                    async with asyncio.timeout(5):
                        await self.client.models.list()
                    _LOGGER.debug("Warmed up %s connection via models list", self.provider)
                except Exception:
                    # Fallback: just establish TCP/SSL connection
                    pass

            elif self.provider == PROVIDER_AZURE and self.client:
                # Azure: similar approach
                try:
                    async with asyncio.timeout(5):
                        await self.client.models.list()
                    _LOGGER.debug("Warmed up Azure OpenAI connection")
                except Exception:
                    pass

            elif self.provider == PROVIDER_ANTHROPIC:
                # Anthropic: lightweight models endpoint
                if self._session:
                    headers = {
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                    }
                    try:
                        async with asyncio.timeout(5):
                            async with self._session.get(
                                f"{self.base_url}/v1/models",
                                headers=headers,
                                timeout=aiohttp.ClientTimeout(total=5),
                            ) as resp:
                                await resp.read()
                        _LOGGER.debug("Warmed up Anthropic connection")
                    except Exception:
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

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: ChatLog,
    ) -> conversation.ConversationResult:
        """Handle an incoming chat message.

        Uses streaming for local providers (LM Studio, Ollama) and
        non-streaming for cloud providers (Google, OpenAI, Anthropic, etc.).
        """
        user_text = user_input.text.strip()
        self._current_user_query = user_text
        self._current_user_input = user_input

        # Local providers use streaming, cloud providers use non-streaming
        local_providers = {PROVIDER_LM_STUDIO, PROVIDER_OLLAMA}
        is_local = self.provider in local_providers

        _LOGGER.info("PureLLM processing: '%s' provider=%s streaming=%s", user_text, self.provider, is_local)

        tools = self._build_tools()
        system_prompt = self._get_effective_system_prompt()
        max_tokens = self._calculate_max_tokens(user_text)

        try:
            if is_local:
                # Use streaming for local providers (LM Studio, Ollama)
                final_response = ""
                stream = self._stream_openai_compatible(user_text, tools, system_prompt, max_tokens)
                async for delta in stream:
                    if isinstance(delta, dict) and delta.get("content"):
                        final_response += delta["content"]
            elif self.provider == PROVIDER_GOOGLE:
                final_response = await self._call_google(user_text, tools, system_prompt, max_tokens)
            elif self.provider in OPENAI_COMPATIBLE_PROVIDERS or self.provider == PROVIDER_AZURE:
                final_response = await self._call_openai(user_text, tools, system_prompt, max_tokens)
            elif self.provider == PROVIDER_ANTHROPIC:
                final_response = await self._call_anthropic(user_text, tools, system_prompt, max_tokens)
            else:
                final_response = "Unknown provider."

            _LOGGER.info("PureLLM response (%d chars): %s", len(final_response) if final_response else 0, (final_response or "")[:100])

        except Exception as err:
            _LOGGER.error("PureLLM error: %s", err, exc_info=True)
            final_response = "Sorry, there was an error."

        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(final_response or "No response.")

        return conversation.ConversationResult(
            response=response,
            conversation_id=str(uuid.uuid4()),
        )

    async def _call_google(self, user_text: str, tools: list, system_prompt: str, max_tokens: int) -> str:
        """Simple non-streaming Google Gemini call."""
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
            {"role": "user", "parts": [{"text": user_text}]},
        ]

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

    async def _call_openai(self, user_text: str, tools: list, system_prompt: str, max_tokens: int) -> str:
        """Simple non-streaming OpenAI call."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        for iteration in range(5):
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            self._track_api_call("llm")
            response = await self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            if message.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in message.tool_calls]
                })

                for tc in message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except:
                        args = {}
                    result = await self._execute_tool(tc.function.name, args)
                    if isinstance(result, dict) and "response_text" in result:
                        content = result["response_text"]
                    else:
                        content = json.dumps(result, ensure_ascii=False)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
                continue

            if message.content:
                return message.content

        return "Could not get response."

    async def _call_anthropic(self, user_text: str, tools: list, system_prompt: str, max_tokens: int) -> str:
        """Simple non-streaming Anthropic call."""
        anthropic_tools = []
        for tool in tools:
            func = tool.get("function", {})
            anthropic_tools.append({
                "name": func.get("name"),
                "description": func.get("description"),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}})
            })

        messages = [{"role": "user", "content": user_text}]

        for iteration in range(5):
            payload = {
                "model": self.model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": messages,
            }
            if anthropic_tools:
                payload["tools"] = anthropic_tools

            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }

            self._track_api_call("llm")

            async with self._session.post(f"{self.base_url}/v1/messages", json=payload, headers=headers, timeout=120) as resp:
                if resp.status != 200:
                    return "Error calling Anthropic API."
                data = await resp.json()

            content_blocks = data.get("content", [])
            text_content = ""
            tool_uses = []

            for block in content_blocks:
                if block.get("type") == "text":
                    text_content += block.get("text", "")
                elif block.get("type") == "tool_use":
                    tool_uses.append(block)

            if tool_uses:
                messages.append({"role": "assistant", "content": content_blocks})
                tool_results = []

                for tu in tool_uses:
                    result = await self._execute_tool(tu["name"], tu.get("input", {}))
                    if isinstance(result, dict) and "response_text" in result:
                        content = result["response_text"]
                    else:
                        content = json.dumps(result, ensure_ascii=False)
                    tool_results.append({"type": "tool_result", "tool_use_id": tu["id"], "content": content})

                messages.append({"role": "user", "content": tool_results})
                continue

            if text_content:
                return text_content

        return "Could not get response."

    def _calculate_max_tokens(self, user_text: str) -> int:
        """Calculate max tokens based on query complexity."""
        base = self.max_tokens
        if len(user_text) < 30:
            return min(base, 500)
        elif len(user_text) < 100:
            return min(base, 1000)
        return base

    # =========================================================================
    # Streaming LLM Provider Methods
    # =========================================================================

    async def _stream_openai_compatible(
        self,
        user_text: str,
        tools: list[dict],
        system_prompt: str,
        max_tokens: int,
    ) -> AsyncGenerator[ContentDelta, None]:
        """Stream from OpenAI-compatible API with tool support."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        called_tools: set[str] = set()

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

                        # Accumulate tool calls
                        if delta.tool_calls:
                            for tc_delta in delta.tool_calls:
                                if tc_delta.index is not None:
                                    while len(tool_calls_buffer) <= tc_delta.index:
                                        tool_calls_buffer.append({
                                            "id": None,
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""}
                                        })

                                    current = tool_calls_buffer[tc_delta.index]
                                    if tc_delta.id:
                                        current["id"] = tc_delta.id
                                    if tc_delta.function:
                                        if tc_delta.function.name:
                                            current["function"]["name"] += tc_delta.function.name
                                        if tc_delta.function.arguments:
                                            current["function"]["arguments"] += tc_delta.function.arguments
                finally:
                    await stream.close()

                # Process tool calls if any
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

                if unique_tool_calls:
                    _LOGGER.info("Processing %d tool call(s)", len(unique_tool_calls))

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
                        else:
                            content = json.dumps(result, ensure_ascii=False)

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

                break

            except Exception as e:
                _LOGGER.error("OpenAI API error: %s", e)
                yield {"content": "Sorry, there was an error processing your request."}
                return

        # If we get here with no content, yield a fallback
        yield {"content": "I apologize, but I couldn't complete that request."}

    async def _stream_anthropic(
        self,
        user_text: str,
        tools: list[dict],
        system_prompt: str,
        max_tokens: int,
    ) -> AsyncGenerator[ContentDelta, None]:
        """Stream from Anthropic Claude API with SSE support."""
        # Convert tools to Anthropic format
        anthropic_tools = []
        for tool in tools:
            func = tool.get("function", {})
            anthropic_tools.append({
                "name": func.get("name"),
                "description": func.get("description"),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}})
            })

        messages = [{"role": "user", "content": user_text}]
        called_tools: set[str] = set()

        for iteration in range(5):
            payload = {
                "model": self.model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": messages,
                "stream": True,  # Enable streaming!
            }
            if anthropic_tools:
                payload["tools"] = anthropic_tools

            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }

            self._track_api_call("llm")

            try:
                async with self._session.post(
                    f"{self.base_url}/v1/messages",
                    json=payload,
                    headers=headers,
                    timeout=180,  # 3 minutes - VLM operations need more time
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        _LOGGER.error("Anthropic API error: %s", error)
                        yield {"content": "Sorry, there was an error with the AI service."}
                        return

                    # Process SSE stream
                    accumulated_content = ""
                    tool_calls = []
                    current_tool: dict | None = None

                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if not line or not line.startswith("data: "):
                            continue

                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str == "[DONE]":
                            break

                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event_type = event.get("type")

                        if event_type == "content_block_start":
                            block = event.get("content_block", {})
                            if block.get("type") == "tool_use":
                                current_tool = {
                                    "id": block.get("id"),
                                    "name": block.get("name"),
                                    "arguments": "",
                                }

                        elif event_type == "content_block_delta":
                            delta = event.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                accumulated_content += text
                                yield {"content": text}
                            elif delta.get("type") == "input_json_delta":
                                if current_tool:
                                    current_tool["arguments"] += delta.get("partial_json", "")

                        elif event_type == "content_block_stop":
                            if current_tool:
                                try:
                                    current_tool["arguments"] = json.loads(current_tool["arguments"])
                                except json.JSONDecodeError:
                                    current_tool["arguments"] = {}
                                tool_calls.append(current_tool)
                                current_tool = None

                # Process tool calls if any
                if tool_calls:
                    unique_tool_calls = []
                    for tc in tool_calls:
                        tool_key = f"{tc['name']}:{tc.get('arguments', '')}"
                        if tool_key not in called_tools:
                            called_tools.add(tool_key)
                            unique_tool_calls.append(tc)
                            _LOGGER.info("Tool call: %s(%s)", tc["name"], tc["arguments"])

                    if unique_tool_calls:
                        # Build assistant content for Anthropic format
                        assistant_content = []
                        if accumulated_content:
                            assistant_content.append({"type": "text", "text": accumulated_content})
                        for tc in unique_tool_calls:
                            assistant_content.append({
                                "type": "tool_use",
                                "id": tc["id"],
                                "name": tc["name"],
                                "input": tc["arguments"],
                            })

                        messages.append({"role": "assistant", "content": assistant_content})

                        # Execute tools in parallel
                        tool_tasks = [
                            self._execute_tool(tc["name"], tc["arguments"])
                            for tc in unique_tool_calls
                        ]
                        results = await asyncio.gather(*tool_tasks, return_exceptions=True)

                        # Add tool results
                        tool_results = []
                        for tc, result in zip(unique_tool_calls, results):
                            if isinstance(result, Exception):
                                content = json.dumps({"error": str(result)}, ensure_ascii=False)
                            elif isinstance(result, dict) and "response_text" in result:
                                content = result["response_text"]
                            else:
                                content = json.dumps(result, ensure_ascii=False)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tc["id"],
                                "content": content
                            })

                        messages.append({"role": "user", "content": tool_results})
                        continue

                # No tool calls - we're done
                if accumulated_content:
                    return

                break

            except Exception as e:
                _LOGGER.error("Anthropic API error: %s", e)
                yield {"content": "Sorry, there was an error processing your request."}
                return

        yield {"content": "I apologize, but I couldn't complete that request."}

    async def _stream_google(
        self,
        user_text: str,
        tools: list[dict],
        system_prompt: str,
        max_tokens: int,
    ) -> AsyncGenerator[ContentDelta, None]:
        """Stream from Google Gemini API."""
        # Convert tools to Gemini format
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
            {"role": "user", "parts": [{"text": user_text}]},
        ]

        called_tools: set[str] = set()

        for iteration in range(5):
            payload = {
                "contents": contents,
                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": self.temperature},
            }

            if function_declarations:
                payload["tools"] = [{"functionDeclarations": function_declarations}]

            # Use streaming endpoint
            url = f"{self.base_url}/models/{self.model}:streamGenerateContent?alt=sse"
            headers = {"x-goog-api-key": self.api_key}

            self._track_api_call("llm")

            try:
                async with self._session.post(url, json=payload, headers=headers, timeout=180) as response:  # 3 minutes - VLM operations need more time
                    if response.status != 200:
                        error = await response.text()
                        _LOGGER.error("Google API error: %s", error)
                        yield {"content": "Sorry, there was an error with the AI service."}
                        return

                    accumulated_content = ""
                    tool_calls = []
                    model_parts = []

                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if not line or not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        candidates = data.get("candidates", [])
                        if not candidates:
                            continue

                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])

                        for part in parts:
                            model_parts.append(part)
                            if "text" in part:
                                text = part["text"]
                                accumulated_content += text
                                yield {"content": text}
                            elif "functionCall" in part:
                                fc = part["functionCall"]
                                tool_calls.append({
                                    "name": fc.get("name"),
                                    "arguments": fc.get("args", {}),
                                })

                # Process tool calls if any
                if tool_calls:
                    unique_tool_calls = []
                    for tc in tool_calls:
                        tool_key = f"{tc['name']}:{tc.get('arguments', '')}"
                        if tool_key not in called_tools:
                            called_tools.add(tool_key)
                            unique_tool_calls.append(tc)
                            _LOGGER.info("Tool call: %s(%s)", tc["name"], tc["arguments"])

                    if unique_tool_calls:
                        contents.append({"role": "model", "parts": model_parts})

                        # Execute tools in parallel
                        tool_tasks = [
                            self._execute_tool(tc["name"], tc["arguments"])
                            for tc in unique_tool_calls
                        ]
                        results = await asyncio.gather(*tool_tasks, return_exceptions=True)

                        # Add function responses
                        function_responses = []
                        for tc, result in zip(unique_tool_calls, results):
                            if isinstance(result, Exception):
                                response_content = {"error": str(result)}
                            elif isinstance(result, dict) and "response_text" in result:
                                response_content = {"text": result["response_text"]}
                            else:
                                response_content = result
                            function_responses.append({
                                "functionResponse": {"name": tc["name"], "response": response_content}
                            })

                        contents.append({"role": "user", "parts": function_responses})
                        continue

                # No tool calls - we're done
                if accumulated_content:
                    return

                break

            except Exception as e:
                _LOGGER.error("Google API error: %s", e)
                yield {"content": "Sorry, there was an error processing your request."}
                return

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
            yelp_url = top_pick.get("yelp_url", "")
            coordinates = top_pick.get("coordinates", {})

            google_maps_url = ""
            if coordinates and coordinates.get("lat") and coordinates.get("lng"):
                lat, lng = coordinates["lat"], coordinates["lng"]
                google_maps_url = f"https://www.google.com/maps/dir/?api=1&destination={lat},{lng}"

            actions = []
            if yelp_url:
                actions.append({"action": "URI", "title": "⭐ #1 Yelp", "uri": yelp_url})
            if len(top_restaurants) > 1 and top_restaurants[1].get("yelp_url"):
                actions.append({"action": "URI", "title": "⭐ #2 Yelp", "uri": top_restaurants[1]["yelp_url"]})
            if len(top_restaurants) > 2 and top_restaurants[2].get("yelp_url"):
                actions.append({"action": "URI", "title": "⭐ #3 Yelp", "uri": top_restaurants[2]["yelp_url"]})
            if google_maps_url:
                actions.append({"action": "URI", "title": "🗺️ Directions #1", "uri": google_maps_url})

            notification_data = self._build_notification_data(
                title, message, actions, yelp_url or google_maps_url
            )
            await self._send_notification(notification_data, "restaurant")

        except Exception as err:
            _LOGGER.error("Error sending restaurant notification: %s", err, exc_info=True)

    async def _send_reservation_notification(self, reservation_result: dict[str, Any]) -> None:
        """Send notification with reservation link to configured devices."""
        try:
            restaurant_name = reservation_result.get("restaurant_name", "Restaurant")
            reservation_url = reservation_result.get("reservation_url", "")
            supports_reservation = reservation_result.get("supports_reservation", False)
            party_size = reservation_result.get("party_size", 2)
            date = reservation_result.get("date", "")
            time = reservation_result.get("time", "")
            phone = reservation_result.get("phone", "")
            address = reservation_result.get("address", "")

            _LOGGER.info("Sending reservation notification for: %s", restaurant_name)

            title = f"🍽️ Reserve at {restaurant_name}" if supports_reservation else f"📞 Book {restaurant_name}"

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
            if not supports_reservation and phone:
                message_parts.append(f"📞 {phone}")
            message = "\n".join(message_parts) if message_parts else f"Book a table at {restaurant_name}"

            actions = []
            if reservation_url:
                button_title = "📅 Reserve Now" if supports_reservation else "🔍 Search Reservations"
                actions.append({"action": "URI", "title": button_title, "uri": reservation_url})
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
                return await handler(arguments, self._session, hass_tz, self._track_api_call)

            # Wikipedia tools
            if tool_name in ("calculate_age", "get_wikipedia_summary"):
                handler = getattr(wikipedia_tool, tool_name)
                return await handler(arguments, self._session, self._track_api_call)

            # Reminder tools
            if tool_name in ("create_reminder", "get_reminders"):
                handler = getattr(reminders_tool, tool_name)
                return await handler(arguments, self.hass, hass_tz)

            # Simple tools with specific handlers
            simple_handlers = {
                "get_weather_forecast": lambda: weather_tool.get_weather_forecast(
                    arguments, self._session, self.openweathermap_api_key,
                    latitude, longitude, self._track_api_call
                ),
                "get_stock_price": lambda: stocks_tool.get_stock_price(
                    arguments, self._session, self._track_api_call
                ),
                "get_news": lambda: news_tool.get_news(
                    arguments, self._session, self.newsapi_key, hass_tz, self._track_api_call
                ),
                "get_calendar_events": lambda: calendar_tool.get_calendar_events(
                    arguments, self.hass, self.calendar_entities, hass_tz
                ),
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
            }

            if tool_name in simple_handlers:
                return await simple_handlers[tool_name]()

            # Tools with notification post-processing
            if tool_name == "find_nearby_places":
                result = await places_tool.find_nearby_places(
                    arguments, self._session, self.google_places_api_key,
                    latitude, longitude, self._track_api_call
                )
                if self.notify_on_places and self.notification_entities and result.get("places"):
                    await self._send_places_notification(result)
                return result

            if tool_name == "get_restaurant_recommendations":
                result = await places_tool.get_restaurant_recommendations(
                    arguments, self._session, self.yelp_api_key,
                    latitude, longitude, self._track_api_call
                )
                if self.notify_on_restaurants and self.notification_entities and result.get("restaurants"):
                    await self._send_restaurant_notification(result)
                return result

            if tool_name == "book_restaurant":
                result = await places_tool.book_restaurant(
                    arguments, self._session, self.yelp_api_key,
                    latitude, longitude, self._track_api_call
                )
                if self.notification_entities and result.get("reservation_url"):
                    await self._send_reservation_notification(result)
                return result

            # Camera tools with notification
            if tool_name in ("check_camera", "quick_camera_check"):
                handler = getattr(camera_tool, tool_name)
                result = await handler(arguments, self.hass, self.camera_friendly_names or None)
                if self.notify_on_camera and self.notification_entities and result.get("snapshot_url"):
                    await self._send_camera_notification(result)
                return result

            # Conditional tools (require configuration)
            if tool_name == "control_sofabaton":
                if not self.sofabaton_activities:
                    return {"error": "No SofaBaton activities configured"}
                session = async_get_clientsession(self.hass)
                return await sofabaton_tool.control_sofabaton(
                    arguments, session, self.sofabaton_activities
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
