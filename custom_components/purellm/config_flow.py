"""Config flow for PureLLM integration."""
from __future__ import annotations

import json
import logging
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector
import homeassistant.helpers.config_validation as cv

from .const import (
    DOMAIN,
    # Provider settings
    CONF_PROVIDER,
    CONF_BASE_URL,
    CONF_API_KEY,
    CONF_MODEL,
    CONF_TEMPERATURE,
    CONF_MAX_TOKENS,
    CONF_TOP_P,
    ALL_PROVIDERS,
    PROVIDER_NAMES,
    PROVIDER_BASE_URLS,
    PROVIDER_DEFAULT_MODELS,
    PROVIDER_MODELS,
    PROVIDER_LM_STUDIO,
    PROVIDER_GOOGLE,
    DEFAULT_PROVIDER,
    DEFAULT_BASE_URL,
    DEFAULT_API_KEY,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P,
    # System settings
    CONF_SYSTEM_PROMPT,
    CONF_CUSTOM_LATITUDE,
    CONF_CUSTOM_LONGITUDE,
    CONF_OPENWEATHERMAP_API_KEY,
    CONF_GOOGLE_PLACES_API_KEY,
    CONF_TAVILY_API_KEY,
    # Feature toggles
    CONF_ENABLE_WEATHER,
    CONF_ENABLE_CALENDAR,
    CONF_ENABLE_CAMERAS,
    CONF_ENABLE_SPORTS,
    CONF_ENABLE_PLACES,
    CONF_ENABLE_RESTAURANTS,
    CONF_ENABLE_THERMOSTAT,
    CONF_ENABLE_DEVICE_STATUS,
    CONF_ENABLE_WIKIPEDIA,
    CONF_ENABLE_MUSIC,
    CONF_ENABLE_SEARCH,
    # Entity config
    CONF_THERMOSTAT_ENTITY,
    CONF_CALENDAR_ENTITIES,
    CONF_ROOM_PLAYER_MAPPING,
    CONF_DEVICE_ALIASES,
    CONF_CAMERA_ENTITIES,
    # Thermostat settings
    CONF_THERMOSTAT_MIN_TEMP,
    CONF_THERMOSTAT_MAX_TEMP,
    CONF_THERMOSTAT_TEMP_STEP,
    CONF_THERMOSTAT_USE_CELSIUS,
    # Defaults
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_CUSTOM_LATITUDE,
    DEFAULT_CUSTOM_LONGITUDE,
    DEFAULT_OPENWEATHERMAP_API_KEY,
    DEFAULT_GOOGLE_PLACES_API_KEY,
    DEFAULT_TAVILY_API_KEY,
    DEFAULT_ENABLE_WEATHER,
    DEFAULT_ENABLE_CALENDAR,
    DEFAULT_ENABLE_CAMERAS,
    DEFAULT_ENABLE_SPORTS,
    DEFAULT_ENABLE_PLACES,
    DEFAULT_ENABLE_RESTAURANTS,
    DEFAULT_ENABLE_THERMOSTAT,
    DEFAULT_ENABLE_DEVICE_STATUS,
    DEFAULT_ENABLE_WIKIPEDIA,
    DEFAULT_ENABLE_MUSIC,
    DEFAULT_ENABLE_SEARCH,
    DEFAULT_THERMOSTAT_ENTITY,
    DEFAULT_CALENDAR_ENTITIES,
    DEFAULT_ROOM_PLAYER_MAPPING,
    DEFAULT_DEVICE_ALIASES,
    DEFAULT_CAMERA_ENTITIES,
    # Thermostat defaults
    DEFAULT_THERMOSTAT_MIN_TEMP,
    DEFAULT_THERMOSTAT_MAX_TEMP,
    DEFAULT_THERMOSTAT_TEMP_STEP,
    DEFAULT_THERMOSTAT_USE_CELSIUS,
    DEFAULT_THERMOSTAT_MIN_TEMP_CELSIUS,
    DEFAULT_THERMOSTAT_MAX_TEMP_CELSIUS,
    DEFAULT_THERMOSTAT_TEMP_STEP_CELSIUS,
    # Notifications
    CONF_NOTIFICATION_ENTITIES,
    CONF_NOTIFY_ON_PLACES,
    CONF_NOTIFY_ON_RESTAURANTS,
    CONF_NOTIFY_ON_SEARCH,
    DEFAULT_NOTIFICATION_ENTITIES,
    DEFAULT_NOTIFY_ON_PLACES,
    DEFAULT_NOTIFY_ON_RESTAURANTS,
    DEFAULT_NOTIFY_ON_SEARCH,
    # Voice Scripts
    CONF_VOICE_SCRIPTS,
    DEFAULT_VOICE_SCRIPTS,
    # Camera Friendly Names
    CONF_CAMERA_FRIENDLY_NAMES,
    DEFAULT_CAMERA_FRIENDLY_NAMES,
    # SofaBaton Activities
    CONF_SOFABATON_ACTIVITIES,
    DEFAULT_SOFABATON_ACTIVITIES,
    # Cast/Chromecast Settings
    CONF_WAKE_CAST_BEFORE_PLAY,
    DEFAULT_WAKE_CAST_BEFORE_PLAY,
    CONF_WAKE_CAST_ADB_ENTITY,
    DEFAULT_WAKE_CAST_ADB_ENTITY,
)

_LOGGER = logging.getLogger(__name__)

# Providers that support dynamic model fetching via /models endpoint
DYNAMIC_MODEL_PROVIDERS = [
    PROVIDER_LM_STUDIO,
    PROVIDER_GOOGLE,
]


async def fetch_provider_models(
    provider: str, base_url: str, api_key: str
) -> list[str]:
    """Fetch available models from provider API.

    Returns a list of model IDs, or empty list if fetch fails.
    """
    if provider not in DYNAMIC_MODEL_PROVIDERS:
        return []

    try:
        async with aiohttp.ClientSession() as session:
            # Google Gemini uses different API format
            if provider == PROVIDER_GOOGLE:
                # Request max models per page - use header auth instead of URL param
                url = f"{base_url}/models?pageSize=1000"
                headers = {"x-goog-api-key": api_key}
            else:
                # OpenAI-compatible providers
                url = f"{base_url}/models"
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status != 200:
                    _LOGGER.warning(
                        "Failed to fetch models from %s: status %s",
                        provider, response.status
                    )
                    return []

                data = await response.json()

                # Google returns {"models": [...]} with "name" field
                # OpenAI-compatible returns {"data": [...]} with "id" field
                if provider == PROVIDER_GOOGLE:
                    models = data.get("models", [])
                    _LOGGER.warning("Google API returned %d total models", len(models))
                    model_ids = []
                    for model in models:
                        # Google format: "models/gemini-1.5-pro" -> "gemini-1.5-pro"
                        name = model.get("name", "")
                        if not name:
                            continue

                        # Extract model ID from full name
                        model_id = name.replace("models/", "")

                        # Include all gemini models (filter out embedding/aqa/imagen)
                        lower_id = model_id.lower()
                        if any(skip in lower_id for skip in ["embedding", "aqa", "imagen"]):
                            continue

                        model_ids.append(model_id)

                    _LOGGER.warning("Google filtered to %d gemini models: %s", len(model_ids), model_ids[:10])
                else:
                    # OpenAI-compatible providers
                    models = data.get("data", [])
                    model_ids = []
                    for model in models:
                        model_id = model.get("id", "")
                        if not model_id:
                            continue

                        # Skip non-chat models
                        lower_id = model_id.lower()
                        if any(skip in lower_id for skip in [
                            "whisper", "tts", "embedding", "embed",
                            "guard", "safeguard", "moderation",
                            "audio", "speech", "vision-preview"
                        ]):
                            continue

                        model_ids.append(model_id)

                # Sort alphabetically for better UX
                model_ids.sort()
                _LOGGER.info("Fetched %d models from %s", len(model_ids), provider)
                return model_ids

    except Exception as e:
        _LOGGER.warning("Error fetching models from %s: %s", provider, e)
        return []


class PureLLMConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for PureLLM."""

    VERSION = 2

    def __init__(self) -> None:
        """Initialize config flow."""
        self._data: dict[str, Any] = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step - provider selection."""
        if user_input is not None:
            self._data[CONF_PROVIDER] = user_input[CONF_PROVIDER]
            return await self.async_step_credentials()

        # Build provider options for selector
        provider_options = [
            selector.SelectOptionDict(value=p, label=PROVIDER_NAMES[p])
            for p in ALL_PROVIDERS
        ]

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_PROVIDER, default=DEFAULT_PROVIDER): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=provider_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                }
            ),
        )

    async def async_step_credentials(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle credentials step."""
        errors = {}
        provider = self._data.get(CONF_PROVIDER, DEFAULT_PROVIDER)
        
        if user_input is not None:
            self._data.update(user_input)
            
            # Set base URL based on provider if not custom
            if not user_input.get(CONF_BASE_URL):
                self._data[CONF_BASE_URL] = PROVIDER_BASE_URLS.get(provider, DEFAULT_BASE_URL)
            
            # Validate connection
            try:
                valid = await self._test_connection(
                    provider,
                    self._data.get(CONF_BASE_URL, PROVIDER_BASE_URLS.get(provider)),
                    self._data.get(CONF_API_KEY, ""),
                )
                if not valid:
                    errors["base"] = "cannot_connect"
            except Exception as e:
                _LOGGER.error("Connection test failed: %s", e)
                errors["base"] = "cannot_connect"
            
            if not errors:
                return self.async_create_entry(
                    title=f"PureLLM ({PROVIDER_NAMES[provider]})",
                    data=self._data,
                )

        # Get defaults for this provider
        default_url = PROVIDER_BASE_URLS.get(provider, DEFAULT_BASE_URL)
        default_model = PROVIDER_DEFAULT_MODELS.get(provider, DEFAULT_MODEL)

        # Show different fields based on provider
        # Show URL for local providers (LM Studio/vLLM)
        show_base_url = provider == PROVIDER_LM_STUDIO

        schema_dict = {}

        if show_base_url:
            schema_dict[vol.Required(CONF_BASE_URL, default=default_url)] = str

        # API key: placeholder for LM Studio, required for others
        if provider == PROVIDER_LM_STUDIO:
            schema_dict[vol.Required(CONF_API_KEY, default="lm-studio")] = str
        else:
            schema_dict[vol.Required(CONF_API_KEY, default="")] = str

        # Build model options for the current provider
        provider_models = PROVIDER_MODELS.get(provider, [default_model])
        model_options = [
            selector.SelectOptionDict(value=m, label=m)
            for m in provider_models
        ]

        schema_dict[vol.Required(CONF_MODEL, default=default_model)] = selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=model_options,
                mode=selector.SelectSelectorMode.DROPDOWN,
                custom_value=True,
            )
        )

        return self.async_show_form(
            step_id="credentials",
            data_schema=vol.Schema(schema_dict),
            errors=errors,
            description_placeholders={
                "provider": PROVIDER_NAMES[provider],
            },
        )

    async def _test_connection(self, provider: str, base_url: str, api_key: str) -> bool:
        """Test connection to the LLM provider."""
        try:
            async with aiohttp.ClientSession() as session:
                if provider == PROVIDER_GOOGLE:
                    # Google Gemini
                    async with session.get(
                        f"{base_url}/models?key={api_key}",
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        return response.status in (200, 400, 403)
                else:
                    # OpenAI-compatible (LM Studio / vLLM)
                    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                    async with session.get(
                        f"{base_url}/models",
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        return response.status in (200, 401)
        except Exception as e:
            _LOGGER.warning("Connection test exception: %s", e)
            # For local providers, connection refused is expected if server isn't running
            if provider == PROVIDER_LM_STUDIO:
                return True  # Allow setup even if server isn't running
            return False

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return PureLLMOptionsFlowHandler(config_entry)


class PureLLMOptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for PureLLM."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self._entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        return self.async_show_menu(
            step_id="init",
            menu_options={
                "connection": "Connection Settings",
                "model": "Model Settings",
                "features": "Enable/Disable Features",
                "entities": "PureLLM Default Entities",
                "device_aliases": "Device Aliases",
                "voice_scripts": "Voice Scripts",
                "camera_names": "Camera Friendly Names",
                "sofabaton": "SofaBaton Activities",
                "music_rooms": "Music Room Mapping",
                "notifications": "Notification Settings",
                "api_keys": "API Keys",
                "location": "Location Settings",
                "advanced": "System Prompt",
            },
        )

    async def async_step_connection(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle connection settings including provider change."""
        if user_input is not None:
            # If provider changed, update base_url to default for new provider
            new_provider = user_input.get(CONF_PROVIDER)
            old_provider = self._entry.data.get(CONF_PROVIDER) or self._entry.options.get(CONF_PROVIDER)
            
            if new_provider and new_provider != old_provider:
                # Set default URL for new provider unless custom URL provided
                if not user_input.get(CONF_BASE_URL) or user_input.get(CONF_BASE_URL) == PROVIDER_BASE_URLS.get(old_provider):
                    user_input[CONF_BASE_URL] = PROVIDER_BASE_URLS.get(new_provider, DEFAULT_BASE_URL)
            
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}
        current_provider = current.get(CONF_PROVIDER, DEFAULT_PROVIDER)

        # Build provider options for selector
        provider_options = [
            selector.SelectOptionDict(value=p, label=PROVIDER_NAMES[p])
            for p in ALL_PROVIDERS
        ]

        return self.async_show_form(
            step_id="connection",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_PROVIDER,
                        default=current_provider,
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=provider_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Required(
                        CONF_BASE_URL,
                        default=current.get(CONF_BASE_URL, PROVIDER_BASE_URLS.get(current_provider, DEFAULT_BASE_URL)),
                    ): str,
                    vol.Required(
                        CONF_API_KEY,
                        default=current.get(CONF_API_KEY, DEFAULT_API_KEY),
                    ): str,
                }
            ),
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle model settings."""
        if user_input is not None:
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}
        current_provider = current.get(CONF_PROVIDER, DEFAULT_PROVIDER)
        current_model = current.get(CONF_MODEL, DEFAULT_MODEL)
        current_base_url = current.get(CONF_BASE_URL, PROVIDER_BASE_URLS.get(current_provider, DEFAULT_BASE_URL))
        current_api_key = current.get(CONF_API_KEY, "")

        # Try to fetch models dynamically from the provider API
        provider_models = await fetch_provider_models(
            current_provider, current_base_url, current_api_key
        )

        # Fall back to static list if dynamic fetch fails or returns empty
        if not provider_models:
            provider_models = list(PROVIDER_MODELS.get(current_provider, [DEFAULT_MODEL]))
            _LOGGER.debug("Using static model list for %s", current_provider)

        # Ensure current model is in the list (for custom models)
        if current_model and current_model not in provider_models:
            provider_models = [current_model] + provider_models

        model_options = [
            selector.SelectOptionDict(value=m, label=m)
            for m in provider_models
        ]

        return self.async_show_form(
            step_id="model",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_MODEL,
                        default=current_model,
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=model_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                            custom_value=True,
                        )
                    ),
                    vol.Optional(
                        CONF_TEMPERATURE,
                        default=current.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                    ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0)),
                    vol.Optional(
                        CONF_MAX_TOKENS,
                        default=current.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
                    ): cv.positive_int,
                    vol.Optional(
                        CONF_TOP_P,
                        default=current.get(CONF_TOP_P, DEFAULT_TOP_P),
                    ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
                }
            ),
        )

    async def async_step_features(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle feature toggles."""
        if user_input is not None:
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        return self.async_show_form(
            step_id="features",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_ENABLE_WEATHER,
                        default=current.get(CONF_ENABLE_WEATHER, DEFAULT_ENABLE_WEATHER),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_ENABLE_CALENDAR,
                        default=current.get(CONF_ENABLE_CALENDAR, DEFAULT_ENABLE_CALENDAR),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_ENABLE_CAMERAS,
                        default=current.get(CONF_ENABLE_CAMERAS, DEFAULT_ENABLE_CAMERAS),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_ENABLE_SPORTS,
                        default=current.get(CONF_ENABLE_SPORTS, DEFAULT_ENABLE_SPORTS),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_ENABLE_PLACES,
                        default=current.get(CONF_ENABLE_PLACES, DEFAULT_ENABLE_PLACES),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_ENABLE_RESTAURANTS,
                        default=current.get(CONF_ENABLE_RESTAURANTS, DEFAULT_ENABLE_RESTAURANTS),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_ENABLE_THERMOSTAT,
                        default=current.get(CONF_ENABLE_THERMOSTAT, DEFAULT_ENABLE_THERMOSTAT),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_ENABLE_DEVICE_STATUS,
                        default=current.get(CONF_ENABLE_DEVICE_STATUS, DEFAULT_ENABLE_DEVICE_STATUS),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_ENABLE_WIKIPEDIA,
                        default=current.get(CONF_ENABLE_WIKIPEDIA, DEFAULT_ENABLE_WIKIPEDIA),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_ENABLE_MUSIC,
                        default=current.get(CONF_ENABLE_MUSIC, DEFAULT_ENABLE_MUSIC),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_ENABLE_SEARCH,
                        default=current.get(CONF_ENABLE_SEARCH, DEFAULT_ENABLE_SEARCH),
                    ): cv.boolean,
                }
            ),
        )

    async def async_step_entities(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle entity configuration."""
        if user_input is not None:
            # Convert entity lists to the format we need
            processed_input = {}

            # Handle thermostat - single entity
            if CONF_THERMOSTAT_ENTITY in user_input:
                processed_input[CONF_THERMOSTAT_ENTITY] = user_input[CONF_THERMOSTAT_ENTITY]

            # Handle calendars - convert list to newline-separated string
            if CONF_CALENDAR_ENTITIES in user_input:
                cal_list = user_input[CONF_CALENDAR_ENTITIES]
                if isinstance(cal_list, list):
                    processed_input[CONF_CALENDAR_ENTITIES] = "\n".join(cal_list)
                else:
                    processed_input[CONF_CALENDAR_ENTITIES] = cal_list

            # Handle cameras - convert list to newline-separated string
            if CONF_CAMERA_ENTITIES in user_input:
                cam_list = user_input[CONF_CAMERA_ENTITIES]
                if isinstance(cam_list, list):
                    processed_input[CONF_CAMERA_ENTITIES] = "\n".join(cam_list)
                else:
                    processed_input[CONF_CAMERA_ENTITIES] = cam_list

            # Handle thermostat settings
            if CONF_THERMOSTAT_MIN_TEMP in user_input:
                processed_input[CONF_THERMOSTAT_MIN_TEMP] = user_input[CONF_THERMOSTAT_MIN_TEMP]
            if CONF_THERMOSTAT_MAX_TEMP in user_input:
                processed_input[CONF_THERMOSTAT_MAX_TEMP] = user_input[CONF_THERMOSTAT_MAX_TEMP]
            if CONF_THERMOSTAT_TEMP_STEP in user_input:
                processed_input[CONF_THERMOSTAT_TEMP_STEP] = user_input[CONF_THERMOSTAT_TEMP_STEP]
            if CONF_THERMOSTAT_USE_CELSIUS in user_input:
                processed_input[CONF_THERMOSTAT_USE_CELSIUS] = user_input[CONF_THERMOSTAT_USE_CELSIUS]

            new_options = {**self._entry.options, **processed_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        # Parse current calendar entities back to list
        current_calendars = current.get(CONF_CALENDAR_ENTITIES, DEFAULT_CALENDAR_ENTITIES)
        if isinstance(current_calendars, str) and current_calendars:
            current_calendars = [c.strip() for c in current_calendars.split("\n") if c.strip()]
        elif not current_calendars:
            current_calendars = []

        # Parse current camera entities back to list
        current_cameras = current.get(CONF_CAMERA_ENTITIES, DEFAULT_CAMERA_ENTITIES)
        if isinstance(current_cameras, str) and current_cameras:
            current_cameras = [c.strip() for c in current_cameras.split("\n") if c.strip()]
        elif not current_cameras:
            current_cameras = []

        # Determine if using Celsius and set appropriate defaults/ranges
        use_celsius = current.get(CONF_THERMOSTAT_USE_CELSIUS, DEFAULT_THERMOSTAT_USE_CELSIUS)
        if use_celsius:
            temp_unit = "°C"
            temp_min_range = 0
            temp_max_range = 50
            default_min = DEFAULT_THERMOSTAT_MIN_TEMP_CELSIUS
            default_max = DEFAULT_THERMOSTAT_MAX_TEMP_CELSIUS
            default_step = DEFAULT_THERMOSTAT_TEMP_STEP_CELSIUS
        else:
            temp_unit = "°F"
            temp_min_range = 32
            temp_max_range = 100
            default_min = DEFAULT_THERMOSTAT_MIN_TEMP
            default_max = DEFAULT_THERMOSTAT_MAX_TEMP
            default_step = DEFAULT_THERMOSTAT_TEMP_STEP

        # Get current values, using appropriate defaults if not set or if switching units
        current_min = current.get(CONF_THERMOSTAT_MIN_TEMP)
        current_max = current.get(CONF_THERMOSTAT_MAX_TEMP)
        current_step = current.get(CONF_THERMOSTAT_TEMP_STEP)

        # If values look like wrong unit (e.g., 60-85 with Celsius enabled), use unit defaults
        if use_celsius:
            if current_min is not None and current_min > 40:  # Likely Fahrenheit value
                current_min = default_min
            if current_max is not None and current_max > 50:  # Likely Fahrenheit value
                current_max = default_max
        else:
            if current_min is not None and current_min < 32:  # Likely Celsius value
                current_min = default_min
            if current_max is not None and current_max < 50:  # Likely Celsius value
                current_max = default_max

        # Use defaults if not set
        if current_min is None:
            current_min = default_min
        if current_max is None:
            current_max = default_max
        if current_step is None:
            current_step = default_step

        return self.async_show_form(
            step_id="entities",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_THERMOSTAT_ENTITY,
                        default=current.get(CONF_THERMOSTAT_ENTITY, DEFAULT_THERMOSTAT_ENTITY),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="climate",
                            multiple=False,
                        )
                    ),
                    vol.Optional(
                        CONF_THERMOSTAT_USE_CELSIUS,
                        default=use_celsius,
                    ): cv.boolean,
                    vol.Optional(
                        CONF_CALENDAR_ENTITIES,
                        default=current_calendars,
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="calendar",
                            multiple=True,
                        )
                    ),
                    vol.Optional(
                        CONF_CAMERA_ENTITIES,
                        default=current_cameras,
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="camera",
                            multiple=True,
                        )
                    ),
                    vol.Optional(
                        CONF_THERMOSTAT_MIN_TEMP,
                        default=current_min,
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=temp_min_range,
                            max=temp_max_range,
                            step=1,
                            unit_of_measurement=temp_unit,
                            mode=selector.NumberSelectorMode.BOX,
                        )
                    ),
                    vol.Optional(
                        CONF_THERMOSTAT_MAX_TEMP,
                        default=current_max,
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=temp_min_range,
                            max=temp_max_range,
                            step=1,
                            unit_of_measurement=temp_unit,
                            mode=selector.NumberSelectorMode.BOX,
                        )
                    ),
                    vol.Optional(
                        CONF_THERMOSTAT_TEMP_STEP,
                        default=current_step,
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            max=10,
                            step=1,
                            unit_of_measurement=temp_unit,
                            mode=selector.NumberSelectorMode.BOX,
                        )
                    ),
                }
            ),
        )

    async def async_step_device_aliases(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle device aliases configuration with add/edit/delete support.

        Device aliases allow users to define custom voice names for entities.
        Uses entity selector dropdown to prevent typos.
        """
        current = {**self._entry.data, **self._entry.options}
        current_aliases_raw = current.get(CONF_DEVICE_ALIASES, DEFAULT_DEVICE_ALIASES)

        # Parse current aliases - support both old text format and new JSON format
        aliases_list: list[dict[str, str]] = []
        if current_aliases_raw:
            try:
                # Try JSON format first (new format)
                aliases_list = json.loads(current_aliases_raw)
                if not isinstance(aliases_list, list):
                    aliases_list = []
            except (json.JSONDecodeError, TypeError):
                # Fall back to old text format: "alias:entity_id" per line
                for line in current_aliases_raw.strip().split("\n"):
                    line = line.strip()
                    if ":" in line:
                        alias, entity_id = line.split(":", 1)
                        aliases_list.append({
                            "alias": alias.strip().lower(),
                            "entity": entity_id.strip()
                        })

        if user_input is not None:
            selected = user_input.get("select_alias", "")
            alias_name = user_input.get("alias_name", "").strip().lower()
            entity_id = user_input.get("entity_id", "")
            action = user_input.get("action", "add")

            # Find index of selected alias
            selected_idx = None
            if selected:
                for i, a in enumerate(aliases_list):
                    if a.get("alias", "").lower() == selected.lower():
                        selected_idx = i
                        break

            if action == "delete" and selected_idx is not None:
                # Delete selected alias
                aliases_list.pop(selected_idx)
            elif action == "update" and selected_idx is not None:
                # Update selected alias
                aliases_list[selected_idx] = {
                    "alias": alias_name if alias_name else selected,
                    "entity": entity_id if entity_id else aliases_list[selected_idx].get("entity", ""),
                }
            elif action == "add" and alias_name and entity_id:
                # Add new alias - check for duplicates first
                existing = [a for a in aliases_list if a.get("alias", "").lower() == alias_name]
                if existing:
                    # Update existing instead of adding duplicate
                    for a in aliases_list:
                        if a.get("alias", "").lower() == alias_name:
                            a["entity"] = entity_id
                            break
                else:
                    aliases_list.append({
                        "alias": alias_name,
                        "entity": entity_id,
                    })
            elif not selected and not alias_name and not entity_id:
                # Empty submit - return to menu
                return self.async_create_entry(title="", data=self._entry.options)

            # Save updated aliases as JSON
            updated_json = json.dumps(aliases_list)
            new_options = {**self._entry.options, CONF_DEVICE_ALIASES: updated_json}
            self.hass.config_entries.async_update_entry(self._entry, options=new_options)

            # Rebuild list for display
            try:
                aliases_list = json.loads(updated_json)
            except (json.JSONDecodeError, TypeError):
                aliases_list = []

        # Build description showing current aliases
        if aliases_list:
            mapping_lines = []
            for a in aliases_list:
                alias = a.get("alias", "")
                entity = a.get("entity", "")
                mapping_lines.append(f"**{alias}** → {entity}")
            description = "**Current device aliases:**\n" + "\n".join(mapping_lines) + "\n\nSelect one to edit/delete, or add a new one below."
        else:
            description = "No device aliases configured. Add your first alias below.\n\n**Examples:**\n- 'receiver' → media_player.living_room_receiver\n- 'front door' → lock.front_door_lock\n- 'kitchen light' → light.kitchen_ceiling"

        # Build select options for existing aliases
        select_options = []
        for a in aliases_list:
            alias = a.get("alias", "")
            entity = a.get("entity", "")
            select_options.append(selector.SelectOptionDict(value=alias, label=f"{alias} → {entity}"))

        return self.async_show_form(
            step_id="device_aliases",
            description_placeholders={"aliases_info": description},
            data_schema=vol.Schema(
                {
                    vol.Optional("select_alias"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=select_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional("alias_name"): selector.TextSelector(
                        selector.TextSelectorConfig(
                            type=selector.TextSelectorType.TEXT,
                        )
                    ),
                    vol.Optional("entity_id"): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            multiple=False,
                        )
                    ),
                    vol.Optional("action", default="add"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                selector.SelectOptionDict(value="add", label="Add New"),
                                selector.SelectOptionDict(value="update", label="Update Selected"),
                                selector.SelectOptionDict(value="delete", label="Delete Selected"),
                            ],
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                }
            ),
        )

    async def async_step_voice_scripts(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle voice scripts configuration with add/edit/delete support.

        Voice scripts map trigger phrases to script entities for custom voice commands.
        Supports bidirectional scripts (open/close) with optional sensor state checking.
        """
        current = {**self._entry.data, **self._entry.options}
        current_scripts_json = current.get(CONF_VOICE_SCRIPTS, DEFAULT_VOICE_SCRIPTS)

        # Parse current scripts from JSON
        try:
            scripts_list = json.loads(current_scripts_json) if current_scripts_json else []
        except (json.JSONDecodeError, TypeError):
            scripts_list = []

        if user_input is not None:
            selected = user_input.get("select_script", "")
            trigger = user_input.get("trigger_phrase", "").strip().lower()
            open_script = user_input.get("open_script", "")
            close_script = user_input.get("close_script", "")
            sensor = user_input.get("sensor_entity", "")
            action = user_input.get("action", "add")

            # Find index of selected script
            selected_idx = None
            if selected:
                for i, s in enumerate(scripts_list):
                    if s.get("trigger", "").lower() == selected.lower():
                        selected_idx = i
                        break

            if action == "delete" and selected_idx is not None:
                # Delete selected script
                scripts_list.pop(selected_idx)
            elif action == "update" and selected_idx is not None:
                # Update selected script
                scripts_list[selected_idx] = {
                    "trigger": trigger if trigger else selected,
                    "open_script": open_script,
                    "close_script": close_script,
                    "sensor": sensor,
                }
            elif action == "add" and trigger and (open_script or close_script):
                # Add new script mapping
                scripts_list.append({
                    "trigger": trigger,
                    "open_script": open_script,
                    "close_script": close_script,
                    "sensor": sensor,
                })
            elif not selected and not trigger and not open_script:
                # Empty submit - return to menu
                return self.async_create_entry(title="", data=self._entry.options)

            # Save updated scripts as JSON
            updated_json = json.dumps(scripts_list)
            new_options = {**self._entry.options, CONF_VOICE_SCRIPTS: updated_json}
            self.hass.config_entries.async_update_entry(self._entry, options=new_options)

            # Rebuild list for display
            try:
                scripts_list = json.loads(updated_json)
            except (json.JSONDecodeError, TypeError):
                scripts_list = []

        # Build description showing current scripts
        if scripts_list:
            mapping_lines = []
            for s in scripts_list:
                trigger = s.get("trigger", "")
                open_s = s.get("open_script", "")
                close_s = s.get("close_script", "")
                sensor = s.get("sensor", "")
                parts = []
                if open_s:
                    parts.append(f"open: {open_s}")
                if close_s:
                    parts.append(f"close: {close_s}")
                if sensor:
                    parts.append(f"sensor: {sensor}")
                mapping_lines.append(f"**{trigger}** → {', '.join(parts)}")
            description = "**Current voice scripts:**\n" + "\n".join(mapping_lines) + "\n\nSelect one to edit/delete, or add a new one below."
        else:
            description = "No voice scripts configured. Add your first voice script below.\n\n**Examples:**\n- Garage door: Set trigger 'garage', open script for opening, close script for closing, and sensor to check state\n- Mute: Set trigger 'mute', open script to your mute-all script\n- Pause: Set trigger 'pause', open script to your pause-all script"

        # Build select options for existing scripts
        select_options = []
        for s in scripts_list:
            trigger = s.get("trigger", "")
            open_s = s.get("open_script", "")
            close_s = s.get("close_script", "")
            label_parts = [trigger]
            if open_s:
                label_parts.append(f"→ {open_s}")
            select_options.append(selector.SelectOptionDict(value=trigger, label=" ".join(label_parts)))

        return self.async_show_form(
            step_id="voice_scripts",
            description_placeholders={"scripts_info": description},
            data_schema=vol.Schema(
                {
                    vol.Optional("select_script"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=select_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional("trigger_phrase"): selector.TextSelector(
                        selector.TextSelectorConfig(
                            type=selector.TextSelectorType.TEXT,
                        )
                    ),
                    vol.Optional("open_script"): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="script",
                            multiple=False,
                        )
                    ),
                    vol.Optional("close_script"): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="script",
                            multiple=False,
                        )
                    ),
                    vol.Optional("sensor_entity"): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="binary_sensor",
                            multiple=False,
                        )
                    ),
                    vol.Optional("action", default="add"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                selector.SelectOptionDict(value="add", label="Add New"),
                                selector.SelectOptionDict(value="update", label="Update Selected"),
                                selector.SelectOptionDict(value="delete", label="Delete Selected"),
                            ],
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                }
            ),
        )

    async def async_step_camera_names(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle camera friendly names configuration with dropdown selection.

        Camera friendly names map camera entity IDs to human-friendly names.
        Uses entity selector dropdown to prevent typos.
        """
        current = {**self._entry.data, **self._entry.options}
        current_names_str = current.get(CONF_CAMERA_FRIENDLY_NAMES, DEFAULT_CAMERA_FRIENDLY_NAMES)

        # Parse current mappings into dict {entity_id: friendly_name}
        names_dict = {}
        if current_names_str:
            for line in current_names_str.split("\n"):
                line = line.strip()
                if ": " in line:
                    entity_id, friendly_name = line.split(": ", 1)
                    names_dict[entity_id.strip()] = friendly_name.strip()

        if user_input is not None:
            selected = user_input.get("select_camera", "")
            new_camera = user_input.get("camera_entity", "")
            new_name = user_input.get("friendly_name", "").strip()
            action = user_input.get("action", "add")

            if action == "delete" and selected:
                # Delete selected camera mapping
                if selected in names_dict:
                    del names_dict[selected]
            elif action == "update" and selected:
                # Update selected camera
                if selected in names_dict:
                    del names_dict[selected]
                camera_key = new_camera if new_camera else selected
                if new_name:
                    names_dict[camera_key] = new_name
            elif action == "add" and new_camera and new_name:
                # Add new camera mapping
                names_dict[new_camera] = new_name
            elif not selected and not new_camera and not new_name:
                # Empty submit - return to menu
                return self.async_create_entry(title="", data=self._entry.options)

            # Save updated mappings
            if names_dict:
                updated_mapping = "\n".join([f"{k}: {v}" for k, v in names_dict.items()])
            else:
                updated_mapping = ""

            new_options = {**self._entry.options, CONF_CAMERA_FRIENDLY_NAMES: updated_mapping}
            self.hass.config_entries.async_update_entry(self._entry, options=new_options)

            # Reload for display
            current_names_str = updated_mapping
            names_dict = {}
            if current_names_str:
                for line in current_names_str.split("\n"):
                    line = line.strip()
                    if ": " in line:
                        entity_id, friendly_name = line.split(": ", 1)
                        names_dict[entity_id.strip()] = friendly_name.strip()

        # Build description showing current mappings
        if names_dict:
            mapping_lines = [f"**{eid}** → {name}" for eid, name in names_dict.items()]
            description = "**Current camera names:**\n" + "\n".join(mapping_lines) + "\n\nSelect one to edit/delete, or add a new one below."
        else:
            description = "No camera names configured. Add your first camera mapping below.\n\nSelect a camera from the dropdown and give it a friendly name like 'Front Porch' or 'Driveway'."

        # Build select options for existing mappings
        select_options = [
            selector.SelectOptionDict(value=eid, label=f"{eid} → {name}")
            for eid, name in names_dict.items()
        ]

        return self.async_show_form(
            step_id="camera_names",
            description_placeholders={"camera_info": description},
            data_schema=vol.Schema(
                {
                    vol.Optional("select_camera"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=select_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional("camera_entity"): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="camera",
                            multiple=False,
                        )
                    ),
                    vol.Optional("friendly_name"): selector.TextSelector(
                        selector.TextSelectorConfig(
                            type=selector.TextSelectorType.TEXT,
                        )
                    ),
                    vol.Optional("action", default="add"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                selector.SelectOptionDict(value="add", label="Add New"),
                                selector.SelectOptionDict(value="update", label="Update Selected"),
                                selector.SelectOptionDict(value="delete", label="Delete Selected"),
                            ],
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                }
            ),
        )

    async def async_step_sofabaton(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle SofaBaton X2 activity configuration.

        SofaBaton activities map custom voice names to switch entities.
        Voice triggers: 'start [name]' or 'turn on [name]' to turn on,
                       'turn off [name]' to turn off.
        """
        current = {**self._entry.data, **self._entry.options}
        current_activities_json = current.get(CONF_SOFABATON_ACTIVITIES, DEFAULT_SOFABATON_ACTIVITIES)

        # Parse current activities from JSON
        try:
            activities_list = json.loads(current_activities_json) if current_activities_json else []
        except (json.JSONDecodeError, TypeError):
            activities_list = []

        if user_input is not None:
            selected = user_input.get("select_activity", "")
            activity_name = user_input.get("activity_name", "").strip()
            entity_id = user_input.get("entity_id", "")
            action = user_input.get("action", "add")

            # Find index of selected activity
            selected_idx = None
            if selected:
                for i, a in enumerate(activities_list):
                    if a.get("name", "").lower() == selected.lower():
                        selected_idx = i
                        break

            if action == "delete" and selected_idx is not None:
                # Delete selected activity
                activities_list.pop(selected_idx)
            elif action == "update" and selected_idx is not None:
                # Update selected activity
                activities_list[selected_idx] = {
                    "name": activity_name if activity_name else selected,
                    "entity_id": entity_id if entity_id else activities_list[selected_idx].get("entity_id", ""),
                }
            elif action == "add" and activity_name and entity_id:
                # Add new activity - check for duplicate names
                existing = [a for a in activities_list if a.get("name", "").lower() == activity_name.lower()]
                if existing:
                    # Update existing instead of adding duplicate
                    for a in activities_list:
                        if a.get("name", "").lower() == activity_name.lower():
                            a["entity_id"] = entity_id
                            break
                else:
                    activities_list.append({
                        "name": activity_name,
                        "entity_id": entity_id,
                    })
            elif not selected and not activity_name and not entity_id:
                # Empty submit - return to menu
                return self.async_create_entry(title="", data=self._entry.options)

            # Save updated activities as JSON
            updated_json = json.dumps(activities_list)
            new_options = {**self._entry.options, CONF_SOFABATON_ACTIVITIES: updated_json}
            self.hass.config_entries.async_update_entry(self._entry, options=new_options)

            # Rebuild list for display
            try:
                activities_list = json.loads(updated_json)
            except (json.JSONDecodeError, TypeError):
                activities_list = []

        # Build description showing current activities
        if activities_list:
            activity_lines = []
            for a in activities_list:
                name = a.get("name", "")
                entity = a.get("entity_id", "")
                activity_lines.append(f"**{name}** → {entity}")
            description = "**Current SofaBaton activities:**\n" + "\n".join(activity_lines) + "\n\nSelect one to edit/delete, or add a new one below.\n\n**Voice triggers:** 'start [name]' or 'turn on [name]' / 'turn off [name]'"
        else:
            description = "No SofaBaton activities configured. Add your first activity below.\n\n**Instructions:**\n1. Select a SofaBaton switch entity from the dropdown\n2. Give it a custom voice name (e.g., 'PC', 'PlayStation', 'Movie Mode')\n\n**Voice triggers:** 'start PC' or 'turn on PC' / 'turn off PC'"

        # Build select options for existing activities
        select_options = []
        for a in activities_list:
            name = a.get("name", "")
            entity = a.get("entity_id", "")
            select_options.append(selector.SelectOptionDict(value=name, label=f"{name} → {entity}"))

        return self.async_show_form(
            step_id="sofabaton",
            description_placeholders={"activities_info": description},
            data_schema=vol.Schema(
                {
                    vol.Optional("select_activity"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=select_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional("activity_name"): selector.TextSelector(
                        selector.TextSelectorConfig(
                            type=selector.TextSelectorType.TEXT,
                        )
                    ),
                    vol.Optional("entity_id"): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="switch",
                            multiple=False,
                        )
                    ),
                    vol.Optional("action", default="add"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                selector.SelectOptionDict(value="add", label="Add New"),
                                selector.SelectOptionDict(value="update", label="Update Selected"),
                                selector.SelectOptionDict(value="delete", label="Delete Selected"),
                            ],
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                }
            ),
        )

    async def async_step_music_rooms(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle music room to player mapping configuration with edit/delete support."""
        current = {**self._entry.data, **self._entry.options}
        current_mapping = current.get(CONF_ROOM_PLAYER_MAPPING, "")

        # Parse current mappings into dict {room_name: entity_id}
        rooms_dict = {}
        if current_mapping:
            for line in current_mapping.split("\n"):
                line = line.strip()
                if ": " in line:
                    room_name, entity_id = line.split(": ", 1)
                    rooms_dict[room_name.strip().lower()] = entity_id.strip()

        if user_input is not None:
            selected = user_input.get("select_room", "")
            new_player = user_input.get("room_player", "")
            new_room = user_input.get("room_name", "").strip().lower()
            action = user_input.get("action", "add")
            wake_cast = user_input.get(CONF_WAKE_CAST_BEFORE_PLAY, DEFAULT_WAKE_CAST_BEFORE_PLAY)
            adb_entity = user_input.get(CONF_WAKE_CAST_ADB_ENTITY, DEFAULT_WAKE_CAST_ADB_ENTITY)

            if action == "delete" and selected:
                # Delete selected room
                if selected in rooms_dict:
                    del rooms_dict[selected]
            elif action == "update" and selected and new_player:
                # Update selected room (delete old, add new)
                if selected in rooms_dict:
                    del rooms_dict[selected]
                room_key = new_room if new_room else selected
                rooms_dict[room_key] = new_player
            elif action == "add" and new_player and new_room:
                # Add new room mapping
                rooms_dict[new_room] = new_player
            elif not selected and not new_player and not new_room:
                # Save wake cast settings and return to menu
                new_options = {**self._entry.options, CONF_WAKE_CAST_BEFORE_PLAY: wake_cast, CONF_WAKE_CAST_ADB_ENTITY: adb_entity}
                return self.async_create_entry(title="", data=new_options)

            # Save updated mappings and wake cast settings
            if rooms_dict:
                updated_mapping = "\n".join([f"{k}: {v}" for k, v in rooms_dict.items()])
            else:
                updated_mapping = ""

            new_options = {**self._entry.options, CONF_ROOM_PLAYER_MAPPING: updated_mapping, CONF_WAKE_CAST_BEFORE_PLAY: wake_cast, CONF_WAKE_CAST_ADB_ENTITY: adb_entity}
            self.hass.config_entries.async_update_entry(self._entry, options=new_options)
            current_mapping = updated_mapping
            # Rebuild dict for display
            rooms_dict = {}
            if current_mapping:
                for line in current_mapping.split("\n"):
                    line = line.strip()
                    if ": " in line:
                        room_name, entity_id = line.split(": ", 1)
                        rooms_dict[room_name.strip().lower()] = entity_id.strip()

        # Build description showing current mappings
        if rooms_dict:
            mapping_display = "\n".join([f"• {k} → {v}" for k, v in rooms_dict.items()])
            description = f"**Current room mappings:**\n{mapping_display}\n\nSelect one to edit/delete, or add a new one below."
        else:
            description = "No room mappings configured. Add your first room below."

        # Build select options for existing rooms (no "none" option - just leave empty to add new)
        select_options = []
        for room_name in rooms_dict.keys():
            select_options.append(selector.SelectOptionDict(value=room_name, label=f"{room_name} → {rooms_dict[room_name]}"))

        return self.async_show_form(
            step_id="music_rooms",
            description_placeholders={"mappings": description},
            data_schema=vol.Schema(
                {
                    vol.Optional("select_room"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=select_options,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional("room_player"): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="media_player",
                            multiple=False,
                        )
                    ),
                    vol.Optional("room_name"): selector.TextSelector(
                        selector.TextSelectorConfig(
                            type=selector.TextSelectorType.TEXT,
                        )
                    ),
                    vol.Optional("action", default="add"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                selector.SelectOptionDict(value="add", label="Add New"),
                                selector.SelectOptionDict(value="update", label="Update Selected"),
                                selector.SelectOptionDict(value="delete", label="Delete Selected"),
                            ],
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_WAKE_CAST_BEFORE_PLAY,
                        default=current.get(CONF_WAKE_CAST_BEFORE_PLAY, DEFAULT_WAKE_CAST_BEFORE_PLAY),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_WAKE_CAST_ADB_ENTITY,
                        default=current.get(CONF_WAKE_CAST_ADB_ENTITY, DEFAULT_WAKE_CAST_ADB_ENTITY),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="media_player",
                            multiple=False,
                        )
                    ),
                }
            ),
        )

    async def async_step_notifications(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle notification settings configuration."""
        if user_input is not None:
            processed_input = {}
            if CONF_NOTIFICATION_ENTITIES in user_input:
                # Convert list to newline-separated string for storage
                entity_list = user_input[CONF_NOTIFICATION_ENTITIES]
                if isinstance(entity_list, list):
                    processed_input[CONF_NOTIFICATION_ENTITIES] = "\n".join(entity_list)
                else:
                    processed_input[CONF_NOTIFICATION_ENTITIES] = entity_list
            if CONF_NOTIFY_ON_PLACES in user_input:
                processed_input[CONF_NOTIFY_ON_PLACES] = user_input[CONF_NOTIFY_ON_PLACES]
            if CONF_NOTIFY_ON_RESTAURANTS in user_input:
                processed_input[CONF_NOTIFY_ON_RESTAURANTS] = user_input[CONF_NOTIFY_ON_RESTAURANTS]
            if CONF_NOTIFY_ON_SEARCH in user_input:
                processed_input[CONF_NOTIFY_ON_SEARCH] = user_input[CONF_NOTIFY_ON_SEARCH]

            new_options = {**self._entry.options, **processed_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        # Parse current notification entities back to list
        current_entities = current.get(CONF_NOTIFICATION_ENTITIES, DEFAULT_NOTIFICATION_ENTITIES)
        if isinstance(current_entities, str) and current_entities:
            current_entities = [e.strip() for e in current_entities.split("\n") if e.strip()]
        elif not current_entities:
            current_entities = []

        # Get available notify services from Home Assistant
        notify_services = []
        services = self.hass.services.async_services()
        if "notify" in services:
            for service_name in services["notify"]:
                # Skip generic services
                if service_name not in ["notify", "persistent_notification"]:
                    notify_services.append(
                        selector.SelectOptionDict(
                            value=service_name,
                            label=f"notify.{service_name}"
                        )
                    )

        # Sort alphabetically
        notify_services.sort(key=lambda x: x["label"])

        return self.async_show_form(
            step_id="notifications",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_NOTIFICATION_ENTITIES,
                        default=current_entities,
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=notify_services,
                            multiple=True,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_NOTIFY_ON_PLACES,
                        default=current.get(CONF_NOTIFY_ON_PLACES, DEFAULT_NOTIFY_ON_PLACES),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_NOTIFY_ON_RESTAURANTS,
                        default=current.get(CONF_NOTIFY_ON_RESTAURANTS, DEFAULT_NOTIFY_ON_RESTAURANTS),
                    ): cv.boolean,
                    vol.Optional(
                        CONF_NOTIFY_ON_SEARCH,
                        default=current.get(CONF_NOTIFY_ON_SEARCH, DEFAULT_NOTIFY_ON_SEARCH),
                    ): cv.boolean,
                }
            ),
        )

    async def async_step_api_keys(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle API key configuration."""
        if user_input is not None:
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        return self.async_show_form(
            step_id="api_keys",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_OPENWEATHERMAP_API_KEY,
                        default=current.get(CONF_OPENWEATHERMAP_API_KEY, DEFAULT_OPENWEATHERMAP_API_KEY),
                    ): str,
                    vol.Optional(
                        CONF_GOOGLE_PLACES_API_KEY,
                        default=current.get(CONF_GOOGLE_PLACES_API_KEY, DEFAULT_GOOGLE_PLACES_API_KEY),
                    ): str,
                    vol.Optional(
                        CONF_TAVILY_API_KEY,
                        default=current.get(CONF_TAVILY_API_KEY, DEFAULT_TAVILY_API_KEY),
                    ): str,
                }
            ),
        )

    async def async_step_location(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle location configuration."""
        if user_input is not None:
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        return self.async_show_form(
            step_id="location",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_CUSTOM_LATITUDE,
                        default=current.get(CONF_CUSTOM_LATITUDE, DEFAULT_CUSTOM_LATITUDE),
                    ): vol.Coerce(float),
                    vol.Optional(
                        CONF_CUSTOM_LONGITUDE,
                        default=current.get(CONF_CUSTOM_LONGITUDE, DEFAULT_CUSTOM_LONGITUDE),
                    ): vol.Coerce(float),
                }
            ),
            description_placeholders={
                "location_note": "Leave as 0 to use Home Assistant's configured location",
            },
        )

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle advanced settings."""
        if user_input is not None:
            new_options = {**self._entry.options, **user_input}
            return self.async_create_entry(title="", data=new_options)

        current = {**self._entry.data, **self._entry.options}

        return self.async_show_form(
            step_id="advanced",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_SYSTEM_PROMPT,
                        description={"suggested_value": current.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT)},
                    ): selector.TemplateSelector(),
                }
            ),
        )