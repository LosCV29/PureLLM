"""Update entity for PureLLM integration."""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import aiohttp

from homeassistant.components.update import (
    UpdateDeviceClass,
    UpdateEntity,
    UpdateEntityFeature,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er

from .const import DOMAIN, get_version

_LOGGER = logging.getLogger(__name__)

GITHUB_REPO = "LosCV29/PureLLM"
SCAN_INTERVAL = timedelta(hours=4)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up PureLLM update entity."""
    async_add_entities([PureLLMUpdateEntity(hass, entry)])


class PureLLMUpdateEntity(UpdateEntity):
    """Update entity for PureLLM."""

    _attr_has_entity_name = True
    _attr_name = "Update"
    _attr_device_class = UpdateDeviceClass.FIRMWARE
    _attr_supported_features = UpdateEntityFeature.INSTALL

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the update entity."""
        self.hass = hass
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_update"
        self._installed_version: str | None = get_version()
        self._latest_version: str | None = None
        self._release_url: str | None = None
        self._release_notes: str | None = None

    @property
    def installed_version(self) -> str | None:
        """Return the installed version."""
        return self._installed_version

    @property
    def latest_version(self) -> str | None:
        """Return the latest version."""
        return self._latest_version

    @property
    def release_url(self) -> str | None:
        """Return the release URL."""
        return self._release_url

    @property
    def release_summary(self) -> str | None:
        """Return the release notes."""
        return self._release_notes

    async def async_added_to_hass(self) -> None:
        """Update device registry sw_version when entity is added."""
        await super().async_added_to_hass()
        # Force update device registry with current version
        # This ensures sw_version stays in sync after updates
        current_version = get_version()
        device_registry = dr.async_get(self.hass)
        device = device_registry.async_get_device(
            identifiers={(DOMAIN, self._entry.entry_id)}
        )
        if device and device.sw_version != current_version:
            device_registry.async_update_device(
                device.id, sw_version=current_version
            )
            _LOGGER.info(
                "Updated PureLLM device version from %s to %s",
                device.sw_version,
                current_version,
            )

    @property
    def device_info(self) -> dict[str, Any]:
        """Return device info."""
        return {
            "identifiers": {(DOMAIN, self._entry.entry_id)},
            "name": "PureLLM",
            "manufacturer": "LosCV29",
            "model": "Voice Assistant",
            "sw_version": get_version(),
        }

    async def async_update(self) -> None:
        """Check GitHub for the latest release."""
        try:
            session = async_get_clientsession(self.hass)
            url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    tag = data.get("tag_name", "")
                    # Remove 'v' prefix if present
                    self._latest_version = tag.lstrip("v")
                    self._release_url = data.get("html_url")

                    # Get release notes (truncate if too long)
                    body = data.get("body", "")
                    if len(body) > 500:
                        self._release_notes = body[:497] + "..."
                    else:
                        self._release_notes = body

                    _LOGGER.debug(
                        "PureLLM update check: installed=%s, latest=%s",
                        self._installed_version,
                        self._latest_version
                    )
                else:
                    _LOGGER.warning("GitHub API returned status %s", response.status)
        except Exception as err:
            _LOGGER.error("Failed to check for updates: %s", err)

    def _find_hacs_update_entity(self) -> str | None:
        """Find the HACS-managed update entity for this repository.

        HACS creates its own update entity for every downloaded repository.
        It is matched here by the GitHub repo slug in its release_url
        attribute, since HACS's entity unique_id is an opaque numeric id.
        """
        registry = er.async_get(self.hass)
        repo = GITHUB_REPO.lower()
        for entry in registry.entities.values():
            if entry.domain != "update" or entry.platform != "hacs":
                continue
            state = self.hass.states.get(entry.entity_id)
            if state is None:
                continue
            release_url = (state.attributes.get("release_url") or "").lower()
            if repo in release_url:
                return entry.entity_id
        return None

    async def async_install(
        self, version: str | None, backup: bool, **kwargs: Any
    ) -> None:
        """Install the update by delegating to HACS.

        HACS removed the ``hacs.download`` service in HACS 2.0, so the
        integration can no longer trigger its own download directly.
        Instead we locate the HACS-managed update entity for this
        repository and call the standard ``update.install`` service on it.
        """
        hacs_entity = self._find_hacs_update_entity()

        if hacs_entity is None:
            # No HACS update entity found (e.g. a manual install).
            # Fall back to instructing the user to update manually.
            _LOGGER.warning(
                "HACS update entity for %s not found; manual update required",
                GITHUB_REPO,
            )
            await self.hass.services.async_call(
                "persistent_notification",
                "create",
                {
                    "title": "PureLLM Update Available",
                    "message": (
                        f"Please update PureLLM to version {version} via HACS:\n\n"
                        "1. Go to HACS\n2. Find PureLLM\n3. Click Update\n"
                        "4. Restart Home Assistant"
                    ),
                    "notification_id": "purellm_update",
                },
            )
            return

        try:
            await self.hass.services.async_call(
                "update",
                "install",
                {"entity_id": hacs_entity},
                blocking=True,
            )
        except Exception as err:
            _LOGGER.error("Failed to install update via HACS: %s", err)
            raise HomeAssistantError(
                f"HACS failed to download the PureLLM update: {err}"
            ) from err

        _LOGGER.info("PureLLM update installed via HACS (%s)", hacs_entity)
        await self.hass.services.async_call(
            "persistent_notification",
            "create",
            {
                "title": "PureLLM Updated",
                "message": (
                    f"PureLLM has been updated to version {version}. "
                    "Please restart Home Assistant to complete the update."
                ),
                "notification_id": "purellm_update",
            },
        )
