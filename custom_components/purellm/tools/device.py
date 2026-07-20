"""Device control and status tool handlers."""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, TYPE_CHECKING

from ..utils.fuzzy_matching import find_entity_by_name
from ..utils.helpers import format_human_readable_state, get_friendly_name
from .fan_speed import LEVEL_MAP as FAN_LEVEL_MAP
from .thermostat import HVAC_MODE_MAP

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


_LOGGER = logging.getLogger(__name__)


async def check_device_status(
    arguments: dict[str, Any],
    hass: "HomeAssistant",
    user_query: str = "",
    temp_format_func: callable = None,
) -> dict[str, Any]:
    """Check the current status of any device.

    Args:
        arguments: Tool arguments (device)
        hass: Home Assistant instance
        user_query: Original user query for better extraction
        temp_format_func: Function to format temperature values

    Returns:
        Device status dict
    """
    device = arguments.get("device", "").strip()

    # Extract device name from original query using patterns
    extracted_device = None
    original_query = user_query.lower()

    patterns = [
        r"(?:what(?:'s| is) the )?status of (?:the )?([a-z ]+?)(?:\?|$)",  # "what's the status of the back door"
        r"(?:what(?:'s| is) the |check (?:the )?|is (?:the )?)([a-z ]+?)(?:\s+(?:status|open|closed|locked|unlocked|on|off|state)|\?|$)",
        r"(?:is |are )(?:the )?([a-z ]+?)(?:\s+(?:open|closed|locked|unlocked|on|off)|\?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, original_query)
        if match:
            extracted_device = match.group(1).strip()
            for suffix in [' status', ' state', ' currently', ' right now']:
                if extracted_device.endswith(suffix):
                    extracted_device = extracted_device[:-len(suffix)].strip()
            break

    if extracted_device and len(extracted_device) > len(device):
        _LOGGER.info("Device extraction: LLM said '%s', extracted '%s'", device, extracted_device)
        device = extracted_device

    if not device:
        return {"error": "No device specified. Please specify a device name like 'front door', 'garage', etc."}

    entity_id, friendly_name = find_entity_by_name(hass, device)

    if not entity_id:
        return {"error": f"Could not find a device matching '{device}'. Try using the exact name as shown in Home Assistant."}

    state = hass.states.get(entity_id)
    if not state:
        return {"error": f"Entity '{entity_id}' not found"}

    domain = entity_id.split(".")[0]
    current_state = state.state

    # Handle climate domain specially (thermostat)
    if domain == "climate":
        target_temp = state.attributes.get("temperature")
        current_temp = state.attributes.get("current_temperature")
        hvac_mode = state.attributes.get("hvac_mode", current_state)

        status_parts = []
        if target_temp:
            formatted_temp = temp_format_func(target_temp) if temp_format_func else f"{target_temp} degrees"
            status_parts.append(f"set to {formatted_temp}")
        if current_temp:
            formatted_temp = temp_format_func(current_temp) if temp_format_func else f"{current_temp} degrees"
            status_parts.append(f"currently {formatted_temp}")
        if hvac_mode:
            status_parts.append(f"mode: {hvac_mode}")

        status = ", ".join(status_parts) if status_parts else current_state.upper()

        _LOGGER.info("Device status check: %s -> %s (%s) status=%s", device, friendly_name, entity_id, status)

        return {
            "device": friendly_name,
            "status": status,
            "target_temperature": target_temp,
            "current_temperature": current_temp,
            "hvac_mode": hvac_mode,
            "entity_id": entity_id
        }

    # Handle sensors with units
    if domain == "sensor":
        unit = state.attributes.get("unit_of_measurement", "")
        if unit:
            status = f"{current_state} {unit}"
        else:
            status = current_state

        _LOGGER.info("Device status check: %s -> %s (%s) status=%s", device, friendly_name, entity_id, status)

        return {
            "device": friendly_name,
            "status": status,
            "entity_id": entity_id
        }

    # Use helper for other domains
    status = format_human_readable_state(entity_id, current_state)

    _LOGGER.info("Device status check: %s -> %s (%s) domain=%s raw_state=%s status=%s",
                 device, friendly_name, entity_id, domain, current_state, status)

    return {
        "device": friendly_name,
        "status": status,
        "entity_id": entity_id
    }


async def control_device(
    arguments: dict[str, Any],
    hass: "HomeAssistant",
    voice_scripts: list[dict[str, str]] | None = None,
    device_id: str | None = None,
    room_player_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Control smart home devices.

    This is the main device control handler that supports:
    - Lights (with brightness, color)
    - Switches
    - Covers/blinds (with position, presets)
    - Locks
    - Fans
    - Media players
    - Climate
    - Vacuums
    - Voice scripts (user-configured trigger phrases)
    - And more...

    Args:
        arguments: Tool arguments
        hass: Home Assistant instance
        voice_scripts: List of voice script configs with trigger, open_script, close_script, sensor
        device_id: Device ID of the satellite that issued the voice command
        room_player_mapping: Room to media player entity mapping

    Returns:
        Control result dict
    """
    if voice_scripts is None:
        voice_scripts = []
    from homeassistant.helpers import entity_registry as er
    from homeassistant.helpers import area_registry as ar
    from homeassistant.helpers import device_registry as dr

    action = arguments.get("action", "").strip().lower()
    brightness = arguments.get("brightness")
    position = arguments.get("position")
    color = arguments.get("color", "").strip().lower()
    color_temp = arguments.get("color_temp")
    volume = arguments.get("volume")
    temperature = arguments.get("temperature")
    hvac_mode_raw = arguments.get("hvac_mode", "").strip().lower()
    fan_speed = arguments.get("fan_speed", "").strip().lower()

    # Normalize HVAC mode aliases to Home Assistant values (shared with thermostat tool)
    hvac_mode = HVAC_MODE_MAP.get(hvac_mode_raw, hvac_mode_raw) if hvac_mode_raw else ""
    _LOGGER.debug("HVAC mode: raw=%s, normalized=%s", hvac_mode_raw, hvac_mode)

    direct_entity_id = arguments.get("entity_id", "").strip()
    entity_ids_list = arguments.get("entity_ids", [])
    area_name = arguments.get("area", "").strip()
    domain_filter = arguments.get("domain", "").strip().lower()
    device_name = arguments.get("device", "").strip()

    # Check if device_name is actually an entity_id (e.g., "light.master_dimmer_switch_light")
    # If so, treat it as a direct entity_id instead of running fuzzy matching
    known_domains = {"light", "switch", "cover", "fan", "lock", "climate", "media_player",
                     "vacuum", "scene", "script", "input_boolean", "automation", "button",
                     "siren", "humidifier", "sensor", "binary_sensor"}
    if device_name and "." in device_name:
        potential_domain = device_name.split(".")[0]
        if potential_domain in known_domains:
            _LOGGER.info("Device name '%s' looks like an entity_id, using directly", device_name)
            direct_entity_id = device_name
            device_name = ""  # Clear device_name so it doesn't go through fuzzy matching

    _LOGGER.debug("control_device: entity=%s, device=%s, area=%s, action=%s",
                  direct_entity_id or entity_ids_list, device_name, area_name, action)

    if not action:
        return {"error": "No action specified."}

    # Satellite-aware volume resolution: when the user says "your volume"
    # style commands (e.g. "raise your volume", "set your volume to 50"),
    # resolve to the satellite's own media_player entity.
    _GENERIC_SPEAKER_NAMES = {
        "speaker", "the speaker", "this speaker", "my speaker",
        "your speaker", "your volume",
        "volume", "the volume", "voice volume", "speaker volume",
    }
    if (action in ("set_volume", "volume_up", "volume_down")
            and device_name.lower().strip() in _GENERIC_SPEAKER_NAMES
            and not direct_entity_id
            and not entity_ids_list
            and device_id):
        from .timer import get_player_for_device
        resolved_player = get_player_for_device(hass, device_id, room_player_mapping)
        if resolved_player:
            _LOGGER.info(
                "Satellite volume resolution: device_id=%s -> %s",
                device_id, resolved_player,
            )
            direct_entity_id = resolved_player
            device_name = ""

    # Normalize actions
    # "run", "execute", "open", "close" support scripts like "open the garage"
    action_aliases = {
        "favorite": "preset",
        "return_home": "dock",
        "activate": "turn_on",
        "run": "turn_on",
        "execute": "turn_on",
        "launch": "trigger",
    }
    action = action_aliases.get(action, action)

    # Service map
    service_map = {
        "light": {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle", "dim": "turn_on",
                  # Models (even large ones) reach for these when asked "set X to N%" —
                  # accept them for lights instead of failing with "unsupported action".
                  "set_position": "turn_on", "set_brightness": "turn_on", "set": "turn_on", "set_level": "turn_on"},
        "switch": {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle"},
        "fan": {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle", "set_speed": "set_percentage", "stop": "turn_off", "pause": "turn_off", "start": "turn_on", "resume": "turn_on"},
        "lock": {"lock": "lock", "unlock": "unlock", "turn_on": "lock", "turn_off": "unlock"},
        "cover": {
            "open": "open_cover", "close": "close_cover", "toggle": "toggle",
            "turn_on": "open_cover", "turn_off": "close_cover",
            "stop": "stop_cover", "set_position": "set_cover_position", "preset": "set_cover_position"
        },
        "climate": {"turn_on": "turn_on", "turn_off": "turn_off", "set_temperature": "set_temperature", "set_hvac_mode": "set_hvac_mode"},
        "media_player": {
            "turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle",
            "play": "media_play", "resume": "media_play", "pause": "media_pause", "stop": "media_stop",
            "next": "media_next_track", "previous": "media_previous_track",
            "volume_up": "volume_up", "volume_down": "volume_down",
            "set_volume": "volume_set", "mute": "volume_mute", "unmute": "volume_mute"
        },
        "vacuum": {"turn_on": "start", "start": "start", "turn_off": "return_to_base", "stop": "pause", "pause": "pause", "resume": "start", "dock": "return_to_base", "locate": "locate"},
        "scene": {"turn_on": "turn_on"},
        "script": {"turn_on": "turn_on", "turn_off": "turn_off", "open": "turn_on", "close": "turn_on", "run": "turn_on", "execute": "turn_on", "mute": "turn_on", "unmute": "turn_on", "toggle": "turn_on", "trigger": "turn_on"},
        "input_boolean": {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle"},
        "automation": {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle", "trigger": "trigger"},
        "button": {"turn_on": "press", "press": "press"},
        "siren": {"turn_on": "turn_on", "turn_off": "turn_off"},
        "humidifier": {"turn_on": "turn_on", "turn_off": "turn_off"},
    }

    action_words = {
        "turn_on": "turned on", "turn_off": "turned off", "toggle": "toggled",
        "open": "opened", "close": "closed",
        "lock": "locked", "unlock": "unlocked",
        "open_cover": "opened", "close_cover": "closed", "stop_cover": "stopped",
        "set_cover_position": "set position for",
        "start": "started", "return_to_base": "sent home", "stop": "stopped", "pause": "paused",
        "locate": "located", "press": "pressed",
        "media_play": "resumed", "media_pause": "paused", "media_stop": "stopped",
        "media_next_track": "skipped to next", "media_previous_track": "went back to previous",
        "volume_up": "turned up", "volume_down": "turned down",
        "volume_set": "set volume for", "volume_mute": "muted/unmuted",
        "set_temperature": "set temperature for", "set_hvac_mode": "set mode for",
        "trigger": "launched",
    }

    color_map = {
        "red": [255, 0, 0], "green": [0, 255, 0], "blue": [0, 0, 255],
        "yellow": [255, 255, 0], "orange": [255, 165, 0], "purple": [128, 0, 128],
        "pink": [255, 192, 203], "white": [255, 255, 255], "cyan": [0, 255, 255],
        "warm": None, "cool": None,
    }

    entities_to_control = []
    matched_voice_script = None

    # Method 1: Direct entity_id
    if direct_entity_id:
        state = hass.states.get(direct_entity_id)
        if state:
            friendly_name = state.attributes.get("friendly_name", direct_entity_id)
            entities_to_control.append((direct_entity_id, friendly_name))
        else:
            return {"error": f"Entity '{direct_entity_id}' not found in Home Assistant."}

    # Method 2: Multiple entity_ids
    elif entity_ids_list:
        for eid in entity_ids_list:
            eid = eid.strip()
            state = hass.states.get(eid)
            if state:
                friendly_name = state.attributes.get("friendly_name", eid)
                entities_to_control.append((eid, friendly_name))
            else:
                _LOGGER.warning("Entity %s not found, skipping", eid)

    # Method 3: Area-based control
    elif area_name:
        # First, check if area_name matches an entity alias (e.g., "downstairs" -> light.downstairs_group)
        # This allows users to create aliases for light groups that the LLM might call as "areas"
        area_alias_entity_id, area_alias_friendly = find_entity_by_name(hass, area_name)

        if area_alias_entity_id:
            state = hass.states.get(area_alias_entity_id)
            if state:
                friendly_name = state.attributes.get("friendly_name", area_alias_entity_id)
                entities_to_control.append((area_alias_entity_id, friendly_name))
                _LOGGER.info("Area name '%s' matched entity alias -> %s", area_name, area_alias_entity_id)

        # If no alias match found, proceed with area registry lookup
        if not entities_to_control:
            ent_reg = er.async_get(hass)
            area_reg = ar.async_get(hass)
            dev_reg = dr.async_get(hass)

            target_area_id = None
            for area in area_reg.async_list_areas():
                if area.name.lower() == area_name.lower():
                    target_area_id = area.id
                    break

            if not target_area_id:
                for area in area_reg.async_list_areas():
                    if area_name.lower() in area.name.lower() or area.name.lower() in area_name.lower():
                        target_area_id = area.id
                        break

            if not target_area_id:
                return {"error": f"Could not find area '{area_name}'."}

            device_areas = {device.id: True for device in dev_reg.devices.values() if device.area_id == target_area_id}

            controllable_domains = ["light", "switch", "fan", "lock", "cover", "media_player", "vacuum", "scene", "script", "input_boolean"]
            if domain_filter and domain_filter != "all":
                controllable_domains = [domain_filter]

            for state in hass.states.async_all():
                eid = state.entity_id
                domain = eid.split(".")[0]

                if domain not in controllable_domains:
                    continue
                if state.state in ("unavailable", "unknown"):
                    continue

                entity_entry = ent_reg.async_get(eid)
                if not entity_entry:
                    continue

                in_area = entity_entry.area_id == target_area_id or (entity_entry.device_id and entity_entry.device_id in device_areas)

                if in_area:
                    friendly_name = state.attributes.get("friendly_name", eid)
                    entities_to_control.append((eid, friendly_name))

            if not entities_to_control:
                return {"error": f"No controllable devices found in area '{area_name}'."}

    # Method 4: Device name matching (uses fuzzy matching with aliases)
    elif device_name:
        device_lower = device_name.lower().strip()
        matched_voice_script = None

        # Media player actions that should skip voice scripts (mute/unmute excluded - they can use toggle scripts)
        media_player_actions = ("play", "pause", "resume", "stop", "next", "previous",
                                "volume_up", "volume_down", "set_volume")

        # Only check voice scripts for non-media-player actions
        if action not in media_player_actions:
            # Check if device name matches any configured voice script trigger
            # Bidirectional: trigger in device OR device in trigger
            # This handles LLM sending shorter names (e.g. "chromecast" for trigger "chromecast screen")
            for vs in voice_scripts:
                trigger = vs.get("trigger", "").lower().strip()
                if trigger and (trigger in device_lower or device_lower in trigger):
                    matched_voice_script = vs
                    _LOGGER.debug("Voice script matched: trigger='%s' for device='%s'", trigger, device_lower)
                    break

        if matched_voice_script:
            # Use configured voice script
            open_script = matched_voice_script.get("open_script", "")
            close_script = matched_voice_script.get("close_script", "")
            trigger_name = matched_voice_script.get("trigger", "").title()

            if action in ("open", "turn_on", "toggle", "mute", "unmute", "trigger") and open_script:
                entities_to_control.append((open_script, trigger_name))
            elif action in ("close", "turn_off") and close_script:
                entities_to_control.append((close_script, trigger_name))
            else:
                # Fall back to normal entity lookup if no script configured for this action
                found_entity_id, friendly_name = find_entity_by_name(hass, device_name)
                if found_entity_id:
                    entities_to_control.append((found_entity_id, friendly_name))
                else:
                    return {"error": f"Could not find a device matching '{device_name}'."}
        else:
            # Launch always means a voice script — if none matched above,
            # the device name the LLM sent doesn't match any configured
            # voice script trigger.  Return a clear error; no fallback to
            # fuzzy-matching media_players which can't handle launch.
            if action == "trigger":
                return {"error": f"No launch script found matching '{device_name}'."}
            else:
                found_entity_id, friendly_name = find_entity_by_name(hass, device_name)

                if found_entity_id:
                    entities_to_control.append((found_entity_id, friendly_name))
                else:
                    return {"error": f"Could not find a device matching '{device_name}'."}

    else:
        return {"error": "No device specified. Provide entity_id, entity_ids, area, or device name."}

    # Check sensor state for voice scripts before executing (e.g., garage door already open)
    for entity_id, friendly_name in entities_to_control:
        # Find matching voice script for this entity
        for vs in voice_scripts:
            open_script = vs.get("open_script", "")
            close_script = vs.get("close_script", "")
            sensor = vs.get("sensor", "")
            trigger_name = vs.get("trigger", "").title()

            if sensor and entity_id in (open_script, close_script):
                sensor_state = hass.states.get(sensor)
                if sensor_state:
                    is_open = sensor_state.state == "on"  # binary_sensor: on = open, off = closed

                    if entity_id == open_script and is_open:
                        return {"response_text": f"The {trigger_name} is already open."}
                    elif entity_id == close_script and not is_open:
                        return {"response_text": f"The {trigger_name} is already closed."}
                break

    # Build service calls first, then execute in parallel
    service_calls: list[tuple[str, str, dict, str]] = []  # (domain, service, data, friendly_name)
    failed = []
    last_service = None

    for entity_id, friendly_name in entities_to_control:
        domain = entity_id.split(".")[0]
        domain_services = service_map.get(domain, {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle"})
        service = domain_services.get(action)

        # Fallback: map common synonyms to turn_on/turn_off when domain doesn't define them
        if not service:
            generic_fallbacks = {"stop": "turn_off", "pause": "turn_off", "start": "turn_on", "resume": "turn_on"}
            fallback_action = generic_fallbacks.get(action)
            if fallback_action:
                service = domain_services.get(fallback_action)

        if not service:
            failed.append(f"{friendly_name} (unsupported action)")
            continue

        service_data = {"entity_id": entity_id}

        # Light controls
        if domain == "light" and action in ("turn_on", "dim", "set_position", "set_brightness", "set", "set_level"):
            if brightness is None and position is not None:
                # Model used the cover-style position param for a light percentage.
                brightness = position
            if brightness is not None:
                if brightness > 100:
                    # Model sent 0-255 scale (e.g. 128 for "50%") — convert, don't clamp
                    # (clamping turned "50%" into 100%).
                    brightness = round(brightness / 255 * 100)
                service_data["brightness_pct"] = max(0, min(100, brightness))
            if color and color in color_map and color_map[color]:
                service_data["rgb_color"] = color_map[color]
            elif color == "warm":
                service_data["color_temp_kelvin"] = 2700
            elif color == "cool":
                service_data["color_temp_kelvin"] = 6500
            if color_temp is not None:
                service_data["color_temp_kelvin"] = max(2000, min(6500, color_temp))

        # Media player controls
        if domain == "media_player":
            if action == "set_volume" and volume is not None:
                service_data["volume_level"] = max(0, min(100, volume)) / 100.0
            if action in ("play", "resume", "pause"):
                # Check current playback state
                state = hass.states.get(entity_id)
                current_state = state.state if state else None

                # Check if already in desired state
                if current_state == "playing" and action in ("play", "resume"):
                    return {"success": True, "response_text": f"The {friendly_name} is already playing."}
                elif current_state == "paused" and action == "pause":
                    return {"success": True, "response_text": f"The {friendly_name} is already paused."}
            if action in ("mute", "unmute"):
                # Check current mute state if available
                state = hass.states.get(entity_id)
                is_currently_muted = state.attributes.get("is_volume_muted") if state else None

                # Only skip if we know the current state for sure
                if is_currently_muted is True and action == "mute":
                    return {"success": True, "response_text": f"The {friendly_name} is already muted."}
                elif is_currently_muted is False and action == "unmute":
                    return {"success": True, "response_text": f"The {friendly_name} is already unmuted."}

                # Explicitly set mute state
                service_data["is_volume_muted"] = (action == "mute")

        # Climate controls
        if domain == "climate":
            _LOGGER.info("Climate control: action=%s, hvac_mode=%s, temperature=%s, entity=%s",
                        action, hvac_mode, temperature, entity_id)

            if action == "set_hvac_mode":
                if not hvac_mode:
                    return {"error": "hvac_mode parameter is required for set_hvac_mode action. Use: heat, cool, heat_cool, or off"}
                service = "set_hvac_mode"
                service_data["hvac_mode"] = hvac_mode
                _LOGGER.info("Setting HVAC mode to %s for %s", hvac_mode, entity_id)
            elif action == "set_temperature" and temperature is not None:
                service_data["temperature"] = temperature
            elif hvac_mode:
                # If hvac_mode provided with any other action, still set it
                service = "set_hvac_mode"
                service_data["hvac_mode"] = hvac_mode
                _LOGGER.info("Setting HVAC mode to %s for %s", hvac_mode, entity_id)

        # Fan controls
        if domain == "fan" and fan_speed:
            if fan_speed in FAN_LEVEL_MAP:
                service_data["percentage"] = FAN_LEVEL_MAP[fan_speed]

        # Cover position
        if domain == "cover" and action == "set_position" and position is not None:
            service_data["position"] = max(0, min(100, position))

        # Cover preset/favorite - try button.{name}_my_position first
        if domain == "cover" and action == "preset":
            cover_object_id = entity_id.split(".")[1]
            my_position_btn = f"button.{cover_object_id}_my_position"

            if hass.states.get(my_position_btn):
                service_calls.append(("button", "press", {"entity_id": my_position_btn}, friendly_name))
                last_service = service
                continue
            else:
                # Fall back to set_cover_position
                state = hass.states.get(entity_id)
                preset_pos = state.attributes.get("preset_position") if state else None
                if preset_pos is None and state:
                    preset_pos = state.attributes.get("favorite_position")
                service_data["position"] = preset_pos if preset_pos is not None else 50

        service_calls.append((domain, service, service_data, friendly_name))
        last_service = service

    # Execute all service calls in parallel
    async def execute_call(call_info: tuple[str, str, dict, str]) -> tuple[str, Exception | None]:
        domain, service, data, name = call_info
        try:
            await hass.services.async_call(domain, service, data, blocking=False)
            _LOGGER.info("Device control: %s.%s on %s", domain, service, name)
            return (name, None)
        except Exception as err:
            _LOGGER.error("Error controlling device %s: %s", name, err)
            return (name, err)

    if service_calls:
        results = await asyncio.gather(*[execute_call(call) for call in service_calls])
        controlled = [name for name, err in results if err is None]
        failed.extend([f"{name} ({str(err)[:30]})" for name, err in results if err is not None])
    else:
        controlled = []

    service = last_service  # For response generation

    # Build response
    if controlled:
        if len(controlled) == 1:
            if action == "preset":
                response = f"I've set the {controlled[0]} to its favorite position."
            elif brightness is not None and action in ("turn_on", "dim", "set_position", "set_brightness", "set", "set_level"):
                response = f"I've set the {controlled[0]} to {brightness}% brightness."
            elif action == "set_position" and position is not None:
                response = f"I've set the {controlled[0]} to {position}% position."
            elif action == "mute":
                response = f"I've muted the {controlled[0]}."
            elif action == "unmute":
                response = f"I've unmuted the {controlled[0]}."
            elif action in ("play", "resume"):
                response = f"The {controlled[0]} is now playing."
            elif action == "pause":
                response = f"The {controlled[0]} is now paused."
            elif hvac_mode:
                response = f"I've set the {controlled[0]} to {hvac_mode} mode."
            elif action == "set_temperature" and temperature is not None:
                response = f"I've set the {controlled[0]} to {temperature} degrees."
            else:
                # For voice scripts, use the original action for the response word
                # (scripts always map to "turn_on" service, which would say "turned on" for close actions)
                if matched_voice_script:
                    action_word = action_words.get(action, action)
                else:
                    action_word = action_words.get(service, action)
                response = f"I've {action_word} the {controlled[0]}."
        else:
            if action == "preset":
                response = f"I've set {len(controlled)} devices to their favorite positions: {', '.join(controlled[:5])}"
            elif action == "set_position" and position is not None:
                response = f"I've set {len(controlled)} devices to {position}%: {', '.join(controlled[:5])}"
            else:
                if matched_voice_script:
                    action_word = action_words.get(action, action)
                else:
                    action_word = action_words.get(service, action)
                response = f"I've {action_word} {len(controlled)} devices: {', '.join(controlled[:5])}"
            if len(controlled) > 5:
                response += f" and {len(controlled) - 5} more"
            response += "."

        result = {
            "success": True,
            "controlled_count": len(controlled),
            "controlled_devices": controlled,
            "response_text": response
        }

        if failed:
            result["failed_count"] = len(failed)
            result["failed_devices"] = failed

        return result
    else:
        return {"error": f"Failed to control any devices. Failures: {', '.join(failed)}"}
