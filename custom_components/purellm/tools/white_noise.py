"""White noise / ambient sound control for PureLLM.

Plays white noise (and related ambient sounds) on configured room speakers
via voice control. Uses Music Assistant's search-based play_media with
radio_mode enabled so playback continues indefinitely — ideal for nurseries,
offices, or sleep.

Supported sounds: white, pink, brown, rain, ocean, fan, thunder, shushing.
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


# Search queries sent to Music Assistant for each sound type. MA resolves
# these against the user's configured providers (Spotify, Apple Music, etc.).
SOUND_QUERIES: dict[str, str] = {
    "white": "white noise",
    "pink": "pink noise",
    "brown": "brown noise",
    "rain": "rain sounds",
    "ocean": "ocean waves",
    "fan": "fan noise",
    "thunder": "thunderstorm sounds",
    "shushing": "baby shushing",
}

DEFAULT_SOUND = "white"


class WhiteNoiseController:
    """Plays ambient sounds (white noise, rain, etc.) on room speakers."""

    def __init__(self, hass: "HomeAssistant", room_player_mapping: dict[str, str]):
        self._hass = hass
        self._players = room_player_mapping

    def _find_target_players(self, room: str) -> list[str]:
        """Resolve a room name to media_player entity_ids (case-insensitive)."""
        if not room:
            return []
        room_lower = room.lower()
        for rname, pid in self._players.items():
            if room_lower == rname.lower():
                return [pid]
        for rname, pid in self._players.items():
            rname_lower = rname.lower()
            if room_lower in rname_lower or rname_lower in room_lower:
                return [pid]
        return []

    def _resolve_sound(self, sound: str | None) -> tuple[str, str]:
        """Return (sound_key, search_query) for a user-supplied sound label."""
        if not sound:
            return DEFAULT_SOUND, SOUND_QUERIES[DEFAULT_SOUND]
        key = sound.lower().strip()
        if key in SOUND_QUERIES:
            return key, SOUND_QUERIES[key]
        # Fuzzy: match any sound key contained in the user's phrase
        for candidate in SOUND_QUERIES:
            if candidate in key or key in candidate:
                return candidate, SOUND_QUERIES[candidate]
        # Unknown label — treat the whole phrase as a free-form search
        return key, key

    async def control_white_noise_deferred(
        self, arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], Callable[[], Awaitable[None]] | None]:
        """Run control_white_noise, deferring the actual play_media call.

        Matches the pattern used by MusicController.control_music_deferred:
        resolve/announce synchronously, then trigger playback after TTS
        finishes so the announcement isn't drowned out.
        """
        captured: list[tuple[str, str]] = []
        original = self._play_on_player

        async def _capture(player: str, query: str) -> None:
            captured.append((player, query))

        self._play_on_player = _capture  # type: ignore[method-assign]
        try:
            result = await self.control_white_noise(arguments)
        finally:
            self._play_on_player = original  # type: ignore[method-assign]

        if not captured:
            return result, None

        async def _do_play() -> None:
            for player, query in captured:
                await original(player, query)

        return result, _do_play

    async def control_white_noise(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle a white-noise tool call."""
        action = (arguments.get("action") or "").lower().strip()
        room = (arguments.get("room") or "").lower().strip()
        sound_arg = arguments.get("sound")

        if not self._players:
            return {"error": "No speakers configured. Set up Room to Player Mapping in PureLLM."}

        all_players = list(self._players.values())
        target_players = self._find_target_players(room) if room else []

        try:
            if action == "play":
                if not target_players:
                    return {
                        "error": (
                            f"Which room? Available: {', '.join(self._players.keys())}"
                            if not room else
                            f"Unknown room '{room}'. Available: {', '.join(self._players.keys())}"
                        )
                    }
                sound_key, query = self._resolve_sound(sound_arg)
                for player in target_players:
                    await self._play_on_player(player, query)
                room_label = self._room_label(target_players[0]) or room
                return {
                    "status": "playing",
                    "sound": sound_key,
                    "response_text": f"Playing {sound_key} noise in the {room_label}",
                }

            if action == "stop":
                players = target_players or all_players
                await self._stop(players)
                return {"status": "stopped", "response_text": "White noise stopped"}

            if action in ("volume_up", "volume_down", "set_volume"):
                players = target_players or all_players
                volume = arguments.get("volume")
                return await self._volume(action, players, volume)

            return {"error": f"Unknown action: {action}"}

        except Exception as err:  # noqa: BLE001
            _LOGGER.error("White noise control error: %s", err, exc_info=True)
            return {"error": f"White noise control failed: {err}"}

    async def _play_on_player(self, player: str, query: str) -> None:
        """Search + play via Music Assistant with radio_mode for continuous loop."""
        _LOGGER.info("White noise: playing '%s' on %s", query, player)
        await self._hass.services.async_call(
            "music_assistant",
            "play_media",
            {
                "media_id": query,
                "media_type": "track",
                "enqueue": "replace",
                "radio_mode": True,
            },
            target={"entity_id": player},
            blocking=True,
        )
        # Belt-and-braces: force repeat so even a single-track result loops.
        try:
            await self._hass.services.async_call(
                "media_player",
                "repeat_set",
                {"entity_id": player, "repeat": "all"},
                blocking=True,
            )
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("repeat_set not supported on %s: %s", player, err)

    async def _stop(self, players: list[str]) -> None:
        for player in players:
            state = self._hass.states.get(player)
            if not state or state.state in ("off", "unavailable", "unknown"):
                continue
            await self._hass.services.async_call(
                "media_player",
                "media_stop",
                {"entity_id": player},
                blocking=True,
            )

    async def _volume(
        self,
        action: str,
        players: list[str],
        volume: int | None,
    ) -> dict[str, Any]:
        """Adjust volume on the target players."""
        STEP = 0.1
        for player in players:
            if action == "set_volume":
                if volume is None:
                    return {"error": "volume param required for set_volume"}
                level = max(0, min(100, int(volume))) / 100
                await self._hass.services.async_call(
                    "media_player",
                    "volume_set",
                    {"entity_id": player, "volume_level": level},
                    blocking=True,
                )
            else:
                state = self._hass.states.get(player)
                current = 0.3
                if state:
                    current = state.attributes.get("volume_level") or 0.3
                delta = STEP if action == "volume_up" else -STEP
                level = max(0.0, min(1.0, current + delta))
                await self._hass.services.async_call(
                    "media_player",
                    "volume_set",
                    {"entity_id": player, "volume_level": level},
                    blocking=True,
                )
        return {"status": "ok", "response_text": "Volume adjusted"}

    def _room_label(self, entity_id: str) -> str | None:
        for rname, pid in self._players.items():
            if pid == entity_id:
                return rname
        return None
