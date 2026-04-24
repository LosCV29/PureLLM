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
#
# Each entry is an ordered list of (query, media_type) fallback attempts.
# We try playlists first (curated, all-ambient) before falling back to
# artist search (whose catalog is expanded via radio_mode). A plain
# "white noise" track search is avoided because streaming libraries
# contain many pop songs literally titled "White Noise" that hijack the
# match and, with radio_mode, pull in similar pop tracks.
SOUND_QUERIES: dict[str, list[tuple[str, str]]] = {
    "white": [
        ("white noise sleep", "playlist"),
        ("White Noise Baby Sleep Sounds", "artist"),
        ("white noise 8 hours", "track"),
    ],
    "pink": [
        ("pink noise sleep", "playlist"),
        ("Pink Noise", "artist"),
        ("pink noise 8 hours", "track"),
    ],
    "brown": [
        ("brown noise sleep", "playlist"),
        ("Brown Noise", "artist"),
        ("brown noise 8 hours", "track"),
    ],
    "rain": [
        ("rain sounds sleep", "playlist"),
        ("Rain Sounds", "artist"),
        ("rain sounds 8 hours", "track"),
    ],
    "ocean": [
        ("ocean sounds sleep", "playlist"),
        ("Ocean Sounds", "artist"),
        ("ocean waves 8 hours", "track"),
    ],
    "fan": [
        ("fan noise sleep", "playlist"),
        ("Fan Sounds", "artist"),
        ("fan sounds 8 hours", "track"),
    ],
    "thunder": [
        ("thunderstorm sleep", "playlist"),
        ("Thunderstorm Sounds", "artist"),
        ("thunderstorm 8 hours", "track"),
    ],
    "shushing": [
        ("Baby Shusher", "artist"),
        ("baby shushing", "track"),
    ],
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

    def _resolve_sound(self, sound: str | None) -> tuple[str, list[tuple[str, str]]]:
        """Return (sound_key, fallback attempts) for a user-supplied label."""
        if not sound:
            return DEFAULT_SOUND, SOUND_QUERIES[DEFAULT_SOUND]
        key = sound.lower().strip()
        if key in SOUND_QUERIES:
            return key, SOUND_QUERIES[key]
        for candidate in SOUND_QUERIES:
            if candidate in key or key in candidate:
                return candidate, SOUND_QUERIES[candidate]
        # Unknown label — treat as free-form with track + artist attempts
        return key, [(key, "playlist"), (key, "artist"), (key, "track")]

    def _ma_config_entry_id(self) -> str | None:
        entries = self._hass.config_entries.async_entries("music_assistant")
        return entries[0].entry_id if entries else None

    async def _search_ma(
        self, config_entry_id: str, query: str, media_type: str, limit: int = 5,
    ) -> list[dict]:
        """Search Music Assistant and return a flat list of result dicts."""
        try:
            result = await self._hass.services.async_call(
                "music_assistant",
                "search",
                {
                    "config_entry_id": config_entry_id,
                    "name": query,
                    "media_type": [media_type],
                    "limit": limit,
                },
                blocking=True,
                return_response=True,
            )
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("MA search failed for '%s' (%s): %s", query, media_type, err)
            return []
        if not result:
            return []
        type_keys = {
            "track": "tracks",
            "album": "albums",
            "artist": "artists",
            "playlist": "playlists",
            "radio": "radio",
        }
        if isinstance(result, dict):
            items = result.get(type_keys.get(media_type, ""), []) or result.get("items", [])
            return items or []
        if isinstance(result, list):
            return result
        return []

    async def _resolve_playable(
        self, attempts: list[tuple[str, str]],
    ) -> tuple[str, str, str] | None:
        """Try each (query, media_type) until MA returns a result.

        Returns (uri, media_type, display_name) for the first non-empty match,
        or None if every attempt came back empty.
        """
        config_entry_id = self._ma_config_entry_id()
        if not config_entry_id:
            return None

        for query, media_type in attempts:
            results = await self._search_ma(config_entry_id, query, media_type)
            for item in results:
                uri = item.get("uri") or item.get("media_id")
                if not uri:
                    continue
                name = item.get("name") or item.get("title") or query
                _LOGGER.info(
                    "White noise: matched '%s' (%s) via query='%s' type=%s",
                    name, uri, query, media_type,
                )
                return uri, media_type, name
        return None

    async def control_white_noise_deferred(
        self, arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], Callable[[], Awaitable[None]] | None]:
        """Run control_white_noise, deferring the actual play_media call.

        Matches the pattern used by MusicController.control_music_deferred:
        resolve/announce synchronously, then trigger playback after TTS
        finishes so the announcement isn't drowned out.
        """
        captured: list[tuple[str, str, str]] = []
        original = self._play_on_player

        async def _capture(player: str, uri: str, media_type: str) -> None:
            captured.append((player, uri, media_type))

        self._play_on_player = _capture  # type: ignore[method-assign]
        try:
            result = await self.control_white_noise(arguments)
        finally:
            self._play_on_player = original  # type: ignore[method-assign]

        if not captured:
            return result, None

        async def _do_play() -> None:
            for player, uri, media_type in captured:
                await original(player, uri, media_type)

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
                sound_key, attempts = self._resolve_sound(sound_arg)
                match = await self._resolve_playable(attempts)
                if not match:
                    return {
                        "error": (
                            f"No {sound_key} noise found in your Music Assistant providers. "
                            "Try adding a streaming provider (Spotify, Apple Music, etc.) "
                            "that carries ambient sleep sounds."
                        )
                    }
                uri, media_type, display_name = match
                for player in target_players:
                    await self._play_on_player(player, uri, media_type)
                room_label = self._room_label(target_players[0]) or room
                return {
                    "status": "playing",
                    "sound": sound_key,
                    "matched": display_name,
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

    async def _play_on_player(self, player: str, uri: str, media_type: str) -> None:
        """Play a resolved MA URI. radio_mode only for artist/track so playlists
        (already curated, all-ambient) aren't expanded into unrelated music."""
        radio_mode = media_type in ("artist", "track")
        _LOGGER.info(
            "White noise: play uri='%s' type=%s radio=%s on %s",
            uri, media_type, radio_mode, player,
        )
        # Shuffle off for playlists/tracks so they start from the beginning;
        # shuffle on for artists so we don't always hear the same opener.
        try:
            await self._hass.services.async_call(
                "media_player", "shuffle_set",
                {"entity_id": player, "shuffle": media_type == "artist"},
                blocking=True,
            )
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("shuffle_set not supported on %s: %s", player, err)

        await self._hass.services.async_call(
            "music_assistant",
            "play_media",
            {
                "media_id": uri,
                "media_type": media_type,
                "enqueue": "replace",
                "radio_mode": radio_mode,
            },
            target={"entity_id": player},
            blocking=True,
        )
        # Belt-and-braces: force repeat so short selections loop.
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
