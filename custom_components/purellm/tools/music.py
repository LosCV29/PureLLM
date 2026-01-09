"""Music control tool handler."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, TYPE_CHECKING

from homeassistant.components.media_player import MediaPlayerEntityFeature

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Feature flags for media player capabilities
PAUSE_FEATURE = MediaPlayerEntityFeature.PAUSE
STOP_FEATURE = MediaPlayerEntityFeature.STOP
PLAY_FEATURE = MediaPlayerEntityFeature.PLAY


class MusicController:
    """Controller for music playback operations.

    This class manages music state (last paused player, debouncing)
    and handles all music control operations via Music Assistant.
    """

    def __init__(self, hass: "HomeAssistant", room_player_mapping: dict[str, str]):
        """Initialize the music controller.

        Args:
            hass: Home Assistant instance
            room_player_mapping: Dict of room name -> media_player entity_id
        """
        self._hass = hass
        self._players = room_player_mapping
        self._last_paused_player: str | None = None
        self._last_music_command: str | None = None
        self._last_music_command_time: datetime | None = None
        self._music_debounce_seconds = 3.0

    async def control_music(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Control music playback.

        Args:
            arguments: Tool arguments (action, query, room, media_type, shuffle)

        Returns:
            Result dict
        """
        action = arguments.get("action", "").lower()
        query = arguments.get("query", "")
        media_type = arguments.get("media_type", "artist")
        room = arguments.get("room", "").lower() if arguments.get("room") else ""
        shuffle = arguments.get("shuffle", False)

        _LOGGER.debug("Music control: action=%s, room=%s, query=%s", action, room, query)

        all_players = list(self._players.values())

        if not all_players:
            _LOGGER.error("No players configured! room_player_mapping is empty")
            return {"error": "No music players configured. Go to PureLLM → Entity Configuration → Room to Player Mapping."}

        # Debounce check
        now = datetime.now()
        debounce_actions = {"skip_next", "skip_previous", "restart_track", "pause", "resume", "stop"}
        if action in debounce_actions:
            if (self._last_music_command == action and
                self._last_music_command_time and
                (now - self._last_music_command_time).total_seconds() < self._music_debounce_seconds):
                _LOGGER.info("DEBOUNCE: Ignoring duplicate '%s' command", action)
                return {"status": "debounced", "message": f"Command '{action}' ignored (duplicate)"}

        self._last_music_command = action
        self._last_music_command_time = now

        try:
            _LOGGER.info("=== MUSIC: %s ===", action.upper())

            # Determine target player(s)
            target_players = self._find_target_players(room)

            if action == "play":
                return await self._play(query, media_type, room, shuffle, target_players)
            elif action == "pause":
                return await self._pause(all_players)
            elif action == "resume":
                return await self._resume(all_players)
            elif action == "stop":
                return await self._stop(all_players)
            elif action == "skip_next":
                return await self._skip_next(all_players)
            elif action == "skip_previous":
                return await self._skip_previous(all_players)
            elif action == "restart_track":
                return await self._restart_track(all_players)
            elif action == "what_playing":
                return await self._what_playing(all_players)
            elif action == "transfer":
                return await self._transfer(all_players, target_players, room)
            elif action == "shuffle":
                return await self._shuffle(query, room, target_players)
            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as err:
            _LOGGER.error("Music control error: %s", err, exc_info=True)
            return {"error": f"Music control failed: {str(err)}"}

    def _find_target_players(self, room: str) -> list[str]:
        """Find target players for a room."""
        if room in self._players:
            return [self._players[room]]
        elif room:
            for rname, pid in self._players.items():
                if room in rname or rname in room:
                    return [pid]
        return []

    def _find_player_by_state(self, target_state: str, all_players: list[str]) -> str | None:
        """Find a player in a specific state from configured players only."""
        for pid in all_players:
            state = self._hass.states.get(pid)
            if state:
                _LOGGER.info("  %s → %s", pid, state.state)
                if state.state == target_state:
                    return pid
        return None

    def _get_transfer_source(self, entity_id: str) -> str:
        """Get the source player entity for transfer operations.

        For transfer_queue, Music Assistant may need the queue ID from active_queue.
        But for pause/stop, we always target the MA wrapper entity directly.
        """
        state = self._hass.states.get(entity_id)
        if state:
            active_queue = state.attributes.get("active_queue", "")
            # If active_queue looks like a queue ID (not an entity), use the entity_id
            # If it's an entity_id, we might use it for transfer source
            if isinstance(active_queue, str) and active_queue.startswith("media_player."):
                _LOGGER.info("Transfer source from active_queue: %s (of %s)", active_queue, entity_id)
                return active_queue

        # Always return the MA wrapper entity - never strip suffix
        # Raw DLNA entities don't support pause/stop/shuffle
        return entity_id

    def _get_room_name(self, entity_id: str) -> str:
        """Get room name from entity_id."""
        for rname, pid in self._players.items():
            if pid == entity_id:
                return rname
        return "unknown"

    async def _play(self, query: str, media_type: str, room: str, shuffle: bool, target_players: list[str]) -> dict:
        """Play music."""
        if not query:
            return {"error": "No music query specified"}
        if not target_players:
            return {"error": f"Unknown room: {room}. Available: {', '.join(self._players.keys())}"}

        for player in target_players:
            _LOGGER.info("Playing '%s' (%s) on %s", query, media_type, player)
            await self._hass.services.async_call(
                "music_assistant", "play_media",
                {"media_id": query, "media_type": media_type, "enqueue": "replace", "radio_mode": False},
                target={"entity_id": player},
                blocking=True
            )
            if shuffle or media_type == "genre":
                await self._hass.services.async_call(
                    "media_player", "shuffle_set",
                    {"entity_id": player, "shuffle": True},
                    blocking=True
                )

        return {"status": "playing", "message": f"Playing {query} in the {room}"}

    async def _pause(self, all_players: list[str]) -> dict:
        """Pause music - matches HA native intent behavior."""
        _LOGGER.info("Looking for player in 'playing' state that supports pause...")

        # Find player that is playing AND supports pause (like HA native intent)
        for pid in all_players:
            state = self._hass.states.get(pid)
            if state and state.state == "playing":
                # Check if player supports PAUSE feature
                supported = state.attributes.get("supported_features", 0)
                supports_pause = bool(supported & PAUSE_FEATURE)
                _LOGGER.info("  %s → playing, supports_pause=%s (features=%s)", pid, supports_pause, supported)

                if supports_pause:
                    _LOGGER.info("Pausing %s", pid)
                    await self._hass.services.async_call(
                        "media_player", "media_pause",
                        {},
                        target={"entity_id": pid},
                        blocking=True
                    )
                    self._last_paused_player = pid
                    return {"status": "paused", "message": f"Paused in {self._get_room_name(pid)}"}
                else:
                    # Player doesn't support pause, try stop instead (like MA does internally)
                    _LOGGER.warning("%s doesn't support pause, trying stop", pid)
                    await self._hass.services.async_call(
                        "media_player", "media_stop",
                        {},
                        target={"entity_id": pid},
                        blocking=True
                    )
                    self._last_paused_player = pid
                    return {"status": "paused", "message": f"Paused in {self._get_room_name(pid)}"}

        return {"error": "No music is currently playing"}

    async def _resume(self, all_players: list[str]) -> dict:
        """Resume music."""
        _LOGGER.info("Looking for player to resume...")

        if self._last_paused_player and self._last_paused_player in all_players:
            _LOGGER.info("Resuming last paused player: %s", self._last_paused_player)
            await self._hass.services.async_call(
                "media_player", "media_play",
                {},
                target={"entity_id": self._last_paused_player},
                blocking=True
            )
            room_name = self._get_room_name(self._last_paused_player)
            self._last_paused_player = None
            return {"status": "resumed", "message": f"Resumed in {room_name}"}

        paused = self._find_player_by_state("paused", all_players)
        if paused:
            await self._hass.services.async_call(
                "media_player", "media_play",
                {},
                target={"entity_id": paused},
                blocking=True
            )
            return {"status": "resumed", "message": f"Resumed in {self._get_room_name(paused)}"}

        return {"error": "No paused music to resume"}

    async def _stop(self, all_players: list[str]) -> dict:
        """Stop music - checks supported features."""
        _LOGGER.info("Looking for player in 'playing' or 'paused' state...")

        for pid in all_players:
            state = self._hass.states.get(pid)
            if state and state.state in ("playing", "paused"):
                supported = state.attributes.get("supported_features", 0)
                supports_stop = bool(supported & STOP_FEATURE)
                _LOGGER.info("  %s → %s, supports_stop=%s", pid, state.state, supports_stop)

                if supports_stop:
                    _LOGGER.info("Stopping %s", pid)
                    await self._hass.services.async_call(
                        "media_player", "media_stop",
                        {},
                        target={"entity_id": pid},
                        blocking=True
                    )
                    return {"status": "stopped", "message": f"Stopped in {self._get_room_name(pid)}"}
                else:
                    # Try turn_off if stop not supported
                    _LOGGER.warning("%s doesn't support stop, trying turn_off", pid)
                    await self._hass.services.async_call(
                        "media_player", "turn_off",
                        {},
                        target={"entity_id": pid},
                        blocking=True
                    )
                    return {"status": "stopped", "message": f"Stopped in {self._get_room_name(pid)}"}

        return {"message": "No music is playing"}

    async def _skip_next(self, all_players: list[str]) -> dict:
        """Skip to next track."""
        _LOGGER.info("Looking for player in 'playing' state...")
        playing = self._find_player_by_state("playing", all_players)
        if playing:
            await self._hass.services.async_call("media_player", "media_next_track", {"entity_id": playing})
            return {"status": "skipped", "message": "Skipped to next track"}
        return {"error": "No music is playing to skip"}

    async def _skip_previous(self, all_players: list[str]) -> dict:
        """Skip to previous track."""
        _LOGGER.info("Looking for player in 'playing' state...")
        playing = self._find_player_by_state("playing", all_players)
        if playing:
            await self._hass.services.async_call("media_player", "media_previous_track", {"entity_id": playing})
            return {"status": "skipped", "message": "Previous track"}
        return {"error": "No music is playing"}

    async def _restart_track(self, all_players: list[str]) -> dict:
        """Restart current track from beginning."""
        _LOGGER.info("Looking for player in 'playing' state to restart track...")
        playing = self._find_player_by_state("playing", all_players)
        if playing:
            await self._hass.services.async_call("media_player", "media_seek", {"entity_id": playing, "seek_position": 0})
            return {"status": "restarted", "message": "Bringing it back from the top"}
        return {"error": "No music is playing"}

    async def _what_playing(self, all_players: list[str]) -> dict:
        """Get currently playing track info."""
        _LOGGER.info("Looking for player in 'playing' state...")
        playing = self._find_player_by_state("playing", all_players)
        if playing:
            state = self._hass.states.get(playing)
            attrs = state.attributes
            return {
                "title": attrs.get("media_title", "Unknown"),
                "artist": attrs.get("media_artist", "Unknown"),
                "album": attrs.get("media_album_name", ""),
                "room": self._get_room_name(playing)
            }
        return {"message": "No music currently playing"}

    async def _transfer(self, all_players: list[str], target_players: list[str], room: str) -> dict:
        """Transfer music to another room."""
        _LOGGER.info("Looking for player in 'playing' state...")
        playing = self._find_player_by_state("playing", all_players)
        if not playing:
            return {"error": "No music playing to transfer"}
        if not target_players:
            return {"error": f"No target room specified. Available: {', '.join(self._players.keys())}"}

        target = target_players[0]

        # For transfer, try the active_queue if available, otherwise use the MA wrapper
        source = self._get_transfer_source(playing)
        _LOGGER.info("Transferring from %s (source: %s) to %s", playing, source, target)

        try:
            await self._hass.services.async_call(
                "music_assistant", "transfer_queue",
                {"source_player": source, "auto_play": True},
                target={"entity_id": target},
                blocking=True
            )
            _LOGGER.info("Transfer complete")
        except Exception as e:
            _LOGGER.error("Transfer failed with source_player: %s", e)
            # Fallback: try with the MA wrapper entity
            try:
                await self._hass.services.async_call(
                    "music_assistant", "transfer_queue",
                    {"source_player": playing, "auto_play": True},
                    target={"entity_id": target},
                    blocking=True
                )
            except Exception as e2:
                _LOGGER.error("Transfer fallback also failed: %s", e2)

        return {"status": "transferred", "message": f"Music transferred to {self._get_room_name(target)}"}

    async def _shuffle(self, query: str, room: str, target_players: list[str]) -> dict:
        """Search and play shuffled playlist."""
        if not query:
            return {"error": "No search query specified for shuffle"}
        if not target_players:
            return {"error": f"No room specified. Available: {', '.join(self._players.keys())}"}

        _LOGGER.info("Searching for playlist matching: %s", query)

        try:
            ma_entries = self._hass.config_entries.async_entries("music_assistant")
            if not ma_entries:
                return {"error": "Music Assistant integration not found"}
            ma_config_entry_id = ma_entries[0].entry_id

            search_result = await self._hass.services.async_call(
                "music_assistant", "search",
                {"config_entry_id": ma_config_entry_id, "name": query, "media_type": ["playlist"], "limit": 5},
                blocking=True, return_response=True
            )

            playlist_name = None
            playlist_uri = None
            media_type_to_use = "playlist"

            if search_result:
                playlists = []
                if isinstance(search_result, dict):
                    playlists = search_result.get("playlists", [])
                    if not playlists and "items" in search_result:
                        playlists = search_result.get("items", [])
                elif isinstance(search_result, list):
                    playlists = search_result

                if playlists:
                    first_playlist = playlists[0]
                    playlist_name = first_playlist.get("name") or first_playlist.get("title", "Unknown Playlist")
                    playlist_uri = first_playlist.get("uri") or first_playlist.get("media_id")

            # Fall back to artist search
            if not playlist_uri:
                _LOGGER.info("No playlist found, searching for artist: %s", query)
                artist_result = await self._hass.services.async_call(
                    "music_assistant", "search",
                    {"config_entry_id": ma_config_entry_id, "name": query, "media_type": ["artist"], "limit": 1},
                    blocking=True, return_response=True
                )
                if artist_result:
                    artists = []
                    if isinstance(artist_result, dict):
                        artists = artist_result.get("artists", [])
                    elif isinstance(artist_result, list):
                        artists = artist_result
                    if artists:
                        playlist_name = artists[0].get("name", query)
                        playlist_uri = artists[0].get("uri") or artists[0].get("media_id")
                        media_type_to_use = "artist"

            if not playlist_uri:
                return {"error": f"Could not find playlist or artist matching '{query}'"}

            player = target_players[0]
            _LOGGER.info("Playing %s (%s) shuffled on %s", playlist_name, media_type_to_use, player)

            await self._hass.services.async_call(
                "music_assistant", "play_media",
                {"media_id": playlist_uri, "media_type": media_type_to_use, "enqueue": "replace", "radio_mode": False},
                target={"entity_id": player},
                blocking=True
            )

            await self._hass.services.async_call(
                "media_player", "shuffle_set",
                {"entity_id": player, "shuffle": True},
                blocking=True
            )

            return {
                "status": "shuffling",
                "playlist_name": playlist_name,
                "room": room,
                "message": f"Shuffling {playlist_name} in the {room}"
            }

        except Exception as search_err:
            _LOGGER.error("Shuffle search/play error: %s", search_err, exc_info=True)
            return {"error": f"Failed to find or play playlist: {str(search_err)}"}
