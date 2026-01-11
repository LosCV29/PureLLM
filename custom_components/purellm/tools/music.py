"""Music control tool handler."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, TYPE_CHECKING

from homeassistant.components.media_player import MediaPlayerEntityFeature
from homeassistant.helpers import entity_registry as er, device_registry as dr

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
        artist = arguments.get("artist", "")
        album = arguments.get("album", "")

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
                return await self._play(query, media_type, room, shuffle, target_players, artist, album)
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

    def _get_area_id(self, entity_id: str) -> str | None:
        """Get area_id for an entity (checks entity, then device)."""
        ent_reg = er.async_get(self._hass)
        dev_reg = dr.async_get(self._hass)

        entity_entry = ent_reg.async_get(entity_id)
        if entity_entry:
            # First check if entity has direct area assignment
            if entity_entry.area_id:
                _LOGGER.info("Entity %s has area_id: %s", entity_id, entity_entry.area_id)
                return entity_entry.area_id
            # Otherwise check the device
            if entity_entry.device_id:
                device = dev_reg.async_get(entity_entry.device_id)
                if device and device.area_id:
                    _LOGGER.info("Entity %s device has area_id: %s", entity_id, device.area_id)
                    return device.area_id

        _LOGGER.warning("Could not find area_id for %s", entity_id)
        return None

    async def _play(self, query: str, media_type: str, room: str, shuffle: bool, target_players: list[str], artist: str = "", album: str = "") -> dict:
        """Play music via Music Assistant with search-first for accuracy.

        Searches Music Assistant first to find the exact track/album/artist,
        then plays the found result and returns the actual name.

        Smart album features:
        - "latest/last/newest album by X" → finds most recent album
        - "first/oldest/debut album by X" → finds earliest album
        """
        if not query:
            return {"error": "No music query specified"}
        if not target_players:
            return {"error": f"Unknown room: {room}. Available: {', '.join(self._players.keys())}"}

        # Enforce valid media types
        valid_types = {"artist", "album", "track"}
        if media_type not in valid_types:
            media_type = "artist"

        # Smart override: if artist is specified with a query but media_type is "artist",
        # user likely wants a track: "Big Pimpin by Jay-Z" = track, not artist
        # BUT if user explicitly said "album", respect that choice
        if artist and query and media_type == "artist":
            media_type = "track"
            _LOGGER.info("Overriding media_type to 'track' since both query and artist specified")

        # Detect smart album modifiers
        album_modifier = None
        query_lower = query.lower()
        latest_keywords = ["latest", "last", "newest", "new", "most recent", "recent", "nuevo", "última", "ultimo", "más reciente"]
        first_keywords = ["first", "oldest", "debut", "earliest", "primero", "primera"]

        if media_type == "album":
            for kw in latest_keywords:
                if kw in query_lower:
                    album_modifier = "latest"
                    _LOGGER.info("Detected album modifier: 'latest' (keyword: %s)", kw)
                    break
            if not album_modifier:
                for kw in first_keywords:
                    if kw in query_lower:
                        album_modifier = "first"
                        _LOGGER.info("Detected album modifier: 'first' (keyword: %s)", kw)
                        break

        try:
            # Get Music Assistant config entry
            ma_entries = self._hass.config_entries.async_entries("music_assistant")
            if not ma_entries:
                return {"error": "Music Assistant integration not found"}
            ma_config_entry_id = ma_entries[0].entry_id

            # Handle smart album search (latest/first album by artist)
            if album_modifier and artist:
                _LOGGER.info("Smart album search: finding %s album by '%s'", album_modifier, artist)

                # Search for albums by artist only
                search_result = await self._hass.services.async_call(
                    "music_assistant", "search",
                    {"config_entry_id": ma_config_entry_id, "name": artist, "media_type": ["album"], "limit": 20},
                    blocking=True, return_response=True
                )

                albums = []
                if search_result:
                    if isinstance(search_result, dict):
                        albums = search_result.get("albums", [])
                    elif isinstance(search_result, list):
                        albums = search_result

                if albums:
                    artist_lower = artist.lower()

                    # Filter to only albums by this artist
                    def get_album_artist(alb):
                        if alb.get("artists"):
                            if isinstance(alb["artists"], list) and alb["artists"]:
                                return (alb["artists"][0].get("name") or "").lower()
                            elif isinstance(alb["artists"], str):
                                return alb["artists"].lower()
                        elif alb.get("artist"):
                            if isinstance(alb["artist"], str):
                                return alb["artist"].lower()
                            else:
                                return (alb["artist"].get("name") or "").lower()
                        return ""

                    matching_albums = [a for a in albums if artist_lower in get_album_artist(a) or get_album_artist(a) in artist_lower]

                    if matching_albums:
                        # Sort by year (try multiple fields that MA might return)
                        def get_year(alb):
                            # Try various fields Music Assistant might use
                            year = alb.get("year") or alb.get("release_date") or alb.get("date") or alb.get("release_year") or ""

                            # Also check in metadata dict if present
                            if not year and alb.get("metadata"):
                                meta = alb["metadata"]
                                year = meta.get("year") or meta.get("release_date") or ""

                            if isinstance(year, str) and len(year) >= 4:
                                try:
                                    return int(year[:4])
                                except ValueError:
                                    return 0
                            elif isinstance(year, int):
                                return year
                            return 0

                        # Log available albums for debugging
                        for alb in matching_albums[:5]:
                            alb_name = alb.get("name") or alb.get("title")
                            alb_year = get_year(alb)
                            _LOGGER.info("  Album candidate: '%s' (year: %s, raw: %s)",
                                        alb_name, alb_year,
                                        alb.get("year") or alb.get("release_date") or "unknown")

                        albums_with_year = [(get_year(a), a) for a in matching_albums]
                        # Include albums even with year=0, but sort them to the end
                        albums_with_year.sort(key=lambda x: (x[0] == 0, -x[0] if album_modifier == "latest" else x[0]))

                        if albums_with_year:
                            best_album = albums_with_year[0][1]
                            found_name = best_album.get("name") or best_album.get("title")
                            found_uri = best_album.get("uri") or best_album.get("media_id")
                            found_artist = artist
                            found_type = "album"

                            year = albums_with_year[0][0]
                            _LOGGER.info("Selected %s album: '%s' (year: %d) by '%s'", album_modifier, found_name, year, found_artist)

                            # Play it
                            for player in target_players:
                                await self._hass.services.async_call(
                                    "music_assistant", "play_media",
                                    {"media_id": found_uri, "media_type": "album", "enqueue": "replace", "radio_mode": False},
                                    target={"entity_id": player},
                                    blocking=True
                                )
                                if shuffle:
                                    await self._hass.services.async_call(
                                        "media_player", "shuffle_set",
                                        {"entity_id": player, "shuffle": True},
                                        blocking=True
                                    )

                            return {"status": "playing", "message": f"Playing {found_name} by {found_artist} in the {room}"}

                return {"error": f"Could not find albums by {artist}"}

            # Standard search (non-modifier path)
            search_query = f"{query} {artist}" if artist else query

            # NO cascading - search ONLY the requested type
            search_types_to_try = [media_type]
            _LOGGER.info("Searching for media_type='%s' only (no cascade)", media_type)

            found_name = None
            found_artist = None
            found_uri = None
            found_type = None

            for try_type in search_types_to_try:
                _LOGGER.info("Searching MA for %s: search_query='%s' (query='%s', artist='%s')", try_type, search_query, query, artist)

                search_result = await self._hass.services.async_call(
                    "music_assistant", "search",
                    {"config_entry_id": ma_config_entry_id, "name": search_query, "media_type": [try_type], "limit": 10},
                    blocking=True, return_response=True
                )

                if not search_result:
                    continue

                # Get the appropriate results list
                results = []
                if isinstance(search_result, dict):
                    if try_type == "track":
                        results = search_result.get("tracks", [])
                    elif try_type == "album":
                        results = search_result.get("albums", [])
                    elif try_type == "artist":
                        results = search_result.get("artists", [])
                    if not results:
                        results = search_result.get("items", [])
                elif isinstance(search_result, list):
                    results = search_result

                if not results:
                    continue

                query_lower = query.lower()
                artist_lower = artist.lower() if artist else ""

                # Score results to find best match
                def score_result(item):
                    score = 0
                    item_name = (item.get("name") or item.get("title") or "").lower()
                    item_artist = ""

                    # Get artist from various possible fields
                    if item.get("artists"):
                        if isinstance(item["artists"], list):
                            item_artist = (item["artists"][0].get("name") or "").lower() if item["artists"] else ""
                        elif isinstance(item["artists"], str):
                            item_artist = item["artists"].lower()
                    elif item.get("artist"):
                        item_artist = (item["artist"] if isinstance(item["artist"], str) else item["artist"].get("name", "")).lower()

                    # Exact query match in name
                    if query_lower == item_name:
                        score += 100
                    elif query_lower in item_name:
                        score += 50

                    # Artist match (if artist was specified)
                    if artist_lower:
                        if artist_lower == item_artist:
                            score += 100
                        elif artist_lower in item_artist or item_artist in artist_lower:
                            score += 50

                    return score

                # Sort by score descending
                scored_results = [(score_result(r), r) for r in results]
                scored_results.sort(key=lambda x: x[0], reverse=True)

                # Only accept if we have a good match (score > 0 means query or artist matched)
                if scored_results and scored_results[0][0] > 0:
                    best_match = scored_results[0][1]
                    found_name = best_match.get("name") or best_match.get("title")
                    found_uri = best_match.get("uri") or best_match.get("media_id")
                    found_type = try_type

                    # Extract artist name from result
                    if best_match.get("artists"):
                        if isinstance(best_match["artists"], list) and best_match["artists"]:
                            found_artist = best_match["artists"][0].get("name")
                        elif isinstance(best_match["artists"], str):
                            found_artist = best_match["artists"]
                    elif best_match.get("artist"):
                        if isinstance(best_match["artist"], str):
                            found_artist = best_match["artist"]
                        else:
                            found_artist = best_match["artist"].get("name")

                    _LOGGER.info("Found %s: '%s' by '%s' (uri: %s, score: %d)", try_type, found_name, found_artist, found_uri, scored_results[0][0])
                    break  # Found a good match, stop searching

            if not found_uri:
                return {"error": f"Could not find track matching '{query}'" + (f" by {artist}" if artist else "")}

            # Build display name from actual found result
            if found_artist and found_type in ("track", "album"):
                display_name = f"{found_name} by {found_artist}"
            else:
                display_name = found_name

            # Play the found media
            for player in target_players:
                _LOGGER.info("Playing: uri='%s', type='%s' on %s", found_uri, found_type, player)
                await self._hass.services.async_call(
                    "music_assistant", "play_media",
                    {"media_id": found_uri, "media_type": found_type, "enqueue": "replace", "radio_mode": False},
                    target={"entity_id": player},
                    blocking=True
                )

                if shuffle:
                    await self._hass.services.async_call(
                        "media_player", "shuffle_set",
                        {"entity_id": player, "shuffle": True},
                        blocking=True
                    )

            return {"status": "playing", "message": f"Playing {display_name} in the {room}"}

        except Exception as e:
            _LOGGER.error("Play search/play error: %s", e, exc_info=True)
            return {"error": f"Failed to find or play music: {str(e)}"}

    async def _pause(self, all_players: list[str]) -> dict:
        """Pause music - uses area targeting like HA native intents."""
        _LOGGER.info("Looking for player in 'playing' state...")

        for pid in all_players:
            state = self._hass.states.get(pid)
            if state and state.state == "playing":
                _LOGGER.info("  %s → playing", pid)

                # Get area_id for area-based targeting (like HA native intents)
                area_id = self._get_area_id(pid)
                if area_id:
                    _LOGGER.info("Pausing via area: %s", area_id)
                    await self._hass.services.async_call(
                        "media_player", "media_pause",
                        {},
                        target={"area_id": area_id},
                        blocking=True
                    )
                else:
                    # Fallback to entity targeting if no area
                    _LOGGER.info("No area found, pausing via entity: %s", pid)
                    await self._hass.services.async_call(
                        "media_player", "media_pause",
                        {},
                        target={"entity_id": pid},
                        blocking=True
                    )

                self._last_paused_player = pid
                return {"status": "paused", "message": f"Paused in {self._get_room_name(pid)}"}

        return {"error": "No music is currently playing"}

    async def _resume(self, all_players: list[str]) -> dict:
        """Resume music - uses area targeting like HA native intents."""
        _LOGGER.info("Looking for player to resume...")

        # Try last paused player first
        if self._last_paused_player and self._last_paused_player in all_players:
            _LOGGER.info("Resuming last paused player: %s", self._last_paused_player)
            area_id = self._get_area_id(self._last_paused_player)
            if area_id:
                await self._hass.services.async_call(
                    "media_player", "media_play",
                    {},
                    target={"area_id": area_id},
                    blocking=True
                )
            else:
                await self._hass.services.async_call(
                    "media_player", "media_play",
                    {},
                    target={"entity_id": self._last_paused_player},
                    blocking=True
                )
            room_name = self._get_room_name(self._last_paused_player)
            self._last_paused_player = None
            return {"status": "resumed", "message": f"Resumed in {room_name}"}

        # Find any paused player
        paused = self._find_player_by_state("paused", all_players)
        if paused:
            area_id = self._get_area_id(paused)
            if area_id:
                await self._hass.services.async_call(
                    "media_player", "media_play",
                    {},
                    target={"area_id": area_id},
                    blocking=True
                )
            else:
                await self._hass.services.async_call(
                    "media_player", "media_play",
                    {},
                    target={"entity_id": paused},
                    blocking=True
                )
            return {"status": "resumed", "message": f"Resumed in {self._get_room_name(paused)}"}

        return {"error": "No paused music to resume"}

    async def _stop(self, all_players: list[str]) -> dict:
        """Stop music - uses area targeting like HA native intents."""
        _LOGGER.info("Looking for player in 'playing' or 'paused' state...")

        for pid in all_players:
            state = self._hass.states.get(pid)
            if state and state.state in ("playing", "paused"):
                _LOGGER.info("  %s → %s", pid, state.state)

                # Get area_id for area-based targeting
                area_id = self._get_area_id(pid)
                if area_id:
                    _LOGGER.info("Stopping via area: %s", area_id)
                    await self._hass.services.async_call(
                        "media_player", "media_stop",
                        {},
                        target={"area_id": area_id},
                        blocking=True
                    )
                else:
                    _LOGGER.info("No area found, stopping via entity: %s", pid)
                    await self._hass.services.async_call(
                        "media_player", "media_stop",
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
        """Search for Spotify playlist by artist or genre and play shuffled.

        IMPORTANT: This ONLY searches for Spotify playlists - no fallback to artist.
        Returns the exact playlist title for verbatim announcement.
        """
        if not query:
            return {"error": "No search query specified for shuffle"}
        if not target_players:
            return {"error": f"No room specified. Available: {', '.join(self._players.keys())}"}

        _LOGGER.info("Searching Spotify for playlist matching: %s", query)

        try:
            ma_entries = self._hass.config_entries.async_entries("music_assistant")
            if not ma_entries:
                return {"error": "Music Assistant integration not found"}
            ma_config_entry_id = ma_entries[0].entry_id

            # Search ONLY for Spotify playlists - no fallback to artist
            search_result = await self._hass.services.async_call(
                "music_assistant", "search",
                {"config_entry_id": ma_config_entry_id, "name": query, "media_type": ["playlist"], "limit": 10},
                blocking=True, return_response=True
            )

            playlist_name = None
            playlist_uri = None

            if search_result:
                playlists = []
                if isinstance(search_result, dict):
                    playlists = search_result.get("playlists", [])
                    if not playlists and "items" in search_result:
                        playlists = search_result.get("items", [])
                elif isinstance(search_result, list):
                    playlists = search_result

                if playlists:
                    # Filter out playlists with "Radio" in the name - we don't want auto-generated radio playlists
                    non_radio_playlists = [
                        p for p in playlists
                        if "radio" not in (p.get("name") or p.get("title") or "").lower()
                    ]

                    query_lower = query.lower()
                    query_words = query_lower.split()

                    def name_matches_query(playlist_name_str: str) -> bool:
                        """Check if playlist name contains query or any significant word from query."""
                        name_lower = playlist_name_str.lower()
                        # Exact query match
                        if query_lower in name_lower:
                            return True
                        # Match on individual words (handles typos like elliot vs elliott)
                        for word in query_words:
                            if len(word) >= 4 and word in name_lower:
                                return True
                        return False

                    # Priority 1: Official Spotify curated playlists ("This Is...", "Best of...", owned by Spotify)
                    official_playlists = [
                        p for p in non_radio_playlists
                        if (p.get("owner") or "").lower() == "spotify"
                        or (p.get("name") or p.get("title") or "").lower().startswith("this is")
                        or (p.get("name") or p.get("title") or "").lower().startswith("best of")
                    ]

                    # Priority 2: Playlists with artist/query name in title
                    matching_name_playlists = [
                        p for p in non_radio_playlists
                        if name_matches_query(p.get("name") or p.get("title") or "")
                    ]

                    # Choose best playlist: Official > Name match > Non-radio > Any
                    if official_playlists:
                        # Among official, prefer ones with query in name
                        official_with_name = [p for p in official_playlists if name_matches_query(p.get("name") or p.get("title") or "")]
                        chosen_playlist = official_with_name[0] if official_with_name else official_playlists[0]
                        _LOGGER.info("Found official Spotify playlist")
                    elif matching_name_playlists:
                        chosen_playlist = matching_name_playlists[0]
                        _LOGGER.info("Found playlist with '%s' in name", query)
                    elif non_radio_playlists:
                        chosen_playlist = non_radio_playlists[0]
                        _LOGGER.info("Using first non-radio playlist")
                    else:
                        chosen_playlist = playlists[0]
                        _LOGGER.info("Falling back to first playlist result")

                    # Get the EXACT playlist title for verbatim announcement
                    playlist_name = chosen_playlist.get("name") or chosen_playlist.get("title")
                    playlist_uri = chosen_playlist.get("uri") or chosen_playlist.get("media_id")
                    _LOGGER.info("Found Spotify playlist: '%s'", playlist_name)

            # NO artist fallback - shuffle is ONLY for playlists
            if not playlist_uri:
                return {"error": f"Could not find a Spotify playlist matching '{query}'. Try a different artist or genre."}

            player = target_players[0]
            _LOGGER.info("Playing playlist '%s' shuffled on %s", playlist_name, player)

            await self._hass.services.async_call(
                "music_assistant", "play_media",
                {"media_id": playlist_uri, "media_type": "playlist", "enqueue": "replace", "radio_mode": False},
                target={"entity_id": player},
                blocking=True
            )

            await self._hass.services.async_call(
                "media_player", "shuffle_set",
                {"entity_id": player, "shuffle": True},
                blocking=True
            )

            # Return the EXACT playlist title for verbatim announcement
            return {
                "status": "shuffling",
                "playlist_title": playlist_name,
                "room": room,
                "announcement": f"Now playing {playlist_name}"
            }

        except Exception as search_err:
            _LOGGER.error("Shuffle search/play error: %s", search_err, exc_info=True)
            return {"error": f"Failed to find or play playlist: {str(search_err)}"}
