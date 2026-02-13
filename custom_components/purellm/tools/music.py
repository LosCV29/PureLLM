"""Music control tool handler."""
from __future__ import annotations

import asyncio
import codecs
import logging
import re
import aiohttp
from datetime import datetime
from typing import Any, TYPE_CHECKING

from homeassistant.components.media_player import MediaPlayerEntityFeature
from homeassistant.helpers import entity_registry as er, device_registry as dr

from ..utils.helpers import COMMON_ROOM_NAMES

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Shared MusicBrainz constants
_MB_HEADERS = {"User-Agent": "PureLLM-HomeAssistant/1.0 (https://github.com/LosCV29/purellm)"}
_MB_BASE = "https://musicbrainz.org/ws/2"


async def _musicbrainz_get(endpoint: str, params: dict, timeout: float = 5) -> dict | None:
    """Make a GET request to MusicBrainz API. Returns JSON data or None."""
    try:
        params["fmt"] = "json"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{_MB_BASE}/{endpoint}", params=params,
                headers=_MB_HEADERS, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    return await response.json()
                _LOGGER.debug("MusicBrainz API returned status %d", response.status)
    except asyncio.TimeoutError:
        _LOGGER.debug("MusicBrainz request timed out: %s", endpoint)
    except Exception as e:
        _LOGGER.debug("MusicBrainz request failed: %s %s", endpoint, e)
    return None


def _parse_ma_results(search_result: Any, media_type: str) -> list:
    """Parse Music Assistant search results into a flat list."""
    if not search_result:
        return []
    type_keys = {"track": "tracks", "album": "albums", "artist": "artists", "playlist": "playlists"}
    if isinstance(search_result, dict):
        results = search_result.get(type_keys.get(media_type, ""), [])
        if not results:
            results = search_result.get("items", [])
        return results
    if isinstance(search_result, list):
        return search_result
    return []


def _extract_artist(item: dict, lowercase: bool = False) -> str:
    """Extract artist name from a Music Assistant item."""
    artist = ""
    if item.get("artists"):
        if isinstance(item["artists"], list) and item["artists"]:
            artist = item["artists"][0].get("name") or ""
        elif isinstance(item["artists"], str):
            artist = item["artists"]
    elif item.get("artist"):
        if isinstance(item["artist"], str):
            artist = item["artist"]
        else:
            artist = item["artist"].get("name") or ""
    if not artist:
        return ""
    if lowercase:
        return artist.lower()
    return _normalize_unicode(artist)

# Feature flags for media player capabilities
PAUSE_FEATURE = MediaPlayerEntityFeature.PAUSE
STOP_FEATURE = MediaPlayerEntityFeature.STOP
PLAY_FEATURE = MediaPlayerEntityFeature.PLAY


def _normalize_unicode(text: str | None) -> str:
    """Normalize Unicode strings to ensure proper character display.

    Handles escaped Unicode sequences like \\u00e1 → á
    """
    if not text:
        return ""

    _LOGGER.debug("Normalizing text (raw repr): %r", text)

    # Method 1: Try regex replacement for \uXXXX patterns
    unicode_pattern = re.compile(r'\\u([0-9a-fA-F]{4})')

    def replace_unicode(match):
        return chr(int(match.group(1), 16))

    if unicode_pattern.search(text):
        try:
            text = unicode_pattern.sub(replace_unicode, text)
            _LOGGER.debug("Unicode normalized via regex: %s", text)
            return text
        except (ValueError, UnicodeError) as e:
            _LOGGER.debug("Regex normalization failed: %s", e)

    # Method 2: Try encode/decode for unicode_escape
    try:
        decoded = text.encode('latin-1').decode('unicode_escape')
        if decoded != text:
            _LOGGER.debug("Unicode normalized via encode/decode: %s", decoded)
            return decoded
    except (UnicodeDecodeError, UnicodeEncodeError) as e:
        _LOGGER.debug("Encode/decode normalization failed: %s", e)

    return text


async def _lookup_album_year_musicbrainz(album_name: str, artist_name: str) -> int:
    """Look up album release year from MusicBrainz API."""
    clean_album = re.sub(r'\s*[\(\[].*?[\)\]]', '', album_name).strip()
    clean_album = re.sub(r'\s*[-–].*?(edition|version|deluxe|remaster).*$', '', clean_album, flags=re.IGNORECASE).strip()

    data = await _musicbrainz_get("release-group", {
        "query": f'releasegroup:"{clean_album}" AND artist:"{artist_name}"',
        "limit": 5,
    })
    if not data:
        return 0

    release_groups = data.get("release-groups", [])
    if not release_groups:
        return 0

    # Find best match - prefer exact title match
    for rg in release_groups:
        rg_title = rg.get("title", "").lower()
        if clean_album.lower() in rg_title or rg_title in clean_album.lower():
            first_release = rg.get("first-release-date", "")
            if first_release and len(first_release) >= 4:
                year = int(first_release[:4])
                _LOGGER.info("MusicBrainz: Found year %d for '%s' by '%s'", year, album_name, artist_name)
                return year

    # Fallback to first result
    first_release = release_groups[0].get("first-release-date", "")
    if first_release and len(first_release) >= 4:
        year = int(first_release[:4])
        _LOGGER.info("MusicBrainz: Found year %d for '%s' by '%s' (fallback)", year, album_name, artist_name)
        return year

    return 0


async def _search_albums_by_tag_musicbrainz(artist_name: str, tag: str) -> list[dict]:
    """Search MusicBrainz for albums by artist with a specific tag (e.g., christmas, holiday)."""
    data = await _musicbrainz_get("release-group", {
        "query": f'artist:"{artist_name}" AND tag:{tag}',
        "limit": 25,
    }, timeout=10)
    if not data:
        return []

    albums = []
    for rg in data.get("release-groups", []):
        primary_type = (rg.get("primary-type") or "").lower()
        if primary_type != "album":
            continue
        first_release = rg.get("first-release-date", "")
        albums.append({
            "name": rg.get("title", ""),
            "year": int(first_release[:4]) if first_release and len(first_release) >= 4 else 0,
            "mbid": rg.get("id", ""),
            "primary_type": primary_type,
        })

    albums.sort(key=lambda x: (x["year"] == 0, x["year"]))
    _LOGGER.warning("MUSIC DEBUG: MusicBrainz tag search for '%s' + tag:'%s' found %d albums",
                   artist_name, tag, len(albums))
    for i, alb in enumerate(albums[:10]):
        _LOGGER.warning("MUSIC DEBUG: MusicBrainz tag [%d] '%s' (%d)", i+1, alb["name"], alb["year"])
    return albums


async def _get_artist_discography_musicbrainz(artist_name: str, album_type: str = None) -> list[dict]:
    """Get artist's discography from MusicBrainz with album types and years."""
    # First, find the artist ID
    artist_data = await _musicbrainz_get("artist", {
        "query": f'artist:"{artist_name}"', "limit": 1,
    })
    if not artist_data:
        return []
    artists = artist_data.get("artists", [])
    if not artists or not artists[0].get("id"):
        return []
    artist_id = artists[0]["id"]

    # MusicBrainz rate limit
    await asyncio.sleep(1)

    # Get artist's release groups
    data = await _musicbrainz_get("release-group", {
        "artist": artist_id, "type": "album|ep", "limit": 100,
    }, timeout=10)
    if not data:
        return []

    discography = []
    for rg in data.get("release-groups", []):
        first_release = rg.get("first-release-date", "")
        year = int(first_release[:4]) if first_release and len(first_release) >= 4 else 0
        primary_type = (rg.get("primary-type") or "").lower()
        secondary_types = [t.lower() for t in (rg.get("secondary-types") or [])]

        is_studio = primary_type == "album" and not secondary_types
        is_live = "live" in secondary_types
        is_compilation = "compilation" in secondary_types
        is_soundtrack = "soundtrack" in secondary_types
        is_ep = primary_type == "ep"

        if album_type:
            type_lower = album_type.lower()
            if type_lower == "studio" and not is_studio:
                continue
            elif type_lower == "live" and not is_live:
                continue
            elif type_lower in ("compilation", "greatest hits", "best of") and not is_compilation:
                continue
            elif type_lower == "soundtrack" and not is_soundtrack:
                continue
            elif type_lower == "ep" and not is_ep:
                continue
        else:
            if primary_type != "album":
                continue

        discography.append({
            "name": rg.get("title", ""),
            "year": year,
            "primary_type": primary_type,
            "secondary_types": secondary_types,
            "is_studio": is_studio,
            "is_live": is_live,
            "is_compilation": is_compilation,
        })

    discography.sort(key=lambda x: (x["year"] == 0, x["year"]))
    _LOGGER.warning("MUSIC DEBUG: MusicBrainz found %d albums for '%s'%s",
                len(discography), artist_name,
                f" (filtered by {album_type})" if album_type else "")
    for i, alb in enumerate(discography[:10]):
        _LOGGER.warning("MUSIC DEBUG: MusicBrainz [%d] '%s' (%d) - type: %s%s",
                       i+1, alb["name"], alb["year"], alb["primary_type"],
                       f" + {alb['secondary_types']}" if alb["secondary_types"] else "")
    return discography


# Ordinal number mapping
ORDINALS = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7,
    "eighth": 8, "8th": 8,
    "ninth": 9, "9th": 9,
    "tenth": 10, "10th": 10,
}

# Album type keywords mapping
ALBUM_TYPE_KEYWORDS = {
    "studio": "studio",
    "live": "live",
    "concert": "live",
    "compilation": "compilation",
    "greatest hits": "compilation",
    "best of": "compilation",
    "hits": "compilation",
    "soundtrack": "soundtrack",
    "ost": "soundtrack",
    "ep": "ep",
}

# Broadway cast recording keywords - used to EXCLUDE these from movie soundtrack searches
BROADWAY_KEYWORDS = [
    "broadway cast",
    "original cast recording",
    "original cast",
    "london cast",
    "west end",
    "revival cast",
    "off-broadway",
    "off broadway",
]

# Movie soundtrack keywords - used to INCLUDE only these for movie soundtrack searches
MOVIE_SOUNDTRACK_KEYWORDS = [
    "motion picture",
    "film soundtrack",
    "movie soundtrack",
    "original soundtrack",
    "music from the film",
    "music from the movie",
]


def _is_broadway_cast_recording(album_name: str) -> bool:
    """Check if an album is a Broadway/theater cast recording.

    Args:
        album_name: Name of the album

    Returns:
        True if the album appears to be a Broadway cast recording
    """
    name_lower = album_name.lower()
    return any(kw in name_lower for kw in BROADWAY_KEYWORDS)


def _is_movie_soundtrack(album_name: str) -> bool:
    """Check if an album is a movie soundtrack (not Broadway).

    Args:
        album_name: Name of the album

    Returns:
        True if the album appears to be a movie soundtrack
    """
    name_lower = album_name.lower()
    # Must contain movie keywords AND not contain broadway keywords
    has_movie_keyword = any(kw in name_lower for kw in MOVIE_SOUNDTRACK_KEYWORDS)
    is_broadway = _is_broadway_cast_recording(album_name)
    return has_movie_keyword and not is_broadway


# Holiday keywords for shuffle playlist search
HOLIDAY_KEYWORDS = {
    # Christmas
    "christmas": ["christmas", "xmas", "holiday"],
    "xmas": ["christmas", "xmas", "holiday"],
    # Halloween
    "halloween": ["halloween", "spooky", "scary", "horror"],
    "spooky": ["halloween", "spooky", "scary"],
    # Thanksgiving
    "thanksgiving": ["thanksgiving", "grateful", "fall"],
    # Easter
    "easter": ["easter", "spring"],
    # Valentine's Day
    "valentine": ["valentine", "valentines", "love", "romantic"],
    "valentines": ["valentine", "valentines", "love", "romantic"],
    "romantic": ["romantic", "love", "valentine"],
    # 4th of July / Independence Day
    "4th of july": ["4th of july", "fourth of july", "independence day", "patriotic", "america"],
    "fourth of july": ["4th of july", "fourth of july", "independence day", "patriotic"],
    "independence day": ["independence day", "4th of july", "patriotic"],
    "patriotic": ["patriotic", "america", "usa"],
    # New Year
    "new year": ["new year", "new years", "party", "celebration"],
    "new years": ["new year", "new years", "party"],
    # St. Patrick's Day
    "st patricks": ["st patricks", "irish", "celtic"],
    "st. patrick": ["st patricks", "irish", "celtic"],
    "irish": ["irish", "celtic", "st patricks"],
    # Cinco de Mayo
    "cinco de mayo": ["cinco de mayo", "mexican", "fiesta"],
    # Summer/seasonal
    "summer": ["summer", "beach", "pool party"],
    "winter": ["winter", "cozy", "fireplace"],
    "fall": ["fall", "autumn", "cozy"],
    "spring": ["spring", "easter"],
}


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
        song_on_album = arguments.get("song_on_album", "")

        # DEBUG: Log raw arguments received from LLM
        _LOGGER.warning("MUSIC DEBUG: Raw arguments from LLM: %s", arguments)
        _LOGGER.warning("MUSIC DEBUG: Extracted - action='%s', query='%s', room='%s'", action, query, room)

        # DEFENSIVE: ALWAYS strip room phrases from query - LLM often includes them
        # This handles cases like query="Young Dolph in the living room"
        # Strip regardless of whether room param is set or not
        if query:
            # Try to extract room from end of query - handles "in the X" pattern
            # Use word boundary matching for multi-word rooms
            room_strip_pattern = r'\s+in\s+the\s+(.+?)\s*$'
            match = re.search(room_strip_pattern, query, flags=re.IGNORECASE)
            _LOGGER.warning("MUSIC DEBUG: Regex match on query='%s': %s", query, match)
            if match:
                potential_room = match.group(1).lower().strip()
                _LOGGER.warning("MUSIC DEBUG: Potential room extracted: '%s'", potential_room)
                configured_rooms = {r.lower() for r in self._players.keys()}
                all_known_rooms = COMMON_ROOM_NAMES | configured_rooms
                _LOGGER.warning("MUSIC DEBUG: Configured rooms: %s", configured_rooms)
                _LOGGER.warning("MUSIC DEBUG: Is '%s' in known rooms? %s", potential_room, potential_room in all_known_rooms)

                if potential_room in all_known_rooms or any(potential_room in r or r in potential_room for r in all_known_rooms):
                    original_query = query
                    query = re.sub(room_strip_pattern, '', query, flags=re.IGNORECASE).strip()
                    if not room:
                        room = potential_room
                    _LOGGER.warning("MUSIC DEBUG: Stripped room - query='%s' → '%s', room='%s'", original_query, query, room)

        _LOGGER.warning("MUSIC DEBUG: Final - action='%s', query='%s', room='%s'", action, query, room)

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
                return {"status": "debounced", "response_text": f"Command '{action}' ignored (duplicate)"}

        self._last_music_command = action
        self._last_music_command_time = now

        try:
            _LOGGER.info("=== MUSIC: %s ===", action.upper())

            # Determine target player(s)
            target_players = self._find_target_players(room)

            if action == "play":
                return await self._play(query, media_type, room, shuffle, target_players, artist, album, song_on_album)
            elif action == "pause":
                return await self._pause(all_players, target_players if target_players else None)
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
        """Find target players for a room (case-insensitive)."""
        room_lower = room.lower()

        # First try exact match (case-insensitive)
        for rname, pid in self._players.items():
            if room_lower == rname.lower():
                return [pid]

        # Then try partial match (case-insensitive)
        if room:
            for rname, pid in self._players.items():
                rname_lower = rname.lower()
                if room_lower in rname_lower or rname_lower in room_lower:
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
        # Raw media player entities may not support all playback controls
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

    async def _wait_for_playback_start(self, player: str, timeout: float = 3.0) -> bool:
        """Wait for player to reach 'playing' state after play_media call.

        Some media players need extra time to initialize after receiving a play command.
        This method polls the player state to ensure playback has actually started.

        Args:
            player: The media_player entity_id
            timeout: Max seconds to wait (default 3.0)

        Returns:
            True if player reached 'playing' state, False if timeout
        """
        poll_interval = 0.3
        elapsed = 0.0
        state = None

        while elapsed < timeout:
            state = self._hass.states.get(player)
            if state and state.state == "playing":
                _LOGGER.info("Player %s confirmed playing after %.1fs", player, elapsed)
                return True
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Log but don't fail - the command was sent, player may still be initializing
        current_state = state.state if state else "unknown"
        _LOGGER.warning("Player %s did not reach 'playing' state within %.1fs (state: %s)",
                       player, timeout, current_state)
        return False

    async def _play_media(
        self,
        player: str,
        media_id: str,
        media_type: str
    ) -> bool:
        """Play media via Music Assistant.

        Args:
            player: The media_player entity_id
            media_id: The URI/ID of the media to play
            media_type: The type of media (track, album, artist, playlist)

        Returns:
            True if command was sent successfully
        """
        _LOGGER.info("Playing media: uri='%s', type='%s' on %s",
                    media_id, media_type, player)

        await self._hass.services.async_call(
            "music_assistant", "play_media",
            {"media_id": media_id, "media_type": media_type, "enqueue": "replace", "radio_mode": False},
            target={"entity_id": player},
            blocking=True
        )

        return True

    async def _play_on_players(self, target_players: list[str], uri: str, media_type: str, shuffle: bool = False) -> None:
        """Play media on target players with appropriate shuffle setting."""
        for player in target_players:
            await self._play_media(player, uri, media_type)
            if media_type == "album":
                await self._hass.services.async_call(
                    "media_player", "shuffle_set",
                    {"entity_id": player, "shuffle": False},
                    blocking=True
                )
            elif shuffle:
                await self._hass.services.async_call(
                    "media_player", "shuffle_set",
                    {"entity_id": player, "shuffle": True},
                    blocking=True
                )

    async def _call_media_service(self, entity_id: str, service: str) -> None:
        """Call a media_player service using area targeting when available."""
        area_id = self._get_area_id(entity_id)
        if area_id:
            _LOGGER.info("%s via area: %s", service, area_id)
            await self._hass.services.async_call(
                "media_player", service, {},
                target={"area_id": area_id}, blocking=True)
        else:
            _LOGGER.info("No area found, %s via entity: %s", service, entity_id)
            await self._hass.services.async_call(
                "media_player", service, {},
                target={"entity_id": entity_id}, blocking=True)

    async def _play(self, query: str, media_type: str, room: str, shuffle: bool, target_players: list[str], artist: str = "", album: str = "", song_on_album: str = "") -> dict:
        """Play music via Music Assistant with search-first for accuracy.

        Searches Music Assistant first to find the exact track/album/artist,
        then plays the found result and returns the actual name.

        Smart album features:
        - "latest/last/newest album by X" → finds most recent album
        - "first/oldest/debut album by X" → finds earliest album
        - song_on_album: finds album containing a specific song
        """
        if not query and not song_on_album:
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
        album_ordinal = None  # For "second", "third", etc.
        album_type_filter = None  # For "studio", "live", "compilation", etc.
        album_year = None
        query_lower = query.lower()
        latest_keywords = ["latest", "last", "newest", "new", "most recent", "recent"]
        first_keywords = ["first", "oldest", "debut", "earliest"]

        if media_type == "album":
            # Check for ordinals (second, third, etc.)
            for ordinal_word, ordinal_num in ORDINALS.items():
                if ordinal_word in query_lower:
                    if ordinal_word in ("first", "1st"):
                        album_modifier = "first"
                        _LOGGER.info("Detected album modifier: 'first' (ordinal)")
                    else:
                        album_ordinal = ordinal_num
                        _LOGGER.info("Detected album ordinal: %d (keyword: %s)", ordinal_num, ordinal_word)
                    break

            # Check for latest/first keywords (if not already set by ordinal)
            if not album_modifier and not album_ordinal:
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

            # Check for album type keywords (studio, live, compilation, etc.)
            for type_keyword, type_value in ALBUM_TYPE_KEYWORDS.items():
                if type_keyword in query_lower:
                    album_type_filter = type_value
                    _LOGGER.info("Detected album type filter: '%s' (keyword: %s)", type_value, type_keyword)
                    break

            # Check for year-based album requests (e.g., "2020 album", "album from 2019")
            album_year = None
            year_patterns = [
                r'\b(19[5-9]\d|20[0-2]\d)\s*album',  # "2020 album"
                r'album\s*(?:from|in|of)?\s*(19[5-9]\d|20[0-2]\d)',  # "album from 2020"
                r'\b(19[5-9]\d|20[0-2]\d)\b',  # Just a year in the query
            ]
            for pattern in year_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    album_year = int(match.group(1))
                    _LOGGER.info("Detected album year: %d", album_year)
                    break

        try:
            # Get Music Assistant config entry
            ma_entries = self._hass.config_entries.async_entries("music_assistant")
            if not ma_entries:
                return {"error": "Music Assistant integration not found"}
            ma_config_entry_id = ma_entries[0].entry_id

            # Handle song_on_album: find album by searching for a song on it
            if song_on_album and media_type == "album":
                _LOGGER.info("Finding album by song: '%s' by '%s'", song_on_album, artist)

                # Search for the track
                track_search_query = f"{song_on_album} {artist}" if artist else song_on_album
                search_result = await self._hass.services.async_call(
                    "music_assistant", "search",
                    {"config_entry_id": ma_config_entry_id, "name": track_search_query, "media_type": ["track"], "limit": 10},
                    blocking=True, return_response=True
                )

                tracks = _parse_ma_results(search_result, "track")
                if tracks:
                        # Score tracks to find best match
                        song_lower = song_on_album.lower()
                        artist_lower = artist.lower() if artist else ""

                        def score_track(track):
                            score = 0
                            track_name = (track.get("name") or track.get("title") or "").lower()
                            track_artist = _extract_artist(track, lowercase=True)

                            # Song name match
                            if song_lower == track_name:
                                score += 100
                            elif song_lower in track_name:
                                score += 50

                            # Artist match
                            if artist_lower:
                                if artist_lower == track_artist:
                                    score += 100
                                elif artist_lower in track_artist or track_artist in artist_lower:
                                    score += 50

                            return score

                        scored_tracks = [(score_track(t), t) for t in tracks]
                        scored_tracks.sort(key=lambda x: x[0], reverse=True)

                        if scored_tracks and scored_tracks[0][0] > 0:
                            best_track = scored_tracks[0][1]

                            # Extract album info from track
                            album_name = None
                            album_uri = None

                            if best_track.get("album"):
                                album_info = best_track["album"]
                                if isinstance(album_info, dict):
                                    album_name = _normalize_unicode(album_info.get("name") or album_info.get("title"))
                                    album_uri = album_info.get("uri") or album_info.get("media_id")
                                elif isinstance(album_info, str):
                                    album_name = album_info

                            # Get artist for display
                            found_artist = artist or _extract_artist(best_track)

                            if album_name:
                                _LOGGER.info("Found album '%s' containing song '%s'", album_name, song_on_album)

                                # If we have a URI, play directly
                                if album_uri:
                                    await self._play_on_players(target_players, album_uri, "album")
                                    return {"status": "playing", "response_text": f"Playing {album_name} by {found_artist} in the {room}"}

                                # Otherwise search for the album by name
                                album_search = await self._hass.services.async_call(
                                    "music_assistant", "search",
                                    {"config_entry_id": ma_config_entry_id, "name": f"{album_name} {found_artist}", "media_type": ["album"], "limit": 5},
                                    blocking=True, return_response=True
                                )

                                albums = _parse_ma_results(album_search, "album")
                                if albums:
                                    found_album = albums[0]
                                    found_album_name = _normalize_unicode(found_album.get("name") or found_album.get("title"))
                                    found_album_uri = found_album.get("uri") or found_album.get("media_id")
                                    await self._play_on_players(target_players, found_album_uri, "album")
                                    return {"status": "playing", "response_text": f"Playing {found_album_name} by {found_artist} in the {room}"}

                return {"error": f"Could not find album containing '{song_on_album}'" + (f" by {artist}" if artist else "")}

            # Handle ordinal album requests (second, third album) or type-filtered requests (studio, live)
            # Also handles tag-based filtering (christmas, holiday albums), year-based and modifier-based requests
            if artist and (album_ordinal or album_type_filter or album or album_year or album_modifier):
                _LOGGER.warning("MUSIC DEBUG: Album search - ordinal=%s, type='%s', artist='%s', album_tag='%s', year=%s",
                               album_ordinal, album_type_filter, artist, album, album_year)

                discography = []

                # If album tag is specified (e.g., "christmas"), use MusicBrainz tag-based search
                # This properly finds albums like "Wrapped in Red" tagged as christmas
                if album:
                    discography = await _search_albums_by_tag_musicbrainz(artist, album)

                # If no results from tag search (or no album tag), get full discography
                # When using latest/first modifier without an explicit type filter, default to
                # studio albums so live performances and compilations are excluded
                if not discography:
                    effective_type_filter = album_type_filter
                    if not effective_type_filter and album_modifier:
                        effective_type_filter = "studio"
                    discography = await _get_artist_discography_musicbrainz(artist, effective_type_filter)

                    # If we have a discography but also an album filter, try name matching as fallback
                    if discography and album:
                        album_filter_lower = album.lower()
                        filtered_discog = [d for d in discography if album_filter_lower in d["name"].lower()]
                        if filtered_discog:
                            _LOGGER.warning("MUSIC DEBUG: Name-filtered discography from %d to %d albums matching '%s'",
                                           len(discography), len(filtered_discog), album)
                            discography = filtered_discog

                # Filter by year if specified
                if discography and album_year:
                    year_filtered = [d for d in discography if d["year"] == album_year]
                    if year_filtered:
                        _LOGGER.warning("MUSIC DEBUG: Year-filtered discography from %d to %d albums from %d",
                                       len(discography), len(year_filtered), album_year)
                        discography = year_filtered

                if not discography:
                    error_parts = []
                    if album:
                        error_parts.append(album)
                    if album_type_filter:
                        error_parts.append(album_type_filter)
                    if album_year:
                        error_parts.append(str(album_year))
                    qualifier = ' '.join(error_parts) + ' ' if error_parts else ''
                    return {"error": f"Could not find {qualifier}albums by {artist}"}

                # Select album based on ordinal or modifier
                target_album_name = None
                target_year = 0
                if album_ordinal:
                    if album_ordinal <= len(discography):
                        target_album_name = discography[album_ordinal - 1]["name"]
                        target_year = discography[album_ordinal - 1]["year"]
                        _LOGGER.warning("MUSIC DEBUG: Selected ordinal #%d album: '%s' (%d)", album_ordinal, target_album_name, target_year)
                    else:
                        return {"error": f"{artist} only has {len(discography)} {album + ' ' if album else ''}albums"}
                elif album_modifier == "latest":
                    target_album_name = discography[-1]["name"]
                    target_year = discography[-1]["year"]
                    _LOGGER.warning("MUSIC DEBUG: Selected latest album: '%s' (%d)", target_album_name, target_year)
                elif album_modifier == "first":
                    target_album_name = discography[0]["name"]
                    target_year = discography[0]["year"]
                    _LOGGER.warning("MUSIC DEBUG: Selected first album: '%s' (%d)", target_album_name, target_year)
                elif album_year and discography:
                    # Year was specified, use the (first) album from that year
                    target_album_name = discography[0]["name"]
                    target_year = discography[0]["year"]
                    _LOGGER.warning("MUSIC DEBUG: Selected album from %d: '%s'", album_year, target_album_name)
                elif discography:
                    # No modifier specified, just use the most recent from the tag/type search
                    target_album_name = discography[-1]["name"]
                    target_year = discography[-1]["year"]
                    _LOGGER.warning("MUSIC DEBUG: No modifier, using most recent: '%s' (%d)", target_album_name, target_year)

                if target_album_name:
                    # Search Music Assistant for this specific album
                    search_result = await self._hass.services.async_call(
                        "music_assistant", "search",
                        {"config_entry_id": ma_config_entry_id, "name": f"{target_album_name} {artist}", "media_type": ["album"], "limit": 5},
                        blocking=True, return_response=True
                    )

                    albums = _parse_ma_results(search_result, "album")
                    if albums:
                        # Find best match
                        target_lower = target_album_name.lower()
                        best_album = None
                        for alb in albums:
                            alb_name = (alb.get("name") or alb.get("title") or "").lower()
                            if target_lower in alb_name or alb_name in target_lower:
                                best_album = alb
                                break
                        if not best_album:
                            best_album = albums[0]

                        found_name = _normalize_unicode(best_album.get("name") or best_album.get("title"))
                        found_uri = best_album.get("uri") or best_album.get("media_id")
                        await self._play_on_players(target_players, found_uri, "album")
                        return {"status": "playing", "response_text": f"Playing {found_name} by {artist} in the {room}"}

                    return {"error": f"Found '{target_album_name}' in MusicBrainz but not in your music library"}

                return {"error": f"Could not find {album + ' ' if album else ''}albums by {artist}"}

            # Handle smart album search (latest/first album by artist)
            if album_modifier and artist:
                _LOGGER.warning("MUSIC DEBUG: Smart album search - modifier='%s', artist='%s', album_filter='%s'", album_modifier, artist, album)

                # Search for albums by artist only
                search_result = await self._hass.services.async_call(
                    "music_assistant", "search",
                    {"config_entry_id": ma_config_entry_id, "name": artist, "media_type": ["album"], "limit": 20},
                    blocking=True, return_response=True
                )

                albums = _parse_ma_results(search_result, "album")

                if albums:
                    artist_lower = artist.lower()
                    matching_albums = [a for a in albums if artist_lower in _extract_artist(a, lowercase=True) or _extract_artist(a, lowercase=True) in artist_lower]
                    _LOGGER.warning("MUSIC DEBUG: Found %d albums, %d match artist '%s'", len(albums), len(matching_albums), artist)

                    # Filter by album name if specified (e.g., "christmas" for christmas albums)
                    if album and matching_albums:
                        album_filter = album.lower()
                        filtered_albums = [
                            a for a in matching_albums
                            if album_filter in (a.get("name") or a.get("title") or "").lower()
                        ]
                        if filtered_albums:
                            _LOGGER.warning("MUSIC DEBUG: Filtered %d albums to %d matching '%s'",
                                        len(matching_albums), len(filtered_albums), album)
                            matching_albums = filtered_albums
                        else:
                            _LOGGER.warning("MUSIC DEBUG: No albums matched filter '%s', using all %d albums by artist",
                                           album, len(matching_albums))

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
                            _LOGGER.warning("MUSIC DEBUG: Album candidate: '%s' (year: %s, raw: %s)",
                                        alb_name, alb_year,
                                        alb.get("year") or alb.get("release_date") or "unknown")

                        albums_with_year = [(get_year(a), a) for a in matching_albums]

                        # Check if all years are 0 (missing) - if so, use MusicBrainz to look up years
                        if all(y == 0 for y, _ in albums_with_year):
                            _LOGGER.warning("MUSIC DEBUG: All years are 0, querying MusicBrainz for release dates...")
                            updated_albums = []
                            for _, alb in albums_with_year:
                                alb_name = alb.get("name") or alb.get("title") or ""
                                mb_year = await _lookup_album_year_musicbrainz(alb_name, artist)
                                updated_albums.append((mb_year, alb))
                                if mb_year > 0:
                                    _LOGGER.warning("MUSIC DEBUG: MusicBrainz found year %d for '%s'", mb_year, alb_name)
                            albums_with_year = updated_albums

                        # Sort: albums with year=0 go to end, then sort by year (desc for latest, asc for first)
                        albums_with_year.sort(key=lambda x: (x[0] == 0, -x[0] if album_modifier == "latest" else x[0]))

                        if albums_with_year:
                            best_album = albums_with_year[0][1]
                            found_name = _normalize_unicode(best_album.get("name") or best_album.get("title"))
                            found_uri = best_album.get("uri") or best_album.get("media_id")
                            found_artist = artist
                            found_type = "album"

                            year = albums_with_year[0][0]
                            _LOGGER.warning("MUSIC DEBUG: Selected %s album: '%s' (year: %d) by '%s'", album_modifier, found_name, year, found_artist)

                            await self._play_on_players(target_players, found_uri, "album")
                            return {"status": "playing", "response_text": f"Playing {found_name} by {found_artist} in the {room}"}

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

                results = _parse_ma_results(search_result, try_type)

                if not results:
                    continue

                # Filter by album name if specified (e.g., "christmas" for christmas albums)
                if album and try_type == "album":
                    album_filter = album.lower()
                    filtered_results = [
                        r for r in results
                        if album_filter in (r.get("name") or r.get("title") or "").lower()
                    ]
                    if filtered_results:
                        _LOGGER.info("Filtered %d albums to %d matching '%s'",
                                    len(results), len(filtered_results), album)
                        results = filtered_results
                    else:
                        _LOGGER.warning("No albums matched filter '%s', using all %d results",
                                       album, len(results))

                query_lower = query.lower()
                artist_lower = artist.lower() if artist else ""

                # Score results to find best match
                def score_result(item):
                    score = 0
                    item_name = (item.get("name") or item.get("title") or "").lower()
                    item_artist = _extract_artist(item, lowercase=True)

                    # Exact query match in name
                    if query_lower == item_name:
                        score += 100
                    elif query_lower in item_name:
                        score += 50
                    else:
                        # Word-based matching: check if key words from query appear in item name
                        # This handles cases like "wicked soundtrack" matching "Wicked: The Soundtrack"
                        query_words = [w for w in query_lower.split() if len(w) > 2]  # Skip short words
                        if query_words:
                            matches = sum(1 for w in query_words if w in item_name)
                            if matches == len(query_words):
                                score += 40  # All key words match
                            elif matches > 0:
                                score += 20 * matches  # Partial word matches

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
                    found_name = _normalize_unicode(best_match.get("name") or best_match.get("title"))
                    found_uri = best_match.get("uri") or best_match.get("media_id")
                    found_type = try_type

                    found_artist = _extract_artist(best_match) or found_artist

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
            await self._play_on_players(target_players, found_uri, found_type, shuffle=shuffle)

            return {"status": "playing", "response_text": f"Playing {display_name} in the {room}"}

        except Exception as e:
            _LOGGER.error("Play search/play error: %s", e, exc_info=True)
            return {"error": f"Failed to find or play music: {str(e)}"}

    async def _pause(self, all_players: list[str], target_players: list[str] | None = None) -> dict:
        """Pause music - uses area targeting like HA native intents.

        Smart selection logic:
        1. If target_players is specified (room was given), pause that specific player
        2. Otherwise, find all playing players and pause the most recently active one
           (based on media_position_updated_at timestamp)
        """
        _LOGGER.info("Looking for player in 'playing' state...")

        # If specific room was requested, only consider those players
        players_to_check = target_players if target_players else all_players

        # Find all playing players with their last update time
        playing_players: list[tuple[str, datetime | None]] = []
        for pid in players_to_check:
            state = self._hass.states.get(pid)
            if state and state.state == "playing":
                # Get the media_position_updated_at timestamp for smart selection
                last_updated = state.attributes.get("media_position_updated_at")
                _LOGGER.info("  %s → playing (last_updated: %s)", pid, last_updated)
                playing_players.append((pid, last_updated))

        if not playing_players:
            if target_players:
                return {"error": f"No music playing in {self._get_room_name(target_players[0])}"}
            return {"error": "No music is currently playing"}

        # Smart selection: pick the most recently active player
        # Sort by last_updated descending (most recent first), with None values last
        def sort_key(item: tuple[str, datetime | None]) -> tuple[int, datetime]:
            pid, ts = item
            if ts is None:
                return (1, datetime.min)  # None timestamps go last
            return (0, ts)

        playing_players.sort(key=sort_key, reverse=True)
        pid = playing_players[0][0]
        _LOGGER.info("Selected player to pause: %s (from %d playing)", pid, len(playing_players))

        await self._call_media_service(pid, "media_pause")

        self._last_paused_player = pid
        return {"status": "paused", "response_text": f"Paused in {self._get_room_name(pid)}"}

    async def _resume(self, all_players: list[str]) -> dict:
        """Resume music - uses area targeting like HA native intents."""
        _LOGGER.info("Looking for player to resume...")

        # Try last paused player first
        if self._last_paused_player and self._last_paused_player in all_players:
            _LOGGER.info("Resuming last paused player: %s", self._last_paused_player)
            await self._call_media_service(self._last_paused_player, "media_play")
            room_name = self._get_room_name(self._last_paused_player)
            self._last_paused_player = None
            return {"status": "resumed", "response_text": f"Resumed in {room_name}"}

        # Find any paused player
        paused = self._find_player_by_state("paused", all_players)
        if paused:
            await self._call_media_service(paused, "media_play")
            return {"status": "resumed", "response_text": f"Resumed in {self._get_room_name(paused)}"}

        return {"error": "No paused music to resume"}

    async def _stop(self, all_players: list[str]) -> dict:
        """Stop music - uses area targeting like HA native intents."""
        _LOGGER.info("Looking for player in 'playing' or 'paused' state...")

        for pid in all_players:
            state = self._hass.states.get(pid)
            if state and state.state in ("playing", "paused"):
                _LOGGER.info("  %s → %s", pid, state.state)

                await self._call_media_service(pid, "media_stop")

                return {"status": "stopped", "response_text": f"Stopped in {self._get_room_name(pid)}"}

        return {"response_text": "No music is playing"}

    async def _skip_next(self, all_players: list[str]) -> dict:
        """Skip to next track."""
        _LOGGER.info("Looking for player in 'playing' state...")
        playing = self._find_player_by_state("playing", all_players)
        if playing:
            await self._hass.services.async_call("media_player", "media_next_track", {"entity_id": playing})
            return {"status": "skipped", "response_text": "Skipped to next track"}
        return {"error": "No music is playing to skip"}

    async def _skip_previous(self, all_players: list[str]) -> dict:
        """Skip to previous track."""
        _LOGGER.info("Looking for player in 'playing' state...")
        playing = self._find_player_by_state("playing", all_players)
        if playing:
            await self._hass.services.async_call("media_player", "media_previous_track", {"entity_id": playing})
            return {"status": "skipped", "response_text": "Previous track"}
        return {"error": "No music is playing"}

    async def _restart_track(self, all_players: list[str]) -> dict:
        """Restart current track from beginning."""
        _LOGGER.info("Looking for player in 'playing' state to restart track...")
        playing = self._find_player_by_state("playing", all_players)
        if playing:
            await self._hass.services.async_call("media_player", "media_seek", {"entity_id": playing, "seek_position": 0})
            return {"status": "restarted", "response_text": "Bringing it back from the top"}
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
        return {"response_text": "No music currently playing"}

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

        return {"status": "transferred", "response_text": f"Music transferred to {self._get_room_name(target)}"}

    async def _shuffle(self, query: str, room: str, target_players: list[str]) -> dict:
        """Search for Spotify playlist by artist, genre, or holiday and play shuffled.

        IMPORTANT: This ONLY searches for Spotify playlists - no fallback to artist.
        Returns the exact playlist title for verbatim announcement.

        Holiday support: Detects holiday keywords (christmas, halloween, etc.) and
        searches for themed playlists with more flexible matching.
        """
        if not query:
            return {"error": "No search query specified for shuffle"}
        if not target_players:
            return {"error": f"No room specified. Available: {', '.join(self._players.keys())}"}

        _LOGGER.info("Searching Spotify for playlist matching: %s", query)

        # Detect holiday keywords in query
        query_lower = query.lower()
        detected_holiday = None
        holiday_search_terms = []
        for keyword, search_terms in HOLIDAY_KEYWORDS.items():
            if keyword in query_lower:
                detected_holiday = keyword
                holiday_search_terms = search_terms
                _LOGGER.info("Detected holiday keyword: '%s', search terms: %s", keyword, search_terms)
                break

        try:
            ma_entries = self._hass.config_entries.async_entries("music_assistant")
            if not ma_entries:
                return {"error": "Music Assistant integration not found"}
            ma_config_entry_id = ma_entries[0].entry_id

            all_playlists = []

            if detected_holiday:
                # For holidays, search using the FULL query first, then fallback to generic terms
                # This ensures "80s christmas" finds "80s Christmas" playlists, not just generic "Christmas Hits"
                search_queries = [query]  # Full query first (e.g., "80s christmas music")
                # Add the primary holiday term if not already the full query
                if holiday_search_terms[0] != query_lower:
                    search_queries.append(holiday_search_terms[0])

                for search_query in search_queries:
                    search_result = await self._hass.services.async_call(
                        "music_assistant", "search",
                        {"config_entry_id": ma_config_entry_id, "name": search_query, "media_type": ["playlist"], "limit": 15},
                        blocking=True, return_response=True
                    )
                    all_playlists.extend(_parse_ma_results(search_result, "playlist"))
                _LOGGER.info("Holiday search for '%s' found %d total playlists", query, len(all_playlists))
            else:
                # Standard search for non-holiday queries
                search_result = await self._hass.services.async_call(
                    "music_assistant", "search",
                    {"config_entry_id": ma_config_entry_id, "name": query, "media_type": ["playlist"], "limit": 10},
                    blocking=True, return_response=True
                )
                all_playlists = _parse_ma_results(search_result, "playlist")

            playlist_name = None
            playlist_uri = None
            playlist_owner = ""
            is_official = False

            if all_playlists:
                # Deduplicate playlists by URI
                seen_uris = set()
                playlists = []
                for p in all_playlists:
                    uri = p.get("uri") or p.get("media_id") or ""
                    if uri and uri not in seen_uris:
                        seen_uris.add(uri)
                        playlists.append(p)

                # Filter out playlists with "Radio" in the name - we don't want auto-generated radio playlists
                non_radio_playlists = [
                    p for p in playlists
                    if "radio" not in (p.get("name") or p.get("title") or "").lower()
                ]

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
                    # For holidays, also check holiday search terms
                    if detected_holiday:
                        for term in holiday_search_terms:
                            if term in name_lower:
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

                is_official = False
                chosen_playlist = None

                if detected_holiday:
                    # For holiday playlists, score by how well they match the FULL query
                    # "80s christmas music" should prefer "80s Christmas" over generic "Christmas Hits"
                    query_words = [w for w in query_lower.split() if len(w) >= 3 and w not in ('the', 'and', 'for', 'music', 'playlist', 'in')]

                    def score_holiday_playlist(p):
                        name = (p.get("name") or p.get("title") or "").lower()
                        score = 0
                        # Score for each query word found in playlist name
                        for word in query_words:
                            if word in name:
                                score += 10
                        # Bonus for official Spotify playlists
                        if (p.get("owner") or "").lower() == "spotify":
                            score += 5
                        return score

                    # Score all playlists
                    scored_playlists = [(score_holiday_playlist(p), p) for p in non_radio_playlists]
                    scored_playlists.sort(key=lambda x: x[0], reverse=True)

                    # Log top matches for debugging
                    for score, p in scored_playlists[:5]:
                        _LOGGER.info("Holiday playlist score %d: '%s'", score, p.get("name") or p.get("title"))

                    if scored_playlists and scored_playlists[0][0] > 0:
                        chosen_playlist = scored_playlists[0][1]
                        is_official = (chosen_playlist.get("owner") or "").lower() == "spotify"
                        _LOGGER.info("Selected holiday playlist by score: '%s' (score: %d)",
                                   chosen_playlist.get("name"), scored_playlists[0][0])
                    elif non_radio_playlists:
                        # Last resort: first available playlist from search
                        chosen_playlist = non_radio_playlists[0]
                        _LOGGER.info("Using first available holiday playlist")
                else:
                    # Standard playlist selection (strict official-only)
                    if official_playlists:
                        # Among official, prefer ones with query in name
                        official_with_name = [p for p in official_playlists if name_matches_query(p.get("name") or p.get("title") or "")]
                        chosen_playlist = official_with_name[0] if official_with_name else official_playlists[0]
                        is_official = True
                        _LOGGER.info("Found official Spotify playlist")

                # If no playlist found
                if not chosen_playlist:
                    if detected_holiday:
                        _LOGGER.warning("No %s playlist found", detected_holiday)
                        return {"error": f"Could not find a {detected_holiday} playlist. Try a different holiday search."}
                    else:
                        _LOGGER.warning("No official Spotify playlist found for '%s'", query)
                        return {"error": f"Could not find an official Spotify playlist for '{query}'. Try 'play {query}' instead to play the artist directly."}

                # Get the EXACT playlist title for verbatim announcement
                playlist_name = chosen_playlist.get("name") or chosen_playlist.get("title")
                playlist_uri = chosen_playlist.get("uri") or chosen_playlist.get("media_id")
                playlist_owner = chosen_playlist.get("owner", "")
                _LOGGER.info("Found Spotify playlist: '%s' (owner: %s)", playlist_name, playlist_owner)

            # NO artist fallback - shuffle is ONLY for playlists
            if not playlist_uri:
                return {"error": f"Could not find a Spotify playlist matching '{query}'. Try a different artist or genre."}

            player = target_players[0]
            _LOGGER.info("Playing playlist '%s' shuffled on %s", playlist_name, player)

            await self._play_media(player, playlist_uri, "playlist")

            await self._hass.services.async_call(
                "media_player", "shuffle_set",
                {"entity_id": player, "shuffle": True},
                blocking=True
            )

            # Return the EXACT playlist title for verbatim announcement
            # Include room name and confirm it's an official Spotify playlist
            room_suffix = f" in the {room}" if room else ""
            return {
                "status": "shuffling",
                "playlist_title": playlist_name,
                "playlist_owner": playlist_owner,
                "is_official_playlist": is_official,
                "room": room,
                "response_text": f"Playing {playlist_name}{room_suffix}"
            }

        except Exception as search_err:
            _LOGGER.error("Shuffle search/play error: %s", search_err, exc_info=True)
            return {"error": f"Failed to find or play playlist: {str(search_err)}"}
