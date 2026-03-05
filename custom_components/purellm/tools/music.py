"""Music control tool handler."""
from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from datetime import datetime
from typing import Any, TYPE_CHECKING

from urllib.parse import quote

from homeassistant.components.media_player import MediaPlayerEntityFeature
from homeassistant.helpers import entity_registry as er, device_registry as dr
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from ..utils.helpers import COMMON_ROOM_NAMES
from ..utils.http_client import fetch_json, CACHE_TTL_LONG

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


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


def _strip_accents(text: str) -> str:
    """Strip accents/diacritics from text for fuzzy matching.

    Converts characters like á→a, é→e, í→i, ó→o, ú→u, ñ→n so that
    accent-free queries (e.g. 'debi tirar mas fotos') match accented
    titles (e.g. 'DeBÍ TiRAR MáS fOtOs').
    """
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# Roman ↔ Arabic numeral mapping for album name matching (e.g. "Culture 3" ↔ "Culture III")
_ROMAN_TO_ARABIC = {
    "i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5",
    "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10",
}
_ARABIC_TO_ROMAN = {v: k for k, v in _ROMAN_TO_ARABIC.items()}


def _normalize_numerals(text: str) -> str:
    """Normalize Roman numerals to Arabic numbers for consistent matching.

    Converts 'III' → '3', 'II' → '2', etc. so that 'Culture 3' matches 'Culture III'.
    Processes longest matches first to avoid 'III' being partially matched as 'I'.
    """
    if not text:
        return ""
    # Replace Arabic → Roman first isn't needed; normalize everything to Arabic.
    # Process longest roman numerals first (viii before vi before i)
    words = text.split()
    result = []
    for word in words:
        lower = word.lower()
        if lower in _ROMAN_TO_ARABIC:
            result.append(_ROMAN_TO_ARABIC[lower])
        elif lower in _ARABIC_TO_ROMAN:
            # Already Arabic, keep as-is
            result.append(word)
        else:
            result.append(word)
    return " ".join(result)


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

# Theme keywords for album filtering (broader than holidays — includes album styles)
ALBUM_THEME_KEYWORDS = {
    "christmas": ["christmas", "xmas", "holiday", "noel", "santa", "jingle", "merry"],
    "xmas": ["christmas", "xmas", "holiday", "noel"],
    "holiday": ["holiday", "christmas", "xmas"],
    "halloween": ["halloween", "spooky", "scary", "horror"],
    "live": ["live", "concert", "unplugged", "acoustic live", "in concert"],
    "acoustic": ["acoustic", "unplugged"],
    "deluxe": ["deluxe", "expanded", "special edition"],
    "remix": ["remix", "remixed", "reimagined"],
    "greatest hits": ["greatest hits", "best of", "essentials", "collection"],
    "soundtrack": ["soundtrack", "motion picture", "original score"],
}

# Ordinal words → index (0-based). "latest"/"newest" use -1 for reverse sort.
_ORDINALS = {
    "first": 0, "1st": 0,
    "second": 1, "2nd": 1,
    "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3,
    "fifth": 4, "5th": 4,
    "latest": -1, "newest": -1, "most recent": -1, "last": -1,
}

# MusicBrainz API configuration
_MB_BASE = "https://musicbrainz.org/ws/2"
_MB_USER_AGENT = "PureLLM-HomeAssistant/7.8.0 ( https://github.com/LosCV29/purellm )"

# Map our theme keys to MusicBrainz tag names for server-side + client-side filtering
_MB_THEME_TAGS: dict[str, list[str]] = {
    "christmas": ["christmas", "xmas", "holiday", "noel"],
    "xmas": ["christmas", "xmas", "holiday", "noel"],
    "holiday": ["holiday", "christmas", "xmas"],
    "halloween": ["halloween"],
    "live": ["live"],
    "acoustic": ["acoustic", "unplugged"],
    "soundtrack": ["soundtrack", "film score"],
}


async def _musicbrainz_themed_albums(
    session: Any,
    artist: str,
    theme: str,
    theme_keywords: list[str],
) -> list[dict]:
    """Query MusicBrainz for themed albums by an artist.

    Returns a list of dicts with keys: name, year, mb_id — sorted by year.
    Returns empty list on any failure (network, no results, etc).
    """
    # Build Lucene query: artist:"X" AND primarytype:album
    mb_tags = _MB_THEME_TAGS.get(theme, [theme])
    tag_clause = " OR ".join(f'tag:"{t}"' for t in mb_tags)
    query = f'artist:"{artist}" AND primarytype:album AND ({tag_clause})'
    url = f"{_MB_BASE}/release-group?query={quote(query)}&fmt=json&limit=100"

    _LOGGER.info("MUSICBRAINZ: Searching: %s", query)
    data, status = await fetch_json(
        session, url,
        headers={"User-Agent": _MB_USER_AGENT, "Accept": "application/json"},
        cache_ttl=CACHE_TTL_LONG,
    )
    if not data or status != 200:
        _LOGGER.warning("MUSICBRAINZ: Search failed (status=%s)", status)
        return []

    release_groups = data.get("release-groups", [])
    if not release_groups:
        _LOGGER.info("MUSICBRAINZ: No release-groups found for query")
        return []

    # Filter and extract relevant albums
    artist_lower = _strip_accents(artist.lower())
    results: list[dict] = []
    seen_titles: set[str] = set()

    for rg in release_groups:
        # Verify artist match (MusicBrainz search can be fuzzy)
        rg_artists = rg.get("artist-credit", [])
        rg_artist_name = ""
        for ac in rg_artists:
            a = ac.get("artist", {}) if isinstance(ac, dict) else {}
            rg_artist_name = a.get("name", "")
            break
        if not rg_artist_name:
            continue
        rg_artist_lower = _strip_accents(rg_artist_name.lower())
        if not (artist_lower in rg_artist_lower or rg_artist_lower in artist_lower):
            continue

        title = rg.get("title", "").strip()
        if not title:
            continue

        # Skip compilations / secondary types we don't want
        secondary = [s.lower() for s in (rg.get("secondary-types") or [])]
        if "compilation" in secondary or "dj-mix" in secondary:
            continue

        # Deduplicate by normalized title
        norm = re.sub(r'\s*\(.*?\)', '', title.lower()).strip()
        if norm in seen_titles:
            continue
        seen_titles.add(norm)

        # Verify theme match: check tags AND title keywords
        rg_tags = {t.get("name", "").lower() for t in (rg.get("tags") or [])}
        tag_match = any(kw in rg_tags for kw in mb_tags)
        title_match = any(kw in title.lower() for kw in theme_keywords)
        if not tag_match and not title_match:
            continue

        # Parse year from first-release-date
        frd = rg.get("first-release-date", "")
        year = 0
        if frd and len(frd) >= 4:
            try:
                year = int(frd[:4])
            except ValueError:
                pass

        results.append({"name": title, "year": year, "mb_id": rg.get("id", "")})
        _LOGGER.info("MUSICBRAINZ: Found '%s' (%d) tags=%s", title, year, rg_tags & set(mb_tags))

    # Sort by year (unknowns at end)
    results.sort(key=lambda r: r["year"] if r["year"] > 0 else 9999)
    _LOGGER.info("MUSICBRAINZ: %d themed albums found: %s", len(results),
                 [(r["name"], r["year"]) for r in results])
    return results


def _parse_ordinal_theme(text: str) -> tuple[int | None, str | None]:
    """Parse ordinal position and theme from user text.

    Returns (ordinal_index, theme) where:
    - ordinal_index: 0-based position (0=first, 1=second, -1=latest), or None
    - theme: theme keyword like "christmas", "live", etc., or None

    Examples:
        "first christmas album" → (0, "christmas")
        "second album" → (1, None)
        "latest live album" → (-1, "live")
        "play culture 3" → (None, None)
    """
    text_lower = text.lower()

    ordinal = None
    for word, idx in _ORDINALS.items():
        if word in text_lower:
            ordinal = idx
            break

    theme = None
    # Check multi-word themes first (e.g. "greatest hits"), then single-word
    for keyword in sorted(ALBUM_THEME_KEYWORDS.keys(), key=len, reverse=True):
        if keyword in text_lower:
            theme = keyword
            break

    return ordinal, theme


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

        # DEBUG: Log raw arguments received from LLM
        _LOGGER.debug("MUSIC: Raw arguments from LLM: %s", arguments)
        _LOGGER.debug("MUSIC: Extracted - action='%s', query='%s', room='%s'", action, query, room)

        # DEFENSIVE: ALWAYS strip room phrases from query - LLM often includes them
        # This handles cases like query="Young Dolph in the living room"
        # Strip regardless of whether room param is set or not
        if query:
            # Try to extract room from end of query - handles "in the X" pattern
            # Use word boundary matching for multi-word rooms
            room_strip_pattern = r'\s+in\s+the\s+(.+?)\s*$'
            match = re.search(room_strip_pattern, query, flags=re.IGNORECASE)
            _LOGGER.debug("MUSIC: Regex match on query='%s': %s", query, match)
            if match:
                potential_room = match.group(1).lower().strip()
                _LOGGER.debug("MUSIC: Potential room extracted: '%s'", potential_room)
                configured_rooms = {r.lower() for r in self._players.keys()}
                all_known_rooms = COMMON_ROOM_NAMES | configured_rooms
                _LOGGER.debug("MUSIC: Configured rooms: %s", configured_rooms)
                _LOGGER.debug("MUSIC: Is '%s' in known rooms? %s", potential_room, potential_room in all_known_rooms)

                if potential_room in all_known_rooms or any(potential_room in r or r in potential_room for r in all_known_rooms):
                    original_query = query
                    query = re.sub(room_strip_pattern, '', query, flags=re.IGNORECASE).strip()
                    if not room:
                        room = potential_room
                    _LOGGER.debug("MUSIC: Stripped room - query='%s' → '%s', room='%s'", original_query, query, room)

        _LOGGER.debug("MUSIC: Final - action='%s', query='%s', room='%s'", action, query, room)

        # DEFENSIVE: If user said "album" in their original request, ensure album param
        # is set so _play() treats it as an album request. The LLM often strips "album"
        # from the query param and may set media_type wrong, which causes the smart
        # override in _play() to convert to "track" (playing a single song).
        # Check BOTH the original user text AND the LLM's query for "album".
        user_text = arguments.pop("_user_text", "")
        album_pattern = r'\balbum\b'
        has_album_intent = (
            action == "play" and
            media_type != "album" and
            (re.search(album_pattern, user_text, flags=re.IGNORECASE) or
             re.search(album_pattern, query, flags=re.IGNORECASE))
        )
        if has_album_intent:
            _LOGGER.info("MUSIC: Detected 'album' in user text, forcing media_type='album'")
            media_type = "album"
            # Strip "album" from query if present so it doesn't interfere with search
            query = re.sub(album_pattern, '', query, flags=re.IGNORECASE).strip()
            if not album:
                album = query

        # Detect ordinal/themed album requests from original user text
        # e.g. "play Kelly Clarkson's first christmas album in the living room"
        ordinal, theme = _parse_ordinal_theme(user_text) if user_text else (None, None)
        if ordinal is not None or theme:
            _LOGGER.info("MUSIC: Detected ordinal=%s, theme=%s from user text", ordinal, theme)

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
                # Try themed/ordinal album search if detected
                # e.g. "play Kelly Clarkson's first christmas album"
                if (ordinal is not None or theme) and artist and media_type == "album":
                    themed_result = await self._find_themed_album(artist, ordinal, theme)
                    if themed_result:
                        found_name = _normalize_unicode(themed_result.get("name") or themed_result.get("title"))
                        found_uri = themed_result.get("uri") or themed_result.get("media_id")
                        found_artist = _extract_artist(themed_result) or artist
                        if found_uri:
                            if not target_players:
                                return {"error": f"Unknown room: {room}. Available: {', '.join(self._players.keys())}"}
                            await self._play_on_players(target_players, found_uri, "album", shuffle=shuffle)
                            display_name = f"{found_name} by {found_artist}"
                            return {"status": "playing", "response_text": f"Playing {display_name} in the {room}"}
                    _LOGGER.info("MUSIC: Themed album search failed, falling back to normal search")

                return await self._play(query, media_type, room, shuffle, target_players, artist, album)
            elif action == "pause":
                return await self._pause(all_players, target_players if target_players else None)
            elif action == "resume":
                return await self._resume(all_players)
            elif action == "stop":
                return await self._stop(all_players, target_players if target_players else None)
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
            # Set shuffle BEFORE playing so the album starts from track 1
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
            await self._play_media(player, uri, media_type)

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

    async def _find_themed_album(
        self, artist: str, ordinal: int | None, theme: str | None,
    ) -> dict | None:
        """Find a themed/ordinal album using MusicBrainz + Music Assistant.

        Strategy:
        1. Query MusicBrainz release-groups to identify the correct album name
           (MusicBrainz has rich genre/tag metadata — e.g. "christmas" tags).
        2. Pick the album by ordinal from the MusicBrainz results (sorted by year).
        3. Search Music Assistant for that exact album name to get a playable URI.
        4. If MusicBrainz fails, fall back to MA-only search with theme filtering.

        Returns the matching MA album dict, or None if not found.
        """
        ma_entries = self._hass.config_entries.async_entries("music_assistant")
        if not ma_entries:
            return None
        ma_config_entry_id = ma_entries[0].entry_id
        theme_keywords = ALBUM_THEME_KEYWORDS.get(theme, [theme]) if theme else []

        # ── Step 1: MusicBrainz lookup (identifies the album name) ──
        mb_album_name: str | None = None
        if theme:
            try:
                session = async_get_clientsession(self._hass)
                mb_albums = await _musicbrainz_themed_albums(
                    session, artist, theme, theme_keywords,
                )
                if mb_albums:
                    # Pick by ordinal
                    if ordinal is not None:
                        if ordinal == -1:
                            mb_pick = mb_albums[-1]
                        elif 0 <= ordinal < len(mb_albums):
                            mb_pick = mb_albums[ordinal]
                        else:
                            _LOGGER.info("MUSICBRAINZ: Ordinal %d out of range (have %d)", ordinal, len(mb_albums))
                            mb_pick = None
                    else:
                        mb_pick = mb_albums[0]

                    if mb_pick:
                        mb_album_name = mb_pick["name"]
                        _LOGGER.info("MUSICBRAINZ: Selected '%s' (%d)", mb_album_name, mb_pick.get("year", 0))
            except Exception as err:
                _LOGGER.warning("MUSICBRAINZ: Lookup failed, will fall back to MA-only: %s", err)

        # ── Step 2: Search Music Assistant for the identified album ──
        if mb_album_name:
            _LOGGER.info("THEMED ALBUM: Searching MA for MusicBrainz pick: '%s' by '%s'", mb_album_name, artist)
            ma_result = await self._search_ma_album_by_name(
                ma_config_entry_id, mb_album_name, artist,
            )
            if ma_result:
                _LOGGER.info("THEMED ALBUM: Found in MA: '%s' (uri=%s)",
                             ma_result.get("name"), ma_result.get("uri") or ma_result.get("media_id"))
                return ma_result
            _LOGGER.info("THEMED ALBUM: MusicBrainz pick '%s' not found in MA, falling back", mb_album_name)

        # ── Step 3: Fallback — MA-only search with theme filtering ──
        _LOGGER.info("THEMED ALBUM: Fallback — searching MA directly for '%s'", artist)
        return await self._find_themed_album_ma_only(
            ma_config_entry_id, artist, ordinal, theme, theme_keywords,
        )

    async def _search_ma_album_by_name(
        self, config_entry_id: str, album_name: str, artist: str,
    ) -> dict | None:
        """Search Music Assistant for a specific album by name and artist."""
        artist_lower = _strip_accents(artist.lower())

        # Try exact album name first, then album + artist
        for query in [album_name, f"{artist} {album_name}"]:
            search_result = await self._hass.services.async_call(
                "music_assistant", "search",
                {"config_entry_id": config_entry_id, "name": query, "media_type": ["album"], "limit": 10},
                blocking=True, return_response=True,
            )
            for r in _parse_ma_results(search_result, "album"):
                item_artist = _strip_accents(_extract_artist(r, lowercase=True))
                if not (artist_lower in item_artist or item_artist in artist_lower):
                    continue
                item_name = _strip_accents((r.get("name") or r.get("title") or "").lower())
                target_name = _strip_accents(album_name.lower())
                # Fuzzy: check if one contains the other (handles "... Deluxe Edition" variants)
                if target_name in item_name or item_name in target_name:
                    return r
        return None

    async def _find_themed_album_ma_only(
        self, config_entry_id: str, artist: str, ordinal: int | None,
        theme: str | None, theme_keywords: list[str],
    ) -> dict | None:
        """Fallback: find themed album using only Music Assistant search + filtering."""
        # Search broadly for albums by this artist
        search_result = await self._hass.services.async_call(
            "music_assistant", "search",
            {"config_entry_id": config_entry_id, "name": artist, "media_type": ["album"], "limit": 50},
            blocking=True, return_response=True,
        )
        results = _parse_ma_results(search_result, "album")

        # Second search with theme keyword for broader coverage
        if theme:
            theme_search = await self._hass.services.async_call(
                "music_assistant", "search",
                {"config_entry_id": config_entry_id, "name": f"{artist} {theme}", "media_type": ["album"], "limit": 25},
                blocking=True, return_response=True,
            )
            seen_uris = {(r.get("uri") or r.get("media_id")) for r in results}
            for r in _parse_ma_results(theme_search, "album"):
                uri = r.get("uri") or r.get("media_id")
                if uri and uri not in seen_uris:
                    results.append(r)
                    seen_uris.add(uri)

        if not results:
            _LOGGER.info("THEMED ALBUM (MA): No albums found for '%s'", artist)
            return None

        # Filter to correct artist, exclude singles/EPs, deduplicate
        artist_lower = _strip_accents(artist.lower())
        seen_names: set[str] = set()
        artist_albums = []
        for r in results:
            item_artist = _strip_accents(_extract_artist(r, lowercase=True))
            if not (artist_lower in item_artist or item_artist in artist_lower):
                continue
            album_name = (r.get("name") or r.get("title") or "").strip()
            album_type_val = (r.get("album_type") or "").lower()
            if album_type_val in ("single", "ep"):
                continue
            if re.search(r'\b-\s*single\b', album_name.lower()):
                continue
            norm_name = re.sub(r'\s*\(.*?\)\s*', '', album_name.lower()).strip()
            norm_name = re.sub(r'\s*[-–]\s*(deluxe|expanded|special|remaster).*$', '', norm_name, flags=re.IGNORECASE).strip()
            if norm_name in seen_names:
                continue
            seen_names.add(norm_name)
            artist_albums.append(r)

        if not artist_albums:
            _LOGGER.info("THEMED ALBUM (MA): No albums matched artist '%s'", artist)
            return None

        # Filter by theme
        if theme and theme_keywords:
            themed = []
            for r in artist_albums:
                name_lower = (r.get("name") or r.get("title") or "").lower()
                name_match = any(kw in name_lower for kw in theme_keywords)
                genre_match = False
                genres = (r.get("metadata") or {}).get("genres") or []
                if isinstance(genres, (list, set)):
                    genres_lower = {g.lower() for g in genres}
                    genre_match = any(kw in genres_lower for kw in theme_keywords)
                album_type = (r.get("album_type") or "").lower()
                type_match = theme in album_type
                if name_match or genre_match or type_match:
                    themed.append(r)
            if not themed:
                _LOGGER.info("THEMED ALBUM (MA): No albums matched theme '%s'", theme)
                return None
            artist_albums = themed

        # Sort by year
        artist_albums.sort(key=lambda r: r.get("year") if isinstance(r.get("year"), int) and r.get("year") > 0 else 9999)
        _LOGGER.info("THEMED ALBUM (MA): Sorted %d albums: %s",
                     len(artist_albums), [(r.get("name"), r.get("year")) for r in artist_albums])

        # Pick by ordinal
        if ordinal is not None:
            if ordinal == -1:
                return artist_albums[-1]
            if 0 <= ordinal < len(artist_albums):
                return artist_albums[ordinal]
            _LOGGER.info("THEMED ALBUM (MA): Ordinal %d out of range (have %d)", ordinal, len(artist_albums))
            return None
        return artist_albums[0]

    async def _play(self, query: str, media_type: str, room: str, shuffle: bool, target_players: list[str], artist: str = "", album: str = "") -> dict:
        """Play music via Music Assistant with search-first for accuracy.

        Searches Music Assistant first to find the exact track/album/artist,
        then plays the found result and returns the actual name.
        """
        # Sync query ↔ album so both code paths work regardless of how LLM maps params
        if not query and album:
            query = album
        if not query:
            return {"error": "No music query specified"}
        if not target_players:
            return {"error": f"Unknown room: {room}. Available: {', '.join(self._players.keys())}"}

        # Enforce valid media types
        valid_types = {"artist", "album", "track"}
        if media_type not in valid_types:
            media_type = "artist"

        # If album parameter was explicitly provided, this is an album request
        if album and media_type != "album":
            _LOGGER.info("Overriding media_type to 'album' since album parameter was specified")
            media_type = "album"

        # Smart override: if artist is specified with a query but media_type is "artist",
        # user likely wants a track: "Big Pimpin by Jay-Z" = track, not artist
        # BUT if user explicitly said "album", respect that choice
        if artist and query and media_type == "artist":
            media_type = "track"
            _LOGGER.info("Overriding media_type to 'track' since both query and artist specified")

        # Sync query → album for album requests so album filter works in search
        if media_type == "album" and query and not album:
            album = query

        try:
            # Get Music Assistant config entry
            ma_entries = self._hass.config_entries.async_entries("music_assistant")
            if not ma_entries:
                return {"error": "Music Assistant integration not found"}
            ma_config_entry_id = ma_entries[0].entry_id

            # Standard search
            # For albums with artist, search by album name alone first (not concatenated)
            # to avoid confusing Music Assistant's search with combined strings.
            # Artist matching is handled by the scoring function below.
            if media_type == "album" and artist:
                search_query = query
            else:
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

                # Fallback for albums: try combined query, then artist-only search
                # (handles accented names like "DeBÍ TiRAR MáS fOtOs")
                if not results and try_type == "album" and artist:
                    _LOGGER.info("Album-name search empty, trying combined query '%s %s'", query, artist)
                    combined_search = await self._hass.services.async_call(
                        "music_assistant", "search",
                        {"config_entry_id": ma_config_entry_id, "name": f"{query} {artist}", "media_type": ["album"], "limit": 10},
                        blocking=True, return_response=True
                    )
                    results = _parse_ma_results(combined_search, "album")

                if not results and try_type == "album" and artist:
                    _LOGGER.info("Combined search empty, trying artist-only search for '%s'", artist)
                    artist_search = await self._hass.services.async_call(
                        "music_assistant", "search",
                        {"config_entry_id": ma_config_entry_id, "name": artist, "media_type": ["album"], "limit": 20},
                        blocking=True, return_response=True
                    )
                    results = _parse_ma_results(artist_search, "album")

                if not results:
                    continue

                # Filter by album name if specified
                if album and try_type == "album":
                    album_filter = _normalize_numerals(_strip_accents(album.lower()))
                    filtered_results = [
                        r for r in results
                        if album_filter in _normalize_numerals(_strip_accents((r.get("name") or r.get("title") or "").lower()))
                    ]
                    if filtered_results:
                        _LOGGER.info("Filtered %d albums to %d matching '%s'",
                                    len(results), len(filtered_results), album)
                        results = filtered_results
                    else:
                        _LOGGER.warning("No albums matched filter '%s', using all %d results",
                                       album, len(results))

                query_lower = _normalize_numerals(_strip_accents(query.lower()))
                artist_lower = _strip_accents(artist.lower()) if artist else ""

                # Score results to find best match
                def score_result(item):
                    score = 0
                    item_name = _normalize_numerals(_strip_accents((item.get("name") or item.get("title") or "").lower()))
                    item_artist = _strip_accents(_extract_artist(item, lowercase=True))

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
                        elif item_artist:
                            # Artist was specified but doesn't match - penalize heavily
                            # to prevent e.g. Bublé's "Christmas" beating Clarkson's album
                            score -= 200

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
                return {"error": f"Could not find {media_type} matching '{query}'" + (f" by {artist}" if artist else "")}

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

    async def _stop(self, all_players: list[str], target_players: list[str] | None = None) -> dict:
        """Stop music - uses area targeting like HA native intents.

        Smart selection logic:
        1. If target_players is specified (room was given), stop that specific player
        2. Otherwise, find all playing/paused players and stop the most recently active one
           (based on media_position_updated_at timestamp)
        """
        _LOGGER.info("Looking for player in 'playing' or 'paused' state...")

        # If specific room was requested, only consider those players
        players_to_check = target_players if target_players else all_players

        # Find all playing/paused players with their last update time
        active_players: list[tuple[str, datetime | None]] = []
        for pid in players_to_check:
            state = self._hass.states.get(pid)
            if state and state.state in ("playing", "paused"):
                last_updated = state.attributes.get("media_position_updated_at")
                _LOGGER.info("  %s → %s (last_updated: %s)", pid, state.state, last_updated)
                active_players.append((pid, last_updated))

        if not active_players:
            if target_players:
                return {"error": f"No music playing in {self._get_room_name(target_players[0])}"}
            return {"error": "No music is currently playing"}

        # Smart selection: pick the most recently active player
        def sort_key(item: tuple[str, datetime | None]) -> tuple[int, datetime]:
            pid, ts = item
            if ts is None:
                return (1, datetime.min)
            return (0, ts)

        active_players.sort(key=sort_key, reverse=True)
        pid = active_players[0][0]
        _LOGGER.info("Selected player to stop: %s (from %d active)", pid, len(active_players))

        room_name = self._get_room_name(pid)
        try:
            await self._call_media_service(pid, "media_stop")
        except Exception as err:
            # Chromecast (and some other players) may throw on media_stop
            # even though the stop command was sent and worked. Give a
            # moment for the state to settle, then check the actual outcome.
            _LOGGER.warning("media_stop raised for %s: %s — checking actual state", pid, err)
            await asyncio.sleep(1)
            state = self._hass.states.get(pid)
            if state and state.state not in ("playing", "paused"):
                _LOGGER.info("Player %s is now %s — stop succeeded despite error", pid, state.state)
                return {"status": "stopped", "response_text": f"Stopped in {room_name}"}
            # Re-raise so the outer handler reports the real error
            raise

        return {"status": "stopped", "response_text": f"Stopped in {room_name}"}

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

        return {"status": "transferred", "response_text": f"Music transferred to {self._get_room_name(target)}"}

    async def _shuffle(self, query: str, room: str, target_players: list[str]) -> dict:
        """Search for Apple Music playlist by artist, genre, or holiday and play shuffled.

        IMPORTANT: This ONLY searches for playlists - no fallback to artist.
        Returns the exact playlist title for verbatim announcement.

        Holiday support: Detects holiday keywords (christmas, halloween, etc.) and
        searches for themed playlists with more flexible matching.
        """
        if not query:
            return {"error": "No search query specified for shuffle"}
        if not target_players:
            return {"error": f"No room specified. Available: {', '.join(self._players.keys())}"}

        _LOGGER.info("Searching Apple Music for playlist matching: %s", query)

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
                    name_lower = _strip_accents(playlist_name_str.lower())
                    query_norm = _strip_accents(query_lower)
                    # Exact query match
                    if query_norm in name_lower:
                        return True
                    # Match on individual words (handles typos like elliot vs elliott)
                    for word in query_words:
                        if len(word) >= 4 and _strip_accents(word) in name_lower:
                            return True
                    # For holidays, also check holiday search terms
                    if detected_holiday:
                        for term in holiday_search_terms:
                            if _strip_accents(term) in name_lower:
                                return True
                    return False

                # Priority 1: Official Apple Music curated playlists ("Essentials", "Best of...", owned by Apple Music)
                official_playlists = [
                    p for p in non_radio_playlists
                    if "apple" in (p.get("owner") or "").lower()
                    or (p.get("name") or p.get("title") or "").lower().endswith("essentials")
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
                        name = _strip_accents((p.get("name") or p.get("title") or "").lower())
                        score = 0
                        # Score for each query word found in playlist name
                        for word in query_words:
                            if _strip_accents(word) in name:
                                score += 10
                        # Bonus for official Apple Music playlists
                        if "apple" in (p.get("owner") or "").lower():
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
                        is_official = "apple" in (chosen_playlist.get("owner") or "").lower()
                        _LOGGER.info("Selected holiday playlist by score: '%s' (score: %d)",
                                   chosen_playlist.get("name"), scored_playlists[0][0])
                    elif non_radio_playlists:
                        # Last resort: first available playlist from search
                        chosen_playlist = non_radio_playlists[0]
                        _LOGGER.info("Using first available holiday playlist")
                else:
                    # Standard playlist selection — HUGELY prefer official Essentials/Best Of playlists,
                    # but fall back to best matching playlist if none exist
                    if official_playlists:
                        # Among official, prefer ones with query in name
                        official_with_name = [p for p in official_playlists if name_matches_query(p.get("name") or p.get("title") or "")]
                        chosen_playlist = official_with_name[0] if official_with_name else official_playlists[0]
                        is_official = True
                        _LOGGER.info("Found official Apple Music playlist")
                    elif matching_name_playlists:
                        # Fallback: best playlist with query/artist in the name
                        chosen_playlist = matching_name_playlists[0]
                        _LOGGER.info("No official playlist; falling back to name-matched: '%s'",
                                   chosen_playlist.get("name") or chosen_playlist.get("title"))
                    elif non_radio_playlists:
                        # Last resort: first non-radio playlist from search results
                        chosen_playlist = non_radio_playlists[0]
                        _LOGGER.info("No official or name-matched playlist; using first available: '%s'",
                                   chosen_playlist.get("name") or chosen_playlist.get("title"))

                # If no playlist found
                if not chosen_playlist:
                    if detected_holiday:
                        _LOGGER.warning("No %s playlist found", detected_holiday)
                        return {"error": f"Could not find a {detected_holiday} playlist. Try a different holiday search."}
                    else:
                        _LOGGER.warning("No official Apple Music playlist found for '%s'", query)
                        return {"error": f"Could not find an official Apple Music playlist for '{query}'. Try 'play {query}' instead to play the artist directly."}

                # Get the EXACT playlist title for verbatim announcement
                playlist_name = chosen_playlist.get("name") or chosen_playlist.get("title")
                playlist_uri = chosen_playlist.get("uri") or chosen_playlist.get("media_id")
                playlist_owner = chosen_playlist.get("owner", "")
                _LOGGER.info("Found Apple Music playlist: '%s' (owner: %s)", playlist_name, playlist_owner)

            # NO artist fallback - shuffle is ONLY for playlists
            if not playlist_uri:
                return {"error": f"Could not find an Apple Music playlist matching '{query}'. Try a different artist or genre."}

            _LOGGER.info("Playing playlist '%s' shuffled on %s", playlist_name, target_players)

            for player in target_players:
                # Set shuffle BEFORE playing so the playlist starts in random order
                await self._hass.services.async_call(
                    "media_player", "shuffle_set",
                    {"entity_id": player, "shuffle": True},
                    blocking=True
                )
                await self._play_media(player, playlist_uri, "playlist")

            # Return the EXACT playlist title for verbatim announcement
            # Include room name and confirm it's an official Apple Music playlist
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
