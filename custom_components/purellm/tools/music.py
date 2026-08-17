"""Music control tool handler."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from urllib.parse import quote

from homeassistant.components.media_player import MediaPlayerEntityFeature
from homeassistant.helpers import entity_registry as er, device_registry as dr
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from ..utils.helpers import COMMON_ROOM_NAMES
from ..utils.http_client import fetch_json, CACHE_TTL_LONG

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# The living-room player rides a Snapdroid snapcast client that can be dead
# (killed by STOP for the screensaver, or a Shield reboot) — the only player
# with a revive path (script.ensure_snapclient).
_SNAPCLIENT_PLAYER = "media_player.shield_android_tv"
# Snapserver JSON-RPC endpoint + the client name to check. The HA entity can
# read available/idle — even 'playing' — while the snapclient is dead, because
# MA's Native output link keeps the player alive and streams to nobody
# (2026-08-08 incident). Snapserver's own connected flag is the only truth.
_SNAPSERVER_ADDR = ("192.168.68.82", 1705)
_SNAPSERVER_CLIENT = "SHIELD Android TV"
# Settle time after a cold-started snapclient shows connected, before play.
_SNAPCLIENT_SETTLE_S = 1.5
# Where play-failure alerts go when playback can't be started at all.
_PLAY_FAILURE_NOTIFY = "mobile_app_pixel_9"


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


# Typographic punctuation → ASCII. Apple Music titles use curly quotes and
# en/em dashes ("I’m Up", "Ain’t It Funny"); voice/text queries arrive with the
# plain ASCII forms. Without this fold the two never compare equal and the
# correct item scores 0 on name and is rejected outright.
_PUNCT_FOLD = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "ʼ": "'", "´": "'", "`": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "―": "-", "−": "-",
    "…": "...", " ": " ",
})


def _strip_accents(text: str) -> str:
    """Strip accents/diacritics and fold typographic punctuation for fuzzy matching.

    Converts characters like á→a, é→e, í→i, ó→o, ú→u, ñ→n so that
    accent-free queries (e.g. 'debi tirar mas fotos') match accented
    titles (e.g. 'DeBÍ TiRAR MáS fOtOs').

    Also folds smart quotes/dashes to ASCII (’→', –→-) so a spoken/typed
    "I'm Up" matches Apple Music's "I’m Up".
    """
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text.translate(_PUNCT_FOLD))
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
        if not _artist_contains(artist_lower, rg_artist_lower):
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


def _artist_norm_variants(text: str) -> tuple[str, str]:
    """Two normalizations of an artist name: punctuation-deleted and punctuation-as-space.

    "K-Camp" → ("kcamp", "k camp"); "O.T. Genasis" → ("ot genasis", "o t genasis").
    Both are needed: deleting punctuation makes O.T.↔OT match, while turning it into
    a space makes K-Camp↔K CAMP match. Callers try every combination.
    """
    base = _strip_accents(text.lower())
    deleted = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", base)).strip()
    spaced = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", base)).strip()
    return deleted, spaced


_ONES_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS_WORDS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def _normalize_spelled_numbers(text: str) -> str:
    """Fold spelled-out numbers to digits: "forty two dog" → "42 dog".

    STT spells numbers out, but artists like 42 Dugg / 21 Savage / 2Pac are
    written with digits — so no text layer ever matched them until this
    existed (2026-07-29: "shuffle forty two dog" played a random playlist
    literally named "Forty two"). Used as an ADDITIONAL variant everywhere,
    never a replacement — "One Direction" style names stay findable raw.
    """
    if not text:
        return text
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i].lower()
        if w in _TENS_WORDS:
            nxt = words[i + 1].lower() if i + 1 < len(words) else ""
            if nxt in _ONES_WORDS and 1 <= _ONES_WORDS[nxt] <= 9:
                out.append(str(_TENS_WORDS[w] + _ONES_WORDS[nxt]))
                i += 2
                continue
            out.append(str(_TENS_WORDS[w]))
        elif w in _ONES_WORDS:
            out.append(str(_ONES_WORDS[w]))
        else:
            out.append(words[i])
        i += 1
    return " ".join(out)


# Music provider preference. Apple Music is the primary subscription; Spotify was
# added 2026-07-26 purely as a BACKUP for things Apple's catalog doesn't carry
# (mixtapes and indie releases — e.g. Wale's "Passive-Aggress Her", which exists
# on Spotify and not on Apple). Lower number = preferred.
#
# This is deliberately a TIE-BREAK, not a score penalty: the provider must never
# override a genuinely better title/artist match, it only decides which copy of
# an equally-good match to play. So Apple wins whenever it has the song, and
# Spotify is reached only when Apple's candidates score lower or don't exist.
_PROVIDER_PRIORITY = {"apple_music": 0, "spotify": 1}
_PROVIDER_FALLBACK_RANK = 5


def _provider_of(uri: str) -> str:
    """Provider domain from a Music Assistant URI.

    MA appends an instance id to the scheme when a provider is configured as an
    instance ("spotify--zjw5KojN://track/123", "apple_music--yQhFUrao://playlist/pl.x"),
    so the scheme cannot be compared directly.
    """
    if not uri or "://" not in uri:
        return ""
    return uri.split("://", 1)[0].split("--", 1)[0]


def _provider_rank(uri: str) -> int:
    """Sort rank for a URI's provider (lower = preferred)."""
    return _PROVIDER_PRIORITY.get(_provider_of(uri), _PROVIDER_FALLBACK_RANK)


def _explicit_rank(item: dict) -> int:
    """Sort rank for censorship (lower = preferred). NEVER play clean over explicit.

    This is an absolute rule, not a preference to be balanced against others:
    when two copies of the same recording exist, the explicit one always wins.
    Music Assistant reports `explicit` as True / False / None, and None is
    COMMON on Apple Music results even when a sibling copy is flagged — so
    unknown must rank between the two, never below clean.

    It sorts ahead of provider rank deliberately: an explicit copy on Spotify
    beats a clean copy on Apple. Do NOT add a tie-break above this one (a
    compilation/"Various Artists" penalty was proposed and REJECTED for exactly
    that reason — it would have picked the clean single over the explicit cut).
    """
    if item.get("explicit") is True:
        return 0
    if item.get("explicit") is False:
        return 2
    return 1


def _search_miss(media_type: str, query: str, artist: str = "") -> dict:
    """Terminal 'not in the catalog' result for a music search.

    A bare "no results" message reads to the model as "try another spelling",
    so it re-searches with variations until the tool-iteration cap and the turn
    dies with "Sorry, the LLM failed to respond" (2026-07-26, "Passive Aggres-Her"
    by Wale — a real song, just not on Apple Music). The instruction field makes
    giving up the explicit next step. Same pattern as tools/search.py.
    """
    label = f"'{query}'" + (f" by {artist}" if artist else "")
    return {
        "results": [],
        "message": f"No {media_type} found for {label}.",
        "instruction": (
            f"This {media_type} is NOT in the music catalog. Do NOT search again "
            f"with a different spelling, a shorter query, or a different media_type "
            f"— the catalog has already been checked. Tell the user you couldn't "
            f"find {label} and stop. Do not play something else instead."
        ),
    }


def _titles_resemble(query: str, title: str) -> bool:
    """Does a MusicBrainz result title plausibly answer the requested title?

    MusicBrainz's general fuzzy fallback ("<title> <artist>") returns ANY recording
    by a matching artist, so a title check is the only thing stopping an unrelated
    song from being treated as the canonical form of the request. Must stay loose
    enough for the misheard-title cases MB exists to fix (Rollin'↔Rolling) and tight
    enough to reject a different song ("Passive Aggressive"↛"Her").
    """
    if not query or not title:
        return False
    norm_q = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", _strip_accents(query.lower()))).strip()
    norm_t = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", _strip_accents(title.lower()))).strip()
    if not norm_q or not norm_t:
        return False
    # Containment covers "Her" ⊂ "Her Fault" and "<title> (Live)" style suffixes
    if norm_q in norm_t or norm_t in norm_q:
        return True
    q_words = norm_q.split()
    t_words = set(norm_t.split())
    # Token coverage: half the requested words must match a title word, allowing
    # a 4+ char shared prefix for spelling variants (rollin↔rolling).
    matched = 0
    for qw in q_words:
        if qw in t_words or any(
            min(len(qw), len(tw)) >= 4 and (qw.startswith(tw[:4]) or tw.startswith(qw[:4]))
            for tw in t_words
        ):
            matched += 1
    if matched * 2 >= len(q_words):
        return True
    # Whole-string similarity as a last resort for heavy STT garble on short
    # titles; folded comparison bridges c/k/z respelling on top of the garble.
    return (SequenceMatcher(None, norm_q, norm_t).ratio() >= 0.7
            or SequenceMatcher(None, _consonant_fold(norm_q), _consonant_fold(norm_t)).ratio() >= 0.7)


# Vowel substitutions ordered by how often speech-to-text confuses them.
_VOWEL_ORDER = "eaiou"
_CONSONANT_SWAPS = (("c", "k"), ("k", "c"), ("s", "z"), ("z", "s"),
                    ("ph", "f"), ("f", "ph"), ("y", "i"), ("i", "y"))
_PHONETIC_VOWELS = "aeiouy"


def _phonetic_key(text: str) -> str:
    """Coarse pronunciation key: consonant skeleton with vowels as placeholders.

    Equal keys mean "these two spellings could be the same word misheard" — NOT
    "these are the same word". Used only to VALIDATE a rescue candidate, never
    to rank one, because it is deliberately lossy ("dreka" and "drake" collide).
    """
    s = re.sub(r"[^a-z0-9\s]", "", _strip_accents((text or "").lower()))
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    s = s.replace("ph", "f").replace("gh", "f")
    s = s.replace("ck", "k").replace("sch", "sk")
    s = re.sub(r"c(?=[eiy])", "s", s)               # cent → sent
    s = s.replace("c", "k").replace("q", "k").replace("x", "ks")
    s = s.replace("z", "s")
    s = re.sub(r"(.)\1+", r"\1", s)                 # collapse doubled letters
    return re.sub(f"[{_PHONETIC_VOWELS}]+", "V", s)


def _phonetic_variants(text: str, limit: int = 8) -> list[str]:
    """Plausible respellings of a possibly-misheard title, most likely first.

    Bridges one or two bad phonemes from STT when NEITHER the catalog nor
    MusicBrainz tolerates the slip (2026-07-29: "Dreka" by Kevin Gates came
    through as "Drica" — Music Assistant search returned only unrelated Kevin
    Gates tracks, and MusicBrainz has no "Dreka" recording at all, so every
    existing fallback was a dead end).

    Every emitted variant shares the original's phonetic key, so this widens
    the search strictly within the set of spellings that SOUND the same. Capped
    at short titles: a long title has too many variants to sweep, and enough
    surviving words for the normal scorer to match on anyway.
    """
    s = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not s or len(s) > 24 or len(s.split()) > 2:
        return []
    key = _phonetic_key(s)
    if not key:
        return []
    seen = {s}
    out: list[str] = []

    def _push(cand: str) -> bool:
        """Add a candidate; return True once the cap is reached."""
        if cand in seen or _phonetic_key(cand) != key:
            return False
        seen.add(cand)
        out.append(cand)
        return len(out) >= limit

    # Consonant-folded seeds first, then vowel substitutions over every seed —
    # "drica" needs BOTH edits (c→k and i→e) before it reaches "dreka".
    seeds = [s]
    for a, b in _CONSONANT_SWAPS:
        if a in s:
            cand = s.replace(a, b)
            if cand not in seeds and _phonetic_key(cand) == key:
                seeds.append(cand)
    for seed in seeds[1:]:
        if _push(seed):
            return out
    for v in _VOWEL_ORDER:
        for seed in seeds:
            for i, ch in enumerate(seed):
                if ch in "aeiou" and ch != v and _push(seed[:i] + v + seed[i + 1:]):
                    return out
    return out


def _artist_contains(a: str, b: str) -> bool:
    """Containment check tolerant of punctuation differences (K-Camp ↔ K CAMP).

    Stricter than _artist_names_match (no word-overlap/prefix fuzz) — used where
    the caller only ever wanted "one name contains the other".
    """
    if not a or not b:
        return False
    for na in _artist_norm_variants(a):
        for nb in _artist_norm_variants(b):
            if len(na) >= 2 and len(nb) >= 2 and (na in nb or nb in na):
                return True
    return False


def _artist_resolution_plausible(requested: str, canonical: str) -> bool:
    """Could `canonical` be what the user meant by `requested`?

    Gate for MA's artist-name canonicalization, which always returns a best
    guess and so hands back an arbitrary artist for any name it doesn't know.
    Looser than _artist_names_match because real canonicalizations can share no
    whole word at all — "Tupac" → "2Pac" is the canonical example (ratio 0.667).

    The 0.6 cutoff was measured against both sets: the loosest true pair that
    _artist_names_match misses is Tupac/2Pac at 0.667, while the tightest wrong
    pair sits at 0.476 ("oh he did"/"The Weeknd") — comfortably separated.
    """
    if not requested or not canonical:
        return False
    if _artist_names_match(requested, canonical):
        return True
    a = _artist_norm_variants(requested)[1]
    b = _artist_norm_variants(canonical)[1]
    if not a or not b:
        return False
    return SequenceMatcher(None, a, b).ratio() >= 0.6


def _consonant_fold(text: str) -> str:
    """Fold consonant spellings STT picks interchangeably (c/k, z/s, ph/f).

    Unlike _phonetic_key this keeps vowels, so it stays precise enough for
    direct name equality: a faithful STT engine hears "Big K.R.I.T." as
    "Big Crit" (2026-08-04, Voxtral), and only the c↔k choice separates the
    normalized forms. Folding both sides makes them literally equal without
    the false-positive surface of vowel-collapsed keys (Dreka↔Drake).
    """
    s = text.replace("ph", "f")
    s = re.sub(r"c(?=[eiy])", "s", s)               # cent → sent
    s = s.replace("ck", "k").replace("c", "k").replace("z", "s")
    return re.sub(r"(.)\1+", r"\1", s)


def _artist_names_match(a: str, b: str) -> bool:
    """Fuzzy artist name comparison for voice/STT variations.

    Handles: OT Genesis↔O.T. Genasis, Jay Z↔JAY-Z, K-Camp↔K CAMP, Tupac↔2Pac Shakur.
    Strips punctuation/dots, then checks word overlap (>50% of words match).
    """
    if not a or not b:
        return False
    # Normalize: strip accents, punctuation, dots → just letters/numbers/spaces.
    # Punctuation is both deleted and spaced-out, and spelled-out numbers are
    # additionally folded to digits ("forty two dog" gains a "42 dog" variant so
    # it can match "42 Dugg" on the shared "42"). A match on any pairing counts.
    # Consonant-folded copies bridge same-sound spellings (Crit↔K.R.I.T.).
    variants_a = set(_artist_norm_variants(a)) | set(_artist_norm_variants(_normalize_spelled_numbers(a)))
    variants_b = set(_artist_norm_variants(b)) | set(_artist_norm_variants(_normalize_spelled_numbers(b)))
    variants_a |= {_consonant_fold(v) for v in variants_a}
    variants_b |= {_consonant_fold(v) for v in variants_b}
    for norm_a in variants_a:
        for norm_b in variants_b:
            if _artist_norm_match(norm_a, norm_b):
                return True
    return False


_GENERIC_ARTIST_TOKENS = frozenset({
    # Rap/hip-hop prefixes so common they identify nothing on their own:
    # "Lil Wayne" must NOT match "Lil Tunechi"/"Lil Baby"/"Lil Durk" on "lil"
    # (2026-07-29: a bootleg act named "Lil Tunechi" won a search this way).
    # Also makes the long-documented Young M.A ↔ Trummy Young reject actually
    # hold — they share only "young".
    "lil", "big", "young", "yung", "dj", "mc", "the",
})


def _artist_norm_match(norm_a: str, norm_b: str) -> bool:
    """Match two already-normalized artist names."""
    # Direct containment after normalization (min 2 chars — a bare "k" matches everything)
    if len(norm_a) >= 2 and len(norm_b) >= 2 and (norm_a in norm_b or norm_b in norm_a):
        return True
    # Word overlap: "ot genesis" vs "ot genasis" → {"ot","genesis"} vs {"ot","genasis"}
    words_a = set(norm_a.split())
    words_b = set(norm_b.split())
    if not words_a or not words_b:
        return False
    # Ignore 1-char tokens: spacing out "O.T. Genasis" → "o t genasis" must not
    # collide with "T-Pain" → "t pain" on the shared "t". Also ignore generic
    # prefix tokens — a shared "lil"/"young" alone is not an artist match.
    if {w for w in words_a & words_b if len(w) >= 2} - _GENERIC_ARTIST_TOKENS:
        return True
    # Prefix matching on individual words: "genesis" ↔ "genasis" (5-char prefix "genes"/"genas" — no)
    # Better: check if any word pair shares a 4+ char prefix. Generic tokens are
    # excluded here too — "young"/"young" is a 4-char prefix of itself, which
    # would resurrect exactly the matches the stoplist above rejects.
    for wa in words_a - _GENERIC_ARTIST_TOKENS:
        for wb in words_b - _GENERIC_ARTIST_TOKENS:
            min_len = min(len(wa), len(wb))
            if min_len >= 4 and (wa.startswith(wb[:4]) or wb.startswith(wa[:4])):
                return True
    return False


async def _musicbrainz_resolve(
    session: Any,
    query: str,
    artist: str,
    media_type: str,
) -> tuple[str | None, str | None]:
    """Resolve a track or album name via MusicBrainz fuzzy search.

    Handles spelling variations (Rollin'↔Rolling, O.T. Genasis↔OT Genesis).
    Tries field-specific query first, then general fuzzy query as fallback.

    Returns (canonical_title, canonical_artist) or (None, None).
    """
    if not query:
        return None, None

    if media_type == "track":
        endpoint, field, results_key = "recording", "recording", "recordings"
    else:
        endpoint, field, results_key = "release-group", "releasegroup", "release-groups"

    # Try two queries: field-specific first, then general fuzzy
    queries = []
    if artist:
        queries.append(f'{field}:"{query}" AND artist:({artist})')
        queries.append(f"{query} {artist}")  # general fuzzy fallback
    else:
        queries.append(f'{field}:"{query}"')
    if media_type == "album":
        queries = [q + " AND primarytype:album" if "AND" in q else q for q in queries]

    for mb_query in queries:
        url = f"{_MB_BASE}/{endpoint}?query={quote(mb_query)}&fmt=json&limit=5"
        _LOGGER.info("MUSICBRAINZ: Searching %s: %s", media_type, mb_query)

        data, status = await fetch_json(
            session, url,
            headers={"User-Agent": _MB_USER_AGENT, "Accept": "application/json"},
            cache_ttl=CACHE_TTL_LONG,
        )
        if not data or status != 200:
            continue

        for item in data.get(results_key, []):
            title = item.get("title", "").strip()
            if not title:
                continue
            item_artist = ""
            for ac in item.get("artist-credit", []):
                a = ac.get("artist", {}) if isinstance(ac, dict) else {}
                item_artist = (a.get("name") or "").strip()
                break
            # Use fuzzy artist matching — "OT Genesis" must match "O.T. Genasis"
            if artist and not _artist_names_match(artist, item_artist):
                continue
            # The general fuzzy query returns ANY recording by the artist, so the
            # title must still resemble what was asked for. Without this,
            # "Passive Aggressive" by Wale "canonicalized" to the unrelated "Her".
            if not _titles_resemble(query, title):
                _LOGGER.debug(
                    "MUSICBRAINZ: Rejected '%s' for query '%s' (title mismatch)",
                    title, query,
                )
                continue
            _LOGGER.info("MUSICBRAINZ: Resolved %s '%s' → '%s' (artist: '%s')",
                         media_type, query, title, item_artist)
            return title, item_artist

    return None, None


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


def _album_title_is_generic(album: str) -> bool:
    """True when the album arg is empty or pure theme/ordinal filler.

    "christmas album" / "first christmas album" are descriptions the themed
    picker should resolve; "Happy Thanksgiving & Merry Christmas" is an
    explicit title that merely CONTAINS a theme word — hijacking it into the
    themed picker played a different album than the one the user named
    (2026-08-10)."""
    if not album:
        return True
    filler = {"album", "albums", "the", "a", "an", "my", "favorite", "favourite"}
    for word in _ORDINALS:
        filler.update(word.split())
    for keyword in ALBUM_THEME_KEYWORDS:
        filler.update(keyword.split())
    words = re.findall(r"[a-z0-9']+", album.lower())
    return all(w in filler for w in words)


# =============================================================================
# CURATED MEDIA — phrase shortcuts that play an exact pinned playlist/artist
# instead of a fuzzy catalog search. Guarantees reliable, kid-safe results
# (e.g. lullabies, baby classical) every time, regardless of search ranking.
# Generic apple_music:// URIs are used so they survive an Apple Music re-auth.
# =============================================================================
CURATED_MEDIA: list[dict] = [
    {
        "id": "lullabies",
        "name": "lullabies",
        # Apple Music editorial playlist "Lullaby Essentials"
        "uri": "apple_music://playlist/pl.bb55dfb4bc4b4247ae7ef9cb9b01fad4",
        "media_type": "playlist",
        "match_any": ["lullaby", "lullabies"],
    },
    {
        "id": "baby_classical",
        "name": "Baby Einstein",
        # The Baby Einstein Music Box Orchestra — soothing classical for babies.
        # Played as an artist with radio_mode off so it stays on Baby Einstein.
        "uri": "apple_music://artist/6839896",
        "media_type": "artist",
        "match_any": ["baby einstein"],
        # ...or "classical" together with a baby/kid word
        # ("children's classical", "classical for babies", "kids classical").
        "match_all_groups": [
            ["classical"],
            ["baby", "babies", "kid", "kids", "child", "children", "childrens", "children's"],
        ],
    },
    {
        "id": "pretty_little_baby",
        "name": "Pretty Little Baby",
        # Connie Francis — the original 1962 recording. Pinned because the title
        # has ~10 near-identical Apple Music releases (remixes/covers credited
        # "Connie Francis & X", phonk/slowed edits, etc.); fuzzy search and MA's
        # per-room radio "similar tracks" would otherwise let different rooms
        # land on different versions of the same song. Pinning the exact track
        # URI guarantees every room plays this recording. radio stays ON so the
        # play-a-song-then-keep-going behavior is preserved.
        "uri": "apple_music://track/1645556889",
        "media_type": "track",
        "radio": True,
        "match_any": ["pretty little baby"],
    },
]


# Music Assistant item URIs are "<provider>://<media_type>/<id>" (e.g.
# "apple_music://song/123", "library://album/5"). Map the path segment back to
# our media_type vocabulary so a media_uri can be played without re-searching.
_URI_TYPE_MAP = {
    "song": "track", "songs": "track", "track": "track", "tracks": "track",
    "album": "album", "albums": "album",
    "artist": "artist", "artists": "artist",
    "playlist": "playlist", "playlists": "playlist",
}


def _media_type_from_uri(uri: str) -> str | None:
    """Infer media_type from a Music Assistant item URI, or None if unknown."""
    if not uri or "://" not in uri:
        return None
    try:
        seg = uri.split("://", 1)[1].split("/", 1)[0].lower()
    except (IndexError, AttributeError):
        return None
    return _URI_TYPE_MAP.get(seg)


def _match_curated(text: str) -> dict | None:
    """Return a CURATED_MEDIA entry if the text names a curated playlist/artist.

    Uses lowercased, accent-stripped substring checks so it works regardless of
    how the LLM split the request across query/artist/album. An entry matches
    when any 'match_any' phrase is present, OR every group in 'match_all_groups'
    has at least one keyword present.
    """
    if not text:
        return None
    t = _strip_accents(text.lower())
    for entry in CURATED_MEDIA:
        if any(kw in t for kw in entry.get("match_any", [])):
            return entry
        groups = entry.get("match_all_groups")
        if groups and all(any(kw in t for kw in group) for group in groups):
            return entry
    return None


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
        # URIs most recently offered to the LLM by search_music, so a media_uri
        # echoed back with a typo can be snapped to the real one.
        self._offered_uris: list[str] = []
        # Pending post-skip resume watchers, keyed by entity_id, so an explicit
        # stop/pause can cancel them instead of having a watcher un-stop the
        # music seconds later (2026-08-10).
        self._resume_tasks: dict[str, asyncio.Task] = {}

    async def control_music_deferred(
        self, arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], Callable[[], Awaitable[None]] | None]:
        """Run control_music, capturing any play_media calls instead of firing.

        Returns (result, play_action). When play_action is not None, the caller
        is responsible for awaiting it to actually start playback. Non-play
        actions (pause, resume, volume, etc.) return play_action=None because
        they don't emit audio that should wait for TTS.
        """
        captured: list[tuple[str, str, str, bool]] = []
        original = self._play_media

        async def _capture(player: str, media_id: str, media_type: str, radio: bool = False) -> bool:
            captured.append((player, media_id, media_type, radio))
            return True

        self._play_media = _capture  # type: ignore[method-assign]
        try:
            result = await self.control_music(arguments)
        finally:
            self._play_media = original  # type: ignore[method-assign]

        if not captured:
            return result, None

        async def _do_play() -> None:
            for player, media_id, media_type, radio in captured:
                await self._play_and_verify(player, media_id, media_type, radio)

        return result, _do_play

    async def _play_and_verify(
        self, player: str, media_id: str, media_type: str, radio: bool,
    ) -> None:
        """Play with outcome verification and one self-heal retry.

        MA can accept play_media yet play nothing — a dead Snapdroid client or
        a wedged MA snapcast player object ("Timed out acquiring playback
        lock") both make MA log a WARNING and return success — so the service
        call result proves nothing. Nor is the entity reaching 'playing'
        enough for the living room: a dead snapclient with the Native output
        link up leaves the entity happily 'playing' while audio streams to
        nobody (2026-08-08), so Shield plays additionally check snapserver's
        connected flag before AND after playing. Since this runs after the
        TTS confirmation has already been spoken, a final failure must be
        surfaced to the user, not just logged.
        """
        ok = False
        try:
            if player == _SNAPCLIENT_PLAYER and await self._snapclient_connected() is False:
                _LOGGER.warning(
                    "Snapclient not connected at snapserver — reviving before play")
                await self._run_ensure_snapclient()
                await self._wait_snapclient_connected()
            await self._play_media(player, media_id, media_type, radio=radio)
            ok = await self._wait_for_playback_start(player, timeout=12.0)
            if ok and player == _SNAPCLIENT_PLAYER:
                ok = await self._verify_snapclient_audio(player)
        except Exception as err:  # noqa: BLE001
            if self._is_content_error(err):
                # MA rejected the item itself (empty/unplayable playlist,
                # usually a bad search match) — the player is fine, so the
                # self-heal + same-URI retry below cannot help (2026-08-10).
                _LOGGER.error(
                    "MA rejected the media item on %s (%s) — bad content "
                    "match, skipping self-heal retry", player, err)
                await self._notify_play_failure(player, content_error=True)
                return
            _LOGGER.warning("Play on %s raised: %s", player, err)
        if ok:
            return

        _LOGGER.warning(
            "Playback did not start on %s — running self-heal and retrying", player)
        if player == _SNAPCLIENT_PLAYER:
            await self._run_ensure_snapclient()
            await self._wait_snapclient_connected()
        try:
            await self._play_media(player, media_id, media_type, radio=radio)
            ok = await self._wait_for_playback_start(player, timeout=12.0)
            if ok and player == _SNAPCLIENT_PLAYER:
                ok = await self._verify_snapclient_audio(player)
                if not ok:
                    _LOGGER.error(
                        "Retry reached 'playing' but the snapclient is still "
                        "not receiving the stream — audio is going nowhere")
        except Exception as err:  # noqa: BLE001
            if self._is_content_error(err):
                _LOGGER.error(
                    "MA rejected the media item on retry on %s (%s) — bad "
                    "content match", player, err)
                await self._notify_play_failure(player, content_error=True)
                return
            _LOGGER.error("Play retry on %s raised: %s", player, err)
        if ok:
            _LOGGER.info("Playback recovered on %s after self-heal retry", player)
            return

        _LOGGER.error("Playback could not be started on %s after retry", player)
        await self._notify_play_failure(player)

    async def _verify_snapclient_audio(self, player: str) -> bool:
        """Post-play truth check for the Shield: entity 'playing' is not enough.

        The snapclient must be connected at snapserver AND its group must be
        on an MA stream — 'connected' but parked on the idle 'default'
        stream, or a Native-link stream set up while the client was still
        (re)connecting, plays silence with every entity signal green
        (2026-08-10, 2026-08-17). If the client is connected but not
        attached, one stop/play bounce re-runs MA's stream assignment and
        has recovered every occurrence so far; re-verify after it.
        Check errors (None) never block playback.
        """
        status = await self._snapclient_status()
        if status is None:
            return True
        connected, stream_id = status
        if not connected:
            _LOGGER.warning(
                "Entity is 'playing' but the snapclient is DISCONNECTED "
                "(Native-link masking) — treating as failure")
            return False
        if stream_id != "default":
            return True
        _LOGGER.warning(
            "Snapclient connected but its group is on snapserver's idle "
            "'default' stream while %s reads 'playing' — silent start; "
            "bouncing", player)
        await self._bounce_snapclient_playback(player)
        if not await self._wait_for_playback_start(player, timeout=12.0):
            return False
        attached = await self._snapclient_attached()
        if attached is False:
            _LOGGER.error(
                "Snapclient still not attached to an MA stream after bounce")
            return False
        _LOGGER.info("Snapclient attached to MA stream after bounce")
        return True

    @staticmethod
    def _is_content_error(err: Exception) -> bool:
        """True when MA rejected the item itself rather than the player failing.

        "No playable item found to start playback" is MA's answer for an
        empty/unplayable playlist or track (seen 2026-08-10 when an STT
        mishear matched a junk Spotify playlist) — retrying the same URI or
        reviving the speaker cannot fix it."""
        msg = str(err).lower()
        return "no playable item" in msg or "no playable items" in msg

    async def _notify_play_failure(self, player: str, content_error: bool = False) -> None:
        """Tell the user playback silently failed (the TTS already claimed success)."""
        state = self._hass.states.get(player)
        name = (state.attributes.get("friendly_name") if state else None) or player
        if content_error:
            message = (
                f"The music I picked couldn't be played on {name} — Music "
                "Assistant reported no playable tracks in it. That's almost "
                "always a bad content match (often a misheard name), not a "
                "player problem. Just ask again, maybe with slightly "
                "different wording."
            )
        else:
            message = (
                f"Music failed to start on {name} even after an automatic restart "
                "of the speaker connection. The Music Assistant player may be "
                "wedged — if it doesn't recover in a few minutes, restart the "
                "Music Assistant add-on."
            )
        try:
            await self._hass.services.async_call(
                "persistent_notification", "create",
                {"title": "Music playback failed", "message": message,
                 "notification_id": f"purellm_play_failure_{player}"},
                blocking=False,
            )
            await self._hass.services.async_call(
                "notify", _PLAY_FAILURE_NOTIFY,
                {"title": "🎵 Music playback failed", "message": message},
                blocking=False,
            )
        except Exception as err:  # noqa: BLE001
            _LOGGER.error("Could not deliver play-failure notification: %s", err)

    async def control_music(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Control music playback.

        Args:
            arguments: Tool arguments (action, query, room, media_type, artist, album, volume)

        Returns:
            Result dict
        """
        action = arguments.get("action", "").lower()
        query = arguments.get("query", "")
        media_type = arguments.get("media_type", "artist")
        room = arguments.get("room", "").lower() if arguments.get("room") else ""
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
                # Strip trailing punctuation (STT often adds periods: "in the Kitchen.")
                potential_room = match.group(1).lower().strip().rstrip(".,!?;:")
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

        # DEFENSIVE: Strip "by {artist}" from query when artist is already a separate param.
        # LLM often includes the full phrase "Picture Me Rolling by Tupac" in query AND
        # also sets artist="Tupac", causing search to be "Picture Me Rolling by Tupac Tupac".
        if query and artist:
            by_artist_pattern = rf'\s+by\s+{re.escape(artist)}\s*$'
            by_match = re.search(by_artist_pattern, query, flags=re.IGNORECASE)
            if by_match:
                original_query = query
                query = query[:by_match.start()].strip()
                _LOGGER.debug("MUSIC: Stripped 'by %s' from query='%s' → '%s'", artist, original_query, query)

        # DEFENSIVE: Strip leading media_type keywords from query.
        # STT often produces "track Picture Me Rolling" or "song Bohemian Rhapsody"
        # and the LLM may include the keyword in the query param.
        if query:
            query = re.sub(r'^(track|song|album|artist)\s+', '', query, flags=re.IGNORECASE).strip()

        # DEFENSIVE: Strip trailing punctuation from query (STT adds periods)
        if query:
            query = query.rstrip(".,!?;:")

        _LOGGER.debug("MUSIC: Final - action='%s', query='%s', room='%s'", action, query, room)

        # Pop the original utterance off the args so it doesn't leak into search
        # strings later. The tool description tells the LLM to route "album"
        # requests itself via media_type='album'; we no longer second-guess it.
        user_text = arguments.pop("_user_text", "")

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

            # Revive any snapcast player killed by the STOP button before ANY
            # play-type action. This must happen here (not just in the deeper
            # helpers) so direct-uri plays, shuffles, curated shortcuts and
            # themed album plays are all covered.
            if target_players and action in ("play", "shuffle", "resume"):
                await self._ensure_players_available(target_players)

            # Direct play of a search_music result: the LLM echoes the chosen
            # candidate's media_uri, so we skip all searching and play it as-is.
            media_uri = self._snap_media_uri((arguments.get("media_uri") or "").strip())
            if media_uri and action in ("play", "shuffle"):
                if not target_players:
                    return {"error": f"Which room? Available: {', '.join(self._players.keys())}"}
                uri_type = _media_type_from_uri(media_uri) or (
                    media_type if media_type in ("track", "album", "artist", "playlist") else "track"
                )
                label = (_normalize_unicode(query) or album or artist or "your music").strip()
                room_suffix = f" in the {room}" if room else ""
                if action == "shuffle":
                    for player in target_players:
                        await self._hass.services.async_call(
                            "media_player", "shuffle_set",
                            {"entity_id": player, "shuffle": True}, blocking=True,
                        )
                        await self._play_media(player, media_uri, uri_type)
                    _LOGGER.info("MUSIC: Direct shuffle of '%s' (uri=%s, type=%s)", label, media_uri, uri_type)
                    return {"status": "shuffling", "playlist_title": label,
                            "response_text": f"Playing {label}{room_suffix}"}
                # Single tracks play in radio mode so similar music follows
                # instead of stopping after one song.
                await self._play_on_players(target_players, media_uri, uri_type, radio=(uri_type == "track"))
                _LOGGER.info("MUSIC: Direct play of '%s' (uri=%s, type=%s)", label, media_uri, uri_type)
                return {"status": "playing", "response_text": f"Playing {label}{room_suffix}"}

            if action == "play":
                # Curated shortcuts (lullabies, baby classical, ...) — play exact
                # pinned media, bypassing fuzzy search for reliable kid-safe results.
                curated = _match_curated(" ".join(filter(None, [query, artist, album, user_text])))
                if curated:
                    if not target_players:
                        return {"error": f"Which room? Available: {', '.join(self._players.keys())}"}
                    await self._play_on_players(
                        target_players, curated["uri"], curated["media_type"],
                        radio=curated.get("radio", False))
                    _LOGGER.info("MUSIC: Curated shortcut '%s' → %s (radio=%s)",
                                 curated["id"], curated["uri"], curated.get("radio", False))
                    return {"status": "playing", "response_text": f"Playing {curated['name']} in the {room}"}

                # Try themed/ordinal album search if detected
                # e.g. "play Kelly Clarkson's first christmas album"
                # — but never when the user named a specific album whose title
                # happens to contain a theme word; exact search handles those.
                if (ordinal is not None or theme) and artist and media_type == "album" \
                        and _album_title_is_generic(album):
                    themed_result = await self._find_themed_album(artist, ordinal, theme)
                    if themed_result:
                        found_name = _normalize_unicode(themed_result.get("name") or themed_result.get("title"))
                        found_uri = themed_result.get("uri") or themed_result.get("media_id")
                        found_artist = _extract_artist(themed_result) or artist
                        if found_uri:
                            if not target_players:
                                return {"error": f"Unknown room: {room}. Available: {', '.join(self._players.keys())}"}
                            await self._play_on_players(target_players, found_uri, "album")
                            display_name = f"{found_name} by {found_artist}"
                            return {"status": "playing", "response_text": f"Playing {display_name} in the {room}"}
                    _LOGGER.info("MUSIC: Themed album search failed, falling back to normal search")

                return await self._play(query, media_type, room, target_players, artist, album)
            elif action == "pause":
                return await self._pause(all_players, target_players if target_players else None)
            elif action == "resume":
                return await self._resume(all_players)
            elif action == "stop":
                return await self._stop(all_players, target_players if target_players else None)
            elif action == "skip_next":
                return await self._skip_next(self._transport_players(all_players, target_players))
            elif action == "skip_previous":
                return await self._skip_previous(self._transport_players(all_players, target_players))
            elif action == "restart_track":
                return await self._restart_track(self._transport_players(all_players, target_players))
            elif action == "what_playing":
                return await self._what_playing(all_players)
            elif action == "transfer":
                return await self._transfer(all_players, target_players, room)
            elif action == "shuffle":
                return await self._shuffle(query, room, target_players)
            elif action in ("volume_up", "volume_down", "set_volume"):
                volume = arguments.get("volume")
                return await self._volume(action, all_players, target_players, volume)
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

    async def _ensure_players_available(self, players: list[str]) -> None:
        """Revive any target player that is 'unavailable'.

        The living-room Shield uses a Snapdroid snapcast client whose
        connection is intentionally killed by the STOP button (so the TV
        screensaver can run). Before playing, if a target player is
        unavailable, fire the HA ``script.ensure_snapclient`` helper (which
        relaunches the client and waits) and give it a moment to register.
        """
        for pid in players:
            state = self._hass.states.get(pid)
            if state is not None and state.state != "unavailable":
                continue
            _LOGGER.info("Player %s unavailable — running ensure_snapclient", pid)
            if not await self._run_ensure_snapclient():
                continue
            # Wait (up to ~16s) for the player to come back.
            elapsed = 0.0
            while elapsed < 16.0:
                state = self._hass.states.get(pid)
                if state is not None and state.state != "unavailable":
                    _LOGGER.info("Player %s available after %.1fs", pid, elapsed)
                    break
                await asyncio.sleep(0.5)
                elapsed += 0.5
            # 'available' only means MA's Native link is up — wait for the
            # snapclient itself to be connected at snapserver (+settle) so
            # play_media doesn't race the cold start (2026-08-17).
            if pid == _SNAPCLIENT_PLAYER:
                await self._wait_snapclient_connected()

    async def _run_ensure_snapclient(self) -> bool:
        """Fire script.ensure_snapclient (blocking) — it always issues the
        Snapdroid ADB start and clears the user-stopped flag."""
        try:
            await self._hass.services.async_call(
                "script", "turn_on",
                {"entity_id": "script.ensure_snapclient"},
                blocking=True,
            )
            return True
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning("ensure_snapclient call failed: %s", err)
            return False

    async def _snapclient_connected(self) -> bool | None:
        """Ask snapserver whether the Shield snapclient is actually connected.

        Returns True/False from snapserver's own connected flag, or None when
        the check itself fails (snapserver unreachable, malformed reply) —
        callers treat None as 'unknown, don't block playback on it'.
        """
        status = await self._snapclient_status()
        return None if status is None else status[0]

    async def _snapclient_status(self) -> tuple[bool, str] | None:
        """(connected, group stream_id) for the Shield snapclient, or None on
        check error. stream_id is what snapserver is actually feeding the
        client — 'default' means it is parked on the idle stream and hears
        nothing even while 'connected'."""
        try:
            return await asyncio.wait_for(self._query_snapserver(), timeout=5.0)
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning("Snapserver connectivity check failed: %s", err)
            return None

    async def _snapclient_attached(self) -> bool | None:
        """True when the snapclient is connected AND its group is on an MA
        stream (not snapserver's idle 'default'); None on check error."""
        status = await self._snapclient_status()
        if status is None:
            return None
        connected, stream_id = status
        return connected and stream_id != "default"

    async def _wait_snapclient_connected(self, timeout: float = 10.0) -> bool:
        """After a Snapdroid cold start, wait until snapserver reports the
        client connected, then let it settle briefly BEFORE play. Firing
        play_media into a half-connected client leaves MA's Native output
        link and stream set up against nobody / the wrong group (2026-08-17
        silent-start analysis) — the entity still reads 'playing'."""
        elapsed = 0.0
        while elapsed < timeout:
            if await self._snapclient_connected():
                _LOGGER.info("Snapclient connected at snapserver after %.1fs", elapsed)
                await asyncio.sleep(_SNAPCLIENT_SETTLE_S)
                return True
            await asyncio.sleep(0.5)
            elapsed += 0.5
        _LOGGER.warning(
            "Snapclient still not connected at snapserver after %.0fs", timeout)
        return False

    async def _bounce_snapclient_playback(self, player: str) -> None:
        """media_stop → media_play on the Shield: restarts the Queue Flow
        stream and re-runs MA's group/stream assignment (the fix that
        recovered every silent-start so far: 2026-08-10, 2026-08-17)."""
        _LOGGER.warning("Bouncing playback on %s (stop/play)", player)
        with contextlib.suppress(Exception):
            await self._hass.services.async_call(
                "media_player", "media_stop", {"entity_id": player}, blocking=True)
        await asyncio.sleep(2.0)
        await self._hass.services.async_call(
            "media_player", "media_play", {"entity_id": player}, blocking=True)

    async def _query_snapserver(self) -> tuple[bool, str] | None:
        reader, writer = await asyncio.open_connection(*_SNAPSERVER_ADDR)
        try:
            writer.write(b'{"id":7,"jsonrpc":"2.0","method":"Server.GetStatus"}\n')
            await writer.drain()
            while True:
                line = await reader.readline()
                if not line:
                    return None
                msg = json.loads(line)
                if msg.get("id") != 7:
                    continue  # interleaved async notifications
                for group in msg["result"]["server"]["groups"]:
                    for client in group["clients"]:
                        if client["host"]["name"] == _SNAPSERVER_CLIENT:
                            return bool(client["connected"]), str(group.get("stream_id", ""))
                return False, ""  # no client entry at all = not connected
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    def _transport_players(
        self, all_players: list[str], target_players: list[str] | None,
    ) -> list[str]:
        """Search order for transport commands (skip / previous / restart).

        When the user named a room ("skip on the kitchen speaker"), that room
        is searched FIRST — previously skip/previous/restart ignored the room
        entirely and just took the first player in 'playing' state from the
        whole house, so a skip aimed at one room could land in another.
        Falls back to every player when the named room isn't playing, so a
        bare "next track" still works from any satellite.
        """
        if not target_players:
            return all_players
        return target_players + [p for p in all_players if p not in target_players]

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
        media_type: str,
        radio: bool = False
    ) -> bool:
        """Play media via Music Assistant.

        Args:
            player: The media_player entity_id
            media_id: The URI/ID of the media to play
            media_type: The type of media (track, album, artist, playlist)
            radio: Enable MA radio mode — after the requested media, keep
                playing dynamically-picked similar tracks

        Returns:
            True if command was sent successfully
        """
        _LOGGER.info("Playing media: uri='%s', type='%s', radio=%s on %s",
                    media_id, media_type, radio, player)

        await self._hass.services.async_call(
            "music_assistant", "play_media",
            {"media_id": media_id, "media_type": media_type, "enqueue": "replace", "radio_mode": radio},
            target={"entity_id": player},
            blocking=True
        )

        return True

    async def _play_on_players(self, target_players: list[str], uri: str, media_type: str, radio: bool = False) -> None:
        """Play media on target players."""
        await self._ensure_players_available(target_players)
        for player in target_players:
            # Clear any lingering repeat mode so a
            # single track doesn't loop forever.
            try:
                await self._hass.services.async_call(
                    "media_player", "repeat_set",
                    {"entity_id": player, "repeat": "off"},
                    blocking=True,
                )
            except Exception as err:  # noqa: BLE001
                _LOGGER.debug("repeat_set not supported on %s: %s", player, err)

            # Albums must start at track 1 — clear any inherited shuffle state
            # before we kick playback so the player doesn't open the album
            # mid-tracklist.
            if media_type == "album":
                await self._hass.services.async_call(
                    "media_player", "shuffle_set",
                    {"entity_id": player, "shuffle": False},
                    blocking=True
                )
            await self._play_media(player, uri, media_type, radio=radio)

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
                if not _artist_contains(artist_lower, item_artist):
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
            if not _artist_contains(artist_lower, item_artist):
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

    async def _resolve_artist_name(self, ma_config_entry_id: str, artist: str) -> str:
        """Resolve voice artist name to MA canonical name (e.g. Tupac → 2Pac).

        MA's artist search is a loose full-text match and ALWAYS returns its best
        guess, so an unrecognized name comes back as something arbitrary — a
        misheard "oh he did" (the band is "Oh He Dead") resolved to "Arctic
        Monkeys", and every downstream search then ran against that wrong artist
        and found nothing. Gate each candidate through _artist_names_match and
        keep the user's original wording when none of them resemble it: the raw
        name still has a chance via the query-only MA search and the MusicBrainz
        fallback, whereas a confidently wrong canonical name has none.

        Same class of bug as the _titles_resemble guard added for MusicBrainz
        title canonicalization in v7.62.3.
        """
        if not artist:
            return artist
        try:
            # Digit-folded form FIRST when it differs ("forty two dog" → "42
            # dog"): its result order is far better for digit-styled artists —
            # "42 dog" puts 42 Dugg at #1, while the raw spelled-out search
            # leads with Dr. Dog, which would win the plausibility gate on the
            # shared word "dog" before 42 Dugg is ever considered.
            names_to_try = []
            folded = _normalize_spelled_numbers(artist)
            if folded.lower() != artist.lower():
                names_to_try.append(folded)
            names_to_try.append(artist)

            rejected: list[str] = []
            for search_name in names_to_try:
                result = await self._hass.services.async_call(
                    "music_assistant", "search",
                    {"config_entry_id": ma_config_entry_id, "name": search_name, "media_type": ["artist"], "limit": 5},
                    blocking=True, return_response=True
                )
                for r in _parse_ma_results(result, "artist"):
                    canonical = (r.get("name") or "").strip()
                    if not canonical:
                        continue
                    if not _artist_resolution_plausible(artist, canonical):
                        rejected.append(canonical)
                        continue
                    _LOGGER.info("MUSIC: Resolved artist '%s' → '%s'", artist, canonical)
                    return canonical
            if rejected:
                _LOGGER.info(
                    "MUSIC: No artist resembling '%s' (ignored %s); keeping original",
                    artist, ", ".join(repr(n) for n in rejected[:5]),
                )
        except Exception as err:
            _LOGGER.debug("MUSIC: Artist resolution failed for '%s': %s", artist, err)
        return artist

    async def _search_ma(
        self, config_entry_id: str, query: str, artist: str,
        media_type: str, album: str = "",
    ) -> dict | None:
        """Search Music Assistant and return best match or None.

        Tries multiple search strategies:
        1. Combined "query artist" (or query-only for albums)
        2. Query-only fallback for tracks
        3. Artist-only fallback for albums
        """
        # Build search queries to try in order
        queries_to_try = []
        if media_type == "album" and artist:
            queries_to_try.append(query)
            queries_to_try.append(f"{query} {artist}")
        elif artist:
            queries_to_try.append(f"{query} {artist}")
            queries_to_try.append(query)  # fallback without artist
        else:
            queries_to_try.append(query)
        if media_type == "album" and artist:
            queries_to_try.append(artist)  # album: try artist-only as last resort

        artist_lower = _strip_accents(artist.lower()) if artist else ""
        query_lower = _normalize_numerals(_strip_accents(query.lower()))

        for search_query in queries_to_try:
            _LOGGER.info("MUSIC: Searching MA for %s: '%s'", media_type, search_query)
            search_result = await self._hass.services.async_call(
                "music_assistant", "search",
                {"config_entry_id": config_entry_id, "name": search_query, "media_type": [media_type], "limit": 10},
                blocking=True, return_response=True
            )
            results = _parse_ma_results(search_result, media_type)
            if not results:
                continue

            # Filter by album name if specified
            if album and media_type == "album":
                album_filter = _normalize_numerals(_strip_accents(album.lower()))
                filtered = [r for r in results
                            if album_filter in _normalize_numerals(_strip_accents((r.get("name") or r.get("title") or "").lower()))]
                if filtered:
                    results = filtered

            best = self._pick_best_match(results, query_lower, artist_lower)
            if best:
                return best

        return None

    # Track variant keywords — penalized unless the user explicitly asks for them
    _VARIANT_KEYWORDS = re.compile(
        r'\b(instrumental|karaoke|acapella|a\s*cappella|backing\s+track|minus\s+one)\b',
        re.IGNORECASE,
    )

    # DJ-mix / continuous-mix compilation markers. These tracks are pre-faded
    # for back-to-back playback, so when MA plays one in isolation it cross-
    # mixes out at the boundary instead of ending cleanly. Apple Music tags
    # the individual tracks "(Mixed)" and the parent album "(DJ Mix)".
    _DJ_MIX_KEYWORDS = re.compile(
        r'\(\s*mixed\s*\)'                                   # "Song (Mixed)"
        r'|\b(?:dj|continuous|nonstop|in\s+the)\s*mix\b'     # "DJ Mix", "Continuous Mix", ...
        r'|\bmegamix\b',
        re.IGNORECASE,
    )

    # Non-album version markers — anything that signals a re-recording, edit, or
    # alternate take rather than the canonical studio-album cut. Apple Music
    # encodes these as parentheticals on the track name (e.g. "Song (Live)",
    # "Song (2009 Remaster)", "Song (Radio Edit)"). We default-prefer the bare
    # album version unless the user's query explicitly asks for a variant.
    _NON_ALBUM_VERSION_KEYWORDS = re.compile(
        r'\b(?:live|remix(?:ed)?|remaster(?:ed)?|acoustic|unplugged|demo'
        r'|radio\s*edit|single\s+(?:version|edit)|extended\s+(?:version|mix)'
        r'|alternate(?:\s+(?:take|version|mix))?'
        r'|mono(?:\s+version)?|stereo\s+version|early\s+version'
        r'|re[\s-]?record(?:ed|ing)?)\b',
        re.IGNORECASE,
    )

    def _score_item(
        self, item: dict, query_lower: str, artist_lower: str,
    ) -> tuple[int, int]:
        """Score one MA result against the query/artist.

        Returns (total_score, name_score). name_score > 0 means the item's name
        actually matched the query — artist-only matches (name_score == 0) play
        the wrong song and must be rejected by callers.
        """

        def _prefix_match(word_a: str, word_b: str) -> bool:
            """Two words match if one is a 4+ char prefix of the other, or —
            for longer words — near-identical spelling. The similarity arm
            covers faithful-STT one-phoneme slips on invented titles
            ("cadillacica" ↔ "cadillactica", 2026-08-04): no prefix rule can
            bridge a dropped mid-word letter, but 0.85 on 6+ chars stays far
            from matching genuinely different words."""
            min_len = min(len(word_a), len(word_b))
            if min_len >= 4 and (word_a.startswith(word_b) or word_b.startswith(word_a)):
                return True
            if min_len < 6:
                return False
            # Compare consonant-folded forms so c/k/z-style respellings don't
            # eat the whole edit budget ("kaddiaktika" ↔ "cadillactica" is 0.61
            # raw but 0.95 folded — same word, different STT spelling choices).
            return (SequenceMatcher(None, word_a, word_b).ratio() >= 0.85
                    or SequenceMatcher(None, _consonant_fold(word_a), _consonant_fold(word_b)).ratio() >= 0.85)

        # Did the user actually ask for a variant version?
        user_wants_variant = bool(self._VARIANT_KEYWORDS.search(query_lower))
        user_wants_djmix = bool(self._DJ_MIX_KEYWORDS.search(query_lower))
        user_wants_non_album = bool(self._NON_ALBUM_VERSION_KEYWORDS.search(query_lower))

        name_score = 0
        artist_score = 0
        item_name = _normalize_numerals(_strip_accents((item.get("name") or item.get("title") or "").lower()))
        item_artist = _strip_accents(_extract_artist(item, lowercase=True))

        # Name scoring
        if query_lower == item_name:
            name_score = 100
        elif query_lower in item_name:
            name_score = 50
        else:
            query_words = [w for w in query_lower.split() if len(w) > 2]
            if query_words:
                matches = sum(1 for w in query_words
                              if w in item_name or any(_prefix_match(w, t) for t in re.split(r"[\s'']+", item_name) if t))
                if matches == len(query_words):
                    name_score = 40
                elif matches > 0:
                    name_score = 20 * matches

        # Artist scoring — use centralized fuzzy match
        collab_penalty = 0
        if artist_lower:
            if artist_lower == item_artist:
                artist_score = 100
            elif _artist_names_match(artist_lower, item_artist):
                artist_score = 50
            elif item_artist:
                artist_score = -200

            # Collaborator / remix-credit penalty: the user asked for a single
            # artist but this credit ADDS extra collaborators (e.g. request
            # "Connie Francis" matching "Connie Francis & LIZOT" or
            # "Connie Francis feat. X"). Those are near-always remixes/covers of
            # the requested song, not the canonical recording — and because
            # _extract_artist keeps the whole "A & B" string, they sneak in via
            # the fuzzy substring match above. Only penalize when the requested
            # artist is genuinely PART of the credit (token subset) AND the
            # credit carries an explicit collaboration separator plus at least
            # one extra name token, so an exact match and an unrelated-artist
            # mismatch are both left untouched, and a user who DID name the
            # collaborators is spared (their tokens are already in artist_lower).
            if artist_lower != item_artist and item_artist:
                _connectors = {"feat", "featuring", "ft", "with", "vs", "x", "and", "the"}
                req_tokens = set(re.sub(r"[^a-z0-9\s]", " ", artist_lower).split())
                item_tokens_list = re.sub(r"[^a-z0-9\s]", " ", item_artist).split()
                item_tokens = set(item_tokens_list)
                if req_tokens and req_tokens <= item_tokens:
                    extra = [t for t in item_tokens_list
                             if t not in req_tokens and t not in _connectors]
                    has_separator = bool(re.search(
                        r"[&,/]|\bfeat\b|\bfeaturing\b|\bft\b|\bwith\b|\bvs\b|\bx\b",
                        item_artist))
                    if extra and has_separator:
                        collab_penalty = -300
                        _LOGGER.debug(
                            "MUSIC: Collaborator penalty applied to artist '%s' (requested '%s')",
                            item_artist, artist_lower)

        # Penalize instrumental/karaoke/etc. variants when user didn't ask for one.
        # Check both track name and the version tag (MA often stores "Instrumental" there).
        variant_penalty = 0
        item_version = (item.get("version") or "").lower()
        if not user_wants_variant:
            if self._VARIANT_KEYWORDS.search(item_name) or self._VARIANT_KEYWORDS.search(item_version):
                variant_penalty = -500
                _LOGGER.debug("MUSIC: Variant penalty applied to '%s' (version='%s')", item_name, item_version)

        # Penalize DJ-mix / continuous-mix compilation tracks (they cross-fade
        # out at the boundary instead of ending). Look in track name, version
        # tag, and the parent album name (Apple Music puts "(DJ Mix)" there).
        album_info = item.get("album") or {}
        if isinstance(album_info, dict):
            item_album = (album_info.get("name") or album_info.get("title") or "").lower()
        elif isinstance(album_info, str):
            item_album = album_info.lower()
        else:
            item_album = ""

        if not user_wants_djmix:
            if (self._DJ_MIX_KEYWORDS.search(item_name)
                    or self._DJ_MIX_KEYWORDS.search(item_version)
                    or self._DJ_MIX_KEYWORDS.search(item_album)):
                variant_penalty -= 800
                _LOGGER.debug(
                    "MUSIC: DJ-mix penalty applied to '%s' (version='%s', album='%s')",
                    item_name, item_version, item_album,
                )

        # Default-prefer the canonical studio-album cut: penalize live /
        # remix / remaster / acoustic / demo / radio-edit / etc. unless
        # the user explicitly asked for that variant. Checked against the
        # track name, MA's version tag, and the album name (e.g. an album
        # titled "MTV Unplugged" or "Live at Wembley").
        if not user_wants_non_album:
            if (self._NON_ALBUM_VERSION_KEYWORDS.search(item_name)
                    or self._NON_ALBUM_VERSION_KEYWORDS.search(item_version)
                    or self._NON_ALBUM_VERSION_KEYWORDS.search(item_album)):
                variant_penalty -= 400
                _LOGGER.debug(
                    "MUSIC: Non-album-version penalty applied to '%s' (version='%s', album='%s')",
                    item_name, item_version, item_album,
                )

        # Prefer explicit over clean when both versions of the same song exist.
        # Apple Music returns explicit as top-level field: True/False/None.
        # Kept small on purpose: this only has to separate two copies of the SAME
        # recording (identical name/artist score), and a larger bonus would let an
        # explicit near-miss title outrank an exact clean one. The absolute
        # never-clean-over-explicit guarantee lives in _explicit_rank, applied as a
        # tie-break where the rest of the score is already equal.
        explicit_bonus = 0
        if item.get("explicit") is True:
            explicit_bonus = 15
        elif item.get("explicit") is False or re.search(r'\bclean\b', item_name):
            explicit_bonus = -15

        total = name_score + artist_score + collab_penalty + variant_penalty + explicit_bonus

        # Collab-credit rescue: some canonical recordings ARE multi-artist
        # credits (e.g. "LADY GAGA" by "Peso Pluma, Gabito Ballesteros &
        # Junior H") — there is no solo original to prefer. The collaborator
        # penalty must demote such credits below a solo original when one
        # exists, not disqualify them outright. If the item matched on both
        # name and artist and ONLY the collab penalty drags it negative
        # (variant penalties still disqualify), keep a small positive score,
        # scaled down so any solo-credit match still outranks it.
        if collab_penalty and total <= 0 and name_score > 0 and artist_score > 0:
            without_collab = total - collab_penalty
            if without_collab > 0:
                total = max(1, without_collab // 10)

        return total, name_score

    def _pick_best_match(
        self, results: list[dict], query_lower: str, artist_lower: str,
    ) -> dict | None:
        """Score MA results and return best match with uri, name, artist. Returns None if no good match."""
        best_score = 0
        best = None
        best_uri = None
        for item in results:
            score, name_score = self._score_item(item, query_lower, artist_lower)
            # Require name to actually match — artist-only matches play wrong songs
            if name_score <= 0:
                continue
            item_uri = str(item.get("uri") or item.get("media_id") or "")
            # Deterministic tie-break: when two candidates score equally (e.g.
            # two near-identical masters both credited to the exact artist), the
            # raw MA result order varies call-to-call, so prefer the primary
            # provider (Apple over Spotify) and then the lexically smallest uri.
            # This makes multi-room playback resolve the SAME recording in every
            # room instead of diverging per call.
            if score > best_score or (
                score == best_score and best_uri is not None
                and (_explicit_rank(item), _provider_rank(item_uri), item_uri)
                < (_explicit_rank(best), _provider_rank(best_uri), best_uri)
            ):
                best_score = score
                best = item
                best_uri = item_uri

        if best and best_score > 0:
            found_name = _normalize_unicode(best.get("name") or best.get("title"))
            found_artist = _extract_artist(best)
            found_uri = best.get("uri") or best.get("media_id")
            _LOGGER.info("MUSIC: Best match: '%s' by '%s' (uri: %s, score: %d)",
                         found_name, found_artist, found_uri, best_score)
            return {"name": found_name, "artist": found_artist, "uri": found_uri, "score": best_score}
        return None

    async def _ma_search_raw(
        self, config_entry_id: str, name: str, media_type: str, limit: int = 10,
    ) -> list:
        """Run a single Music Assistant search and return the parsed result list."""
        if not name:
            return []
        search_result = await self._hass.services.async_call(
            "music_assistant", "search",
            {"config_entry_id": config_entry_id, "name": name, "media_type": [media_type], "limit": limit},
            blocking=True, return_response=True,
        )
        return _parse_ma_results(search_result, media_type)

    def _remember_candidates(self, results: list[dict]) -> None:
        """Record the media_uris search_music just offered the LLM."""
        for r in results:
            uri = r.get("media_uri")
            if uri and uri not in self._offered_uris:
                self._offered_uris.append(uri)
        del self._offered_uris[:-40]

    def _snap_media_uri(self, media_uri: str) -> str:
        """Repair a media_uri the LLM mis-copied from a search_music candidate.

        search_music hands the model a full URI to echo back into control_music.
        Apple ids are short and numeric, but Spotify ids are 22 characters of
        base62 and the model does sometimes drop a character on the way through
        (2026-07-29: '…/4pB4dmSs…' came back as '…/4pBBdmSs…' and Music Assistant
        answered "No playable items found"). The right URI is one we just handed
        out, so snap to it rather than failing the turn. Requires the same
        provider and an otherwise near-identical id, so an unrelated URI the
        model invented outright is still played (or rejected) as given.
        """
        if not media_uri or media_uri in self._offered_uris:
            return media_uri
        scheme = _provider_of(media_uri)
        best, best_ratio = "", 0.0
        for candidate in self._offered_uris:
            if _provider_of(candidate) != scheme:
                continue
            ratio = SequenceMatcher(None, media_uri, candidate).ratio()
            if ratio > best_ratio:
                best, best_ratio = candidate, ratio
        if best and best_ratio >= 0.9:
            _LOGGER.warning(
                "MUSIC: media_uri '%s' was not offered; snapping to '%s' (similarity %.2f)",
                media_uri, best, best_ratio)
            return best
        return media_uri

    async def _phonetic_rescue(
        self, config_entry_id: str, query: str, artist: str, media_type: str,
    ) -> list[tuple[str, list]]:
        """Last-resort search for a title STT probably misheard.

        Only runs when every normal path (MA search, MusicBrainz canonicalization)
        found nothing AND the artist is known — the artist is what makes this safe,
        since the phonetic key alone is lossy enough to match unrelated songs.

        Sweeps the respellings in parallel (one round-trip of latency, not eight)
        and keeps only results whose title sounds like what was asked for.
        Returns [(matched_title, [items])] groups, closest spelling first.
        """
        if media_type not in ("track", "album") or not artist:
            return []
        variants = _phonetic_variants(query)
        if not variants:
            return []

        _LOGGER.info("MUSIC: Phonetic rescue for %s '%s' by '%s' — trying %s",
                     media_type, query, artist, variants)
        searches = await asyncio.gather(
            *(self._ma_search_raw(config_entry_id, f"{v} {artist}", media_type)
              for v in variants),
            return_exceptions=True,
        )

        key = _phonetic_key(query)
        artist_lower = _strip_accents(artist.lower())
        groups: dict[str, list] = {}
        seen_uris: set[str] = set()
        for result in searches:
            if isinstance(result, BaseException):
                continue
            for item in result:
                uri = item.get("uri") or item.get("media_id")
                name = item.get("name") or item.get("title") or ""
                if not uri or uri in seen_uris or _phonetic_key(name) != key:
                    continue
                # The artist gate is load-bearing — without it a phonetic match
                # happily plays a different artist's similarly-named song.
                if not _artist_names_match(artist_lower, _strip_accents(_extract_artist(item, lowercase=True))):
                    continue
                seen_uris.add(uri)
                groups.setdefault(_normalize_unicode(name), []).append(item)

        if not groups:
            _LOGGER.info("MUSIC: Phonetic rescue found nothing for '%s'", query)
            return []
        # Closest spelling to what was actually heard wins.
        ordered = sorted(
            groups.items(),
            key=lambda kv: -SequenceMatcher(None, query.lower(), kv[0].lower()).ratio(),
        )
        _LOGGER.info("MUSIC: Phonetic rescue matched '%s' → %s",
                     query, [title for title, _ in ordered])
        return ordered

    def _rank_matches(
        self, results: list[dict], query_lower: str, artist_lower: str,
        media_type: str, limit: int = 5, alt_query_lower: str = "",
    ) -> list[dict]:
        """Score and rank MA results, returning the top `limit` as candidate dicts.

        Each candidate carries the fields the LLM needs to pick and play:
        name, artist, album, media_type, media_uri, explicit.

        `alt_query_lower` is an optional second title to score against (the
        MusicBrainz-canonical form of a misheard query). Each item keeps its
        better score, so a candidate found only via the canonical title is not
        rejected for failing to match the raw spoken one.
        """
        scored: list[tuple[int, dict]] = []
        for item in results:
            score, name_score = self._score_item(item, query_lower, artist_lower)
            if alt_query_lower and alt_query_lower != query_lower:
                alt_score, alt_name = self._score_item(item, alt_query_lower, artist_lower)
                if alt_name > 0 and alt_score > 0 and (name_score <= 0 or alt_score > score):
                    score, name_score = alt_score, alt_name
            if name_score <= 0 or score <= 0:
                continue
            scored.append((score, item))
        # Sort by score desc, then explicit-over-clean, then preferred provider,
        # then uri asc — a deterministic order so the candidate list the LLM sees
        # is stable across identical searches. Explicit sorts above provider so
        # the cross-provider dedupe below can never drop an explicit copy in
        # favour of a clean one (see _explicit_rank).
        scored.sort(key=lambda x: (
            -x[0],
            _explicit_rank(x[1]),
            _provider_rank(str(x[1].get("uri") or x[1].get("media_id") or "")),
            str(x[1].get("uri") or x[1].get("media_id") or ""),
        ))

        out: list[dict] = []
        seen_uris: set[str] = set()
        seen_items: set[tuple[str, str]] = set()
        for score, item in scored:
            uri = item.get("uri") or item.get("media_id")
            if not uri or uri in seen_uris:
                continue
            # Cross-provider dedupe: with Apple + Spotify both configured the
            # same song comes back twice, and two copies of one recording would
            # eat two of the five candidate slots the LLM gets to choose from.
            # Sorted preferred-provider-first, so the copy kept is the Apple one.
            ident = (
                _strip_accents((item.get("name") or item.get("title") or "").lower()).strip(),
                _strip_accents(_extract_artist(item, lowercase=True)).strip(),
            )
            if ident in seen_items:
                _LOGGER.debug("MUSIC: Dropping duplicate %s from %s (already have it)",
                              ident, _provider_of(uri))
                continue
            seen_items.add(ident)
            seen_uris.add(uri)
            album_info = item.get("album") or {}
            if isinstance(album_info, dict):
                album_name = _normalize_unicode(album_info.get("name") or album_info.get("title") or "")
            elif isinstance(album_info, str):
                album_name = _normalize_unicode(album_info)
            else:
                album_name = ""
            candidate = {
                "name": _normalize_unicode(item.get("name") or item.get("title")),
                "artist": _extract_artist(item),
                "media_type": media_type,
                "media_uri": uri,
                "explicit": bool(item.get("explicit")),
            }
            if media_type == "track" and album_name:
                candidate["album"] = album_name
            out.append(candidate)
            if len(out) >= limit:
                break
        return out

    def _lyric_alias_candidates(
        self, items: list[dict], query_lower: str, artist_lower: str,
        media_type: str, limit: int = 2,
    ) -> list[dict]:
        """Exact-artist hits the catalog ranked ABOVE the first title match.

        When the user asks by a lyric or alternate title, MA's relevance
        search puts the real song first while the only title-matching items
        are bootlegs/covers further down. Those top differently-titled hits
        are returned as `possible_alias` candidates for the LLM to judge.

        Deliberately narrow:
        - needs a title-matching item to exist somewhere in the list (the
          bootleg pattern) — with no cutoff, every artist-only hit for a
          plain miss like "Drica" would qualify, which is exactly the
          wrong-song trap the name_score>0 invariant exists to prevent;
        - the item's artist must EXACTLY equal the resolved artist (no fuzz);
        - variant/DJ-mix/live junk is excluded the same way the scorer would.
        """
        if not artist_lower or media_type not in ("track", "album"):
            return []
        cutoff = None
        for i, item in enumerate(items):
            _, name_score = self._score_item(item, query_lower, artist_lower)
            if name_score > 0:
                cutoff = i
                break
        if not cutoff:  # no title match at all, or the title match is already #1
            return []

        def _sort_key(it: dict) -> tuple:
            uri = str(it.get("uri") or it.get("media_id") or "")
            return (_explicit_rank(it), _provider_rank(uri), uri)

        picked: dict[tuple[str, str], dict] = {}
        order: list[tuple[str, str]] = []
        for item in items[:cutoff]:
            item_artist = _strip_accents(_extract_artist(item, lowercase=True))
            if item_artist != artist_lower:
                continue
            uri = item.get("uri") or item.get("media_id")
            name = item.get("name") or item.get("title") or ""
            if not uri or not name:
                continue
            album_info = item.get("album") or {}
            album_name = ((album_info.get("name") or album_info.get("title") or "")
                          if isinstance(album_info, dict) else (album_info or ""))
            blob = f"{name} {item.get('version') or ''} {album_name}"
            if (self._VARIANT_KEYWORDS.search(blob)
                    or self._DJ_MIX_KEYWORDS.search(blob)
                    or self._NON_ALBUM_VERSION_KEYWORDS.search(blob)):
                continue
            ident = (_strip_accents(name.lower()).strip(), item_artist)
            if ident not in picked:
                order.append(ident)
                picked[ident] = item
            elif _sort_key(item) < _sort_key(picked[ident]):
                # Same song from both providers — keep the explicit/preferred copy.
                picked[ident] = item

        out: list[dict] = []
        for ident in order[:limit]:
            item = picked[ident]
            album_info = item.get("album") or {}
            album_name = ((album_info.get("name") or album_info.get("title") or "")
                          if isinstance(album_info, dict) else (album_info or ""))
            candidate = {
                "name": _normalize_unicode(item.get("name") or item.get("title")),
                "artist": _extract_artist(item),
                "media_type": media_type,
                "media_uri": item.get("uri") or item.get("media_id"),
                "explicit": bool(item.get("explicit")),
                "possible_alias": True,
            }
            if media_type == "track" and album_name:
                candidate["album"] = _normalize_unicode(album_name)
            out.append(candidate)
        return out

    async def search_catalog(
        self, query: str, media_type: str = "track", artist: str = "",
    ) -> dict:
        """Master search for the search_music tool.

        Resolves vague/misheard queries through MusicBrainz (for track/album/
        artist), then searches Music Assistant for playable candidates. Returns
        {"results": [candidate, ...]} where each candidate has a media_uri the
        LLM echoes back to control_music to play. Playlists are MA-only since
        MusicBrainz does not index streaming playlists.
        """
        query = (query or "").strip()
        if not query:
            return {"error": "No search query specified"}
        if media_type not in ("track", "album", "artist", "playlist"):
            media_type = "track"
        artist = (artist or "").strip()

        ma_entries = self._hass.config_entries.async_entries("music_assistant")
        if not ma_entries:
            return {"error": "Music Assistant integration not found"}
        ma_config_entry_id = ma_entries[0].entry_id

        # Playlists: Apple Music / MA only (not in MusicBrainz).
        if media_type == "playlist":
            return await self._search_playlists_for_tool(ma_config_entry_id, query)

        # Resolve voice artist name to MA canonical (Tupac → 2Pac).
        resolved_artist = await self._resolve_artist_name(ma_config_entry_id, artist) if artist else ""

        # Gather candidates from MA using the raw query (+ artist variants).
        seen_uris: set[str] = set()
        candidates: list[dict] = []

        def _add(items: list) -> None:
            for it in items:
                uri = it.get("uri") or it.get("media_id")
                if uri and uri not in seen_uris:
                    seen_uris.add(uri)
                    candidates.append(it)

        # The artist-scoped search is kept separately (in MA's own relevance
        # order) for lyric-alias detection below.
        artist_query_results: list[dict] = []
        if resolved_artist and media_type != "artist":
            artist_query_results = await self._ma_search_raw(
                ma_config_entry_id, f"{query} {resolved_artist}", media_type)
            _add(artist_query_results)
        _add(await self._ma_search_raw(ma_config_entry_id, query, media_type))

        # MusicBrainz canonicalization for vague/misheard track & album names.
        mb_canonical = ""
        if media_type in ("track", "album"):
            try:
                session = async_get_clientsession(self._hass)
                mb_title, mb_artist = await _musicbrainz_resolve(
                    session, query, resolved_artist, media_type,
                )
                if mb_title and mb_title.lower() != query.lower():
                    _LOGGER.info("SEARCH: MusicBrainz resolved '%s' → '%s' (artist: '%s')",
                                 query, mb_title, mb_artist or resolved_artist)
                    mb_canonical = mb_title
                    _add(await self._ma_search_raw(
                        ma_config_entry_id, f"{mb_title} {mb_artist or resolved_artist}".strip(), media_type,
                    ))
                    _add(await self._ma_search_raw(ma_config_entry_id, mb_title, media_type))
            except Exception as err:  # noqa: BLE001
                _LOGGER.debug("SEARCH: MusicBrainz resolution failed: %s", err)

        query_lower = _normalize_numerals(_strip_accents(query.lower()))
        artist_lower = _strip_accents(resolved_artist.lower()) if resolved_artist else ""
        alt_query_lower = (
            _normalize_numerals(_strip_accents(mb_canonical.lower())) if mb_canonical else ""
        )
        ranked = self._rank_matches(
            candidates, query_lower, artist_lower, media_type, limit=5,
            alt_query_lower=alt_query_lower,
        )

        # Nothing matched — the title may simply have been misheard. Sweep
        # same-sounding respellings before declaring it absent from the catalog.
        if not ranked:
            for matched_title, items in await self._phonetic_rescue(
                ma_config_entry_id, query, resolved_artist, media_type,
            ):
                ranked = self._rank_matches(
                    items, query_lower, artist_lower, media_type, limit=5,
                    alt_query_lower=_normalize_numerals(_strip_accents(matched_title.lower())),
                )
                if ranked:
                    _LOGGER.info("SEARCH: Phonetic rescue resolved '%s' → '%s'",
                                 query, matched_title)
                    break

        # Lyric-alias detection: when the user asks by a lyric or alternate
        # title ("Good Kush and Alcohol" for Lil Wayne's "Love Me"), the
        # catalog's own relevance search puts the REAL song at the top — but
        # its title doesn't match the query, so the scorer above rejects it
        # while a title-matching bootleg/cover survives. Surface those
        # differently-titled exact-artist hits as flagged candidates and let
        # the LLM decide — it knows the official titles; we don't.
        alias = self._lyric_alias_candidates(
            artist_query_results, query_lower, artist_lower, media_type)
        alias = [c for c in alias
                 if c["media_uri"] not in {r["media_uri"] for r in ranked}]

        if not ranked and not alias:
            return _search_miss(media_type, query, artist)

        # Order: when the best title match is only a weak partial (not all
        # requested words — "Kush" for "Good Kush and Alcohol"), the catalog's
        # own relevance ranking put the alias candidates above it, so mirror
        # that and list them first; a partial-word match is usually a different
        # song. A strong title match (exact/containment/all-words) stays first.
        def _name_score_of(name: str) -> int:
            best = self._score_item({"name": name}, query_lower, "")[1]
            if alt_query_lower:
                best = max(best, self._score_item({"name": name}, alt_query_lower, "")[1])
            return best

        strong_title = any(_name_score_of(c["name"]) >= 40 for c in ranked)
        ordered = (ranked + alias) if (strong_title or not alias) else (alias + ranked)
        result: dict[str, Any] = {"results": ordered}
        if alias:
            result["instruction"] = (
                "The requested title did not exactly match any catalog track. "
                "Candidates marked possible_alias are the catalog's TOP "
                "relevance hits for this exact request — when a user asks by "
                "a lyric or an alternate title, the official song appears "
                "there. Unmarked candidates only partially matched the "
                "requested words and are often a different song entirely. "
                "Play the candidate that IS the song the user asked for; if "
                "none of them is, say you couldn't find it."
            )
        _LOGGER.info("SEARCH: %d candidates for %s '%s': %s",
                     len(ordered), media_type, query,
                     [c["name"] + (" (alias)" if c.get("possible_alias") else "")
                      for c in ordered])
        self._remember_candidates(ordered)
        return result

    async def _search_playlists_for_tool(
        self, config_entry_id: str, query: str,
    ) -> dict:
        """Find playlist candidates for search_music (official Apple Music first)."""
        results = await self._ma_search_raw(config_entry_id, query, "playlist", limit=15)
        seen_uris: set[str] = set()
        out: list[dict] = []
        for p in results:
            uri = p.get("uri") or p.get("media_id")
            name = p.get("name") or p.get("title") or ""
            if not uri or uri in seen_uris or "radio" in name.lower():
                continue
            seen_uris.add(uri)
            name_l = name.lower()
            official = ("apple" in (p.get("owner") or "").lower()
                        or name_l.endswith("essentials") or name_l.startswith("best of"))
            out.append({
                "name": _normalize_unicode(name),
                "owner": p.get("owner", ""),
                "media_type": "playlist",
                "media_uri": uri,
                "official": official,
            })
        # Official curated playlists first, then the rest.
        out.sort(key=lambda x: 0 if x["official"] else 1)
        if not out:
            return _search_miss("playlist", query)
        self._remember_candidates(out[:5])
        return {"results": out[:5]}

    async def _play(self, query: str, media_type: str, room: str, target_players: list[str], artist: str = "", album: str = "") -> dict:
        """Play music via Music Assistant.

        Strategy:
        1. Resolve artist name via MA (Tupac → 2Pac)
        2. Search MA for the track/album
        3. If MA fails, resolve canonical name via MusicBrainz, then retry MA
        """
        if not query and album:
            query = album
        if not query:
            return {"error": "No music query specified"}
        if not target_players:
            return {"error": f"Unknown room: {room}. Available: {', '.join(self._players.keys())}"}

        # Revive any snapcast player killed by the STOP button before we search.
        await self._ensure_players_available(target_players)

        # Enforce valid media types
        valid_types = {"artist", "album", "track"}
        if media_type not in valid_types:
            media_type = "artist"
        if album and media_type != "album":
            media_type = "album"
        if media_type == "album" and query and not album:
            album = query

        try:
            ma_entries = self._hass.config_entries.async_entries("music_assistant")
            if not ma_entries:
                return {"error": "Music Assistant integration not found"}
            ma_config_entry_id = ma_entries[0].entry_id

            # Step 1: Resolve artist via MA (handles Tupac→2Pac, Jay Z→JAY-Z)
            resolved_artist = await self._resolve_artist_name(ma_config_entry_id, artist)

            # Step 2: Search MA directly (explicit versions preferred via scoring)
            match = await self._search_ma(ma_config_entry_id, query, resolved_artist, media_type, album)

            # Step 3: MusicBrainz fallback — resolve canonical name, retry MA
            if not match and media_type in ("track", "album"):
                try:
                    session = async_get_clientsession(self._hass)
                    mb_title, mb_artist = await _musicbrainz_resolve(session, query, resolved_artist, media_type)
                    # Use MB artist if MA couldn't resolve it (e.g. OT Genesis → O.T. Genasis)
                    final_artist = mb_artist or resolved_artist
                    if mb_title:
                        _LOGGER.info("MUSIC: MusicBrainz resolved '%s' → '%s' (artist: '%s')", query, mb_title, final_artist)
                        match = await self._search_ma(ma_config_entry_id, mb_title, final_artist, media_type, album)
                except Exception as err:
                    _LOGGER.debug("MUSIC: MusicBrainz fallback failed: %s", err)

            # Step 4: phonetic rescue — MA and MusicBrainz both came up empty, so
            # the title itself was probably misheard ("Drica" for "Dreka").
            if not match:
                for matched_title, items in await self._phonetic_rescue(
                    ma_config_entry_id, query, resolved_artist, media_type,
                ):
                    match = self._pick_best_match(
                        items,
                        _normalize_numerals(_strip_accents(matched_title.lower())),
                        _strip_accents(resolved_artist.lower()) if resolved_artist else "",
                    )
                    if match:
                        _LOGGER.info("MUSIC: Phonetic rescue resolved '%s' → '%s'",
                                     query, matched_title)
                        break

            if not match:
                miss = _search_miss(media_type, query, artist)
                return {"error": miss["message"], "instruction": miss["instruction"]}

            # Play the found media
            found_name = match["name"]
            found_artist = match.get("artist")
            found_uri = match["uri"]
            display_name = f"{found_name} by {found_artist}" if found_artist and media_type in ("track", "album") else found_name
            # Single tracks play in radio mode so similar music follows
            # instead of stopping after one song.
            await self._play_on_players(target_players, found_uri, media_type, radio=(media_type == "track"))

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
        self._cancel_pending_resumes()
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
        self._cancel_pending_resumes()
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
            if not await self._queue_has_next(playing):
                _LOGGER.info("Queue on %s has no next item; reporting end of queue", playing)
                return {"status": "end_of_queue", "response_text": "That was the last track in the queue"}
            await self._hass.services.async_call("media_player", "media_next_track", {"entity_id": playing}, blocking=True)
            self._schedule_resume_if_idle(playing)
            return {"status": "skipped", "response_text": "Skipped to next track"}
        return {"error": "No music is playing to skip"}

    async def _skip_previous(self, all_players: list[str]) -> dict:
        """Skip to previous track."""
        _LOGGER.info("Looking for player in 'playing' state...")
        playing = self._find_player_by_state("playing", all_players)
        if playing:
            await self._hass.services.async_call("media_player", "media_previous_track", {"entity_id": playing}, blocking=True)
            self._schedule_resume_if_idle(playing)
            return {"status": "skipped", "response_text": "Previous track"}
        return {"error": "No music is playing"}

    async def _queue_has_next(self, entity_id: str) -> bool:
        """True if the MA queue behind entity_id has a track after the current one.

        Skipping past the last item advances MA's queue index with nothing to
        play: no new flow stream starts, the current song plays out, then the
        queue clears (observed 2026-07-22) — while we'd have claimed success.
        Fail open: any error (non-MA player, service missing) allows the skip."""
        try:
            resp = await self._hass.services.async_call(
                "music_assistant", "get_queue", {"entity_id": entity_id},
                blocking=True, return_response=True,
            )
            queue = (resp or {}).get(entity_id) or {}
            if not queue:
                return True
            if queue.get("repeat_mode") and queue["repeat_mode"] != "off":
                return True
            return queue.get("next_item") is not None
        except Exception as err:  # noqa: BLE001 - best effort
            _LOGGER.debug("get_queue check failed for %s: %s", entity_id, err)
            return True

    def _schedule_resume_if_idle(self, entity_id: str) -> None:
        """Fire-and-forget the post-skip resume so the spoken reply isn't delayed."""
        # The resume dance is a workaround for Music Assistant flow-mode queues
        # only. On other players (e.g. the Shield's androidtv entity fronting a
        # snapcast client) the nudge itself breaks the skip: play makes the
        # client re-join the live snapcast stream seconds in (2026-07-16).
        entry = er.async_get(self._hass).async_get(entity_id)
        if not entry or entry.platform != "music_assistant":
            _LOGGER.debug("Skipping post-skip resume for %s (not a Music Assistant player)", entity_id)
            return
        old = self._resume_tasks.pop(entity_id, None)
        if old and not old.done():
            old.cancel()
        task = self._hass.async_create_background_task(
            self._resume_if_idle(entity_id), name=f"purellm_resume_{entity_id}"
        )
        self._resume_tasks[entity_id] = task
        task.add_done_callback(
            lambda t, eid=entity_id: self._resume_tasks.pop(eid, None)
            if self._resume_tasks.get(eid) is t else None
        )

    def _cancel_pending_resumes(self) -> None:
        """Cancel post-skip resume watchers on an explicit stop/pause.

        The watcher polls for up to 90s after a skip; a stop inside that
        window looks exactly like the flow-stream stall it exists to fix, so
        without cancellation it resumes the music the user just stopped
        (observed 2026-08-10: stop at :48, watcher resumed at :53)."""
        for eid, task in list(self._resume_tasks.items()):
            if not task.done():
                _LOGGER.info("Cancelling pending post-skip resume for %s (explicit stop/pause)", eid)
                task.cancel()
        self._resume_tasks.clear()

    async def _resume_if_idle(self, entity_id: str) -> None:
        """Restart playback if a manual skip left the player idle.

        Music Assistant queues that stream to a player in flow mode (e.g. an
        ESPHome speaker driven via MA's Home Assistant provider) advance the
        queue index on a manual next/previous but do NOT restart the flow
        stream, so the player drops to 'idle' and goes silent. The transition
        can lag well behind the skip (observed ~5s when the flow stream aborts
        while the TTS reply plays on the same device), so poll long enough to
        outlast it and nudge playback back on if it goes idle. On players that
        keep playing through a skip this never fires media_play and just exits."""
        consecutive_idle = 0
        # 90s window: on the Shield's flow-stream path a stalled skip can keep
        # the OLD stream audibly playing long after the queue advanced — the
        # player only dropped to idle 46s after the skip (observed 2026-07-22),
        # well past the original 20s window, so the stall went un-nudged.
        for _ in range(180):
            await asyncio.sleep(0.5)
            state = self._hass.states.get(entity_id)
            if not (state and state.state == "idle"):
                consecutive_idle = 0
                continue
            # Some players blip to idle for a second or two while MA restarts
            # its flow stream and recover on their own; nudging during the blip
            # makes snapcast clients re-join the live stream mid-song
            # (2026-07-16). Only treat SUSTAINED idle as a genuine stall.
            consecutive_idle += 1
            if consecutive_idle >= 10:
                _LOGGER.info("Player %s idle 5s after skip; resuming playback", entity_id)
                # MA keeps advancing the queue item's position clock while the
                # flow stream is down, so a plain resume starts seconds into
                # the track. Seeking to 0 WHILE IDLE resets that clock and is
                # accepted by MA (verified 2026-07-16); never seek after the
                # resume - that kills the stream on some protocol paths.
                try:
                    await self._hass.services.async_call(
                        "media_player", "media_seek", {"entity_id": entity_id, "seek_position": 0}, blocking=True
                    )
                except Exception as err:  # noqa: BLE001 - best effort
                    _LOGGER.debug("Pre-resume seek-to-start failed on %s: %s", entity_id, err)
                await self._hass.services.async_call("media_player", "media_play", {"entity_id": entity_id}, blocking=True)
                return
        _LOGGER.debug("Player %s never went idle within resume window", entity_id)

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
        return {"message": "No music currently playing"}

    async def _volume(self, action: str, all_players: list[str], target_players: list[str] | None, volume: int | None) -> dict:
        """Control music volume on active or specified player."""
        _LOGGER.info("Volume control: action=%s, volume=%s", action, volume)

        # If specific room was requested, use that player
        players_to_check = target_players if target_players else all_players

        # Find the active (playing/paused) player
        active_player = None
        for pid in players_to_check:
            state = self._hass.states.get(pid)
            if state and state.state in ("playing", "paused"):
                active_player = pid
                break

        if not active_player:
            # Fall back to first target player if specified, otherwise error
            if target_players:
                active_player = target_players[0]
            else:
                return {"error": "No music is currently playing. To control speaker voice volume, say 'set speaker volume to' followed by a number."}

        if action == "set_volume" and volume is not None:
            vol_level = max(0, min(100, volume)) / 100.0
            await self._hass.services.async_call(
                "media_player", "volume_set",
                {"entity_id": active_player, "volume_level": vol_level},
                blocking=True,
            )
            return {"status": "volume_set", "response_text": f"Volume set to {volume} percent in {self._get_room_name(active_player)}"}
        elif action == "volume_up":
            await self._hass.services.async_call(
                "media_player", "volume_up",
                {"entity_id": active_player},
                blocking=True,
            )
            return {"status": "volume_up", "response_text": f"Volume up in {self._get_room_name(active_player)}"}
        elif action == "volume_down":
            await self._hass.services.async_call(
                "media_player", "volume_down",
                {"entity_id": active_player},
                blocking=True,
            )
            return {"status": "volume_down", "response_text": f"Volume down in {self._get_room_name(active_player)}"}

        return {"error": f"Unknown volume action: {action}"}

    async def _transfer(self, all_players: list[str], target_players: list[str], room: str) -> dict:
        """Transfer music to another room."""
        _LOGGER.info("Looking for player in 'playing' state...")
        playing = self._find_player_by_state("playing", all_players)
        if not playing:
            return {"error": "No music playing to transfer"}
        if not target_players:
            return {"error": f"No target room specified. Available: {', '.join(self._players.keys())}"}

        target = target_players[0]

        # Revive the target if its snapcast client was stopped (STOP button) —
        # play actions do this, but transfer previously skipped it, so
        # transferring into a stopped room failed while playing there worked.
        await self._ensure_players_available(target_players)

        # For transfer, try the active_queue if available, otherwise use the MA wrapper
        source = self._get_transfer_source(playing)
        _LOGGER.info("Transferring from %s (source: %s) to %s", playing, source, target)

        try:
            # MA wrapper entity first — transfer_queue wants an MA player as
            # source; the raw speaker entity's queue id is often unknown to MA.
            await self._hass.services.async_call(
                "music_assistant", "transfer_queue",
                {"source_player": playing, "auto_play": True},
                target={"entity_id": target},
                blocking=True
            )
            _LOGGER.info("Transfer complete")
        except Exception as e:
            _LOGGER.error("Transfer failed with MA wrapper source: %s", e)
            # Fallback: try the active_queue-derived source entity
            try:
                await self._hass.services.async_call(
                    "music_assistant", "transfer_queue",
                    {"source_player": source, "auto_play": True},
                    target={"entity_id": target},
                    blocking=True
                )
            except Exception as e2:
                _LOGGER.error("Transfer fallback also failed: %s", e2)
                # Both attempts failed — tell the truth instead of claiming success
                # (previously fell through to the success response, so the user
                # heard "transferred" while the music just stopped).
                return {
                    "error": f"Transfer to {self._get_room_name(target)} failed: {e2}",
                    "response_text": f"I couldn't transfer the music to the {self._get_room_name(target)} — the speaker there isn't accepting playback right now.",
                }

        if target == _SNAPCLIENT_PLAYER:
            # The 2026-08-10 silent-transfer variant: entity playing, client
            # connected, no audio. Verify in the background (don't hold TTS)
            # and bounce if the client isn't attached to an MA stream.
            self._hass.async_create_task(self._verify_transfer_audio(target))

        return {"status": "transferred", "response_text": f"Music transferred to {self._get_room_name(target)}"}

    async def _verify_transfer_audio(self, player: str) -> None:
        try:
            if not await self._wait_for_playback_start(player, timeout=12.0):
                _LOGGER.warning("Transfer target %s never reached 'playing'", player)
                return
            if not await self._verify_snapclient_audio(player):
                await self._notify_play_failure(player)
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning("Post-transfer audio verify failed: %s", err)

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
            resolved = ""

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
                # Resolve the query as an artist name first — "shuffle <artist>"
                # is the dominant use, and STT renderings like "forty two dog"
                # find no playlist while the canonical "42 Dugg" has an official
                # Essentials. Non-artist queries (genres, moods) resolve to
                # nothing plausible and fall through with the raw query.
                resolved = await self._resolve_artist_name(ma_config_entry_id, query)
                search_name = resolved if resolved and resolved.lower() != query_lower else query
                if search_name != query:
                    _LOGGER.info("Shuffle query '%s' resolved to artist '%s'", query, search_name)
                search_result = await self._hass.services.async_call(
                    "music_assistant", "search",
                    {"config_entry_id": ma_config_entry_id, "name": search_name, "media_type": ["playlist"], "limit": 10},
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
                _playlist_stopwords = {"the", "and", "for", "music", "playlist", "mix", "songs", "some"}

                def name_matches_query(playlist_name_str: str) -> bool:
                    """Does the playlist name plausibly answer the request?

                    Full containment of the query (raw or artist-resolved), or
                    ALL significant query words present. A single shared word
                    is NOT enough — "shuffle forty two dog" once matched a
                    random playlist literally named "Forty two" on "forty".
                    """
                    name_lower = _strip_accents(playlist_name_str.lower())
                    for whole in (query_lower, resolved.lower() if resolved else ""):
                        if whole and _strip_accents(whole) in name_lower:
                            return True
                    significant = [w for w in query_words
                                   if len(w) >= 3 and w not in _playlist_stopwords]
                    if significant and all(_strip_accents(w) in name_lower for w in significant):
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
