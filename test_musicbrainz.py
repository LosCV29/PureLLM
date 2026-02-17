#!/usr/bin/env python3
"""Standalone test for MusicBrainz API infrastructure.

Run: python3 test_musicbrainz.py

Tests _escape_lucene (unit), then makes real API calls to verify
rate limiting, retry logic, escaping, and caching all work.

No Home Assistant dependency — imports only the MusicBrainz functions
by monkey-patching the HA imports out.
"""
import asyncio
import sys
import time
import types

# Stub out Home Assistant imports so music.py can load standalone
ha_stub = types.ModuleType("homeassistant")
ha_comp = types.ModuleType("homeassistant.components")
ha_mp = types.ModuleType("homeassistant.components.media_player")
ha_mp.MediaPlayerEntityFeature = type("MediaPlayerEntityFeature", (), {"PAUSE": 1, "STOP": 2, "PLAY": 4})
ha_helpers = types.ModuleType("homeassistant.helpers")
ha_er = types.ModuleType("homeassistant.helpers.entity_registry")
ha_dr = types.ModuleType("homeassistant.helpers.device_registry")
ha_core = types.ModuleType("homeassistant.core")

sys.modules["homeassistant"] = ha_stub
sys.modules["homeassistant.components"] = ha_comp
sys.modules["homeassistant.components.media_player"] = ha_mp
sys.modules["homeassistant.helpers"] = ha_helpers
sys.modules["homeassistant.helpers.entity_registry"] = ha_er
sys.modules["homeassistant.helpers.device_registry"] = ha_dr
sys.modules["homeassistant.core"] = ha_core

# Stub the parent package so relative imports work
purellm_pkg = types.ModuleType("custom_components.purellm")
purellm_pkg.__path__ = ["custom_components/purellm"]
utils_pkg = types.ModuleType("custom_components.purellm.utils")
helpers_mod = types.ModuleType("custom_components.purellm.utils.helpers")
helpers_mod.COMMON_ROOM_NAMES = set()
sys.modules["custom_components"] = types.ModuleType("custom_components")
sys.modules["custom_components"].__path__ = ["custom_components"]
sys.modules["custom_components.purellm"] = purellm_pkg
sys.modules["custom_components.purellm.utils"] = utils_pkg
sys.modules["custom_components.purellm.utils.helpers"] = helpers_mod

sys.path.insert(0, ".")

from custom_components.purellm.tools.music import (
    _escape_lucene,
    _mb_rate_limit,
    _musicbrainz_get,
    _mb_artist_id_cache,
    _lookup_album_year_musicbrainz,
    _get_artist_discography_musicbrainz,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
INFO = "\033[94mINFO\033[0m"

failures = 0


def assert_eq(label, got, expected):
    global failures
    if got == expected:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}")
        print(f"         expected: {expected!r}")
        print(f"         got:      {got!r}")
        failures += 1


def assert_true(label, value):
    global failures
    if value:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}")
        failures += 1


# -----------------------------------------------------------------------
# 1. Unit tests for _escape_lucene
# -----------------------------------------------------------------------
print("\n=== 1. _escape_lucene unit tests ===")

assert_eq("Plain text unchanged",     _escape_lucene("Taylor Swift"),       "Taylor Swift")
assert_eq("P!nk escapes !",           _escape_lucene("P!nk"),              "P\\!nk")
assert_eq("AC/DC escapes /",          _escape_lucene("AC/DC"),             "AC\\/DC")
assert_eq("Guns N' Roses (clean)",    _escape_lucene("Guns N' Roses"),     "Guns N' Roses")
assert_eq("Ke$ha ($ is not special)", _escape_lucene("Ke$ha"),             "Ke$ha")
assert_eq("Parens escaped",           _escape_lucene("Album (Deluxe)"),    "Album \\(Deluxe\\)")
assert_eq("Brackets escaped",         _escape_lucene("Song [Remix]"),      "Song \\[Remix\\]")
assert_eq("Colon escaped",            _escape_lucene("Vol: 1"),            "Vol\\: 1")
assert_eq("Quotes escaped",           _escape_lucene('Say "Hello"'),       'Say \\"Hello\\"')
assert_eq("Plus escaped",             _escape_lucene("C+C Music"),         "C\\+C Music")
assert_eq("Asterisk escaped",         _escape_lucene("B*Witched"),         "B\\*Witched")
assert_eq("Tilde escaped",            _escape_lucene("test~2"),            "test\\~2")
assert_eq("Question mark escaped",    _escape_lucene("Who?"),              "Who\\?")
assert_eq("&& double-escaped",        _escape_lucene("Tom && Jerry"),      "Tom \\&\\& Jerry")
assert_eq("|| double-escaped",        _escape_lucene("This || That"),      "This \\|\\| That")
assert_eq("Single & not escaped",     _escape_lucene("Tom & Jerry"),       "Tom \\& Jerry")
assert_eq("Backslash escaped",        _escape_lucene("back\\slash"),       "back\\\\slash")
assert_eq("Caret escaped",            _escape_lucene("test^3"),            "test\\^3")
assert_eq("Hyphen escaped",           _escape_lucene("Jay-Z"),             "Jay\\-Z")
assert_eq("Empty string",             _escape_lucene(""),                  "")


# -----------------------------------------------------------------------
# 2. Rate limiter test
# -----------------------------------------------------------------------
print("\n=== 2. Rate limiter test ===")


async def test_rate_limiter():
    """Verify rate limiter enforces ~1.1s gap between calls."""
    t0 = time.monotonic()
    await _mb_rate_limit()
    t1 = time.monotonic()
    await _mb_rate_limit()
    t2 = time.monotonic()

    first_gap = t1 - t0
    second_gap = t2 - t1

    print(f"  {INFO}  First call took {first_gap:.2f}s")
    print(f"  {INFO}  Second call took {second_gap:.2f}s")
    assert_true(f"Second call waited >= 1.0s (got {second_gap:.2f}s)", second_gap >= 1.0)

asyncio.run(test_rate_limiter())


# -----------------------------------------------------------------------
# 3. Live API tests (real MusicBrainz calls)
# -----------------------------------------------------------------------
print("\n=== 3. Live MusicBrainz API calls ===")
print(f"  {INFO}  These hit the real API - expect ~1s between each call\n")


async def test_live_api():
    global failures

    # 3a. Basic search — Taylor Swift / 1989
    print("  --- 3a. Search release-group: Taylor Swift / 1989 ---")
    data = await _musicbrainz_get("release-group", {
        "query": f'artist:"{_escape_lucene("Taylor Swift")}" AND releasegroup:"{_escape_lucene("1989")}"',
        "limit": 3,
    })
    assert_true("Got response", data is not None)
    if data:
        rgs = data.get("release-groups", [])
        assert_true(f"Found {len(rgs)} release groups", len(rgs) > 0)
        if rgs:
            title = rgs[0].get("title", "")
            print(f"  {INFO}  First result: '{title}'")
            assert_true("Title contains '1989'", "1989" in title)

    # 3b. Escaped characters — P!nk
    print("\n  --- 3b. Search artist: P!nk (tests ! escaping) ---")
    data = await _musicbrainz_get("artist", {
        "query": f'artist:"{_escape_lucene("P!nk")}"', "limit": 1,
    })
    assert_true("Got response", data is not None)
    if data:
        artists = data.get("artists", [])
        assert_true(f"Found {len(artists)} artists", len(artists) > 0)
        if artists:
            name = artists[0].get("name", "")
            print(f"  {INFO}  Found: '{name}'")
            assert_true("Found P!nk or Pink", "pink" in name.lower() or "p!nk" in name.lower())

    # 3c. Escaped characters — AC/DC
    print("\n  --- 3c. Search artist: AC/DC (tests / escaping) ---")
    data = await _musicbrainz_get("artist", {
        "query": f'artist:"{_escape_lucene("AC/DC")}"', "limit": 1,
    })
    assert_true("Got response", data is not None)
    if data:
        artists = data.get("artists", [])
        assert_true(f"Found {len(artists)} artists", len(artists) > 0)
        if artists:
            name = artists[0].get("name", "")
            print(f"  {INFO}  Found: '{name}'")
            assert_true("Found AC/DC", "ac/dc" in name.lower() or "ac dc" in name.lower())

    # 3d. Full discography lookup (tests artist ID cache + browse)
    print("\n  --- 3d. Discography: Radiohead (tests artist cache) ---")
    _mb_artist_id_cache.clear()
    disco = await _get_artist_discography_musicbrainz("Radiohead", "studio")
    assert_true("Got discography", len(disco) > 0)
    if disco:
        print(f"  {INFO}  Found {len(disco)} studio albums:")
        for d in disco[:5]:
            print(f"         {d['year']}  {d['name']}")
        assert_true("OK Computer is in there", any("ok computer" in d["name"].lower() for d in disco))

    # Verify cache was populated
    assert_true("Artist ID cache populated", "radiohead" in _mb_artist_id_cache)
    cached_id = _mb_artist_id_cache.get("radiohead")
    print(f"  {INFO}  Cached artist ID: {cached_id}")

    # Call again — should use cache (artist lookup skipped)
    disco2 = await _get_artist_discography_musicbrainz("Radiohead", "studio")
    assert_true(f"Second call still works ({len(disco2)} albums)", len(disco2) > 0)

    # 3e. Album year lookup
    print("\n  --- 3e. Album year: 'OK Computer' by Radiohead ---")
    year = await _lookup_album_year_musicbrainz("OK Computer", "Radiohead")
    print(f"  {INFO}  Year: {year}")
    assert_true("Year is 1997", year == 1997)

    # 3f. Album year with special chars in name
    print("\n  --- 3f. Album year: 'Funhouse' by P!nk ---")
    year = await _lookup_album_year_musicbrainz("Funhouse", "P!nk")
    print(f"  {INFO}  Year: {year}")
    assert_true("Year is 2008 or 2009", year in (2008, 2009))

asyncio.run(test_live_api())


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*50}")
if failures == 0:
    print(f"  {PASS}  All tests passed!")
else:
    print(f"  {FAIL}  {failures} test(s) failed")
print(f"{'='*50}\n")

sys.exit(1 if failures else 0)
