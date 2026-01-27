"""Music information tool using MusicBrainz API.

MusicBrainz is a free, community-maintained music database with
accurate discography information.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING
from urllib.parse import quote

from ..const import API_TIMEOUT
from ..utils.http_client import log_and_error

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)

MUSICBRAINZ_API = "https://musicbrainz.org/ws/2"
USER_AGENT = "PureLLM/1.0 (Home Assistant Integration)"


async def get_music_info(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    track_api_call: callable,
) -> dict[str, Any]:
    """Get music/artist information from MusicBrainz.

    Args:
        arguments: Tool arguments containing:
            - artist: Artist name (for album queries)
            - song: Song title (for "who sings" queries)
            - query_type: "latest_album", "discography", "song_artist"
        session: aiohttp session
        track_api_call: Callback to track API usage

    Returns:
        Music info dict with artist and album/song details
    """
    artist_name = arguments.get("artist", "").strip()
    song_title = arguments.get("song", "").strip()
    query_type = arguments.get("query_type", "latest_album")

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    try:
        track_api_call("musicbrainz")

        # Handle song lookup ("who sings X")
        if query_type == "song_artist" or (song_title and not artist_name):
            if not song_title:
                return {"error": "No song title provided"}

            # Search for recording (song)
            search_url = f"{MUSICBRAINZ_API}/recording?query={quote(song_title)}&fmt=json&limit=10"

            async with asyncio.timeout(API_TIMEOUT):
                async with session.get(search_url, headers=headers) as response:
                    if response.status != 200:
                        return {"error": f"MusicBrainz search failed: HTTP {response.status}"}
                    data = await response.json()

            recordings = data.get("recordings", [])
            if not recordings:
                return {
                    "instruction": "Start your response with 'According to MusicBrainz, ...'",
                    "message": f"Song '{song_title}' not found on MusicBrainz",
                }

            # Get the top result
            recording = recordings[0]
            song_name = recording.get("title", song_title)

            # Get artist credits
            artist_credits = recording.get("artist-credit", [])
            if artist_credits:
                artists = []
                for credit in artist_credits:
                    if "artist" in credit:
                        artists.append(credit["artist"].get("name", "Unknown"))
                artist_display = " & ".join(artists) if artists else "Unknown"
            else:
                artist_display = "Unknown"

            # Get album info if available
            releases = recording.get("releases", [])
            album_name = releases[0].get("title") if releases else None

            result = {
                "instruction": "Start your response with 'According to MusicBrainz, ...'",
                "song": song_name,
                "artist": artist_display,
            }
            if album_name:
                result["album"] = album_name

            return result

        # Handle artist/album queries
        if not artist_name:
            return {"error": "No artist name provided"}

        # Step 1: Search for the artist
        search_url = f"{MUSICBRAINZ_API}/artist?query={quote(artist_name)}&fmt=json&limit=5"

        async with asyncio.timeout(API_TIMEOUT):
            async with session.get(search_url, headers=headers) as response:
                if response.status != 200:
                    return {"error": f"MusicBrainz search failed: HTTP {response.status}"}
                data = await response.json()

        artists = data.get("artists", [])
        if not artists:
            return {"error": f"Artist '{artist_name}' not found on MusicBrainz"}

        # Find best match (prioritize exact matches and high scores)
        artist = None
        for a in artists:
            if a.get("name", "").lower() == artist_name.lower():
                artist = a
                break
        if not artist:
            artist = artists[0]  # Take top result

        artist_id = artist["id"]
        artist_display_name = artist.get("name", artist_name)

        _LOGGER.info("MusicBrainz: Found artist '%s' (id=%s)", artist_display_name, artist_id)

        # Step 2: Get release groups (albums) for this artist
        # type=album filters to only studio albums (excludes singles, EPs, compilations)
        releases_url = (
            f"{MUSICBRAINZ_API}/release-group"
            f"?artist={artist_id}"
            f"&type=album"
            f"&fmt=json"
            f"&limit=100"
        )

        # MusicBrainz rate limit: 1 request per second
        await asyncio.sleep(1.1)

        async with asyncio.timeout(API_TIMEOUT):
            async with session.get(releases_url, headers=headers) as response:
                if response.status != 200:
                    return {"error": f"MusicBrainz releases failed: HTTP {response.status}"}
                releases_data = await response.json()

        release_groups = releases_data.get("release-groups", [])

        if not release_groups:
            return {
                "instruction": "Start your response with 'According to MusicBrainz, ...'",
                "artist": artist_display_name,
                "message": f"No studio albums found for {artist_display_name}",
            }

        # Parse and sort albums by date
        albums = []
        for rg in release_groups:
            album_name = rg.get("title", "Unknown")
            first_release = rg.get("first-release-date", "")

            # Parse date (can be YYYY, YYYY-MM, or YYYY-MM-DD)
            year = None
            if first_release:
                try:
                    year = int(first_release[:4])
                except (ValueError, IndexError):
                    pass

            albums.append({
                "name": album_name,
                "release_date": first_release or "Unknown",
                "year": year,
                "type": rg.get("primary-type", "Album"),
            })

        # Sort by date (newest first), putting unknown dates last
        albums.sort(key=lambda x: (x["year"] is None, -(x["year"] or 0)))

        # Build response based on query type
        if query_type == "latest_album" and albums:
            latest = albums[0]
            return {
                "instruction": "Start your response with 'According to MusicBrainz, ...'",
                "artist": artist_display_name,
                "latest_album": latest["name"],
                "release_date": latest["release_date"],
            }

        elif query_type in ("discography", "albums"):
            # Return full discography
            album_list = [
                f"{a['name']} ({a['release_date']})" for a in albums[:15]
            ]
            return {
                "instruction": "Start your response with 'According to MusicBrainz, ...'",
                "artist": artist_display_name,
                "total_albums": len(albums),
                "albums": album_list,
            }

        else:
            # Default: return latest with some context
            latest = albums[0] if albums else None
            return {
                "instruction": "Start your response with 'According to MusicBrainz, ...'",
                "artist": artist_display_name,
                "latest_album": latest["name"] if latest else "Unknown",
                "release_date": latest["release_date"] if latest else "Unknown",
                "total_albums": len(albums),
            }

    except asyncio.TimeoutError:
        return log_and_error("MusicBrainz request timed out", exc_info=False)
    except Exception as err:
        return log_and_error("MusicBrainz lookup failed", err)
