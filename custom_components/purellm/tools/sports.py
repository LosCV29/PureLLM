"""Sports tool handlers."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, TYPE_CHECKING

from ..const import API_TIMEOUT
from ..utils.http_client import fetch_json, log_and_error, CACHE_TTL_SHORT

if TYPE_CHECKING:
    import aiohttp
    from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)

# Common ESPN API headers
ESPN_HEADERS = {"User-Agent": "HomeAssistant-PolyVoice/1.0"}

# Cache TTL for team lists (teams don't change often)
TEAMS_CACHE_TTL = 3600  # 1 hour


async def _fetch_teams_for_league(
    session: "aiohttp.ClientSession",
    sport: str,
    league: str,
) -> tuple[str, str, list[dict]]:
    """Fetch teams for a single league. Returns (sport, league, teams_list)."""
    teams_url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/teams?limit=1000"
    data, status = await fetch_json(session, teams_url, headers=ESPN_HEADERS, cache_ttl=TEAMS_CACHE_TTL)
    if data and status == 200:
        teams = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
        return (sport, league, teams)
    return (sport, league, [])


async def _fetch_soccer_scoreboard_for_date(
    session: "aiohttp.ClientSession",
    league: str,
    date_str: str,
    days_ahead: int,
) -> tuple[int, str, dict | None]:
    """Fetch soccer scoreboard for a single date. Returns (days_ahead, date_str, data)."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard?dates={date_str}"
    data, status = await fetch_json(session, url, headers=ESPN_HEADERS, cache_ttl=CACHE_TTL_SHORT)
    if data and status == 200:
        return (days_ahead, date_str, data)
    return (days_ahead, date_str, None)


# ============ ESPN API Functions ============


async def get_sports_info(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    hass_timezone,
    track_api_call: callable,
    tavily_api_key: str = None,
) -> dict[str, Any]:
    """Get sports info from ESPN API with dynamic team search.

    Args:
        arguments: Tool arguments (team_name, query_type)
        session: aiohttp session
        hass_timezone: Home Assistant timezone
        track_api_call: Callback to track API usage
        tavily_api_key: Unused, kept for API compatibility

    Returns:
        Sports data dict
    """
    from homeassistant.util import dt as dt_util

    team_name = arguments.get("team_name", "")
    query_type = arguments.get("query_type", "both")

    if not team_name:
        return {"error": "No team name provided"}

    try:
        track_api_call("sports")
        team_key = team_name.lower().strip()

        # Detect sport type from query BEFORE removing noise words
        wants_football = "football" in team_key
        wants_basketball = "basketball" in team_key
        wants_baseball = "baseball" in team_key
        wants_hockey = "hockey" in team_key

        # Remove common extra words that don't help with team matching
        noise_words = ["game", "match", "next", "last", "the", "play", "playing", "fixture", "fixtures", "schedule",
                       "mens", "womens", "men's", "women's", "basketball", "football", "baseball", "hockey",
                       "team", "university", "of", "college", "state"]
        for word in noise_words:
            team_key = team_key.replace(word, "").strip()
        # Clean up multiple spaces
        team_key = " ".join(team_key.split())

        # Check for UEFA Champions League ONLY - these keywords mean user wants UCL
        ucl_keywords = ["champions league", "ucl", "champions", "uefa"]
        wants_ucl = any(kw in team_key for kw in ucl_keywords)

        # Remove UCL keywords from team_key if present
        if wants_ucl:
            for kw in ucl_keywords:
                team_key = team_key.replace(kw, "").strip()

        # Define leagues to search - order based on sport keyword in query
        if wants_ucl:
            # User explicitly wants Champions League - ONLY search UCL
            leagues_to_try = [("soccer", "uefa.champions")]
        elif wants_football:
            # User asked about football - ONLY search football leagues
            leagues_to_try = [
                ("football", "nfl"),
                ("football", "college-football"),
            ]
        elif wants_basketball:
            # User asked about basketball - ONLY search basketball leagues
            leagues_to_try = [
                ("basketball", "nba"),
                ("basketball", "mens-college-basketball"),
            ]
        elif wants_baseball:
            # User asked about baseball - ONLY search MLB
            leagues_to_try = [
                ("baseball", "mlb"),
            ]
        elif wants_hockey:
            # User asked about hockey - ONLY search NHL
            leagues_to_try = [
                ("hockey", "nhl"),
            ]
        else:
            # Default order - soccer first, then American sports
            leagues_to_try = [
                ("soccer", "eng.1"),      # Premier League
                ("soccer", "esp.1"),      # La Liga
                ("soccer", "ger.1"),      # Bundesliga
                ("soccer", "ita.1"),      # Serie A
                ("soccer", "fra.1"),      # Ligue 1
                ("basketball", "nba"),
                ("football", "nfl"),
                ("baseball", "mlb"),
                ("hockey", "nhl"),
                ("basketball", "mens-college-basketball"),  # NCAA Basketball
                ("football", "college-football"),  # NCAA Football
            ]

        team_found = False
        url = None
        full_name = team_name
        team_leagues = []
        team_id = None
        team_abbrev = None  # Store abbreviation for backup matching

        search_words = team_key.split()

        # Fetch all league teams in parallel (major optimization)
        league_tasks = [
            _fetch_teams_for_league(session, sport, league)
            for sport, league in leagues_to_try
        ]
        all_league_results = await asyncio.gather(*league_tasks, return_exceptions=True)

        # Search through results for matching team
        for match_type in ["abbrev", "name"]:
            if team_found:
                break
            for result in all_league_results:
                if team_found:
                    break
                if isinstance(result, Exception):
                    continue
                sport, league, teams = result
                for team in teams:
                    t = team.get("team", {})
                    match = False
                    if match_type == "abbrev":
                        match = team_key == t.get("abbreviation", "").lower()
                    else:
                        display_name = t.get("displayName", "").lower()
                        short_name = t.get("shortDisplayName", "").lower()
                        nickname = t.get("nickname", "").lower()
                        combined = f"{display_name} {short_name} {nickname}"
                        match = all(word in combined for word in search_words)

                    if match:
                        team_id = t.get("id", "")
                        team_abbrev = t.get("abbreviation", "").lower()
                        full_name = t.get("displayName", team_name)
                        url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/teams/{team_id}/schedule"
                        team_leagues.append((sport, league))
                        if not team_found:
                            team_found = True
                        break

        if not team_found:
            if wants_ucl:
                return {"error": f"Team '{team_name}' not found in UEFA Champions League."}
            return {"error": f"Team '{team_name}' not found. Try the full team name (e.g., 'Miami Heat', 'Manchester City', 'Real Madrid')."}

        result = {"team": full_name}

        # Fetch standings if requested
        if query_type == "standings" and team_id:
            found_sport, found_league = team_leagues[0] if team_leagues else (None, None)
            if found_sport and found_league:
                try:
                    standings_url = f"https://site.api.espn.com/apis/v2/sports/{found_sport}/{found_league}/standings"
                    async with session.get(standings_url, headers=ESPN_HEADERS) as standings_resp:
                        if standings_resp.status == 200:
                            standings_data = await standings_resp.json()

                            # Navigate standings structure to find the team
                            for child in standings_data.get("children", []):
                                for division in child.get("standings", {}).get("entries", []):
                                    team_info = division.get("team", {})
                                    if team_info.get("id") == team_id:
                                        stats = {s.get("name"): s.get("displayValue", s.get("value")) for s in division.get("stats", [])}

                                        # Prefer 'overall' record if available (college sports), otherwise use wins-losses
                                        if "overall" in stats:
                                            record = stats["overall"]
                                        else:
                                            wins = stats.get("wins", "0")
                                            losses = stats.get("losses", "0")
                                            record = f"{wins}-{losses}"
                                            if "OTL" in stats or "otLosses" in stats:  # NHL format
                                                otl = stats.get("OTL", stats.get("otLosses", "0"))
                                                record = f"{wins}-{losses}-{otl}"

                                        # Get conference/division rank if available
                                        conf_rank = stats.get("playoffSeed", stats.get("divisionRank", ""))

                                        standing_text = f"{full_name} is {record}"
                                        if conf_rank:
                                            standing_text += f", #{conf_rank} seed"

                                        result["standings"] = {
                                            "record": record,
                                            "rank": conf_rank,
                                            "summary": standing_text
                                        }
                                        break
                                if "standings" in result:
                                    break

                            # Also check flat standings structure (some leagues use this)
                            if "standings" not in result:
                                for entry in standings_data.get("standings", {}).get("entries", []):
                                    team_info = entry.get("team", {})
                                    if team_info.get("id") == team_id:
                                        stats = {s.get("name"): s.get("displayValue", s.get("value")) for s in entry.get("stats", [])}
                                        conf_rank = stats.get("playoffSeed", stats.get("divisionRank", ""))

                                        # Prefer 'overall' record if available (college sports)
                                        if "overall" in stats:
                                            record = stats["overall"]
                                        else:
                                            wins = stats.get("wins", "0")
                                            losses = stats.get("losses", "0")
                                            record = f"{wins}-{losses}"

                                        standing_text = f"{full_name} is {record}"
                                        if conf_rank:
                                            standing_text += f", #{conf_rank} seed"

                                        result["standings"] = {
                                            "record": record,
                                            "rank": conf_rank,
                                            "summary": standing_text
                                        }
                                        break
                except Exception as standings_err:
                    _LOGGER.warning("Failed to fetch standings for %s: %s", full_name, standings_err)

        # Check scoreboard for live/upcoming games
        live_game_from_scoreboard = None
        next_game_from_scoreboard = None

        try:
            found_sport, found_league = team_leagues[0] if team_leagues else (None, None)
            scoreboards_to_check = [(found_sport, found_league)]
            _LOGGER.debug("Sports: Checking scoreboard for %s (id=%s, abbrev=%s) in %s/%s",
                         full_name, team_id, team_abbrev, found_sport, found_league)

            for sb_sport, sb_league in scoreboards_to_check:
                if live_game_from_scoreboard and next_game_from_scoreboard:
                    break

                scoreboard_url = f"https://site.api.espn.com/apis/site/v2/sports/{sb_sport}/{sb_league}/scoreboard"
                async with session.get(scoreboard_url, headers=ESPN_HEADERS) as sb_resp:
                    if sb_resp.status != 200:
                        continue
                    sb_data = await sb_resp.json()

                    for sb_event in sb_data.get("events", []):
                        sb_comp = sb_event.get("competitions", [{}])[0]
                        sb_status = sb_comp.get("status", {}).get("type", {})
                        sb_state = sb_status.get("state", "")

                        sb_competitors = sb_comp.get("competitors", [])

                        # Match by ID, abbreviation, or display name (ESPN IDs can differ between endpoints for college sports)
                        team_match = False
                        for c in sb_competitors:
                            c_team = c.get("team", {})
                            c_id = c_team.get("id", "")
                            c_abbrev = c_team.get("abbreviation", "").lower()
                            c_name = c_team.get("displayName", "").lower()
                            if c_id == team_id:
                                team_match = True
                                break
                            if team_abbrev and c_abbrev == team_abbrev:
                                team_match = True
                                break
                            if full_name.lower() in c_name or c_name in full_name.lower():
                                team_match = True
                                break

                        if not team_match:
                            # Log what we're checking for debugging
                            competitor_names = [c.get("team", {}).get("displayName", "?") for c in sb_competitors]
                            _LOGGER.debug("Sports: Scoreboard event %s - no match (looking for %s)",
                                         competitor_names, full_name)
                            continue

                        _LOGGER.debug("Sports: Found team match on scoreboard, state=%s", sb_state)
                        home_team_sb = next((c for c in sb_competitors if c.get("homeAway") == "home"), {})
                        away_team_sb = next((c for c in sb_competitors if c.get("homeAway") == "away"), {})
                        home_name = home_team_sb.get("team", {}).get("displayName", "Home")
                        away_name = away_team_sb.get("team", {}).get("displayName", "Away")

                        if sb_state == "in":
                            home_score = home_team_sb.get("score", "0")
                            away_score = away_team_sb.get("score", "0")
                            if isinstance(home_score, dict):
                                home_score = home_score.get("displayValue", "0")
                            if isinstance(away_score, dict):
                                away_score = away_score.get("displayValue", "0")

                            status_detail = sb_status.get("detail", "In Progress")
                            result["live_game"] = {
                                "home_team": home_name,
                                "away_team": away_name,
                                "home_score": home_score,
                                "away_score": away_score,
                                "status": status_detail,
                                "summary": f"LIVE: {away_name} {away_score} @ {home_name} {home_score} ({status_detail})"
                            }
                            live_game_from_scoreboard = True

                        elif sb_state == "pre" and not next_game_from_scoreboard:
                            game_date_str = sb_event.get("date", "")
                            if game_date_str:
                                try:
                                    game_dt = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                                    game_dt_local = game_dt.astimezone(hass_timezone)
                                    now_local = datetime.now(hass_timezone)

                                    game_date_only = game_dt_local.date()
                                    today_date = now_local.date()
                                    tomorrow_date = today_date + timedelta(days=1)

                                    time_str = game_dt_local.strftime("%I:%M %p").lstrip("0")
                                    if game_date_only == today_date:
                                        formatted_date = f"Today at {time_str}"
                                    elif game_date_only == tomorrow_date:
                                        formatted_date = f"Tomorrow at {time_str}"
                                    else:
                                        formatted_date = game_dt_local.strftime("%A, %B %d at %I:%M %p")
                                except (ValueError, KeyError, TypeError, AttributeError):
                                    formatted_date = sb_status.get("detail", "TBD")
                            else:
                                formatted_date = sb_status.get("detail", "TBD")

                            venue = sb_comp.get("venue", {}).get("fullName", "")
                            result["next_game"] = {
                                "date": formatted_date,
                                "home_team": home_name,
                                "away_team": away_name,
                                "venue": venue,
                                "summary": f"{away_name} @ {home_name} - {formatted_date}"
                            }
                            next_game_from_scoreboard = True

            # For soccer: check upcoming dates if no next game found (ESPN soccer schedule API only shows completed games)
            # Fetch all 22 days in parallel for major performance improvement
            major_soccer_leagues = ["eng.1", "esp.1", "fra.1", "ger.1", "ita.1", "uefa.champions"]
            if found_sport == "soccer" and not next_game_from_scoreboard and found_league in major_soccer_leagues:
                now_local = datetime.now(hass_timezone)

                # Build tasks for all 22 days in parallel
                future_tasks = []
                for days_ahead in range(0, 22):
                    future_date = now_local + timedelta(days=days_ahead)
                    date_str = future_date.strftime("%Y%m%d")
                    future_tasks.append(
                        _fetch_soccer_scoreboard_for_date(session, found_league, date_str, days_ahead)
                    )

                # Fetch all dates in parallel
                future_results = await asyncio.gather(*future_tasks, return_exceptions=True)

                # Sort by days_ahead and process to find earliest upcoming game
                valid_results = [r for r in future_results if not isinstance(r, Exception) and r[2] is not None]
                valid_results.sort(key=lambda x: x[0])  # Sort by days_ahead

                for days_ahead, date_str, future_data in valid_results:
                    if next_game_from_scoreboard:
                        break
                    for fut_event in future_data.get("events", []):
                        if next_game_from_scoreboard:
                            break
                        fut_comp = fut_event.get("competitions", [{}])[0]
                        # Skip completed or in-progress games
                        fut_status = fut_comp.get("status", {}).get("type", {})
                        if fut_status.get("state", "") != "pre":
                            continue
                        fut_competitors = fut_comp.get("competitors", [])
                        # Match by ID, abbreviation, or display name (ESPN IDs can differ between endpoints)
                        team_match = False
                        for c in fut_competitors:
                            c_team = c.get("team", {})
                            c_id = c_team.get("id", "")
                            c_abbrev = c_team.get("abbreviation", "").lower()
                            c_name = c_team.get("displayName", "").lower()
                            if c_id == team_id:
                                team_match = True
                                break
                            if team_abbrev and c_abbrev == team_abbrev:
                                team_match = True
                                break
                            if full_name.lower() in c_name or c_name in full_name.lower():
                                team_match = True
                                break
                        if not team_match:
                            continue
                        # Found upcoming game
                        home_team_fut = next((c for c in fut_competitors if c.get("homeAway") == "home"), {})
                        away_team_fut = next((c for c in fut_competitors if c.get("homeAway") == "away"), {})
                        home_name = home_team_fut.get("team", {}).get("displayName", "Home")
                        away_name = away_team_fut.get("team", {}).get("displayName", "Away")
                        game_date_str = fut_event.get("date", "")
                        try:
                            game_dt = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                            game_dt_local = game_dt.astimezone(hass_timezone)
                            game_date_only = game_dt_local.date()
                            today_date = now_local.date()
                            tomorrow_date = today_date + timedelta(days=1)
                            time_str = game_dt_local.strftime("%I:%M %p").lstrip("0")
                            if game_date_only == today_date:
                                formatted_date = f"Today at {time_str}"
                            elif game_date_only == tomorrow_date:
                                formatted_date = f"Tomorrow at {time_str}"
                            else:
                                formatted_date = game_dt_local.strftime("%A, %B %d at %I:%M %p")
                        except (ValueError, KeyError, TypeError, AttributeError):
                            formatted_date = "TBD"
                        venue = fut_comp.get("venue", {}).get("fullName", "")
                        result["next_game"] = {
                            "date": formatted_date,
                            "home_team": home_name,
                            "away_team": away_name,
                            "venue": venue,
                            "summary": f"{away_name} @ {home_name} - {formatted_date}"
                        }
                        next_game_from_scoreboard = True
                        break

        except Exception as e:
            _LOGGER.warning("Failed to check scoreboard for live games: %s", e)

        # Get schedule for last game
        if url:
            async with session.get(url, headers=ESPN_HEADERS) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    events = data.get("events", [])

                    if query_type in ["last_game", "both"]:
                        # Find last completed game
                        last_game = None
                        for event in events:
                            status_info = event.get("competitions", [{}])[0].get("status", {}).get("type", {})
                            is_completed = status_info.get("completed", False)
                            if is_completed:
                                game_date_str = event.get("date", "")
                                if game_date_str:
                                    try:
                                        game_date = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                                        if last_game is None:
                                            last_game = event
                                        else:
                                            last_date = datetime.fromisoformat(last_game.get("date", "2000-01-01").replace("Z", "+00:00"))
                                            if game_date > last_date:
                                                last_game = event
                                    except (ValueError, KeyError, TypeError):
                                        pass

                        if last_game:
                            comp = last_game.get("competitions", [{}])[0]
                            competitors = comp.get("competitors", [])
                            home_team = next((c for c in competitors if c.get("homeAway") == "home"), {})
                            away_team = next((c for c in competitors if c.get("homeAway") == "away"), {})

                            home_name = home_team.get("team", {}).get("displayName", "Home")
                            away_name = away_team.get("team", {}).get("displayName", "Away")
                            home_score_raw = home_team.get("score", "0")
                            away_score_raw = away_team.get("score", "0")
                            home_score = home_score_raw.get("displayValue", home_score_raw) if isinstance(home_score_raw, dict) else home_score_raw
                            away_score = away_score_raw.get("displayValue", away_score_raw) if isinstance(away_score_raw, dict) else away_score_raw

                            # Format last game date with relative dates
                            game_date_str = last_game.get("date", "")
                            formatted_last_date = game_date_str[:10]  # Default fallback
                            if game_date_str:
                                try:
                                    # Handle both full ISO timestamps and date-only strings
                                    if "T" in game_date_str:
                                        # Full timestamp: "2026-01-07T00:30:00Z"
                                        game_dt = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                                    else:
                                        # Date only: "2026-01-07" - treat as UTC midnight
                                        game_dt = datetime.fromisoformat(game_date_str).replace(tzinfo=timezone.utc)

                                    game_dt_local = game_dt.astimezone(hass_timezone)
                                    now_local = datetime.now(hass_timezone)

                                    game_date_only = game_dt_local.date()
                                    today_date = now_local.date()
                                    yesterday_date = today_date - timedelta(days=1)

                                    if game_date_only == today_date:
                                        formatted_last_date = "today"
                                    elif game_date_only == yesterday_date:
                                        formatted_last_date = "yesterday"
                                    else:
                                        formatted_last_date = game_dt_local.strftime("%A, %B %d")
                                except (ValueError, KeyError, TypeError, AttributeError):
                                    pass

                            result["last_game"] = {
                                "date": formatted_last_date,
                                "home_team": home_name,
                                "away_team": away_name,
                                "home_score": home_score,
                                "away_score": away_score,
                                "summary": f"{away_name} {away_score} @ {home_name} {home_score}"
                            }

                    # Find next upcoming game from schedule if not found on scoreboard
                    if not next_game_from_scoreboard and query_type in ["next_game", "both"]:
                        now_utc = datetime.now(timezone.utc)
                        next_game = None
                        next_game_date = None
                        _LOGGER.debug("Sports: Checking schedule for next game, %d events total", len(events))

                        for event in events:
                            status_info = event.get("competitions", [{}])[0].get("status", {}).get("type", {})
                            is_completed = status_info.get("completed", False)
                            state = status_info.get("state", "")
                            event_date = event.get("date", "")[:10]
                            _LOGGER.debug("Sports: Schedule event date=%s, state=%s, completed=%s",
                                         event_date, state, is_completed)

                            # Accept any game that isn't completed and isn't currently in progress
                            # College sports may use different states than "pre"
                            if not is_completed and state != "in" and state != "post":
                                game_date_str = event.get("date", "")
                                if game_date_str:
                                    try:
                                        game_dt = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                                        if game_dt > now_utc:
                                            if next_game is None or game_dt < next_game_date:
                                                next_game = event
                                                next_game_date = game_dt
                                    except (ValueError, KeyError, TypeError):
                                        pass

                        if next_game:
                            comp = next_game.get("competitions", [{}])[0]
                            competitors = comp.get("competitors", [])
                            home_team = next((c for c in competitors if c.get("homeAway") == "home"), {})
                            away_team = next((c for c in competitors if c.get("homeAway") == "away"), {})
                            home_name = home_team.get("team", {}).get("displayName", "Home")
                            away_name = away_team.get("team", {}).get("displayName", "Away")

                            # Format the date with relative dates
                            game_dt_local = next_game_date.astimezone(hass_timezone)
                            now_local = datetime.now(hass_timezone)
                            game_date_only = game_dt_local.date()
                            today_date = now_local.date()
                            tomorrow_date = today_date + timedelta(days=1)

                            time_str = game_dt_local.strftime("%I:%M %p").lstrip("0")
                            if game_date_only == today_date:
                                formatted_date = f"Today at {time_str}"
                            elif game_date_only == tomorrow_date:
                                formatted_date = f"Tomorrow at {time_str}"
                            else:
                                formatted_date = game_dt_local.strftime("%A, %B %d at %I:%M %p")

                            venue = comp.get("venue", {}).get("fullName", "")
                            result["next_game"] = {
                                "date": formatted_date,
                                "home_team": home_name,
                                "away_team": away_name,
                                "venue": venue,
                                "summary": f"{away_name} @ {home_name} - {formatted_date}"
                            }

        # Build response text
        response_parts = []
        if "standings" in result:
            response_parts.append(result["standings"]["summary"])
        if "live_game" in result:
            response_parts.append(result["live_game"]["summary"])
        if "last_game" in result:
            lg = result["last_game"]
            date_str = lg['date']
            # Build natural sentence for last game
            home = lg['home_team']
            away = lg['away_team']
            h_score = lg['home_score']
            a_score = lg['away_score']
            # Determine winner and build natural response
            try:
                if int(h_score) > int(a_score):
                    winner, loser = home, away
                    w_score, l_score = h_score, a_score
                else:
                    winner, loser = away, home
                    w_score, l_score = a_score, h_score
                # Natural phrasing the LLM won't want to change
                if date_str in ["today", "yesterday"]:
                    response_parts.append(f"{winner} beat {loser} {w_score}-{l_score} {date_str}")
                else:
                    response_parts.append(f"{winner} beat {loser} {w_score}-{l_score} on {date_str}")
            except ValueError:
                if date_str in ["today", "yesterday"]:
                    response_parts.append(f"{lg['summary']} {date_str}")
                else:
                    response_parts.append(f"{lg['summary']} on {date_str}")
        if "next_game" in result:
            ng = result["next_game"]
            next_text = f"Next game: {ng['summary']}"
            if ng.get('venue'):
                next_text += f" at {ng['venue']}"
            response_parts.append(next_text)

        result["response_text"] = ". ".join(response_parts) if response_parts else f"No game info found for {full_name}"

        _LOGGER.info("Sports info for %s: %s", full_name, result.get("response_text", ""))
        # Return only response_text to prevent LLM from reformatting dates
        return {"response_text": result["response_text"], "team": result.get("team", "")}

    except Exception as err:
        return log_and_error("Failed to get sports info", err)


async def get_ufc_info(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    hass_timezone,
    track_api_call: callable,
    tavily_api_key: str = None,
) -> dict[str, Any]:
    """Get UFC/MMA event information from ESPN API.

    Args:
        arguments: Tool arguments (query_type)
        session: aiohttp session
        hass_timezone: Home Assistant timezone
        track_api_call: Callback to track API usage

    Returns:
        UFC event data dict
    """
    query_type = arguments.get("query_type", "next_event")

    try:
        track_api_call("sports")
        events_url = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"

        data, status = await fetch_json(session, events_url, headers=ESPN_HEADERS)
        if data is None:
            return {"error": f"ESPN UFC API error: {status}"}

        leagues = data.get("leagues", [{}])
        calendar = leagues[0].get("calendar", []) if leagues else []

        if not calendar:
            return {"error": "No upcoming UFC events found"}

        result = {"events": []}
        now_local = datetime.now(hass_timezone)
        today_date = now_local.date()
        tomorrow_date = today_date + timedelta(days=1)
        yesterday_date = today_date - timedelta(days=1)

        for event in calendar[:5]:
            event_info = {
                "name": event.get("label", "Unknown Event"),
                "date": event.get("startDate", "")[:10] if event.get("startDate") else "TBD"
            }
            if event_info["date"] and event_info["date"] != "TBD":
                try:
                    event_dt = datetime.fromisoformat(event.get("startDate", "").replace("Z", "+00:00"))
                    event_dt_local = event_dt.astimezone(hass_timezone)
                    event_date_only = event_dt_local.date()

                    if event_date_only == today_date:
                        event_info["formatted_date"] = "Today"
                    elif event_date_only == tomorrow_date:
                        event_info["formatted_date"] = "Tomorrow"
                    elif event_date_only == yesterday_date:
                        event_info["formatted_date"] = "Yesterday"
                    else:
                        event_info["formatted_date"] = event_dt_local.strftime("%A, %B %d")
                except:
                    event_info["formatted_date"] = event_info["date"]
            result["events"].append(event_info)

        if query_type == "next_event" and result["events"]:
            next_evt = result["events"][0]
            result["response_text"] = f"The next UFC event is {next_evt['name']} on {next_evt.get('formatted_date', next_evt['date'])}."
        elif query_type == "upcoming" and result["events"]:
            event_list = [f"{e['name']} ({e.get('formatted_date', e['date'])})" for e in result["events"]]
            result["response_text"] = "Upcoming UFC events: " + ", ".join(event_list)
        else:
            result["response_text"] = "No upcoming UFC events found."

        _LOGGER.info("UFC info: %s", result.get("response_text", ""))
        return result

    except Exception as err:
        return log_and_error("Failed to get UFC info", err)


# League code mappings for ESPN API
LEAGUE_CODES = {
    # American sports
    "nfl": ("football", "nfl"),
    "nba": ("basketball", "nba"),
    "mlb": ("baseball", "mlb"),
    "nhl": ("hockey", "nhl"),
    "mls": ("soccer", "usa.1"),
    # Soccer/Football
    "premier league": ("soccer", "eng.1"),
    "epl": ("soccer", "eng.1"),
    "la liga": ("soccer", "esp.1"),
    "bundesliga": ("soccer", "ger.1"),
    "serie a": ("soccer", "ita.1"),
    "ligue 1": ("soccer", "fra.1"),
    "champions league": ("soccer", "uefa.champions"),
    "ucl": ("soccer", "uefa.champions"),
}


def _parse_league_and_date(arguments: dict[str, Any], hass_timezone) -> tuple:
    """Parse league and date from arguments. Returns (league_display, sport, league_code, date_label, date_str, error)."""
    league_input = arguments.get("league", "").lower().strip()
    date_input = arguments.get("date", "today").lower().strip()

    if not league_input:
        return None, None, None, None, None, "No league specified. Try: NFL, NBA, MLB, NHL, Premier League, etc."

    # Map league name to ESPN codes
    league_key = None
    for key in LEAGUE_CODES:
        if key in league_input or league_input in key:
            league_key = key
            break

    if not league_key:
        available = ", ".join(sorted(set(k.upper() for k in LEAGUE_CODES.keys())))
        return None, None, None, None, None, f"Unknown league '{league_input}'. Available: {available}"

    sport, league_code = LEAGUE_CODES[league_key]
    league_display = league_key.upper()

    # Determine date
    now_local = datetime.now(hass_timezone)
    today_date = now_local.date()
    tomorrow_date = today_date + timedelta(days=1)

    if date_input in ["today", "tonight", "now"]:
        target_date = today_date
        date_label = "today"
    elif date_input in ["tomorrow"]:
        target_date = tomorrow_date
        date_label = "tomorrow"
    else:
        target_date = today_date
        date_label = "today"

    date_str = target_date.strftime("%Y%m%d")
    return league_display, sport, league_code, date_label, date_str, None


async def check_league_games(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    hass_timezone,
    track_api_call: callable,
    tavily_api_key: str = None,
) -> dict[str, Any]:
    """Check if there are games for a league (count only, no game list).

    Use for: "any NFL games today?", "is there NBA tonight?", "how many MLB games?"
    """
    league_display, sport, league_code, date_label, date_str, error = _parse_league_and_date(arguments, hass_timezone)
    if error:
        return {"error": error}

    try:
        track_api_call("sports")
        url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league_code}/scoreboard?dates={date_str}"

        data, status = await fetch_json(session, url, headers=ESPN_HEADERS)
        if data is None:
            return {"error": f"ESPN API error: {status}"}

        game_count = len(data.get("events", []))

        if game_count == 0:
            response_text = f"No {league_display} games {date_label}."
        elif game_count == 1:
            response_text = f"Yes, there's 1 {league_display} game {date_label}."
        else:
            response_text = f"Yes, there are {game_count} {league_display} games {date_label}."

        return {
            "league": league_display,
            "date": date_label,
            "game_count": game_count,
            "response_text": response_text
        }

    except Exception as err:
        return log_and_error(f"Failed to check {league_display} games", err)


async def list_league_games(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    hass_timezone,
    track_api_call: callable,
    tavily_api_key: str = None,
) -> dict[str, Any]:
    """List all games for a league with matchups and times.

    Use for: "what NFL games are today?", "show me NBA games", "list MLB schedule"
    """
    league_display, sport, league_code, date_label, date_str, error = _parse_league_and_date(arguments, hass_timezone)
    if error:
        return {"error": error}

    try:
        track_api_call("sports")
        url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league_code}/scoreboard?dates={date_str}"

        data, status = await fetch_json(session, url, headers=ESPN_HEADERS)
        if data is None:
            return {"error": f"ESPN API error: {status}"}

        events = data.get("events", [])
        game_count = len(events)

        if game_count == 0:
            return {
                "league": league_display,
                "date": date_label,
                "game_count": 0,
                "response_text": f"No {league_display} games {date_label}."
            }

        # Build game list
        game_summaries = []
        for event in events:
            comp = event.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            status_info = comp.get("status", {}).get("type", {})
            state = status_info.get("state", "")

            home_team = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away_team = next((c for c in competitors if c.get("homeAway") == "away"), {})

            home_name = home_team.get("team", {}).get("shortDisplayName", "Home")
            away_name = away_team.get("team", {}).get("shortDisplayName", "Away")

            if state == "in":
                home_score = home_team.get("score", "0")
                away_score = away_team.get("score", "0")
                status_detail = status_info.get("detail", "Live")
                summary = f"{away_name} {away_score} @ {home_name} {home_score} - {status_detail}"
            elif state == "post":
                home_score = home_team.get("score", "0")
                away_score = away_team.get("score", "0")
                summary = f"{away_name} {away_score} @ {home_name} {home_score} (Final)"
            else:
                game_date_str = event.get("date", "")
                time_str = "TBD"
                if game_date_str:
                    try:
                        game_dt = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                        game_dt_local = game_dt.astimezone(hass_timezone)
                        time_str = game_dt_local.strftime("%I:%M %p").lstrip("0")
                    except:
                        pass
                summary = f"{away_name} @ {home_name} - {time_str}"

            game_summaries.append(summary)

        games_list = ", ".join(game_summaries)
        if game_count == 1:
            response_text = f"There's 1 {league_display} game {date_label}: {games_list}"
        else:
            response_text = f"There are {game_count} {league_display} games {date_label}: {games_list}"

        return {
            "league": league_display,
            "date": date_label,
            "game_count": game_count,
            "response_text": response_text
        }

    except Exception as err:
        return log_and_error(f"Failed to list {league_display} games", err)
