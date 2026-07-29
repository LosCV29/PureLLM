"""Weather tool handler."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, TYPE_CHECKING

from ..const import API_TIMEOUT
from ..utils.helpers import format_time_remaining
from ..utils.http_client import fetch_json, log_and_error

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)

# US state name to abbreviation mapping
US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}

# Set of valid state abbreviations for detection
US_STATE_ABBREVS = set(US_STATES.values())

# forecast_type synonyms → canonical value. A weak local brain will not
# reliably emit one of the enum values; it emits whatever word the user said
# ("rain", "week", "sunset", "warning"). Normalizing here is far cheaper than
# widening the enum, which would grow the tool definition for every request.
_FORECAST_TYPE_ALIASES = {
    "current": "current", "now": "current", "today": "today", "day": "today",
    "daily": "today", "rain": "today", "precipitation": "today",
    "hourly": "hourly", "hour": "hourly", "next_hour": "hourly",
    "hours": "hourly", "timeline": "hourly", "afternoon": "hourly",
    "morning": "hourly", "evening": "hourly", "tonight": "hourly",
    "tomorrow": "tomorrow",
    "weekly": "weekly", "week": "weekly", "forecast": "weekly",
    "weekend": "weekly", "5day": "weekly", "7day": "weekly", "both": "weekly",
    "sun_times": "sun_times", "sun": "sun_times", "sunrise": "sun_times",
    "sunset": "sun_times", "daylight": "sun_times",
    "alerts": "alerts", "alert": "alerts", "warning": "alerts",
    "warnings": "alerts", "advisory": "alerts", "severe": "alerts",
}

_WEEKDAYS = [
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
]

# An HOUR at or above this pop is a wet hour for window-scanning purposes.
_RAIN_LIKELY_POP = 0.30

# A DAY's `pop` at or above this is called out as "expect rain" even when no
# single hour crosses the bar. Deliberately higher than the hourly threshold:
# OWM's daily pop is the max over 24h, so a 36% day in South Florida routinely
# means "one hour somewhere might see a shower", not "it will rain today".
# 2026-07-29 live check: pop=0.36 with every hourly pop at 0-2% and OWM's own
# summary reading "There will be partly cloudy today" — answering "yes" there
# is simply wrong.
_RAIN_LIKELY_DAILY_POP = 0.55

# Hours ahead scanned when building the rain window / hourly timeline.
_HOURLY_WINDOW = 12


def _normalize_location(query: str) -> str:
    """Normalize location query for OpenWeatherMap geocoding API.

    Converts full state names to abbreviations and adds ',US' suffix
    for US locations to improve geocoding accuracy.
    """
    if not query:
        return query

    parts = [p.strip() for p in query.split(",")]

    if len(parts) >= 2:
        # Check if second part is a US state (full name or abbrev)
        state_part = parts[1].lower()

        # Convert full state name to abbreviation
        if state_part in US_STATES:
            parts[1] = US_STATES[state_part]
            # Add US suffix if not already present
            if len(parts) == 2:
                parts.append("US")
        # If it's already a state abbreviation, add US suffix
        elif parts[1].upper() in US_STATE_ABBREVS:
            parts[1] = parts[1].upper()
            if len(parts) == 2:
                parts.append("US")

    return ",".join(parts)


def _normalize_forecast_type(raw: Any) -> str:
    """Map whatever the model emitted onto a canonical forecast_type."""
    if not isinstance(raw, str) or not raw.strip():
        return "current"
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if key in _FORECAST_TYPE_ALIASES:
        return _FORECAST_TYPE_ALIASES[key]
    # Substring fallback: "rain_today", "7_day_forecast", "sunset_time".
    # Longest alias first, otherwise "7_day_forecast" matches "day" (→ today)
    # before it reaches "forecast" (→ weekly).
    for alias in sorted(_FORECAST_TYPE_ALIASES, key=len, reverse=True):
        if alias in key:
            return _FORECAST_TYPE_ALIASES[alias]
    _LOGGER.debug("Unrecognized forecast_type %r → 'current'", raw)
    return "current"


def _local_dt(timestamp: int, tz_offset: int) -> datetime:
    """Convert a UNIX timestamp to the *forecast location's* local time.

    The One Call API returns UTC epochs plus a `timezone_offset`. The previous
    implementation used bare datetime.fromtimestamp(), which renders every
    timestamp in the Home Assistant host's timezone — correct for local
    weather, silently wrong for "what time is sunset in Tokyo".
    """
    return datetime.fromtimestamp(timestamp, tz=timezone(timedelta(seconds=tz_offset)))


def _fmt_time(dt: datetime) -> str:
    """Format as '7:04 AM' without a leading zero (platform-independent)."""
    return dt.strftime("%I:%M %p").lstrip("0")


def _fmt_hour(dt: datetime) -> str:
    """Format as '4 PM' without a leading zero."""
    return dt.strftime("%I %p").lstrip("0")


def _day_block(day_data: dict[str, Any], tz_offset: int) -> dict[str, Any]:
    """Build a compact per-day forecast block."""
    dt = _local_dt(day_data.get("dt", 0), tz_offset)
    block = {
        "day": dt.strftime("%A"),
        "date": dt.strftime("%B %d"),
        "high": round(day_data.get("temp", {}).get("max", 0)),
        "low": round(day_data.get("temp", {}).get("min", 0)),
        "conditions": day_data.get("weather", [{}])[0].get("description", "Unknown").title(),
        "rain_chance": round(day_data.get("pop", 0) * 100),
    }
    # NOT "wind_speed": a daily figure sitting next to current.wind_speed under
    # the same key reads as a contradiction (live: current 7, daily 16).
    if day_data.get("wind_speed"):
        block["max_wind_speed"] = round(day_data["wind_speed"])
    if day_data.get("summary"):
        block["summary"] = day_data["summary"]
    return block


def _minutely_nowcast(minutely: list[dict[str, Any]], tz_offset: int) -> str | None:
    """Turn the 60-minute precipitation nowcast into one plain sentence.

    This is what actually answers "is it about to rain" / "do I have time to
    walk the dog" — a daily probability cannot.
    """
    if not minutely:
        return None

    raining_now = minutely[0].get("precipitation", 0) > 0

    if raining_now:
        for idx, minute in enumerate(minutely):
            if minute.get("precipitation", 0) <= 0:
                if idx == 0:
                    break
                return f"Raining now, stopping in about {idx} minutes"
        return "Raining now, continuing for at least the next hour"

    for idx, minute in enumerate(minutely):
        if minute.get("precipitation", 0) > 0:
            if idx <= 2:
                return "Rain starting within a few minutes"
            return f"Rain starting in about {idx} minutes"

    return "No rain in the next hour"


def _is_wet_hour(hour: dict[str, Any]) -> bool:
    """True if this hourly entry is likely to see precipitation."""
    return hour.get("pop", 0) >= _RAIN_LIKELY_POP or "rain" in hour or "snow" in hour


def _hours_left_today(
    hourly: list[dict[str, Any]],
    tz_offset: int,
) -> list[dict[str, Any]]:
    """Hourly entries that still fall on the location's current calendar day.

    "Will it rain today" asked at 8 PM must not be answered from tomorrow
    morning's hours, which a fixed 12-entry slice would happily include.
    """
    if not hourly:
        return []
    today = _local_dt(hourly[0].get("dt", 0), tz_offset).date()
    return [h for h in hourly if _local_dt(h.get("dt", 0), tz_offset).date() == today]


def _rain_window(
    hourly: list[dict[str, Any]],
    tz_offset: int,
    hours: int = _HOURLY_WINDOW,
) -> str | None:
    """Describe the next stretch of likely rain, e.g. '2 PM to 5 PM (70%)'.

    Answers the timing half of "will it rain today" — a bare daily percentage
    leaves the model to guess *when*, and it guesses badly.
    """
    if not hourly:
        return None

    window = hourly[:hours]
    start_idx = None

    for idx, hour in enumerate(window):
        if _is_wet_hour(hour):
            start_idx = idx
            break

    if start_idx is None:
        return None

    end_idx = start_idx
    for idx in range(start_idx, len(window)):
        if _is_wet_hour(window[idx]):
            end_idx = idx
        else:
            break

    peak = round(max(h.get("pop", 0) for h in window[start_idx:end_idx + 1]) * 100)
    start_dt = _local_dt(window[start_idx].get("dt", 0), tz_offset)

    if start_idx == 0:
        end_dt = _local_dt(window[end_idx].get("dt", 0), tz_offset) + timedelta(hours=1)
        return f"now through about {_fmt_hour(end_dt)} ({peak}% chance)"

    end_dt = _local_dt(window[end_idx].get("dt", 0), tz_offset) + timedelta(hours=1)
    if end_idx == start_idx:
        return f"around {_fmt_hour(start_dt)} ({peak}% chance)"
    return f"{_fmt_hour(start_dt)} to {_fmt_hour(end_dt)} ({peak}% chance)"


def _rain_today_answer(
    today: dict[str, Any],
    hourly: list[dict[str, Any]],
    current: dict[str, Any],
    tz_offset: int,
) -> str:
    """One authoritative sentence answering "will it rain today".

    Deliberately a single string rather than a percentage plus a boolean plus a
    window: when those are emitted as three independent fields they can
    disagree, and a small brain will happily read out the contradiction. Fusing
    the daily probability with the hour-by-hour scan here means there is
    exactly one answer to repeat.
    """
    pop = round(today.get("pop", 0) * 100)
    remaining = _hours_left_today(hourly, tz_offset)

    if "rain" in current or "snow" in current:
        return f"Yes — it is raining right now ({pop}% chance today)"

    window = _rain_window(remaining, tz_offset, len(remaining) or 1) if remaining else None

    if window:
        return f"Yes — rain likely {window}"

    if not remaining:
        return f"The day is essentially over; today's chance was {pop}%"

    if pop >= _RAIN_LIKELY_DAILY_POP * 100:
        return (
            f"Possibly — {pop}% chance today, though no single hour in the "
            "remaining forecast shows likely rain"
        )

    return (
        f"No — only a {pop}% chance today, and no rain in the hour-by-hour "
        "forecast for the rest of the day"
    )


def _hourly_timeline(
    hourly: list[dict[str, Any]],
    tz_offset: int,
    hours: int = _HOURLY_WINDOW,
) -> list[dict[str, Any]]:
    """Compact hour-by-hour list for 'this afternoon' / 'tonight' questions."""
    timeline = []
    for hour in hourly[:hours]:
        dt = _local_dt(hour.get("dt", 0), tz_offset)
        timeline.append({
            "time": _fmt_hour(dt),
            "temp": round(hour.get("temp", 0)),
            "conditions": hour.get("weather", [{}])[0].get("description", "Unknown").title(),
            "rain_chance": round(hour.get("pop", 0) * 100),
        })
    return timeline


# Alert events worth interrupting an unrelated weather answer for. NWS uses a
# strict Warning > Watch > Advisory severity ladder, so matching on "warning"
# plus the few life-threatening event names covers it.
_URGENT_ALERT_TERMS = (
    "warning", "tornado", "hurricane", "evacuation", "emergency",
)


def _is_urgent_alert(event: str) -> bool:
    """True if this alert should surface even when alerts weren't asked about."""
    return any(term in event.lower() for term in _URGENT_ALERT_TERMS)


def _format_alerts(alerts: list[dict[str, Any]], tz_offset: int) -> list[dict[str, Any]]:
    """Compact active government weather alerts."""
    formatted = []
    for alert in alerts[:3]:
        entry = {
            "event": alert.get("event", "Weather Alert"),
            "source": alert.get("sender_name", ""),
        }
        if alert.get("end"):
            entry["until"] = _fmt_time(_local_dt(alert["end"], tz_offset))
        description = (alert.get("description") or "").strip().replace("\n", " ")
        if description:
            entry["details"] = description[:300]
        formatted.append(entry)
    return formatted


def _resolve_requested_day(
    day: str,
    daily: list[dict[str, Any]],
    tz_offset: int,
) -> dict[str, Any] | None:
    """Find the daily entry matching a spoken day name ('saturday')."""
    target = day.strip().lower()
    if not target:
        return None

    if target in ("today", "tonight"):
        return daily[0] if daily else None
    if target == "tomorrow":
        return daily[1] if len(daily) > 1 else None

    for name in _WEEKDAYS:
        if name in target:
            for day_data in daily:
                if _local_dt(day_data.get("dt", 0), tz_offset).strftime("%A").lower() == name:
                    return day_data
            return None
    return None


async def get_weather_forecast(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    api_key: str,
    latitude: float,
    longitude: float,
    user_query: str = "",
) -> dict[str, Any]:
    """Get weather forecast from OpenWeatherMap."""
    forecast_type = _normalize_forecast_type(arguments.get("forecast_type"))
    requested_day = (arguments.get("day") or "").strip()
    location_query = arguments.get("location", "").strip()

    if not api_key:
        return {"error": "OpenWeatherMap API key not configured. Add it in Settings → PureLLM → API Keys."}

    # Validate that the location was actually mentioned by the user
    # This prevents models from hallucinating locations like "New York" when none was specified
    if location_query:
        user_query_lower = user_query.lower()
        location_lower = location_query.lower()
        # Extract just the city name (before any comma)
        city_name = location_lower.split(",")[0].strip()
        # Check if any part of the location appears in the user's query
        if city_name not in user_query_lower and location_lower not in user_query_lower:
            _LOGGER.warning(
                "Ignoring hallucinated location '%s' - not found in user query: '%s'",
                location_query, user_query
            )
            location_query = ""  # Reset to use default coordinates

    location_name = None

    # If user specified a location, geocode it
    if location_query:
        # Normalize: convert full state names to abbreviations, add US suffix
        location_query = _normalize_location(location_query)
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location_query}&limit=1&appid={api_key}"
        geo_data, status = await fetch_json(session, geo_url)
        if geo_data is None:
            return {"error": f"Geocoding failed for: {location_query}"}
        if geo_data and len(geo_data) > 0:
            latitude = geo_data[0]["lat"]
            longitude = geo_data[0]["lon"]
            location_name = geo_data[0].get("name", location_query)
            if geo_data[0].get("state"):
                location_name += f", {geo_data[0]['state']}"
            if geo_data[0].get("country"):
                location_name += f", {geo_data[0]['country']}"
            _LOGGER.info("Geocoded '%s' to %s (%s, %s)", location_query, location_name, latitude, longitude)
        else:
            return {"error": f"Could not find location: {location_query}"}
    else:
        # No location specified - reverse geocode the default coordinates to get city name
        reverse_geo_url = f"http://api.openweathermap.org/geo/1.0/reverse?lat={latitude}&lon={longitude}&limit=1&appid={api_key}"
        geo_data, status = await fetch_json(session, reverse_geo_url)
        if geo_data and len(geo_data) > 0:
            location_name = geo_data[0].get("name", "Current Location")
            if geo_data[0].get("state"):
                location_name += f", {geo_data[0]['state']}"
            _LOGGER.info("Reverse geocoded to: %s", location_name)

    try:
        result: dict[str, Any] = {}

        async with asyncio.timeout(API_TIMEOUT):
            # One Call API 3.0. `minutely` and `alerts` are NO LONGER excluded:
            # minutely is the only source that can answer "is it about to
            # rain", and an active severe-weather alert must never be silently
            # dropped from a weather answer.
            onecall_url = (
                f"https://api.openweathermap.org/data/3.0/onecall?"
                f"lat={latitude}&lon={longitude}&appid={api_key}&units=imperial"
            )

            async with session.get(onecall_url) as response:
                if response.status != 200:
                    _LOGGER.error("One Call API error: %s", response.status)
                    return {"error": f"Weather API error: {response.status}"}

                data = await response.json()

            tz_offset = data.get("timezone_offset", 0)
            now = datetime.now(timezone.utc)

            hourly = data.get("hourly", [])
            daily = data.get("daily", [])
            minutely = data.get("minutely", [])
            alerts = data.get("alerts", [])

            # ---------- current conditions ----------
            current = data.get("current", {})
            result["current"] = {
                "temperature": round(current.get("temp", 0)),
                "feels_like": round(current.get("feels_like", 0)),
                "humidity": current.get("humidity", 0),
                "conditions": current.get("weather", [{}])[0].get("description", "Unknown").title(),
                "wind_speed": round(current.get("wind_speed", 0)),
                "location": location_name or "Current Location",
                "local_time": _fmt_time(_local_dt(int(now.timestamp()), tz_offset)),
            }

            if current.get("wind_gust"):
                result["current"]["wind_gust"] = round(current["wind_gust"])
            if current.get("uvi") is not None:
                result["current"]["uv_index"] = round(current["uvi"], 1)

            # Explicit boolean beats making the model parse "light rain" out of
            # a description string.
            result["current"]["raining_now"] = "rain" in current or "snow" in current
            if "rain" in current:
                result["current"]["rain_1h"] = current["rain"].get("1h", 0)
            if "snow" in current:
                result["current"]["snow_1h"] = current["snow"].get("1h", 0)

            # ---------- today (always present) ----------
            # This block is the fix for "will it rain today". Previously the
            # only rain figures returned were the next-hour pop and an 8-hour
            # average, so the model had to extrapolate a whole-day answer from
            # a partial window.
            if daily:
                today = daily[0]
                today_block = _day_block(today, tz_offset)
                today_block["will_it_rain"] = _rain_today_answer(
                    today, hourly, current, tz_offset
                )
                if today.get("uvi") is not None:
                    today_block["max_uv_index"] = round(today["uvi"], 1)
                if today.get("rain"):
                    today_block["rain_total_mm"] = today["rain"]

                result["today"] = today_block
                _LOGGER.info(
                    "Today: %s/%s, rain %s%% (%s)",
                    today_block["high"], today_block["low"],
                    today_block["rain_chance"], today_block["will_it_rain"],
                )

            # ---------- 60-minute nowcast ----------
            nowcast = _minutely_nowcast(minutely, tz_offset)
            if nowcast:
                result["next_hour"] = nowcast
            elif hourly:
                # Nowcast is not available everywhere (mainly US/EU coverage).
                # Fall back to the current hour's probability so "is it about
                # to rain" still gets a grounded answer.
                result["next_hour"] = f"{round(hourly[0].get('pop', 0) * 100)}% chance of rain this hour"

            # ---------- active alerts ----------
            # Names always (cheap, and severe weather must surface no matter
            # what was asked); full text only when alerts were the question.
            if alerts:
                if forecast_type == "alerts":
                    result["alerts"] = _format_alerts(alerts, tz_offset)
                else:
                    # Only WARNINGS ride along on an unrelated weather question.
                    # A live check on 2026-07-29 had an active Heat Advisory,
                    # which in South Florida is close to a daily occurrence all
                    # summer — attaching it to "what's the temperature" would
                    # make the assistant nag about it several times a day.
                    # Watches and advisories still come back in full when
                    # alerts are what was actually asked for.
                    urgent = [
                        a.get("event", "Weather Alert") for a in alerts
                        if _is_urgent_alert(a.get("event", ""))
                    ]
                    if urgent:
                        result["active_alerts"] = urgent[:3]
            elif forecast_type == "alerts":
                result["alerts"] = []
                result["alerts_summary"] = "No active weather alerts for this area"

            # ---------- sunrise / sunset ----------
            if forecast_type == "sun_times":
                sun: dict[str, Any] = {}
                if "sunrise" in current:
                    sunrise_dt = _local_dt(current["sunrise"], tz_offset)
                    sun["sunrise"] = _fmt_time(sunrise_dt)
                    if current["sunrise"] > now.timestamp():
                        sun["time_until_sunrise"] = format_time_remaining(
                            current["sunrise"] - now.timestamp()
                        )
                    else:
                        sun["sunrise_passed"] = True

                if "sunset" in current:
                    sunset_dt = _local_dt(current["sunset"], tz_offset)
                    sun["sunset"] = _fmt_time(sunset_dt)
                    if current["sunset"] > now.timestamp():
                        sun["time_until_sunset"] = format_time_remaining(
                            current["sunset"] - now.timestamp()
                        )
                    else:
                        sun["sunset_passed"] = True

                if "sunrise" in current and "sunset" in current:
                    sun["daylight_hours"] = round(
                        (current["sunset"] - current["sunrise"]) / 3600, 1
                    )

                result["sun"] = sun

            # ---------- hour-by-hour ----------
            if forecast_type == "hourly":
                result["hourly"] = _hourly_timeline(hourly, tz_offset)

            # ---------- a specific named day ----------
            if requested_day and daily:
                day_data = _resolve_requested_day(requested_day, daily, tz_offset)
                if day_data:
                    result["requested_day"] = _day_block(day_data, tz_offset)
                else:
                    result["requested_day_error"] = (
                        f"No forecast available for '{requested_day}' "
                        "(forecast only covers the next 7 days)"
                    )

            # ---------- tomorrow ----------
            if forecast_type == "tomorrow" and len(daily) > 1:
                result["tomorrow"] = _day_block(daily[1], tz_offset)
                _LOGGER.info("Tomorrow's forecast: %s", result["tomorrow"])

            # ---------- 7-day ----------
            if forecast_type == "weekly":
                result["forecast"] = [_day_block(d, tz_offset) for d in daily[:7]]
                _LOGGER.info("Weather forecast: %d days", len(result["forecast"]))

            _LOGGER.info("Current weather: %s", result["current"])

        if not result:
            return {"error": "No weather data retrieved"}

        return result

    except Exception as err:
        return log_and_error("Failed to get weather", err)
