"""Weather tool handler."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
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


async def get_weather_forecast(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    api_key: str,
    latitude: float,
    longitude: float,
    track_api_call: callable,
    user_query: str = "",
) -> dict[str, Any]:
    """Get weather forecast from OpenWeatherMap.

    Args:
        arguments: Tool arguments (location, forecast_type)
        session: aiohttp session
        api_key: OpenWeatherMap API key
        latitude: Default latitude
        longitude: Default longitude
        track_api_call: Callback to track API usage
        user_query: Original user query for validation

    Returns:
        Weather data dict
    """
    forecast_type = arguments.get("forecast_type", "current")
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
        result = {}
        track_api_call("weather")

        async with asyncio.timeout(API_TIMEOUT):
            # Use One Call API 3.0 for accurate daily min/max temps
            onecall_url = f"https://api.openweathermap.org/data/3.0/onecall?lat={latitude}&lon={longitude}&appid={api_key}&units=imperial&exclude=minutely,alerts"

            async with session.get(onecall_url) as response:
                if response.status != 200:
                    _LOGGER.error("One Call API error: %s", response.status)
                    return {"error": f"Weather API error: {response.status}"}

                data = await response.json()
                now = datetime.now()

                # Process current weather from One Call API
                current = data.get("current", {})
                result["current"] = {
                    "temperature": round(current.get("temp", 0)),
                    "feels_like": round(current.get("feels_like", 0)),
                    "humidity": current.get("humidity", 0),
                    "conditions": current.get("weather", [{}])[0].get("description", "Unknown").title(),
                    "wind_speed": round(current.get("wind_speed", 0)),
                    "location": location_name or "Current Location"
                }

                # Add rain if present
                if "rain" in current:
                    result["current"]["rain_1h"] = current["rain"].get("1h", 0)

                # Add sunrise/sunset times only when explicitly requested
                if forecast_type == "sun_times":
                    if "sunrise" in current:
                        sunrise_dt = datetime.fromtimestamp(current["sunrise"])
                        result["current"]["sunrise"] = sunrise_dt.strftime("%-I:%M %p")

                        if sunrise_dt > now:
                            result["current"]["time_until_sunrise"] = format_time_remaining(
                                (sunrise_dt - now).total_seconds()
                            )
                        else:
                            result["current"]["sunrise_passed"] = True

                    if "sunset" in current:
                        sunset_dt = datetime.fromtimestamp(current["sunset"])
                        result["current"]["sunset"] = sunset_dt.strftime("%-I:%M %p")

                        if sunset_dt > now:
                            result["current"]["time_until_sunset"] = format_time_remaining(
                                (sunset_dt - now).total_seconds()
                            )
                        else:
                            result["current"]["sunset_passed"] = True

                    # Calculate daylight hours
                    if "sunrise" in current and "sunset" in current:
                        daylight_seconds = current["sunset"] - current["sunrise"]
                        daylight_hours = daylight_seconds / 3600
                        result["current"]["daylight_hours"] = round(daylight_hours, 1)

                # Process hourly data for rain chances
                hourly = data.get("hourly", [])
                if hourly:
                    # Next hour rain chance
                    result["current"]["rain_chance_next_hour"] = round(hourly[0].get("pop", 0) * 100)

                    # Average rain chance for next 8 hours
                    rain_chances_8hr = [h.get("pop", 0) * 100 for h in hourly[:8]]
                    if rain_chances_8hr:
                        result["current"]["avg_rain_chance_8hr"] = round(sum(rain_chances_8hr) / len(rain_chances_8hr))
                    else:
                        result["current"]["avg_rain_chance_8hr"] = 0

                # Process daily data for proper high/low temps
                daily = data.get("daily", [])
                if daily:
                    # Today's high/low from daily[0] - this is the REAL daily min/max
                    today = daily[0]
                    result["current"]["todays_high"] = round(today.get("temp", {}).get("max", 0))
                    result["current"]["todays_low"] = round(today.get("temp", {}).get("min", 0))

                    _LOGGER.info("Today's high/low from One Call API: %s/%s",
                                result["current"]["todays_high"], result["current"]["todays_low"])

                    # Format tomorrow's forecast (when user asks about tomorrow)
                    if forecast_type == "tomorrow" and len(daily) > 1:
                        tomorrow = daily[1]
                        tomorrow_dt = datetime.fromtimestamp(tomorrow.get("dt", 0))
                        result["tomorrow"] = {
                            "day": tomorrow_dt.strftime("%A"),
                            "date": tomorrow_dt.strftime("%B %d"),
                            "high": round(tomorrow.get("temp", {}).get("max", 0)),
                            "low": round(tomorrow.get("temp", {}).get("min", 0)),
                            "conditions": tomorrow.get("weather", [{}])[0].get("description", "Unknown").title(),
                            "rain_chance": round(tomorrow.get("pop", 0) * 100)
                        }
                        _LOGGER.info("Tomorrow's forecast: %s", result["tomorrow"])

                    # Format weekly forecast (only if requested)
                    if forecast_type in ["weekly", "both"]:
                        forecast_list = []
                        for day_data in daily[:7]:  # Up to 7 days
                            dt = datetime.fromtimestamp(day_data.get("dt", 0))
                            forecast_list.append({
                                "day": dt.strftime("%A"),
                                "date": dt.strftime("%B %d"),
                                "high": round(day_data.get("temp", {}).get("max", 0)),
                                "low": round(day_data.get("temp", {}).get("min", 0)),
                                "conditions": day_data.get("weather", [{}])[0].get("description", "Unknown").title(),
                                "rain_chance": round(day_data.get("pop", 0) * 100)
                            })

                        result["forecast"] = forecast_list
                        _LOGGER.info("Weather forecast: %d days", len(forecast_list))

                _LOGGER.info("Current weather: %s", result["current"])

                # Build pre-formatted response_text for current weather
                # so the LLM doesn't inconsistently truncate the response
                if forecast_type == "current":
                    c = result["current"]
                    result["response_text"] = (
                        f"It's currently {c['temperature']}°F and feels like {c['feels_like']}°F "
                        f"with {c['conditions']} in {c['location']}. "
                        f"There is a {c.get('rain_chance_next_hour', 0)}% chance of rain in the next hour. "
                        f"Today's high is {c.get('todays_high', 'N/A')}°F with an overnight low of {c.get('todays_low', 'N/A')}°F."
                    )

        if not result:
            return {"error": "No weather data retrieved"}

        return result

    except Exception as err:
        return log_and_error("Failed to get weather", err)
