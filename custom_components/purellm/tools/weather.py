"""Weather tool handler."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, TYPE_CHECKING

from ..const import API_TIMEOUT

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
) -> dict[str, Any]:
    """Get weather forecast from OpenWeatherMap.

    Args:
        arguments: Tool arguments (location, forecast_type)
        session: aiohttp session
        api_key: OpenWeatherMap API key
        latitude: Default latitude
        longitude: Default longitude
        track_api_call: Callback to track API usage

    Returns:
        Weather data dict
    """
    forecast_type = arguments.get("forecast_type", "current")
    location_query = arguments.get("location", "").strip()

    if not api_key:
        return {"error": "OpenWeatherMap API key not configured. Add it in Settings → PolyVoice → API Keys."}

    location_name = None

    # If user specified a location, geocode it
    if location_query:
        # Normalize: convert full state names to abbreviations, add US suffix
        location_query = _normalize_location(location_query)
        try:
            geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location_query}&limit=1&appid={api_key}"
            async with session.get(geo_url) as geo_response:
                if geo_response.status == 200:
                    geo_data = await geo_response.json()
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
                    return {"error": f"Geocoding failed for: {location_query}"}
        except Exception as geo_err:
            _LOGGER.error("Geocoding error: %s", geo_err)
            return {"error": f"Could not geocode location: {location_query}"}

    try:
        result = {}
        track_api_call("weather")

        async with asyncio.timeout(API_TIMEOUT):
            # PARALLEL fetch: current weather AND forecast simultaneously
            current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=imperial"
            forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={latitude}&lon={longitude}&appid={api_key}&units=imperial"

            # Fire both requests at once
            current_task = session.get(current_url)
            forecast_task = session.get(forecast_url)

            async with current_task as current_response, forecast_task as forecast_response:
                # Process current weather
                if current_response.status == 200:
                    data = await current_response.json()

                    result["current"] = {
                        "temperature": round(data["main"]["temp"]),
                        "feels_like": round(data["main"]["feels_like"]),
                        "humidity": data["main"]["humidity"],
                        "conditions": data["weather"][0]["description"].title(),
                        "wind_speed": round(data["wind"]["speed"]),
                        "location": location_name or data["name"]
                    }

                    # Add rain if present
                    if "rain" in data:
                        result["current"]["rain_1h"] = data["rain"].get("1h", 0)

                    # Add sunrise/sunset times
                    if "sys" in data:
                        now = datetime.now()

                        # Extract and format sunrise
                        if "sunrise" in data["sys"]:
                            sunrise_ts = data["sys"]["sunrise"]
                            sunrise_dt = datetime.fromtimestamp(sunrise_ts)
                            result["current"]["sunrise"] = sunrise_dt.strftime("%-I:%M %p")

                            # Calculate time until sunrise (if it hasn't happened yet today)
                            if sunrise_dt > now:
                                time_until = sunrise_dt - now
                                hours, remainder = divmod(int(time_until.total_seconds()), 3600)
                                minutes = remainder // 60
                                if hours > 0:
                                    result["current"]["time_until_sunrise"] = f"{hours}h {minutes}m"
                                else:
                                    result["current"]["time_until_sunrise"] = f"{minutes}m"
                            else:
                                # Sunrise already passed today
                                result["current"]["sunrise_passed"] = True

                        # Extract and format sunset
                        if "sunset" in data["sys"]:
                            sunset_ts = data["sys"]["sunset"]
                            sunset_dt = datetime.fromtimestamp(sunset_ts)
                            result["current"]["sunset"] = sunset_dt.strftime("%-I:%M %p")

                            # Calculate time until sunset (if it hasn't happened yet today)
                            if sunset_dt > now:
                                time_until = sunset_dt - now
                                hours, remainder = divmod(int(time_until.total_seconds()), 3600)
                                minutes = remainder // 60
                                if hours > 0:
                                    result["current"]["time_until_sunset"] = f"{hours}h {minutes}m"
                                else:
                                    result["current"]["time_until_sunset"] = f"{minutes}m"
                            else:
                                # Sunset already passed today
                                result["current"]["sunset_passed"] = True

                        # Calculate daylight hours
                        if "sunrise" in data["sys"] and "sunset" in data["sys"]:
                            daylight_seconds = data["sys"]["sunset"] - data["sys"]["sunrise"]
                            daylight_hours = daylight_seconds / 3600
                            result["current"]["daylight_hours"] = round(daylight_hours, 1)

                    _LOGGER.info("Current weather: %s", result["current"])
                else:
                    _LOGGER.error("Weather API error: %s", current_response.status)
                    return {"error": "Could not get current weather"}

                # Process forecast data
                if forecast_response.status == 200:
                    data = await forecast_response.json()

                    # Get NEXT HOUR rain chance from first forecast entry
                    next_hour_rain = 0
                    if data["list"] and len(data["list"]) > 0:
                        next_hour_rain = round(data["list"][0].get("pop", 0) * 100)
                    result["current"]["rain_chance_next_hour"] = next_hour_rain

                    # Calculate AVERAGE rain chance for next 8 hours
                    rain_chances_8hr = []
                    for i, item in enumerate(data["list"][:3]):
                        rain_chances_8hr.append(item.get("pop", 0) * 100)
                    if rain_chances_8hr:
                        result["current"]["avg_rain_chance_8hr"] = round(sum(rain_chances_8hr) / len(rain_chances_8hr))
                    else:
                        result["current"]["avg_rain_chance_8hr"] = 0

                    # Get TODAY's date for extracting today's high/low
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    today_temps = []

                    # Group by day for weekly forecast
                    daily_forecasts = {}
                    for item in data["list"]:
                        dt = datetime.strptime(item["dt_txt"], "%Y-%m-%d %H:%M:%S")
                        day_key = dt.strftime("%A")
                        item_date = dt.strftime("%Y-%m-%d")

                        # Collect today's temps
                        if item_date == today_str:
                            today_temps.append(item["main"]["temp_max"])
                            today_temps.append(item["main"]["temp_min"])

                        if day_key not in daily_forecasts:
                            daily_forecasts[day_key] = {
                                "date": dt.strftime("%B %d"),
                                "high": item["main"]["temp_max"],
                                "low": item["main"]["temp_min"],
                                "conditions": item["weather"][0]["description"].title(),
                                "rain_chance": item.get("pop", 0) * 100
                            }
                        else:
                            daily_forecasts[day_key]["high"] = max(
                                daily_forecasts[day_key]["high"],
                                item["main"]["temp_max"]
                            )
                            daily_forecasts[day_key]["low"] = min(
                                daily_forecasts[day_key]["low"],
                                item["main"]["temp_min"]
                            )
                            # Take noon conditions if available
                            if dt.hour == 12:
                                daily_forecasts[day_key]["conditions"] = item["weather"][0]["description"].title()
                                daily_forecasts[day_key]["rain_chance"] = item.get("pop", 0) * 100

                    # ADD TODAY'S HIGH/LOW TO CURRENT
                    current_day = datetime.now().strftime("%A")
                    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%A")

                    if current_day in daily_forecasts:
                        result["current"]["todays_high"] = round(daily_forecasts[current_day]["high"])
                        if tomorrow in daily_forecasts:
                            result["current"]["todays_low"] = round(daily_forecasts[tomorrow]["low"])
                        else:
                            result["current"]["todays_low"] = round(daily_forecasts[current_day]["low"])
                    elif today_temps:
                        result["current"]["todays_high"] = round(max(today_temps))
                        result["current"]["todays_low"] = round(min(today_temps))
                    else:
                        first_day = list(daily_forecasts.values())[0] if daily_forecasts else None
                        if first_day:
                            result["current"]["todays_high"] = round(first_day["high"])
                            result["current"]["todays_low"] = round(first_day["low"])

                    # Format weekly forecast (only if requested)
                    if forecast_type in ["weekly", "both"]:
                        forecast_list = []
                        for day, forecast in list(daily_forecasts.items())[:5]:
                            forecast_list.append({
                                "day": day,
                                "date": forecast["date"],
                                "high": round(forecast["high"]),
                                "low": round(forecast["low"]),
                                "conditions": forecast["conditions"],
                                "rain_chance": round(forecast["rain_chance"])
                            })

                        result["forecast"] = forecast_list
                        _LOGGER.info("Weather forecast: %d days", len(forecast_list))

        if not result:
            return {"error": "No weather data retrieved"}

        return result

    except Exception as err:
        _LOGGER.error("Error getting weather: %s", err, exc_info=True)
        return {"error": f"Failed to get weather: {str(err)}"}
