"""Places and restaurant tool handlers."""
from __future__ import annotations

import asyncio
import logging
import re
import urllib.parse
from datetime import datetime, time as dt_time
from typing import Any, TYPE_CHECKING

from ..utils.helpers import calculate_distance_miles

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)

API_TIMEOUT = 15

# Price level mapping
PRICE_LEVELS = {
    "PRICE_LEVEL_FREE": "Free",
    "PRICE_LEVEL_INEXPENSIVE": "$",
    "PRICE_LEVEL_MODERATE": "$$",
    "PRICE_LEVEL_EXPENSIVE": "$$$",
    "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$",
}

# Day of week mapping (Google uses 0=Sunday)
DAYS_OF_WEEK = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def _parse_opening_hours(opening_hours: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """Parse opening hours into a smart, useful format.

    Args:
        opening_hours: The currentOpeningHours or regularOpeningHours dict from Google
        now: Current datetime (for testing), defaults to datetime.now()

    Returns:
        Dict with parsed hours info including status, closing time, hours remaining, etc.
    """
    if now is None:
        now = datetime.now()

    result = {
        "is_open": opening_hours.get("openNow", False),
        "status": "Unknown",
        "closing_time": None,
        "opening_time": None,
        "hours_remaining": None,
        "next_change": None,
        "today_hours": None,
        "weekday_descriptions": opening_hours.get("weekdayDescriptions", []),
    }

    # Get basic open/closed status
    is_open = opening_hours.get("openNow", False)
    result["is_open"] = is_open

    # Get today's day of week (Python: 0=Monday, Google: 0=Sunday)
    # Convert Python's weekday to Google's format
    python_day = now.weekday()  # Monday=0, Sunday=6
    google_day = (python_day + 1) % 7  # Sunday=0, Monday=1, etc.

    # Parse periods to find today's hours and closing time
    periods = opening_hours.get("periods", [])
    current_time = now.time()

    today_periods = []
    next_open_period = None
    current_period = None

    for period in periods:
        open_info = period.get("open", {})
        close_info = period.get("close", {})

        open_day = open_info.get("day")
        close_day = close_info.get("day") if close_info else None

        # Check if this period applies to today
        if open_day == google_day:
            open_hour = open_info.get("hour", 0)
            open_minute = open_info.get("minute", 0)
            open_time = dt_time(open_hour, open_minute)

            close_time = None
            if close_info:
                close_hour = close_info.get("hour", 0)
                close_minute = close_info.get("minute", 0)
                # Handle midnight (0:00) as end of day
                if close_hour == 0 and close_minute == 0 and close_day != open_day:
                    close_time = dt_time(23, 59)
                else:
                    close_time = dt_time(close_hour, close_minute)

            today_periods.append({
                "open": open_time,
                "close": close_time,
                "open_str": open_time.strftime("%-I:%M %p").replace(":00", ""),
                "close_str": close_time.strftime("%-I:%M %p").replace(":00", "") if close_time else "Unknown",
            })

            # Check if we're currently in this period
            if close_time:
                if open_time <= current_time <= close_time:
                    current_period = today_periods[-1]
                elif current_time < open_time and (next_open_period is None or open_time < next_open_period["open"]):
                    next_open_period = today_periods[-1]

    # Build today's hours string
    if today_periods:
        hours_strs = [f"{p['open_str']} - {p['close_str']}" for p in today_periods]
        result["today_hours"] = ", ".join(hours_strs)

    # Calculate status and time remaining
    if is_open and current_period and current_period.get("close"):
        close_time = current_period["close"]
        result["closing_time"] = current_period["close_str"]

        # Calculate hours remaining
        close_dt = now.replace(hour=close_time.hour, minute=close_time.minute, second=0)
        remaining = close_dt - now
        remaining_minutes = remaining.total_seconds() / 60

        if remaining_minutes > 0:
            hours = int(remaining_minutes // 60)
            minutes = int(remaining_minutes % 60)

            if hours > 0:
                result["hours_remaining"] = f"{hours}h {minutes}m"
                result["status"] = f"Open · Closes at {current_period['close_str']} ({hours}h {minutes}m remaining)"
            else:
                result["hours_remaining"] = f"{minutes}m"
                result["status"] = f"Open · Closes at {current_period['close_str']} ({minutes}m remaining)"
        else:
            result["status"] = f"Open · Closes at {current_period['close_str']}"
    elif is_open:
        result["status"] = "Open now"
    elif next_open_period:
        result["opening_time"] = next_open_period["open_str"]
        result["status"] = f"Closed · Opens at {next_open_period['open_str']}"
    else:
        # Check tomorrow's opening
        tomorrow_google_day = (google_day + 1) % 7
        for period in periods:
            open_info = period.get("open", {})
            if open_info.get("day") == tomorrow_google_day:
                open_hour = open_info.get("hour", 0)
                open_minute = open_info.get("minute", 0)
                open_time = dt_time(open_hour, open_minute)
                result["opening_time"] = open_time.strftime("%-I:%M %p").replace(":00", "")
                result["status"] = f"Closed · Opens tomorrow at {result['opening_time']}"
                break
        else:
            result["status"] = "Closed"

    return result


async def find_nearby_places(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    api_key: str,
    latitude: float,
    longitude: float,
    track_api_call: callable,
) -> dict[str, Any]:
    """Find nearby places using Google Places API.

    Args:
        arguments: Tool arguments (query, max_results)
        session: aiohttp session
        api_key: Google Places API key
        latitude: Search center latitude
        longitude: Search center longitude
        track_api_call: Callback to track API usage

    Returns:
        Places data dict
    """
    query = arguments.get("query", "")
    max_results = min(arguments.get("max_results", 5), 20)

    if not api_key:
        return {"error": "Google Places API key not configured. Add it in Settings → PolyVoice → API Keys."}

    if not query:
        return {"error": "No search query provided"}

    try:
        url = "https://places.googleapis.com/v1/places:searchText"

        # Request comprehensive field mask for rich place data
        field_mask = ",".join([
            "places.displayName",
            "places.formattedAddress",
            "places.shortFormattedAddress",
            "places.location",
            "places.rating",
            "places.userRatingCount",
            "places.priceLevel",
            "places.currentOpeningHours",
            "places.regularOpeningHours",
            "places.businessStatus",
            "places.internationalPhoneNumber",
            "places.websiteUri",
            "places.googleMapsUri",
        ])

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": field_mask,
        }

        body = {
            "textQuery": query,
            "locationBias": {
                "circle": {
                    "center": {"latitude": latitude, "longitude": longitude},
                    "radius": 10000.0
                }
            },
            "maxResultCount": max_results,
            "rankPreference": "DISTANCE"
        }

        track_api_call("places")

        async with asyncio.timeout(API_TIMEOUT):
            async with session.post(url, json=body, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    _LOGGER.error("Google Places HTTP error: %s - %s", response.status, error_text)
                    return {"error": f"Google Places API error: {response.status}"}

                data = await response.json()

        places = data.get("places", [])

        if not places:
            return {"results": f"No results found for '{query}' near you."}

        results = []
        detailed_places = []

        for idx, place in enumerate(places[:max_results], 1):
            place_name = place.get("displayName", {}).get("text", "Unknown")
            address = place.get("formattedAddress", "Address not available")
            short_address = place.get("shortFormattedAddress", address)
            rating = place.get("rating")
            rating_count = place.get("userRatingCount", 0)
            price_level = PRICE_LEVELS.get(place.get("priceLevel"), "")
            business_status = place.get("businessStatus", "")
            phone = place.get("internationalPhoneNumber", "")
            website = place.get("websiteUri", "")
            maps_url = place.get("googleMapsUri", "")

            # Parse opening hours smartly
            hours_info = {"status": "Hours unknown"}
            if "currentOpeningHours" in place:
                hours_info = _parse_opening_hours(place["currentOpeningHours"])
            elif "regularOpeningHours" in place:
                hours_info = _parse_opening_hours(place["regularOpeningHours"])

            # Calculate distance
            place_lat = place.get("location", {}).get("latitude")
            place_lng = place.get("location", {}).get("longitude")
            distance_miles = None
            distance_str = ""

            if place_lat and place_lng:
                distance_miles = calculate_distance_miles(latitude, longitude, place_lat, place_lng)
                distance_str = f"{distance_miles:.1f} mi"

            # Build rating string
            rating_str = ""
            if rating:
                rating_str = f"★ {rating}"
                if rating_count:
                    rating_str += f" ({rating_count:,} reviews)"

            # Build concise result text for voice/text response
            parts = [f"{idx}. {place_name}"]

            if distance_str:
                parts.append(f"- {distance_str}")

            if short_address and short_address != place_name:
                parts.append(f"at {short_address}")

            status_parts = []
            if hours_info.get("status") and hours_info["status"] != "Unknown":
                status_parts.append(hours_info["status"])
            if rating_str:
                status_parts.append(rating_str)
            if price_level:
                status_parts.append(price_level)

            if status_parts:
                parts.append(f"({', '.join(status_parts)})")

            result_text = " ".join(parts)
            results.append(result_text)

            # Build detailed place object for structured data
            place_detail = {
                "name": place_name,
                "address": address,
                "short_address": short_address,
                "distance_miles": distance_miles,
                "rating": rating,
                "rating_count": rating_count,
                "price_level": price_level,
                "is_open": hours_info.get("is_open"),
                "hours_status": hours_info.get("status"),
                "closing_time": hours_info.get("closing_time"),
                "opening_time": hours_info.get("opening_time"),
                "hours_remaining": hours_info.get("hours_remaining"),
                "today_hours": hours_info.get("today_hours"),
                "phone": phone,
                "website": website,
                "directions_url": maps_url,
                "coordinates": {"lat": place_lat, "lng": place_lng} if place_lat else None,
            }

            # Add weekly hours if available
            if hours_info.get("weekday_descriptions"):
                place_detail["weekly_hours"] = hours_info["weekday_descriptions"]

            detailed_places.append(place_detail)

        response_text = f"Found {len(results)} places for '{query}' near you:\n\n" + "\n".join(results)

        # Add helpful tips based on results
        open_places = [p for p in detailed_places if p.get("is_open")]
        if detailed_places and not open_places:
            response_text += "\n\nNote: All results appear to be closed right now."
        elif open_places and len(open_places) < len(detailed_places):
            response_text += f"\n\n{len(open_places)} of {len(detailed_places)} places are currently open."

        return {
            "results": response_text,
            "places": detailed_places,
            "query": query,
            "total_found": len(detailed_places),
        }

    except Exception as err:
        _LOGGER.error("Error calling Google Places API: %s", err, exc_info=True)
        return {"error": f"Failed to search for places: {str(err)}"}


async def get_restaurant_recommendations(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    api_key: str,
    latitude: float,
    longitude: float,
    track_api_call: callable,
) -> dict[str, Any]:
    """Get restaurant recommendations from Yelp API.

    Args:
        arguments: Tool arguments (query, sort_by, price, max_results)
        session: aiohttp session
        api_key: Yelp API key
        latitude: Search center latitude
        longitude: Search center longitude
        track_api_call: Callback to track API usage

    Returns:
        Restaurant data dict
    """
    query = arguments.get("query", "")
    sort_by = arguments.get("sort_by", "rating")  # rating, review_count, distance, best_match
    price = arguments.get("price", "")  # 1,2,3,4 or combinations like "1,2"
    max_results = min(arguments.get("max_results", 3), 10)

    if not query:
        return {"error": "No restaurant/food type specified"}

    if not api_key:
        return {"error": "Yelp API key not configured. Add it in Settings → PolyVoice → API Keys."}

    # Validate sort_by
    valid_sorts = ["rating", "review_count", "distance", "best_match"]
    if sort_by not in valid_sorts:
        sort_by = "rating"

    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.yelp.com/v3/businesses/search?term={encoded_query}&latitude={latitude}&longitude={longitude}&limit={max_results}&sort_by={sort_by}"

        # Add price filter if specified
        if price:
            # Clean and validate price (should be like "1", "2", "1,2", "3,4", etc.)
            price_clean = ",".join([p.strip() for p in price.split(",") if p.strip() in ["1", "2", "3", "4"]])
            if price_clean:
                url += f"&price={price_clean}"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }

        _LOGGER.info("Searching Yelp for: %s (sort=%s, price=%s)", query, sort_by, price or "any")

        track_api_call("restaurants")

        async with asyncio.timeout(API_TIMEOUT):
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    businesses = data.get("businesses", [])

                    if not businesses:
                        return {"message": f"No restaurants found for '{query}'"}

                    results = []
                    for biz in businesses:
                        result = {
                            "name": biz.get("name"),
                            "rating": biz.get("rating"),
                            "review_count": biz.get("review_count"),
                            "price": biz.get("price", "N/A"),
                            "address": ", ".join(biz.get("location", {}).get("display_address", [])),
                            "yelp_url": biz.get("url", ""),
                            "phone": biz.get("display_phone", ""),
                            "image_url": biz.get("image_url", ""),
                        }

                        # Get coordinates for maps links
                        coordinates = biz.get("coordinates", {})
                        if coordinates:
                            result["coordinates"] = {
                                "lat": coordinates.get("latitude"),
                                "lng": coordinates.get("longitude"),
                            }

                        categories = [cat.get("title") for cat in biz.get("categories", [])]
                        if categories:
                            result["cuisine"] = ", ".join(categories[:2])

                        distance_meters = biz.get("distance", 0)
                        distance_miles = distance_meters / 1609.34
                        result["distance"] = f"{distance_miles:.1f} miles"

                        if not biz.get("is_closed", True):
                            result["status"] = "Open now"

                        # Capture reservation availability from transactions
                        transactions = biz.get("transactions", [])
                        result["supports_reservation"] = "restaurant_reservation" in transactions
                        result["supports_delivery"] = "delivery" in transactions
                        result["supports_pickup"] = "pickup" in transactions

                        # Build Yelp reservation URL if supported
                        if result["supports_reservation"]:
                            # Yelp reservation URL pattern
                            biz_alias = biz.get("alias", "")
                            if biz_alias:
                                result["yelp_reservation_url"] = f"https://www.yelp.com/reservations/{biz_alias}"

                        results.append(result)

                    _LOGGER.info("Yelp found %d restaurants", len(results))
                    return {
                        "query": query,
                        "count": len(results),
                        "restaurants": results
                    }
                else:
                    _LOGGER.error("Yelp API error: %s", response.status)
                    return {"error": f"Yelp API returned status {response.status}"}

    except Exception as err:
        _LOGGER.error("Error searching Yelp: %s", err, exc_info=True)
        return {"error": f"Failed to search restaurants: {str(err)}"}


async def book_restaurant(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    api_key: str,
    latitude: float,
    longitude: float,
    track_api_call: callable,
) -> dict[str, Any]:
    """Get reservation link for a restaurant.

    Searches Yelp for the restaurant and returns reservation URL if available,
    otherwise falls back to Google search.

    Args:
        arguments: Tool arguments (restaurant_name, party_size, date, time)
        session: aiohttp session
        api_key: Yelp API key
        latitude: Search center latitude
        longitude: Search center longitude
        track_api_call: Callback to track API usage

    Returns:
        Reservation data dict with URLs
    """
    restaurant_name = arguments.get("restaurant_name", "")
    location = arguments.get("location", "")  # Optional city/area override
    party_size = arguments.get("party_size", 2)
    date = arguments.get("date", "")  # YYYY-MM-DD format
    time = arguments.get("time", "")  # HH:MM format (24hr) or natural like "7pm"

    if not restaurant_name:
        return {"error": "No restaurant name provided"}

    if not api_key:
        return {"error": "Yelp API key not configured. Add it in Settings → PolyVoice → API Keys."}

    # Extract location from restaurant_name if it contains "in [city]"
    # e.g., "Sunny's Steak House in Miami" -> name="Sunny's Steak House", location="Miami"
    location_match = re.search(r'\s+in\s+([A-Za-z\s]+)$', restaurant_name, re.IGNORECASE)
    if location_match and not location:
        location = location_match.group(1).strip()
        restaurant_name = restaurant_name[:location_match.start()].strip()
        _LOGGER.info("Extracted location '%s' from restaurant name, searching for '%s'", location, restaurant_name)

    try:
        # Include location in search term for better Yelp results
        search_term = f"{restaurant_name} {location}" if location else restaurant_name
        encoded_query = urllib.parse.quote(search_term)

        # Use location text if provided, otherwise fall back to lat/long
        if location:
            encoded_location = urllib.parse.quote(location)
            url = f"https://api.yelp.com/v3/businesses/search?term={encoded_query}&location={encoded_location}&limit=10&categories=restaurants,food"
            _LOGGER.info("Searching Yelp in location '%s' for: %s", location, restaurant_name)
        else:
            url = f"https://api.yelp.com/v3/businesses/search?term={encoded_query}&latitude={latitude}&longitude={longitude}&limit=10&categories=restaurants,food"
            _LOGGER.info("Searching Yelp near coordinates for: %s", restaurant_name)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }

        _LOGGER.info("Searching Yelp for reservation: %s", restaurant_name)
        track_api_call("book_restaurant")

        async with asyncio.timeout(API_TIMEOUT):
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    _LOGGER.error("Yelp API error: %s", response.status)
                    return _build_fallback_response(restaurant_name, party_size, date, time, latitude, longitude)

                data = await response.json()
                businesses = data.get("businesses", [])

                if not businesses:
                    return _build_fallback_response(restaurant_name, party_size, date, time, latitude, longitude)

                # Find best match by name similarity - strict matching for specific restaurant bookings
                search_name = restaurant_name.lower().strip()
                search_words = search_name.replace("'s", "s").replace("'", "").split()
                first_word = search_words[0] if search_words else ""
                best_match = None
                best_score = 0

                for b in businesses:
                    biz_name_lower = b.get("name", "").lower()
                    biz_name_normalized = biz_name_lower.replace("'s", "s").replace("'", "")
                    biz_words = biz_name_normalized.split()
                    biz_first_word = biz_words[0] if biz_words else ""

                    # First word must match exactly (sunny != sonny)
                    if first_word and biz_first_word and first_word != biz_first_word:
                        # Skip if first words don't match
                        continue

                    # Score based on how much of the search term appears in the business name
                    if search_name in biz_name_lower or search_name.replace("'s", "s") in biz_name_normalized:
                        score = 100  # Exact substring match
                    elif biz_name_lower in search_name or biz_name_normalized in search_name.replace("'s", "s"):
                        score = 90  # Business name is part of search
                    else:
                        # Count matching words
                        matching = len(set(search_words) & set(biz_words))
                        score = matching * 20

                    if score > best_score:
                        best_score = score
                        best_match = b

                # Fall back to first result only if we found no match at all
                biz = best_match if best_match else businesses[0]
                _LOGGER.info("Best match for '%s': %s (score: %d)", restaurant_name, biz.get("name"), best_score)
                biz_name = biz.get("name", restaurant_name)
                biz_alias = biz.get("alias", "")
                transactions = biz.get("transactions", [])
                yelp_url = biz.get("url", "")
                phone = biz.get("display_phone", "")
                address = ", ".join(biz.get("location", {}).get("display_address", []))

                result = {
                    "restaurant_name": biz_name,
                    "address": address,
                    "phone": phone,
                    "party_size": party_size,
                    "date": date,
                    "time": time,
                    "yelp_url": yelp_url,
                }

                # Check if Yelp reservations are supported
                if "restaurant_reservation" in transactions and biz_alias:
                    # Build Yelp reservation URL with parameters
                    yelp_resy_url = f"https://www.yelp.com/reservations/{biz_alias}"

                    # Add query params if we have date/time/covers
                    params = []
                    if party_size:
                        params.append(f"covers={party_size}")
                    if date:
                        params.append(f"date={date}")
                    if time:
                        # Normalize time to HH:MM format
                        params.append(f"time={_normalize_time(time)}")

                    if params:
                        yelp_resy_url += "?" + "&".join(params)

                    result["reservation_url"] = yelp_resy_url
                    result["reservation_source"] = "Yelp"
                    result["supports_reservation"] = True
                    result["message"] = f"Reservation available at {biz_name} via Yelp!"
                    result["response_text"] = f"I found {biz_name} and sent a reservation link to your phone."

                    _LOGGER.info("Found Yelp reservation for %s: %s", biz_name, yelp_resy_url)
                else:
                    # No Yelp reservation - fall back to Google
                    result["supports_reservation"] = False
                    fallback = _build_fallback_response(biz_name, party_size, date, time, latitude, longitude)
                    result["reservation_url"] = fallback["reservation_url"]
                    result["reservation_source"] = "Google Search"
                    if phone:
                        result["message"] = f"{biz_name} doesn't support online reservations through Yelp. Try searching Google or call them directly at {phone}."
                        result["response_text"] = f"{biz_name} doesn't support online reservations. I sent a link to your phone, or you can call them at {phone}."
                    else:
                        result["message"] = f"{biz_name} doesn't support online reservations through Yelp. Use the Google link to search for booking options."
                        result["response_text"] = f"{biz_name} doesn't support online reservations. I sent a search link to your phone."

                    _LOGGER.info("No Yelp reservation for %s, using Google fallback", biz_name)

                return result

    except Exception as err:
        _LOGGER.error("Error booking restaurant: %s", err, exc_info=True)
        return _build_fallback_response(restaurant_name, party_size, date, time, latitude, longitude)


def _normalize_time(time_str: str) -> str:
    """Normalize time string to HH:MM format.

    Handles inputs like "7pm", "7:30 PM", "19:00", etc.
    """
    if not time_str:
        return ""

    time_str = time_str.strip().lower().replace(" ", "")

    # Already in HH:MM format
    if ":" in time_str and len(time_str) <= 5 and time_str.replace(":", "").isdigit():
        return time_str

    # Handle 12-hour format with am/pm
    is_pm = "pm" in time_str
    is_am = "am" in time_str
    time_str = time_str.replace("pm", "").replace("am", "")

    try:
        if ":" in time_str:
            parts = time_str.split(":")
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
        else:
            hour = int(time_str)
            minute = 0

        # Convert to 24-hour format
        if is_pm and hour < 12:
            hour += 12
        elif is_am and hour == 12:
            hour = 0

        return f"{hour:02d}:{minute:02d}"
    except ValueError:
        return time_str  # Return as-is if we can't parse


def _build_fallback_response(
    restaurant_name: str,
    party_size: int,
    date: str,
    time: str,
    latitude: float,
    longitude: float,
) -> dict[str, Any]:
    """Build a fallback response with Google search URL."""
    # Build Google search query
    search_query = f"{restaurant_name} reservations"
    encoded_query = urllib.parse.quote(search_query)
    google_url = f"https://www.google.com/search?q={encoded_query}"

    return {
        "restaurant_name": restaurant_name,
        "party_size": party_size,
        "date": date,
        "time": time,
        "reservation_url": google_url,
        "reservation_source": "Google Search",
        "supports_reservation": False,
        "message": f"No direct reservation link found for {restaurant_name}. Use the Google link to search for booking options, or call the restaurant directly.",
        "response_text": f"I couldn't find a direct reservation link for {restaurant_name}, but I sent a search link to your phone."
    }
