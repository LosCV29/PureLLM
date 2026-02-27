"""Places and restaurant tool handlers."""
from __future__ import annotations

import logging
import re
import urllib.parse
from datetime import datetime, time as dt_time
from typing import Any, TYPE_CHECKING

from ..utils.helpers import calculate_distance_miles, format_time_remaining
from ..utils.http_client import post_json, log_and_error

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)

# Google Places API URL
PLACES_API_URL = "https://places.googleapis.com/v1/places:searchText"

# Price level mapping
PRICE_LEVELS = {
    "PRICE_LEVEL_FREE": "Free",
    "PRICE_LEVEL_INEXPENSIVE": "$",
    "PRICE_LEVEL_MODERATE": "$$",
    "PRICE_LEVEL_EXPENSIVE": "$$$",
    "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$",
}

# Reverse mapping for price filter
PRICE_LEVEL_FILTERS = {
    "1": "PRICE_LEVEL_INEXPENSIVE",
    "2": "PRICE_LEVEL_MODERATE",
    "3": "PRICE_LEVEL_EXPENSIVE",
    "4": "PRICE_LEVEL_VERY_EXPENSIVE",
}

# Day of week mapping (Google uses 0=Sunday)
DAYS_OF_WEEK = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# Common field masks for different use cases
FIELD_MASK_FULL = [
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
]

FIELD_MASK_RESTAURANT = FIELD_MASK_FULL + ["places.primaryType", "places.types"]

FIELD_MASK_BOOKING = [
    "places.displayName",
    "places.formattedAddress",
    "places.shortFormattedAddress",
    "places.location",
    "places.internationalPhoneNumber",
    "places.websiteUri",
    "places.googleMapsUri",
]


def _build_headers(api_key: str, field_mask: list[str]) -> dict[str, str]:
    """Build common headers for Google Places API requests."""
    return {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": ",".join(field_mask),
    }


def _build_location_bias(latitude: float, longitude: float, radius: float = 10000.0) -> dict:
    """Build location bias structure for Google Places API."""
    return {
        "circle": {
            "center": {"latitude": latitude, "longitude": longitude},
            "radius": radius
        }
    }


async def _search_places(
    session: "aiohttp.ClientSession",
    api_key: str,
    query: str,
    latitude: float,
    longitude: float,
    field_mask: list[str],
    max_results: int = 5,
    rank_preference: str = "DISTANCE",
    included_type: str | None = None,
    price_levels: list[str] | None = None,
    radius: float = 10000.0,
) -> tuple[list[dict] | None, int | str]:
    """Execute a Google Places API search.

    Returns:
        Tuple of (places list, status). places is None on error.
    """
    headers = _build_headers(api_key, field_mask)

    body = {
        "textQuery": query,
        "locationBias": _build_location_bias(latitude, longitude, radius),
        "maxResultCount": max_results,
        "rankPreference": rank_preference,
    }

    if included_type:
        body["includedType"] = included_type

    if price_levels:
        body["priceLevels"] = price_levels

    data, status = await post_json(session, PLACES_API_URL, body, headers=headers)

    if data is None:
        return None, status

    return data.get("places", []), status


def _extract_place_basics(place: dict) -> dict[str, Any]:
    """Extract common fields from a Google Places result."""
    return {
        "name": place.get("displayName", {}).get("text", "Unknown"),
        "address": place.get("formattedAddress", "Address not available"),
        "short_address": place.get("shortFormattedAddress", place.get("formattedAddress", "Address not available")),
        "rating": place.get("rating"),
        "rating_count": place.get("userRatingCount", 0),
        "price_level": PRICE_LEVELS.get(place.get("priceLevel"), ""),
        "phone": place.get("internationalPhoneNumber", ""),
        "website": place.get("websiteUri", ""),
        "maps_url": place.get("googleMapsUri", ""),
        "latitude": place.get("location", {}).get("latitude"),
        "longitude": place.get("location", {}).get("longitude"),
    }


def _get_hours_info(place: dict) -> dict[str, Any]:
    """Get parsed opening hours from a place result."""
    if "currentOpeningHours" in place:
        return _parse_opening_hours(place["currentOpeningHours"])
    elif "regularOpeningHours" in place:
        return _parse_opening_hours(place["regularOpeningHours"])
    return {"status": "Hours unknown"}


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
        remaining_seconds = (close_dt - now).total_seconds()

        if remaining_seconds > 0:
            time_remaining = format_time_remaining(remaining_seconds)
            result["hours_remaining"] = time_remaining
            result["status"] = f"Open · Closes at {current_period['close_str']} ({time_remaining} remaining)"
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
        return {"error": "Google Places API key not configured. Add it in Settings → PureLLM → API Keys."}

    if not query:
        return {"error": "No search query provided"}

    try:
        track_api_call("places")

        places, status = await _search_places(
            session, api_key, query, latitude, longitude,
            field_mask=FIELD_MASK_FULL,
            max_results=max_results,
            rank_preference="DISTANCE",
        )

        if places is None:
            _LOGGER.error("Google Places HTTP error: %s", status)
            return {"error": f"Google Places API error: {status}"}

        if not places:
            return {"results": f"No results found for '{query}' near you."}

        results = []
        detailed_places = []

        for idx, place in enumerate(places[:max_results], 1):
            # Extract common fields using helper
            basics = _extract_place_basics(place)
            hours_info = _get_hours_info(place)

            # Calculate distance
            distance_miles = None
            distance_str = ""
            if basics["latitude"] and basics["longitude"]:
                distance_miles = calculate_distance_miles(
                    latitude, longitude, basics["latitude"], basics["longitude"]
                )
                distance_str = f"{distance_miles:.1f} mi"

            # Build rating string
            rating_str = ""
            if basics["rating"]:
                rating_str = f"★ {basics['rating']}"
                if basics["rating_count"]:
                    rating_str += f" ({basics['rating_count']:,} reviews)"

            # Build concise result text for voice/text response
            parts = [f"{idx}. {basics['name']}"]

            if distance_str:
                parts.append(f"- {distance_str}")

            if basics["short_address"] and basics["short_address"] != basics["name"]:
                parts.append(f"at {basics['short_address']}")

            status_parts = []
            if hours_info.get("status") and hours_info["status"] != "Unknown":
                status_parts.append(hours_info["status"])
            if rating_str:
                status_parts.append(rating_str)
            if basics["price_level"]:
                status_parts.append(basics["price_level"])

            if status_parts:
                parts.append(f"({', '.join(status_parts)})")

            result_text = " ".join(parts)
            results.append(result_text)

            # Build detailed place object for structured data
            place_detail = {
                "name": basics["name"],
                "address": basics["address"],
                "short_address": basics["short_address"],
                "distance_miles": distance_miles,
                "rating": basics["rating"],
                "rating_count": basics["rating_count"],
                "price_level": basics["price_level"],
                "is_open": hours_info.get("is_open"),
                "hours_status": hours_info.get("status"),
                "closing_time": hours_info.get("closing_time"),
                "opening_time": hours_info.get("opening_time"),
                "hours_remaining": hours_info.get("hours_remaining"),
                "today_hours": hours_info.get("today_hours"),
                "phone": basics["phone"],
                "website": basics["website"],
                "directions_url": basics["maps_url"],
                "coordinates": {"lat": basics["latitude"], "lng": basics["longitude"]} if basics["latitude"] else None,
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
        return log_and_error("Failed to search for places", err)


async def get_restaurant_recommendations(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    api_key: str,
    latitude: float,
    longitude: float,
    track_api_call: callable,
) -> dict[str, Any]:
    """Get restaurant recommendations from Google Places API.

    Args:
        arguments: Tool arguments (query, sort_by, price, max_results)
        session: aiohttp session
        api_key: Google Places API key
        latitude: Search center latitude
        longitude: Search center longitude
        track_api_call: Callback to track API usage

    Returns:
        Restaurant data dict
    """
    query = arguments.get("query", "")
    sort_by = arguments.get("sort_by", "rating")  # rating, review_count, distance
    price = arguments.get("price", "")  # 1,2,3,4 or combinations like "1,2"
    max_results = min(arguments.get("max_results", 3), 10)

    if not query:
        return {"error": "No restaurant/food type specified"}

    if not api_key:
        return {"error": "Google Places API key not configured. Add it in Settings → PureLLM → API Keys."}

    # Validate sort_by
    valid_sorts = ["rating", "review_count", "distance"]
    if sort_by not in valid_sorts:
        sort_by = "rating"

    try:
        # Build search query with restaurant context
        search_query = f"{query} restaurant"

        # Map sort_by to Google's rankPreference
        rank_preference = "DISTANCE" if sort_by == "distance" else "RELEVANCE"

        # Parse price filter using the mapping
        price_levels = None
        if price:
            price_levels = [
                PRICE_LEVEL_FILTERS[p.strip()]
                for p in price.split(",")
                if p.strip() in PRICE_LEVEL_FILTERS
            ]
            price_levels = price_levels if price_levels else None

        _LOGGER.info("Searching Google Places for restaurants: %s (sort=%s, price=%s)", query, sort_by, price or "any")
        track_api_call("restaurants")

        places, status = await _search_places(
            session, api_key, search_query, latitude, longitude,
            field_mask=FIELD_MASK_RESTAURANT,
            max_results=max_results,
            rank_preference=rank_preference,
            included_type="restaurant",
            price_levels=price_levels,
        )

        if places is None:
            _LOGGER.error("Google Places API error: %s", status)
            return {"error": f"Google Places API returned status {status}"}

        if not places:
            return {"message": f"No restaurants found for '{query}'"}

        # Sort by rating or review count if requested (Google doesn't support this natively)
        if sort_by == "rating":
            places.sort(key=lambda x: x.get("rating", 0), reverse=True)
        elif sort_by == "review_count":
            places.sort(key=lambda x: x.get("userRatingCount", 0), reverse=True)

        results = []
        for place in places[:max_results]:
            # Extract common fields using helper
            basics = _extract_place_basics(place)
            hours_info = _get_hours_info(place)

            # Calculate distance
            distance_str = ""
            if basics["latitude"] and basics["longitude"]:
                distance_miles = calculate_distance_miles(
                    latitude, longitude, basics["latitude"], basics["longitude"]
                )
                distance_str = f"{distance_miles:.1f} miles"

            # Get cuisine from types
            cuisine = ""
            place_types = place.get("types", [])
            cuisine_types = [t.replace("_", " ").title() for t in place_types
                           if t not in ["restaurant", "food", "point_of_interest", "establishment"]]
            if cuisine_types:
                cuisine = ", ".join(cuisine_types[:2])

            result = {
                "name": basics["name"],
                "rating": basics["rating"],
                "review_count": basics["rating_count"],
                "price": basics["price_level"],
                "address": basics["address"],
                "short_address": basics["short_address"],
                "phone": basics["phone"],
                "website": basics["website"],
                "directions_url": basics["maps_url"],
                "distance": distance_str,
                "status": hours_info.get("status", ""),
                "is_open": hours_info.get("is_open"),
            }

            if cuisine:
                result["cuisine"] = cuisine

            if basics["latitude"] and basics["longitude"]:
                result["coordinates"] = {"lat": basics["latitude"], "lng": basics["longitude"]}

            results.append(result)

        _LOGGER.info("Google Places found %d restaurants", len(results))
        return {
            "query": query,
            "count": len(results),
            "restaurants": results
        }

    except Exception as err:
        return log_and_error("Failed to search restaurants", err)


async def book_restaurant(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    api_key: str,
    latitude: float,
    longitude: float,
    track_api_call: callable,
) -> dict[str, Any]:
    """Get reservation link for a restaurant.

    Searches Google Places for the restaurant and returns a Google search link
    for reservation options.

    Args:
        arguments: Tool arguments (restaurant_name, party_size, date, time)
        session: aiohttp session
        api_key: Google Places API key
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
        return {"error": "Google Places API key not configured. Add it in Settings → PureLLM → API Keys."}

    # Extract location from restaurant_name if it contains "in [city]"
    # e.g., "Sunny's Steak House in Miami" -> name="Sunny's Steak House", location="Miami"
    location_match = re.search(r'\s+in\s+([A-Za-z\s]+)$', restaurant_name, re.IGNORECASE)
    if location_match and not location:
        location = location_match.group(1).strip()
        restaurant_name = restaurant_name[:location_match.start()].strip()
        _LOGGER.info("Extracted location '%s' from restaurant name, searching for '%s'", location, restaurant_name)

    try:
        # Build search query
        search_query = f"{restaurant_name} restaurant"
        if location:
            search_query += f" {location}"
            _LOGGER.info("Searching Google Places in location '%s' for: %s", location, restaurant_name)
        else:
            _LOGGER.info("Searching Google Places near coordinates for: %s", restaurant_name)

        _LOGGER.info("Searching Google Places for reservation: %s", restaurant_name)
        track_api_call("book_restaurant")

        places, status = await _search_places(
            session, api_key, search_query, latitude, longitude,
            field_mask=FIELD_MASK_BOOKING,
            max_results=5,
            rank_preference="RELEVANCE",
            included_type="restaurant",
            radius=50000.0,  # Larger radius for specific restaurant search
        )

        if places is None:
            _LOGGER.error("Google Places API error: %s", status)
            return _build_fallback_response(restaurant_name, party_size, date, time)

        if not places:
            return _build_fallback_response(restaurant_name, party_size, date, time)

        # Find best match by name similarity
        search_name = restaurant_name.lower().strip()
        search_words = search_name.replace("'s", "s").replace("'", "").split()
        first_word = search_words[0] if search_words else ""
        best_match = None
        best_score = 0

        for p in places:
            place_name = p.get("displayName", {}).get("text", "")
            place_name_lower = place_name.lower()
            place_name_normalized = place_name_lower.replace("'s", "s").replace("'", "")
            place_words = place_name_normalized.split()
            place_first_word = place_words[0] if place_words else ""

            # First word must match exactly
            if first_word and place_first_word and first_word != place_first_word:
                continue

            # Score based on how much of the search term appears in the place name
            if search_name in place_name_lower or search_name.replace("'s", "s") in place_name_normalized:
                score = 100  # Exact substring match
            elif place_name_lower in search_name or place_name_normalized in search_name.replace("'s", "s"):
                score = 90  # Place name is part of search
            else:
                # Count matching words
                matching = len(set(search_words) & set(place_words))
                score = matching * 20

            if score > best_score:
                best_score = score
                best_match = p

        # Fall back to first result only if we found no match at all
        place = best_match if best_match else places[0]
        place_name = place.get("displayName", {}).get("text", restaurant_name)
        _LOGGER.info("Best match for '%s': %s (score: %d)", restaurant_name, place_name, best_score)

        phone = place.get("internationalPhoneNumber", "")
        address = place.get("formattedAddress", "")
        maps_url = place.get("googleMapsUri", "")

        # Build Google search URL for reservations
        search_query = f"{place_name} reservations"
        encoded_query = urllib.parse.quote(search_query)
        reservation_url = f"https://www.google.com/search?q={encoded_query}"

        result = {
            "restaurant_name": place_name,
            "address": address,
            "phone": phone,
            "party_size": party_size,
            "date": date,
            "time": time,
            "directions_url": maps_url,
            "reservation_url": reservation_url,
            "reservation_source": "Google Search",
        }

        if phone:
            result["message"] = f"Found {place_name}! Use the search link to find reservation options, or call them directly at {phone}."
            result["response_text"] = f"I found {place_name}. I sent a reservation search link to your phone, or you can call them at {phone}."
        else:
            result["message"] = f"Found {place_name}! Use the Google link to search for booking options."
            result["response_text"] = f"I found {place_name} and sent a reservation search link to your phone."

        _LOGGER.info("Found restaurant %s, providing Google search for reservations", place_name)

        return result

    except Exception as err:
        _LOGGER.error("Error booking restaurant: %s", err, exc_info=True)
        return _build_fallback_response(restaurant_name, party_size, date, time)


def _build_fallback_response(
    restaurant_name: str,
    party_size: int,
    date: str,
    time: str,
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
