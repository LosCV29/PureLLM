"""Fuzzy matching utilities for PolyVoice.

This module handles device name matching with:
- Synonym expansion (blind/shade/curtain/cover are interchangeable)
- Stopword removal
- Direct entity matching (NO room fuzzy logic - causes cross-room confusion)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

from homeassistant.helpers import entity_registry as er

_LOGGER = logging.getLogger(__name__)

# Domains that should be prioritized for device control (actual controllable devices)
# Lower number = higher priority
DOMAIN_PRIORITY = {
    "light": 1,
    "switch": 1,
    "fan": 1,
    "cover": 1,
    "lock": 1,
    "climate": 2,
    "vacuum": 2,
    "media_player": 2,
    "scene": 3,
    "script": 3,
    "automation": 4,
    "input_boolean": 5,
    "humidifier": 2,
    "siren": 2,
    # Helper/sensor domains - lowest priority for device control
    "select": 10,
    "number": 10,
    "input_number": 10,
    "input_select": 10,
    "input_text": 10,
    "sensor": 20,
    "binary_sensor": 20,
}

# Stopwords to remove from queries (articles, possessives, prepositions)
STOPWORDS = frozenset([
    "the", "my", "a", "an", "in", "on", "at", "to", "for", "of",
    "please", "can", "you", "could", "would"
])

# Number word to digit mapping for voice command normalization
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}

# Synonym groups for fuzzy entity matching
# When searching for entities, these synonyms are treated as equivalent
DEVICE_SYNONYMS = {
    # Cover synonyms (blind/shade/curtain/cover are interchangeable)
    "blind": ["shade", "curtain", "cover", "drape", "roller", "blinds", "shades"],
    "shade": ["blind", "curtain", "cover", "drape", "roller", "blinds", "shades"],
    "curtain": ["blind", "shade", "cover", "drape", "curtains", "blinds", "shades"],
    "cover": ["blind", "shade", "curtain", "drape", "covers", "blinds", "shades"],
    "drape": ["blind", "shade", "curtain", "cover", "drapes", "blinds", "shades"],
    "roller": ["blind", "shade", "curtain", "cover", "rollers", "blinds", "shades"],
    # Plural forms
    "blinds": ["shades", "curtains", "covers", "drapes", "rollers", "blind", "shade"],
    "shades": ["blinds", "curtains", "covers", "drapes", "rollers", "blind", "shade"],
    "curtains": ["blinds", "shades", "covers", "drapes", "curtain", "blind", "shade"],
    "covers": ["blinds", "shades", "curtains", "drapes", "cover", "blind", "shade"],
    "drapes": ["blinds", "shades", "curtains", "covers", "drape", "blind", "shade"],
    "rollers": ["blinds", "shades", "curtains", "covers", "roller", "blind", "shade"],
    # Light synonyms
    "light": ["lamp", "bulb", "fixture", "lights", "lamps"],
    "lights": ["lamps", "bulbs", "fixtures", "light", "lamp"],
    "lamp": ["light", "bulb", "lamps", "lights"],
    "lamps": ["lights", "bulbs", "lamp", "light"],
    "bulb": ["light", "lamp", "bulbs"],
    "bulbs": ["lights", "lamps", "bulb"],
    # Lock synonyms
    "lock": ["deadbolt", "latch", "locks"],
    "locks": ["deadbolts", "latches", "lock"],
    # Climate synonyms
    "thermostat": ["climate", "hvac", "ac", "heater", "temp", "temperature"],
    "ac": ["air conditioner", "air conditioning", "climate", "thermostat", "cooling"],
    "air conditioner": ["ac", "climate", "thermostat"],
    "heater": ["heating", "thermostat", "climate", "heat"],
    "heat": ["heater", "heating", "thermostat"],
    # Fan synonyms
    "fan": ["ceiling fan", "exhaust fan", "fans"],
    "fans": ["ceiling fans", "fan"],
    "ceiling fan": ["fan"],
    # Door synonyms
    "door": ["doors"],
    "doors": ["door"],
    "gate": ["gates"],
    "gates": ["gate"],
    "garage": ["garage door"],
    "garage door": ["garage"],
    # TV/Media synonyms
    "tv": ["television", "telly", "screen"],
    "television": ["tv", "telly"],
    "speaker": ["media player", "sonos", "echo", "speakers"],
    "speakers": ["media players", "speaker"],
    # Switch synonyms
    "switch": ["outlet", "plug", "switches"],
    "switches": ["outlets", "plugs", "switch"],
    "outlet": ["switch", "plug", "outlets"],
    "plug": ["outlet", "switch", "plugs"],
}


def _strip_stopwords(query: str) -> str:
    """Remove stopwords from query for better matching."""
    words = query.lower().split()
    filtered = [w for w in words if w not in STOPWORDS]
    return " ".join(filtered) if filtered else query.lower()


def normalize_numbers(query: str) -> str:
    """Convert number words to digits in query.

    "master shade one" -> "master shade 1"
    "bedroom light two" -> "bedroom light 2"
    """
    words = query.lower().split()
    normalized = [NUMBER_WORDS.get(word, word) for word in words]
    return " ".join(normalized)


def normalize_cover_query(query: str) -> list[str]:
    """Generate query variations with device synonyms.

    For "living room blinds" generates:
    - "living room blinds" (original)
    - "living room shades" (synonym substitution)
    - "living room curtains", "living room covers", etc.
    """
    query_lower = query.lower().strip()
    variations = set()

    # Start with original and stopwords-stripped
    variations.add(query_lower)
    stripped = _strip_stopwords(query_lower)
    variations.add(stripped)

    # Generate synonym variations
    for base_query in [query_lower, stripped]:
        words = base_query.split()
        for i, word in enumerate(words):
            if word in DEVICE_SYNONYMS:
                for replacement in DEVICE_SYNONYMS[word]:
                    new_words = words.copy()
                    new_words[i] = replacement
                    variations.add(" ".join(new_words))

    # Return shorter (stopwords stripped) first
    result = list(variations)
    result.sort(key=len)
    return result


def _words_match(query: str, target: str) -> bool:
    """Check if all query words appear in target (with synonym and number expansion)."""
    query_words = set(query.lower().split()) - STOPWORDS
    target_words = set(target.lower().split()) - STOPWORDS

    if not query_words:
        return False

    # Normalize numbers in both query and target for matching
    # e.g., "one" -> "1" so "shade one" matches "shade 1"
    query_words_normalized = {NUMBER_WORDS.get(w, w) for w in query_words}
    target_words_normalized = {NUMBER_WORDS.get(w, w) for w in target_words}

    # Expand target words with their synonyms for matching
    expanded_target = set(target_words) | target_words_normalized
    for word in target_words:
        if word in DEVICE_SYNONYMS:
            expanded_target.update(DEVICE_SYNONYMS[word])

    # All query words (or their number-normalized forms) must be in expanded target
    return query_words_normalized <= expanded_target


def find_entity_by_name(
    hass: HomeAssistant,
    query: str,
    device_aliases: dict[str, str]
) -> tuple[str | None, str | None]:
    """Search for entity by name.

    Returns (entity_id, friendly_name) or (None, None) if not found.
    """
    # Strip trailing punctuation the LLM may include
    query = query.strip().rstrip(".,!?;:")

    # Build list of query variations to try
    queries_to_try = [query]

    # Add number-normalized version (e.g., "shade one" -> "shade 1")
    number_normalized = normalize_numbers(query)
    if number_normalized.lower() != query.lower():
        queries_to_try.append(number_normalized)

    # Try each query variation
    for q in queries_to_try:
        # Try direct query first
        result = _find_entity_by_query(hass, q, device_aliases)
        if result[0] is not None:
            return result

        # Try synonym variations
        for query_var in normalize_cover_query(q):
            if query_var.lower() == q.lower():
                continue
            result = _find_entity_by_query(hass, query_var, device_aliases)
            if result[0] is not None:
                return result

    return (None, None)


def _find_entity_by_query(
    hass: HomeAssistant,
    query: str,
    device_aliases: dict[str, str]
) -> tuple[str | None, str | None]:
    """Internal entity search - direct matching only, no fuzzy logic."""
    query_lower = query.lower().strip()
    _LOGGER.warning("DEBUG Fuzzy search for: '%s' with aliases: %s", query_lower, device_aliases)

    # PRIORITY 1: Exact match in configured device aliases
    if query_lower in device_aliases:
        entity_id = device_aliases[query_lower]
        state = hass.states.get(entity_id)
        friendly_name = state.attributes.get("friendly_name", query) if state else query
        _LOGGER.warning("DEBUG Found via device alias: %s -> %s", query_lower, entity_id)
        return (entity_id, friendly_name)

    # PRIORITY 2: Partial match in device aliases (all words present)
    for alias, entity_id in device_aliases.items():
        if _words_match(query_lower, alias) or _words_match(alias, query_lower):
            state = hass.states.get(entity_id)
            friendly_name = state.attributes.get("friendly_name", alias) if state else alias
            _LOGGER.info("Found via partial alias: %s -> %s", alias, entity_id)
            return (entity_id, friendly_name)

    # Single pass through entity registry
    ent_reg = er.async_get(hass)
    all_states = {s.entity_id: s for s in hass.states.async_all()}

    # partial_matches: (match_priority, domain_priority, entity_id, friendly_name)
    partial_matches: list[tuple[int, int, str, str]] = []

    def get_domain_priority(entity_id: str) -> int:
        """Get priority for an entity's domain (lower = better for device control)."""
        domain = entity_id.split(".")[0] if "." in entity_id else ""
        return DOMAIN_PRIORITY.get(domain, 8)  # Default priority 8 for unknown domains

    for entity_entry in ent_reg.entities.values():
        state = all_states.get(entity_entry.entity_id)
        friendly_name = state.attributes.get("friendly_name", "") if state else ""
        domain_pri = get_domain_priority(entity_entry.entity_id)

        # PRIORITY 3: Exact match on entity registry alias
        if entity_entry.aliases:
            for alias in entity_entry.aliases:
                if alias.lower() == query_lower:
                    # For exact matches, still consider domain priority
                    _LOGGER.info("Exact alias match: '%s' -> %s", alias, entity_entry.entity_id)
                    partial_matches.append((3, domain_pri, entity_entry.entity_id, friendly_name or alias))
                elif _words_match(query_lower, alias.lower()):
                    _LOGGER.info("Partial alias match: '%s' contains '%s' -> %s", alias, query_lower, entity_entry.entity_id)
                    partial_matches.append((4, domain_pri, entity_entry.entity_id, friendly_name or alias))

        # PRIORITY 5: Exact match on friendly name
        if friendly_name:
            fn_lower = friendly_name.lower()
            if fn_lower == query_lower:
                partial_matches.append((5, domain_pri, entity_entry.entity_id, friendly_name))
            elif _words_match(query_lower, fn_lower):
                partial_matches.append((6, domain_pri, entity_entry.entity_id, friendly_name))

    # Check states not in entity registry
    for entity_id, state in all_states.items():
        if entity_id not in {e.entity_id for e in ent_reg.entities.values()}:
            friendly_name = state.attributes.get("friendly_name", "")
            domain_pri = get_domain_priority(entity_id)
            if friendly_name:
                fn_lower = friendly_name.lower()
                if fn_lower == query_lower:
                    partial_matches.append((5, domain_pri, entity_id, friendly_name))
                elif _words_match(query_lower, fn_lower):
                    partial_matches.append((6, domain_pri, entity_id, friendly_name))

    # Return best match - sort by match priority first, then domain priority
    if partial_matches:
        partial_matches.sort(key=lambda x: (x[0], x[1]))
        _LOGGER.warning("DEBUG Best match for '%s': %s (%s)", query_lower, partial_matches[0][2], partial_matches[0][3])
        return (partial_matches[0][2], partial_matches[0][3])

    _LOGGER.warning("No entity found for query: '%s'", query_lower)
    return (None, None)
