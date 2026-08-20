"""Spoken message relay — say a message out loud on a chosen speaker.

The phone-relay sibling of `partner_message`: same "one household member wants
to reach the other" job, but delivered as TTS in a room instead of a push
notification. The motivating case (2026-08-20) is Elise texting Assist from
outside the house — "tell Carlos I'm trying to call him on the kitchen
speaker" — when his phone is not being answered. A notification would land in
the same place that is already being ignored; a speaker in the room he is
standing in will not.

Deliberately a separate tool from `send_partner_message` rather than an extra
argument on it: the local brain picks small single-purpose tools far more
reliably than it picks options out of a big one (see
feedback_purellm_dedicated_tool_for_weak_local), and the delivery channels have
opposite privacy properties — one is private to a phone, this one is audible to
whoever is in the room. Keeping them distinct means a misrouted call is a
harmless wrong-channel, never a private text broadcast to the house.

The message body is relayed verbatim (same rule as partner_message); only a
"Message from <sender>." attribution is prepended so whoever hears it knows who
it came from.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Spoken forms that mean "every speaker", not a room name.
_EVERYWHERE = {
    "everywhere", "every speaker", "every room", "all", "all speakers",
    "all the speakers", "all rooms", "the whole house", "whole house",
    "the house", "house", "everywhere in the house",
}

# Trailing nouns people attach to a room name ("the kitchen speaker", "the
# nursery TV"). Stripped only as a FALLBACK pass — "master bathroom speaker"
# must first get a chance to match a mapping key that itself contains the word.
_TARGET_NOUNS = (
    "speakers", "speaker", "media player", "player", "display", "screen",
    "satellite", "sat", "tv", "television", "assist",
)

# TTS fallbacks, best first, used only when the pipeline's own engine cannot be
# resolved. The local wyoming_openai/Chatterbox entities come first; HA Cloud is
# a last resort and ElevenLabs is never acceptable (feedback_tts_engine — the
# ElevenLabs entities exist in the registry but are unavailable/unused).
_TTS_PREFERENCE = ("tts.openai_streaming", "tts.openai_tts")
_TTS_NEVER = ("elevenlabs",)
_TTS_LAST_RESORT = ("cloud", "google")


def _normalize_room(room: str) -> str:
    """Lowercase, strip articles and possessives from a spoken room name."""
    text = " ".join((room or "").lower().replace("_", " ").split())
    for article in ("the ", "my ", "our ", "in the ", "on the ", "in ", "on "):
        if text.startswith(article):
            text = text[len(article):]
    return text.strip(" .,!?")


def _strip_target_noun(room: str) -> str:
    """Drop a trailing device noun: 'kitchen speaker' -> 'kitchen'."""
    for noun in _TARGET_NOUNS:
        if room.endswith(" " + noun):
            return room[: -(len(noun) + 1)].strip()
    return room


def _area_names(hass: "HomeAssistant", entity_id: str) -> list[str]:
    """Area name(s) an entity belongs to, via its own or its device's area."""
    try:
        from homeassistant.helpers import area_registry as ar
        from homeassistant.helpers import device_registry as dr
        from homeassistant.helpers import entity_registry as er
    except Exception:  # noqa: BLE001 - non-HA context (unit tests)
        return []

    try:
        ent_reg = er.async_get(hass)
        entry = ent_reg.async_get(entity_id)
        if not entry:
            return []
        area_id = entry.area_id
        if not area_id and entry.device_id:
            device = dr.async_get(hass).async_get(entry.device_id)
            area_id = device.area_id if device else None
        if not area_id:
            return []
        area = ar.async_get(hass).async_get_area(area_id)
        return [area.name.lower()] if area and area.name else []
    except Exception as err:  # noqa: BLE001 - registry lookups are best-effort
        _LOGGER.debug("announce: area lookup failed for %s: %s", entity_id, err)
        return []


def _match_mapping(room: str, mapping: dict[str, str]) -> list[str]:
    """Match a room name against the configured room->player mapping."""
    if not mapping:
        return []
    for rname, pid in mapping.items():
        if room == rname.lower():
            return [pid]
    for rname, pid in mapping.items():
        rname_lower = rname.lower()
        if room in rname_lower or rname_lower in room:
            return [pid]
    return []


def _match_media_players(hass: "HomeAssistant", room: str) -> list[str]:
    """Fuzzy-match any media_player by friendly name or area.

    Scored so an exact name/area hit always beats a substring one, and an
    unavailable player never beats an available one — announcing into a dead
    entity is silent failure, which is the worst outcome for this tool.
    """
    best: tuple[int, str] | None = None
    for state in hass.states.async_all("media_player"):
        name = (state.attributes.get("friendly_name") or state.entity_id).lower()
        names = [name] + _area_names(hass, state.entity_id)

        score = 0
        if any(room == n for n in names):
            score = 3
        elif any(room in n for n in names):
            score = 2
        elif any(n in room for n in names if n):
            score = 1
        if not score:
            continue
        if state.state not in ("unavailable", "unknown"):
            score += 4

        if best is None or score > best[0]:
            best = (score, state.entity_id)
    return [best[1]] if best else []


def _resolve_targets(
    hass: "HomeAssistant", room: str, mapping: dict[str, str],
) -> list[str]:
    """Resolve a spoken room name to the media_player(s) to speak on.

    The configured mapping points at Music Assistant players, which is what we
    want: an MA announcement ducks and resumes whatever was playing.
    """
    if room in _EVERYWHERE:
        # Only the curated mapping — "everywhere" must never fan out to every
        # media_player in the house (TVs, cast groups, random Chromecasts).
        return list(dict.fromkeys(mapping.values()))

    for candidate in (room, _strip_target_noun(room)):
        if not candidate:
            continue
        players = _match_mapping(candidate, mapping)
        if players:
            return players

    for candidate in (room, _strip_target_noun(room)):
        if not candidate:
            continue
        players = _match_media_players(hass, candidate)
        if players:
            return players
    return []


def _resolve_tts_entity(hass: "HomeAssistant") -> str | None:
    """Pick the TTS entity: the pipeline's engine, else the local ones.

    The naive 'first tts.* entity' fallback is a trap here: this house has
    seven TTS entities registered and the first one enumerated can easily be
    HA Cloud or ElevenLabs, neither of which is wanted.
    """
    try:
        from .. import _resolve_tts_entity as resolve_from_pipeline

        entity = resolve_from_pipeline(hass)
        if entity and not any(bad in entity for bad in _TTS_NEVER):
            return entity
    except Exception as err:  # noqa: BLE001 - fall through to the static picks
        _LOGGER.debug("announce: pipeline TTS lookup failed: %s", err)

    for candidate in _TTS_PREFERENCE:
        state = hass.states.get(candidate)
        if state is not None and state.state != "unavailable":
            return candidate

    fallback = None
    for state in hass.states.async_all("tts"):
        eid = state.entity_id
        if state.state == "unavailable" or any(bad in eid for bad in _TTS_NEVER):
            continue
        if any(late in eid for late in _TTS_LAST_RESORT):
            fallback = fallback or eid
            continue
        return eid
    return fallback


async def _resolve_sender(
    hass: "HomeAssistant", user_id: str | None, override: str | None,
) -> str | None:
    """Human first name of whoever sent the message, for the attribution."""
    if override and override.strip():
        return override.strip().split()[0].title()
    if not user_id:
        return None
    try:
        user = await hass.auth.async_get_user(user_id)
    except Exception as err:  # noqa: BLE001 - auth lookup is best-effort
        _LOGGER.debug("announce: user lookup failed for %s: %s", user_id, err)
        return None
    if not user or not user.name:
        return None
    # First name only — "Message from Elise." reads better than the full name,
    # and the household has no two people sharing one.
    return user.name.strip().split()[0].title()


def _spoken_target(
    hass: "HomeAssistant",
    players: list[str],
    room: str,
    mapping: dict[str, str],
) -> str:
    """How to describe where it was announced, in the confirmation.

    Prefers the configured ROOM name over the entity's friendly name: the
    mapping points at Music Assistant players called "Kitchen Music" / "Office
    Music", and "Announced in the kitchen." reads far better spoken aloud than
    "Announced on the Kitchen Music."
    """
    if len(players) > 1:
        return "every speaker"
    for rname, pid in mapping.items():
        if pid == players[0]:
            return f"the {rname}"
    state = hass.states.get(players[0])
    name = (state.attributes.get("friendly_name") if state else None) or room
    return f"the {name}" if not name.lower().startswith("the ") else name


async def announce_on_speaker(
    arguments: dict[str, Any],
    hass: "HomeAssistant",
    room_player_mapping: dict[str, str] | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Speak a message out loud on a named speaker.

    Args:
        arguments: room (where to say it), message (exact words), optional
            sender (name to attribute it to, when it isn't the HA user).
        hass: Home Assistant instance.
        room_player_mapping: Configured room -> media_player entity mapping.
        user_id: HA user id of whoever sent the request, used to attribute the
            announcement ("Message from Elise.").

    Returns:
        response_text confirmation, or error.
    """
    mapping = room_player_mapping or {}
    room = _normalize_room(arguments.get("room") or "")
    # Verbatim beyond outer whitespace — emphasis and punctuation are part of
    # what was said (same rule as send_partner_message).
    message = (arguments.get("message") or "").strip()

    if not message:
        return {"error": "No message given. Ask what they want said."}
    if not room:
        known = ", ".join(mapping) or "no rooms configured"
        return {
            "error": f"No speaker named. Ask which speaker to say it on ({known}).",
        }

    players = _resolve_targets(hass, room, mapping)
    if not players:
        known = ", ".join(mapping) or "none configured"
        _LOGGER.warning("announce_on_speaker: no player for room %r", room)
        return {
            "error": f"I couldn't find a speaker called '{room}'. "
                     f"Rooms I can announce in: {known}."
        }

    tts_entity = _resolve_tts_entity(hass)
    if not tts_entity:
        return {"error": "No text-to-speech engine is available to announce that."}

    sender = await _resolve_sender(hass, user_id, arguments.get("sender"))
    if sender:
        spoken = f"Message from {sender}. {message}"
    else:
        spoken = message

    delivered: list[str] = []
    failures: list[str] = []
    for player in players:
        try:
            await hass.services.async_call(
                "tts", "speak",
                {
                    "entity_id": tts_entity,
                    "media_player_entity_id": player,
                    "message": spoken,
                },
                blocking=True,
            )
            delivered.append(player)
        except Exception as err:  # noqa: BLE001 - surface delivery failure
            _LOGGER.error("announce_on_speaker: tts.speak on %s failed: %s", player, err)
            failures.append(player)

    if not delivered:
        return {"error": f"Could not play the announcement on {room}."}

    _LOGGER.info(
        "announce_on_speaker: spoke %d chars on %s via %s (sender=%s, failed=%s)",
        len(spoken), ", ".join(delivered), tts_entity, sender or "unknown",
        ", ".join(failures) or "none",
    )
    # Confirmation only — never echo the message back. When this is used from a
    # voice satellite the reply is spoken aloud there, which would say the
    # message twice (and in the wrong room).
    where = _spoken_target(hass, delivered, room, mapping)
    preposition = "on" if where == "every speaker" else "in"
    return {"status": "ok", "response_text": f"Announced {preposition} {where}."}
