"""Verbatim message relay between the two paired household members.

Sends what the speaker actually said to the other person's phone via their
mobile_app notify service. Deliberately dumb: no rewriting, no summarising, no
tone-policing. The whole point of the channel is that it is private between two
consenting adults who both own the devices involved, so the relay must not
editorialise — see the matching Hermes "couples-messaging" skill.

Phone-only by design (chosen 2026-07-27): a speaker announcement would be
audible to anyone else in the room, which is the wrong default for this.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


async def send_partner_message(
    arguments: dict[str, Any],
    hass: "HomeAssistant",
    partner_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Relay a message verbatim to the named person's phone.

    Args:
        arguments: recipient (name) and message (exact words to send)
        hass: Home Assistant instance
        partner_map: lowercase name/alias -> notify service (e.g. "elise" ->
            "notify.mobile_app_elise_munoz")

    Returns:
        response_text confirmation, or error.
    """
    partner_map = partner_map or {}
    recipient = (arguments.get("recipient") or "").strip().lower()
    message = arguments.get("message") or ""

    # Do NOT strip/normalise the message beyond outer whitespace — the relay is
    # verbatim and punctuation/emphasis is part of what was said.
    message = message.strip()

    if not recipient:
        return {"error": "No recipient given. Ask who the message is for."}
    if not message:
        return {"error": "No message text given. Ask what they want to say."}

    # Aliases ("my wife", "the wife", "her") are resolved by the caller building
    # partner_map; anything unknown is refused rather than guessed at, so a
    # misheard name can never send private text to the wrong contact.
    service = partner_map.get(recipient)
    if not service:
        known = ", ".join(sorted({k for k in partner_map})) or "nobody"
        _LOGGER.warning(
            "send_partner_message: unknown recipient %r (known: %s)", recipient, known
        )
        return {
            "error": f"'{recipient}' is not a known messaging contact. "
                     f"This tool can only message: {known}."
        }

    service_name = service.split(".", 1)[1] if service.startswith("notify.") else service

    try:
        await hass.services.async_call(
            "notify",
            service_name,
            {"message": message, "title": "Message"},
            blocking=True,
        )
    except Exception as err:  # noqa: BLE001 - surface any delivery failure to the user
        _LOGGER.error("send_partner_message: notify.%s failed: %s", service_name, err)
        return {"error": f"Could not send the message: {err}"}

    _LOGGER.info(
        "send_partner_message: delivered %d chars to %s via notify.%s",
        len(message), recipient, service_name,
    )
    # Confirmation only. Never echo the message back — it would be spoken aloud
    # on the satellite that just sent it, which defeats the point of a private
    # phone-only channel.
    return {"response_text": f"Sent to {recipient.title()}."}
