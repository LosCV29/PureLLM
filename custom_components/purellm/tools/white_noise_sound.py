"""Dedicated white-noise-machine sound tool for PureLLM.

Plays a named sound (tone) on the Tuya/Momcozy white noise machine via its
tuya-local `siren` entity, optionally setting volume, and can list the sounds
it supports. Kept separate from `control_device` on purpose (same rationale as
fan_speed): weak local LLMs reliably call a small single-purpose tool, and
control_device's action enum cannot express "play rain on the sound machine"
at all. Plain on/off of the machine stays with control_device.

NOT a pre-LLM short-circuit and NOT an ambient-speaker feature — this drives
the physical machine only (see feedback_no_prellm_shortcircuits /
project_tuya_white_noise_machine).
"""
from __future__ import annotations

import logging

_LOGGER = logging.getLogger(__name__)


def _find_machine(hass) -> tuple[str | None, list[str]]:
    """Locate the white-noise-machine siren entity.

    Prefer a siren whose friendly name mentions white noise / sound machine;
    fall back to any siren that advertises a tone list. Returns
    (entity_id, available_tones).
    """
    candidates = []
    for state in hass.states.async_all("siren"):
        tones = state.attributes.get("available_tones") or []
        if not tones:
            continue
        name = (state.attributes.get("friendly_name") or state.entity_id).lower()
        score = 1 if ("white noise" in name or "sound machine" in name or "noise machine" in name) else 0
        candidates.append((score, state.entity_id, list(tones)))
    if not candidates:
        return None, []
    candidates.sort(reverse=True)
    _, eid, tones = candidates[0]
    return eid, tones


def _match_tone(requested: str, tones: list[str]) -> str | None:
    """Fuzzy-map the user's sound name onto an available tone."""
    req = requested.strip().lower()
    if not req:
        return None
    low = {t.lower(): t for t in tones}
    # 1) exact
    if req in low:
        return low[req]
    # 2) singular/plural nudge ("ocean waves" -> "Ocean wave", "bird" -> "Birds")
    for cand in (req.rstrip("s"), req + "s"):
        if cand in low:
            return low[cand]
    # 3) substring either way — prefer the SHORTEST tone name so "rain"
    #    picks "Rain", not "Rain on roof".
    # 2.5) strong token overlap first ("rain on the roof" shares two words
    #      with "Rain on roof" but only substring-matches "Rain")
    req_words = set(req.replace("-", " ").split())
    overlaps = [(len(req_words & set(tl.split())), t) for tl, t in low.items()]
    top_n = max(n for n, _ in overlaps)
    if top_n >= 2:
        return min((t for n, t in overlaps if n == top_n), key=len)
    reqs = req.rstrip("s")
    subs = [t for tl, t in low.items() if req in tl or tl in req or reqs in tl]
    if subs:
        # Prefer the hit sharing the most words ("rain on the roof" ->
        # "Rain on roof", not "Rain"); tie-break on shortest name.
        req_words = set(req.split())
        return max(subs, key=lambda t: (len(req_words & set(t.lower().split())), -len(t)))
    # 4) token overlap ("rain sounds" -> "Rain", "camp fire" -> "Campfire")
    req_tokens = set(req.replace("-", " ").split())
    best, best_n = None, 0
    for tl, t in low.items():
        n = len(req_tokens & set(tl.split()))
        # also credit joined tokens (camp+fire == campfire)
        if "".join(sorted(req_tokens)) == "".join(sorted(tl.split())):
            n += 1
        if n > best_n or (n == best_n and best and n and len(t) < len(best)):
            if n:
                best, best_n = t, n
    return best


async def set_white_noise_sound(arguments: dict, hass) -> str:
    """Play a sound on the white noise machine, set its volume, or list sounds."""
    eid, tones = _find_machine(hass)
    if not eid:
        return "I couldn't find the white noise machine's sound entity."

    requested = (arguments.get("sound") or "").strip()
    volume = arguments.get("volume")

    # List mode: no sound and no volume requested.
    if not requested and volume is None:
        return "The white noise machine can play: " + ", ".join(tones) + "."

    data: dict = {"entity_id": eid}
    spoken_bits = []

    if requested:
        tone = _match_tone(requested, tones)
        if not tone:
            return (
                f"The machine doesn't have a '{requested}' sound. "
                "It can play: " + ", ".join(tones) + "."
            )
        data["tone"] = tone
        spoken_bits.append(f"playing {tone}")

    if volume is not None:
        try:
            vol = max(0, min(100, int(volume)))
        except (TypeError, ValueError):
            return f"'{volume}' isn't a valid volume — use 0 to 100."
        data["volume_level"] = round(vol / 100.0, 2)
        spoken_bits.append(f"volume {vol}%")

    _LOGGER.info("White noise sound: siren.turn_on %s", data)
    await hass.services.async_call("siren", "turn_on", data, blocking=True)
    return "White noise machine " + " at ".join(spoken_bits) + "."
