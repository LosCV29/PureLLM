"""Plant status tool handler.

Read-only queries against the Olen homeassistant-plant integration
(https://github.com/Olen/homeassistant-plant).

Leans on the integration's own computed status — each ``plant.<name>``
entity already reports per-metric ``*_status`` attributes (``ok`` /
``Low`` / ``High`` / ``null``), so threshold comparisons happen inside
the integration rather than being re-implemented here. Each plant also
exposes companion ``sensor.<slug>_*`` readings and
``number.<slug>_min/max_*`` thresholds that we read for specific-value
queries.
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant, State

_LOGGER = logging.getLogger(__name__)


# Map user-facing metric names to the plant.* status-attribute key,
# the companion sensor suffix, and the number.* threshold suffix.
# "status" and "thresholds" are virtual metrics handled specially.
METRIC_MAP: dict[str, dict[str, str]] = {
    "moisture": {
        "status_attr": "moisture_status",
        "sensor_suffix": "soil_moisture",
        "threshold_suffix": "soil_moisture",
    },
    "temperature": {
        "status_attr": "temperature_status",
        "sensor_suffix": "temperature",
        "threshold_suffix": "temperature",
    },
    "conductivity": {
        "status_attr": "conductivity_status",
        "sensor_suffix": "conductivity",
        "threshold_suffix": "conductivity",
    },
    "illuminance": {
        "status_attr": "illuminance_status",
        "sensor_suffix": "illuminance",
        "threshold_suffix": "illuminance",
    },
    "humidity": {
        "status_attr": "humidity_status",
        "sensor_suffix": "air_humidity",
        "threshold_suffix": "air_humidity",
    },
    "dli": {
        "status_attr": "dli_status",
        "sensor_suffix": "dli_24h",  # 24h rolling value is the meaningful one
        "threshold_suffix": "dli",
    },
}


def _plant_slug(entity_id: str) -> str:
    """Get the slug portion of a plant entity id (``plant.boogie`` -> ``boogie``)."""
    return entity_id.split(".", 1)[1] if "." in entity_id else entity_id


def _plant_name(state: "State") -> str:
    """Return the human-readable plant name."""
    return state.attributes.get("friendly_name") or _plant_slug(state.entity_id).replace("_", " ").title()


def _discover_plants(hass: "HomeAssistant") -> list["State"]:
    """Return all ``plant.*`` entity states currently in the registry."""
    return [s for s in hass.states.async_all() if s.entity_id.startswith("plant.")]


_PLANT_NOISE_WORDS = (
    "the plant", "plant", "my", "the",
)


def _clean_plant_query(query: str) -> str:
    """Strip filler words so 'boogie the plant' / 'my boogie plant' match 'boogie'."""
    q = query.strip().lower()
    # Strip trailing/leading noise words iteratively
    changed = True
    while changed:
        changed = False
        for noise in _PLANT_NOISE_WORDS:
            if q.endswith(" " + noise):
                q = q[: -(len(noise) + 1)].strip()
                changed = True
            elif q.startswith(noise + " "):
                q = q[len(noise) + 1:].strip()
                changed = True
    return q


def _match_plant(hass: "HomeAssistant", query: str) -> "State | None":
    """Fuzzy-match a plant by name or slug. Case-insensitive, filler-tolerant."""
    if not query:
        return None
    plants = _discover_plants(hass)
    q = _clean_plant_query(query)
    if not q:
        return None

    # Exact match on slug or friendly name
    for p in plants:
        if _plant_slug(p.entity_id).lower() == q:
            return p
        if _plant_name(p).lower() == q:
            return p

    # Two-way substring match — handles both "boog" -> "boogie" AND
    # "boogie the plant" (if noise stripping missed something) -> "boogie"
    for p in plants:
        slug = _plant_slug(p.entity_id).lower()
        name = _plant_name(p).lower()
        if q in slug or slug in q or q in name or name in q:
            return p

    return None


def _read_sensor(hass: "HomeAssistant", entity_id: str) -> tuple[Any, str | None]:
    """Read a sensor's current value and unit. Returns (value, unit) or (None, None)."""
    state = hass.states.get(entity_id)
    if not state or state.state in ("unknown", "unavailable", ""):
        return None, None
    raw = state.state
    unit = state.attributes.get("unit_of_measurement")
    # Try numeric conversion for cleaner output
    try:
        val = float(raw)
        val = int(val) if val.is_integer() else round(val, 2)
    except (ValueError, TypeError):
        val = raw
    return val, unit


def _battery_for_plant(hass: "HomeAssistant", plant_slug: str) -> tuple[int | None, str | None]:
    """Find the battery sensor on the underlying plant sensor device.

    Olen's integration stores the physical sensor entity id in the
    ``external_sensor`` attribute of the moisture sensor. The battery
    sensor lives on the same device.
    """
    from homeassistant.helpers import entity_registry as er, device_registry as dr

    moisture_id = f"sensor.{plant_slug}_soil_moisture"
    moisture = hass.states.get(moisture_id)
    if not moisture:
        return None, None

    external = moisture.attributes.get("external_sensor")
    if not external:
        return None, None

    ent_reg = er.async_get(hass)
    ext_entry = ent_reg.async_get(external)
    if not ext_entry or not ext_entry.device_id:
        return None, None

    # Find any battery-class sensor on the same device
    for entry in ent_reg.entities.values():
        if entry.device_id != ext_entry.device_id:
            continue
        if entry.entity_id.split(".", 1)[0] != "sensor":
            continue
        st = hass.states.get(entry.entity_id)
        if not st:
            continue
        if st.attributes.get("device_class") == "battery":
            val, unit = _read_sensor(hass, entry.entity_id)
            return val, unit

    return None, None


def _metric_block(hass: "HomeAssistant", plant_state: "State", metric: str) -> dict[str, Any]:
    """Build a single-metric reading block: value, unit, thresholds, status."""
    slug = _plant_slug(plant_state.entity_id)
    spec = METRIC_MAP[metric]

    value, unit = _read_sensor(hass, f"sensor.{slug}_{spec['sensor_suffix']}")
    min_val, _ = _read_sensor(hass, f"number.{slug}_min_{spec['threshold_suffix']}")
    max_val, _ = _read_sensor(hass, f"number.{slug}_max_{spec['threshold_suffix']}")
    status = plant_state.attributes.get(spec["status_attr"]) or "ok"

    block: dict[str, Any] = {
        "metric": metric,
        "value": value,
        "status": status,
    }
    if unit is not None:
        block["unit"] = unit
    if min_val is not None:
        block["min"] = min_val
    if max_val is not None:
        block["max"] = max_val
    return block


def _all_metric_blocks(hass: "HomeAssistant", plant_state: "State") -> dict[str, dict[str, Any]]:
    """Gather every known metric for a plant into a dict keyed by metric name."""
    blocks: dict[str, dict[str, Any]] = {}
    for metric in METRIC_MAP:
        block = _metric_block(hass, plant_state, metric)
        # Skip metrics that have no sensor data AND no threshold (integration doesn't track them for this plant)
        if block["value"] is None and "min" not in block and "max" not in block:
            continue
        blocks[metric] = block
    return blocks


def _plant_issues(plant_state: "State") -> list[str]:
    """Return a list of non-ok status descriptions like ['moisture Low', 'dli Low']."""
    issues: list[str] = []
    for metric, spec in METRIC_MAP.items():
        status = plant_state.attributes.get(spec["status_attr"])
        if status and status != "ok":
            issues.append(f"{metric} {status}")
    return issues


def list_plant_names(hass: "HomeAssistant") -> list[str]:
    """Return friendly names of all discovered plants. Used for tool description injection."""
    return sorted({_plant_name(p) for p in _discover_plants(hass)})


# Keywords that strongly imply the user is asking about moisture/watering.
# Used as a backstop when the LLM omits metric='moisture' for a water query.
_WATER_KEYWORDS = (
    "water", "watering", "moisture", "soil", "thirsty", "dry",
    "wet", "hydrat", "overwater", "underwater", "drown",
)


def _infer_metric_from_query(user_query: str) -> str | None:
    """Infer the intended metric from the user's natural-language query.

    Only fires when the LLM didn't provide one — meant as a safety net, not
    a replacement for the LLM choosing the right param.
    """
    if not user_query:
        return None
    q = user_query.lower()
    if any(kw in q for kw in _WATER_KEYWORDS):
        return "moisture"
    return None


def _join_names(entries: list[dict[str, Any]]) -> str:
    return ", ".join(e["plant"] for e in entries)


def _pluralize(n: int, singular: str, plural: str) -> str:
    return singular if n == 1 else plural


def _moisture_sweep(
    hass: "HomeAssistant",
    plants: list["State"],
    problems_only: bool,
) -> dict[str, Any]:
    """Group every plant's moisture status into underwatered / overwatered /
    ok / unavailable, and produce a response_text ready for TTS.

    This is the star query — "does any plant need water" — so we go
    out of our way to format it so the LLM can't lose the thread.
    """
    underwatered: list[dict[str, Any]] = []
    overwatered: list[dict[str, Any]] = []
    ok_plants: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []

    for p in plants:
        block = _metric_block(hass, p, "moisture")
        entry = {"plant": _plant_name(p), **block}
        if block["value"] is None:
            unavailable.append(entry)
        elif block["status"] == "Low":
            underwatered.append(entry)
        elif block["status"] == "High":
            overwatered.append(entry)
        else:
            ok_plants.append(entry)

    # Build a natural-language response_text the LLM can repeat verbatim
    parts: list[str] = []
    if underwatered:
        verb = _pluralize(len(underwatered), "needs", "need")
        parts.append(
            f"{len(underwatered)} {_pluralize(len(underwatered), 'plant', 'plants')} "
            f"{verb} water ({_join_names(underwatered)})"
        )
    if overwatered:
        verb = _pluralize(len(overwatered), "is", "are")
        parts.append(
            f"{len(overwatered)} {_pluralize(len(overwatered), 'plant', 'plants')} "
            f"{verb} overwatered ({_join_names(overwatered)})"
        )

    if not parts:
        response_text = f"All {len(plants)} plants have adequate moisture."
    else:
        response_text = "; ".join(parts) + "."

    if unavailable:
        response_text += (
            f" {len(unavailable)} "
            f"{_pluralize(len(unavailable), 'sensor is', 'sensors are')} unavailable."
        )

    result: dict[str, Any] = {
        "metric": "moisture",
        "underwatered_plants": underwatered,
        "overwatered_plants": overwatered,
        "total_checked": len(plants),
        "response_text": response_text,
        "instruction": "Use response_text VERBATIM. Do not re-label or combine underwatered/overwatered plants.",
    }
    if unavailable:
        result["unavailable_sensors"] = unavailable
    if not problems_only:
        result["ok_plants"] = ok_plants
    return result


async def check_plant_status(
    arguments: dict[str, Any],
    hass: "HomeAssistant",
    user_query: str = "",
) -> dict[str, Any]:
    """Query plant status via the Olen homeassistant-plant integration.

    Args:
        arguments: Tool arguments:
            plant (str, optional): Plant name. Fuzzy matched. Omit to query all plants.
            metric (str, optional): Specific metric to read
                (moisture | temperature | conductivity | illuminance |
                 humidity | dli | battery | status | thresholds).
            problems_only (bool, optional): When True, only return plants whose
                overall status is ``problem`` (or, with a ``metric``, plants whose
                that metric's status is Low/High).
        hass: Home Assistant instance.
        user_query: Original user utterance. Used as a safety net to infer
            ``metric='moisture'`` if the LLM forgot to pass it for a water
            question like "any plants dry".

    Returns:
        Dict with the query result. Shape depends on arguments:
        - plant set, metric set: single reading
        - plant set, no metric: full readings for one plant
        - no plant: overview of all plants (optionally filtered to problems)
    """
    plant_arg = (arguments.get("plant") or "").strip()
    metric_arg = (arguments.get("metric") or "").strip().lower()
    problems_only = bool(arguments.get("problems_only"))

    # Safety net: if the LLM didn't pick a metric but the user clearly asked
    # about moisture, infer it. Prevents the LLM from silently falling into
    # the generic overview path and giving muddy answers.
    if not metric_arg:
        inferred = _infer_metric_from_query(user_query)
        if inferred:
            _LOGGER.info(
                "check_plant_status: inferring metric=%r from user query %r",
                inferred, user_query,
            )
            metric_arg = inferred

    _LOGGER.info(
        "check_plant_status called: plant=%r metric=%r problems_only=%s (query=%r)",
        plant_arg, metric_arg, problems_only, user_query,
    )

    plants = _discover_plants(hass)
    if not plants:
        return {"error": "No plant entities found. This tool requires the Olen homeassistant-plant integration."}

    # Validate metric if provided
    if metric_arg and metric_arg not in METRIC_MAP and metric_arg not in ("battery", "status", "thresholds"):
        return {
            "error": f"Unknown metric '{metric_arg}'. "
                     f"Valid: {', '.join(list(METRIC_MAP.keys()) + ['battery', 'status', 'thresholds'])}"
        }

    # ========== Single plant ==========
    if plant_arg:
        plant = _match_plant(hass, plant_arg)
        if not plant:
            available = ", ".join(list_plant_names(hass))
            return {"error": f"No plant matching '{plant_arg}'. Available: {available}"}

        name = _plant_name(plant)
        slug = _plant_slug(plant.entity_id)

        # Specific metric ------------------------------------------------
        if metric_arg in METRIC_MAP:
            block = _metric_block(hass, plant, metric_arg)
            value = block["value"]
            unit = block.get("unit", "")
            status = block["status"]
            if value is None:
                response_text = f"{name}'s {metric_arg} sensor is unavailable."
            else:
                status_phrase = ""
                if status == "Low":
                    status_phrase = f" (below minimum of {block.get('min', '?')}{unit})"
                elif status == "High":
                    status_phrase = f" (above maximum of {block.get('max', '?')}{unit})"
                response_text = f"{name}'s {metric_arg} is {value}{unit}{status_phrase}."
            return {
                "plant": name,
                "species": plant.attributes.get("species"),
                "response_text": response_text,
                **block,
            }

        if metric_arg == "battery":
            val, unit = _battery_for_plant(hass, slug)
            if val is None:
                return {"plant": name, "metric": "battery", "value": None,
                        "response_text": f"{name}'s plant sensor battery reading is unavailable."}
            return {"plant": name, "metric": "battery", "value": val, "unit": unit or "%",
                    "response_text": f"{name}'s plant sensor battery is at {val}{unit or '%'}."}

        if metric_arg == "thresholds":
            thresholds: dict[str, Any] = {}
            for metric, spec in METRIC_MAP.items():
                mn, mn_unit = _read_sensor(hass, f"number.{slug}_min_{spec['threshold_suffix']}")
                mx, mx_unit = _read_sensor(hass, f"number.{slug}_max_{spec['threshold_suffix']}")
                if mn is None and mx is None:
                    continue
                thresholds[metric] = {"min": mn, "max": mx, "unit": mn_unit or mx_unit}
            return {"plant": name, "thresholds": thresholds}

        # No metric → full readout
        issues = _plant_issues(plant)
        state = plant.state
        if state == "ok" and not issues:
            response_text = f"{name} is doing well — no issues detected."
        else:
            response_text = f"{name}: {', '.join(issues) if issues else state}."
        return {
            "plant": name,
            "species": plant.attributes.get("species"),
            "state": state,
            "issues": issues,
            "metrics": _all_metric_blocks(hass, plant),
            "response_text": response_text,
        }

    # ========== All plants ==========
    # Moisture sweep gets special grouped treatment (under/over/ok/unavailable)
    if metric_arg == "moisture":
        return _moisture_sweep(hass, plants, problems_only)

    # Other metric sweeps
    if metric_arg in METRIC_MAP:
        results = []
        for p in plants:
            block = _metric_block(hass, p, metric_arg)
            if problems_only and block["status"] == "ok":
                continue
            results.append({"plant": _plant_name(p), **block})
        if not results:
            response_text = (
                f"All {len(plants)} plants are within their {metric_arg} range."
                if problems_only else
                f"No {metric_arg} readings available."
            )
        else:
            names = ", ".join(f"{r['plant']} ({r['status']})" for r in results)
            response_text = f"{metric_arg.title()} issues: {names}."
        return {
            "metric": metric_arg,
            "plants": results,
            "total_checked": len(plants),
            "problems_only": problems_only,
            "response_text": response_text,
        }

    if metric_arg == "battery":
        results = []
        for p in plants:
            val, unit = _battery_for_plant(hass, _plant_slug(p.entity_id))
            if val is None:
                continue
            entry = {"plant": _plant_name(p), "value": val, "unit": unit or "%"}
            if problems_only and (not isinstance(val, (int, float)) or val >= 20):
                continue
            results.append(entry)
        if not results:
            response_text = ("All plant sensor batteries are above 20%."
                             if problems_only else "No battery readings available.")
        else:
            names = ", ".join(f"{r['plant']} ({r['value']}{r.get('unit', '%')})" for r in results)
            response_text = f"Low batteries: {names}." if problems_only else f"Plant sensor batteries: {names}."
        return {"metric": "battery", "plants": results, "total_checked": len(plants),
                "problems_only": problems_only, "response_text": response_text}

    # No metric → overview of every plant
    overview = []
    for p in plants:
        issues = _plant_issues(p)
        if problems_only and p.state != "problem" and not issues:
            continue
        overview.append({
            "plant": _plant_name(p),
            "state": p.state,
            "issues": issues,
        })
    problem_count = sum(1 for p in plants if p.state == "problem")
    if not overview:
        response_text = f"All {len(plants)} plants are doing well."
    else:
        lines = [f"{e['plant']}: {', '.join(e['issues']) if e['issues'] else e['state']}" for e in overview]
        response_text = "; ".join(lines) + "."
    return {
        "plants": overview,
        "total": len(plants),
        "problem_count": problem_count,
        "problems_only": problems_only,
        "response_text": response_text,
    }
