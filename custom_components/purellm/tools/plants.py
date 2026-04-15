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


def _match_plant(hass: "HomeAssistant", query: str) -> "State | None":
    """Fuzzy-match a plant by name or slug. Case-insensitive substring match."""
    if not query:
        return None
    q = query.strip().lower()
    plants = _discover_plants(hass)

    # Exact match on slug or friendly name
    for p in plants:
        if _plant_slug(p.entity_id).lower() == q:
            return p
        if _plant_name(p).lower() == q:
            return p

    # Substring match on slug or friendly name (e.g. "boog" -> "boogie")
    for p in plants:
        if q in _plant_slug(p.entity_id).lower() or q in _plant_name(p).lower():
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


async def check_plant_status(
    arguments: dict[str, Any],
    hass: "HomeAssistant",
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

    Returns:
        Dict with the query result. Shape depends on arguments:
        - plant set, metric set: single reading
        - plant set, no metric: full readings for one plant
        - no plant: overview of all plants (optionally filtered to problems)
    """
    plant_arg = (arguments.get("plant") or "").strip()
    metric_arg = (arguments.get("metric") or "").strip().lower()
    problems_only = bool(arguments.get("problems_only"))

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
            return {
                "plant": name,
                "species": plant.attributes.get("species"),
                **block,
            }

        if metric_arg == "battery":
            val, unit = _battery_for_plant(hass, slug)
            if val is None:
                return {"plant": name, "metric": "battery", "value": None,
                        "message": "No battery sensor found for this plant."}
            return {"plant": name, "metric": "battery", "value": val, "unit": unit or "%"}

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
        return {
            "plant": name,
            "species": plant.attributes.get("species"),
            "state": plant.state,  # "ok" or "problem"
            "issues": issues,
            "metrics": _all_metric_blocks(hass, plant),
        }

    # ========== All plants ==========
    # Specific metric across every plant
    if metric_arg in METRIC_MAP:
        results = []
        for p in plants:
            block = _metric_block(hass, p, metric_arg)
            if problems_only and block["status"] == "ok":
                continue
            results.append({"plant": _plant_name(p), **block})
        return {
            "metric": metric_arg,
            "plants": results,
            "total_checked": len(plants),
            "problems_only": problems_only,
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
        return {"metric": "battery", "plants": results, "total_checked": len(plants),
                "problems_only": problems_only,
                "note": "Batteries below 20% flagged as problems" if problems_only else None}

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
    return {
        "plants": overview,
        "total": len(plants),
        "problem_count": sum(1 for p in plants if p.state == "problem"),
        "problems_only": problems_only,
    }
