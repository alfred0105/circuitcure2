"""Load explicit component rating data without inferring missing limits."""

import json
import math
from pathlib import Path
from typing import Any

RATING_KEYS = {"max_abs_voltage_v", "max_current_a", "max_power_w"}
RATING_KINDS = {"continuous", "peak", "unknown"}


def load_component_specs(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a component-rating JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("components"), dict):
        raise ValueError("ratings JSON must contain a components object")
    for reference, component in data["components"].items():
        if not isinstance(component, dict) or not isinstance(component.get("ratings", {}), dict):
            raise ValueError(f"ratings for {reference} must be an object")
        ratings = component.get("ratings", {})
        unknown_keys = set(ratings) - RATING_KEYS
        if unknown_keys:
            raise ValueError(f"unsupported rating key(s) for {reference}: {', '.join(sorted(unknown_keys))}")
        if any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0)
            for value in ratings.values()
        ):
            raise ValueError(f"ratings for {reference} must be positive numbers or null")
        source = component.get("source")
        if any(value is not None for value in ratings.values()) and (
            not isinstance(source, dict) or not isinstance(source.get("type"), str) or not source["type"].strip()
        ):
            raise ValueError(f"component {reference} has ratings but no source metadata")
        rating_sources = component.get("rating_sources", {})
        if not isinstance(rating_sources, dict) or set(rating_sources) - RATING_KEYS:
            raise ValueError(f"rating_sources for {reference} must use supported rating keys")
        kinds = component.get("rating_types", {})
        if not isinstance(kinds, dict) or set(kinds) - RATING_KEYS:
            raise ValueError(f"rating_types for {reference} must use supported rating keys")
        if any(kind not in RATING_KINDS for kind in kinds.values()):
            raise ValueError(f"rating_types for {reference} must be continuous, peak, or unknown")
    return data
