"""Canonical measurement source codes and stable observation-set identity."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

SOURCE_LABELS = {
    "실제 측정": "user_measurement",
    "시뮬레이션": "simulation",
    "가상 데모": "simulated_demo",
}
KNOWN_SOURCES = {"user_measurement", "simulation", "simulated_demo", "imported_measurement", "unknown"}


def normalize_measurement(row: dict[str, Any]) -> dict[str, Any]:
    node = str(row.get("node", "")).strip()
    if not node:
        raise ValueError("Measurement node is required.")
    value = row.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("Measurement value must be a finite number.")
    uncertainty = row.get("uncertainty")
    if uncertainty is not None and (isinstance(uncertainty, bool) or not isinstance(uncertainty, (int, float))
                                   or not math.isfinite(float(uncertainty)) or float(uncertainty) < 0):
        raise ValueError("Measurement uncertainty must be finite and non-negative.")
    original_source = row.get("source")
    source = SOURCE_LABELS.get(original_source, original_source)
    if source not in KNOWN_SOURCES:
        source = "unknown"
    normalized = {**row, "node": node, "value": float(value), "source": source,
                  "verification_status": "USER_REPORTED" if source == "user_measurement" else
                  "SIMULATED" if source == "simulated_demo" else
                  "MODEL_OUTPUT" if source == "simulation" else "UNVERIFIED"}
    if original_source != source:
        normalized["source_original"] = original_source
    if uncertainty is not None:
        normalized["uncertainty"] = float(uncertainty)
    return normalized


def normalize_measurements(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_measurement(row) for row in rows]


def measurement_set_sha256(rows: list[dict[str, Any]]) -> str:
    normalized = normalize_measurements(rows)
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def measurement_source_class(rows: list[dict[str, Any]]) -> str:
    sources = {row["source"] for row in normalize_measurements(rows)}
    if not sources:
        return "NONE"
    return next(iter(sources)) if len(sources) == 1 else "MIXED_SOURCES"
