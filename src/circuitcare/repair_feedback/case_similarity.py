"""Deterministic duplicate and compatibility checks for repair cases."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any


def case_fingerprint(case: dict[str, Any]) -> str:
    """Exact-event fingerprint excludes free text, photos, and timestamps."""
    cause = case.get("root_cause") or {}
    interventions = sorted((row.get("component_ref"), row.get("action"), row.get("replacement_mpn"))
                           for row in case.get("interventions", []) if isinstance(row, dict))
    payload = (case.get("circuit_hash"), case.get("board_instance_id"), cause.get("component_ref"),
               cause.get("fault_type"), tuple(interventions), _measurement_signature(case.get("observations", [])),
               _measurement_signature(case.get("post_repair_tests", [])))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def find_duplicate_cases(case: dict[str, Any], existing: list[dict[str, Any]], *, review_days: int = 30) -> dict[str, Any]:
    fingerprint = case_fingerprint(case)
    for other in existing:
        if (other.get("case_id") == case.get("case_id") or
            case.get("source_case_id") == other.get("case_id") or
            other.get("source_case_id") == case.get("case_id") or
            (case.get("source_case_id") and case.get("source_case_id") == other.get("source_case_id")) or
            case_fingerprint(other) == fingerprint):
            return {"status": "DUPLICATE_CASE", "matched_case_id": other.get("case_id"), "basis": "case_reference_or_event_fingerprint"}
    cause = case.get("root_cause") or {}
    current_time = _date(case.get("created_at"))
    for other in existing:
        other_cause = other.get("root_cause") or {}
        if (case.get("board_instance_id") and case.get("board_instance_id") == other.get("board_instance_id")
            and case.get("circuit_hash") == other.get("circuit_hash")
            and cause.get("component_ref") == other_cause.get("component_ref")
            and cause.get("fault_type") == other_cause.get("fault_type")):
            other_time = _date(other.get("created_at"))
            if current_time and other_time and abs(current_time - other_time) <= timedelta(days=review_days):
                return {"status": "REVIEW_REQUIRED", "matched_case_id": other.get("case_id"), "basis": "same_board_circuit_cause_within_time_window"}
    return {"status": "UNIQUE", "matched_case_id": None, "basis": "no_exact_or_suspected_duplicate"}


def compatible_cases(cases: list[dict[str, Any]], *, group_id: str, fault_types: set[str]) -> list[dict[str, Any]]:
    return [case for case in cases if case.get("circuit_family") == group_id
            and (case.get("root_cause") or {}).get("fault_type") in fault_types]


def _measurement_signature(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return sorted((row.get("node"), row.get("value"), row.get("unit"),
                   json.dumps(row.get("conditions", {}), sort_keys=True), row.get("source"))
                  for row in rows if isinstance(row, dict))


def _date(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None
    except (ValueError, TypeError):
        return None
