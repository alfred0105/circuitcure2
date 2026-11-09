"""Validation and provenance checks for repair cases."""

from __future__ import annotations

import math
from typing import Any

from .models import CASE_SOURCES, CASE_STATUSES, OUTCOMES, VERIFICATION_STATUSES

_UNITS = {"V", "mV", "A", "mA", "uA", "W", "mW", "Ohm", "kOhm", "MOhm", "s", "ms", "Hz", "kHz", "°C", "degC", "%"}


def validate_case(case: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for key in ("case_id", "circuit_id", "circuit_hash"):
        if not isinstance(case.get(key), str) or not case[key].strip():
            issues.append(f"{key} is required.")
    if case.get("schema_version") != 1:
        issues.append("Unsupported repair case schema_version.")
    if case.get("data_source") not in CASE_SOURCES:
        issues.append("data_source is not a supported provenance value.")
    if case.get("case_status") not in CASE_STATUSES:
        issues.append("case_status is not supported.")
    if case.get("verification_status") not in VERIFICATION_STATUSES:
        issues.append("verification_status is not supported.")
    if case.get("outcome") not in OUTCOMES:
        issues.append("outcome is not supported.")
    for key in ("reported_symptoms", "observations", "interventions", "post_repair_tests"):
        if not isinstance(case.get(key, []), list):
            issues.append(f"{key} must be a list.")
    observations = case.get("observations", [])
    observations = observations if isinstance(observations, list) else []
    for index, row in enumerate(observations):
        if not isinstance(row, dict):
            issues.append(f"observation {index} must be an object.")
            continue
        if not str(row.get("node", "")).strip() or not str(row.get("unit", "")).strip():
            issues.append(f"observation {index} requires node and unit.")
        elif row.get("unit") not in _UNITS:
            issues.append(f"observation {index} uses an unsupported unit.")
        for value_key in ("value", "uncertainty"):
            value = row.get(value_key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                issues.append(f"observation {index} {value_key} must be finite numeric.")
            elif value_key == "uncertainty" and value <= 0:
                issues.append(f"observation {index} uncertainty must be positive.")
        if row.get("source") not in {"user_measurement", "simulated_demo", "simulation", "user_reported", "instrument_recorded", "unknown"}:
            issues.append(f"observation {index} requires an explicit recognized source.")
    post_tests = case.get("post_repair_tests", [])
    post_tests = post_tests if isinstance(post_tests, list) else []
    for test_index, test in enumerate(post_tests):
        if not isinstance(test, dict) or not isinstance(test.get("passed"), bool):
            issues.append(f"post_repair_test {test_index} requires an explicit boolean passed value.")
            continue
        if not isinstance(test.get("observations", []), list):
            issues.append(f"post_repair_test {test_index} observations must be a list.")
        for measurement in test.get("observations", []):
            if not isinstance(measurement, dict):
                issues.append(f"post_repair_test {test_index} has a non-object measurement.")
                continue
            if not str(measurement.get("node", "")).strip() or measurement.get("unit") not in _UNITS:
                issues.append(f"post_repair_test {test_index} measurements require a node and supported unit.")
            if measurement.get("source") not in {"user_measurement", "simulated_demo", "simulation", "user_reported", "instrument_recorded", "unknown"}:
                issues.append(f"post_repair_test {test_index} measurement requires a recognized source.")
            for key in ("value", "uncertainty"):
                value = measurement.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (key == "uncertainty" and value <= 0):
                    issues.append(f"post_repair_test {test_index} measurement {key} must be finite and valid.")
    known_refs = case.get("component_refs")
    if known_refs is not None:
        if not isinstance(known_refs, list) or any(not isinstance(ref, str) for ref in known_refs):
            issues.append("component_refs must be a list of known reference strings.")
        else:
            known = set(known_refs)
            used_refs = [row.get("component_ref") for row in case.get("interventions", []) if isinstance(row, dict) and row.get("component_ref")]
            root_cause = case.get("root_cause")
            cause_ref = root_cause.get("component_ref") if isinstance(root_cause, dict) else None
            if cause_ref:
                used_refs.append(cause_ref)
            if any(ref not in known for ref in used_refs):
                issues.append("An intervention or reviewed root cause uses a Reference absent from the circuit.")
    if case.get("verification_status") == "ROOT_CAUSE_REVIEWED":
        cause = case.get("root_cause")
        if not isinstance(cause, dict) or not cause.get("component_ref") or not cause.get("fault_type") or cause.get("verified") is not True:
            issues.append("ROOT_CAUSE_REVIEWED requires an explicitly verified root_cause component and fault type.")
    if case.get("training_eligibility", {}).get("status") == "ELIGIBLE":
        cause = case.get("root_cause") or {}
        if (case.get("data_source") != "real_repair" or case.get("verification_status") != "ROOT_CAUSE_REVIEWED"
            or not (case.get("training_review") or {}).get("approved") or not isinstance(cause, dict) or cause.get("verified") is not True):
            issues.append("Only a reviewed real_repair case can be marked training eligible.")
    return issues


def assert_valid_case(case: dict[str, Any]) -> None:
    issues = validate_case(case)
    if issues:
        raise ValueError("Invalid repair case: " + " ".join(issues))
