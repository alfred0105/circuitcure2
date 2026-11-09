"""Fail-closed validation before any generated hypothesis reaches SPICE."""

import hashlib
import json
from typing import Any

from circuitcare.fault_injector import inject_fault
from circuitcare.fault_models import SUPPORTED_FAULT_TYPES, create_scenario

_PARAMETERS = {
    "resistor_open": set(), "resistor_short": set(), "resistance_drift": {"fault_resistance"},
    "regulator_output_rise": {"output_voltage"}, "regulator_output_drop": {"output_voltage"},
    "regulator_output_loss": set(), "stuck_open": set(), "stuck_on": set(),
}


def validate_hypothesis(candidate: dict[str, Any], *, component_refs: set[str],
                        component_mapping: dict[str, dict[str, Any]], spice_text: str,
                        allowed_catalog: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    row = dict(candidate)
    reasons: list[str] = []
    ref, fault_type = row.get("component_ref"), row.get("fault_type")
    if not isinstance(ref, str) or ref not in component_refs:
        status = "INVALID_REFERENCE"
        reasons.append("Reference is absent from the loaded circuit.")
    elif not isinstance(fault_type, str) or fault_type not in SUPPORTED_FAULT_TYPES:
        status = "UNSUPPORTED_FAULT_TYPE"
        reasons.append("Fault type is not supported by the existing Fault Injector.")
    else:
        mapping = component_mapping.get(ref)
        if not isinstance(mapping, dict):
            status = "MISSING_SPICE_MAPPING"
            reasons.append("No explicit KiCad-to-SPICE mapping exists for this component.")
        else:
            params = row.get("parameters", {})
            if not isinstance(params, dict) or set(params) - _PARAMETERS[fault_type] or _PARAMETERS[fault_type] - set(params):
                status = "INVALID_PARAMETERS"
                reasons.append("Parameters do not match the allowlisted schema for this fault type.")
            elif allowed_catalog is not None and not any(
                item.get("component_ref") == ref and item.get("fault_type") == fault_type
                and _parameters_match(item.get("parameters", {}), params)
                and item.get("validation_status") == "VALIDATED_EXECUTABLE"
                for item in allowed_catalog
            ):
                status = "INSUFFICIENT_EVIDENCE"
                reasons.append("The exact component, fault mode, and parameters are absent from the preflighted catalog.")
            else:
                try:
                    scenario = create_scenario(ref, fault_type, params)
                    injected = inject_fault(spice_text, scenario, component_mapping)
                    if not injected.changes:
                        raise ValueError("Fault Injector produced no netlist change")
                except (ValueError, KeyError, TypeError) as exc:
                    status = "INCOMPATIBLE_MODEL"
                    reasons.append(str(exc))
                else:
                    status = "VALIDATED_EXECUTABLE"
    row.update(validation_status=status, validation_reasons=reasons,
               simulation_supported=status == "VALIDATED_EXECUTABLE",
               simulation_status="SIMULATION_NOT_RUN")
    if status == "VALIDATED_EXECUTABLE":
        stable = json.dumps({"ref": ref, "fault_type": fault_type, "parameters": row.get("parameters", {})}, sort_keys=True)
        row["hypothesis_id"] = "H_" + hashlib.sha256(stable.encode()).hexdigest()[:12].upper()
    return row


def validate_hypotheses(candidates: list[dict[str, Any]], **context: Any) -> list[dict[str, Any]]:
    results = []
    seen = set()
    for candidate in candidates:
        row = validate_hypothesis(candidate, **context)
        key = (row.get("component_ref"), row.get("fault_type"), _canonical_parameters(row.get("parameters", {})))
        if key in seen:
            row.update(validation_status="DUPLICATE_HYPOTHESIS", simulation_supported=False,
                       validation_reasons=["Equivalent hypothesis already exists."])
        else:
            seen.add(key)
        results.append(row)
    return results


def _parameters_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    if set(expected) != set(actual):
        return False
    for key, value in expected.items():
        candidate = actual[key]
        try:
            if float(value) == float(candidate):
                continue
        except (TypeError, ValueError):
            pass
        if str(value).strip().casefold() != str(candidate).strip().casefold():
            return False
    return True


def _canonical_parameters(params: dict[str, Any]) -> str:
    normalized = {}
    for key, value in params.items():
        try:
            normalized[key] = f"{float(value):.12g}"
        except (TypeError, ValueError):
            normalized[key] = str(value).strip().casefold()
    return json.dumps(normalized, sort_keys=True)
