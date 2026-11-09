"""Catalog supported fault modes and create deterministic seed hypotheses."""

import re
from typing import Any

from circuitcare.fault_models import SUPPORTED_FAULT_TYPES, create_scenario
from circuitcare.models import Circuit

FAULT_MODELS = {
    "resistor_open": "resistor",
    "resistor_short": "resistor",
    "resistance_drift": "resistor",
    "regulator_output_rise": "regulator_output",
    "regulator_output_drop": "regulator_output",
    "regulator_output_loss": "regulator_output",
    "stuck_open": "switch",
    "stuck_on": "switch",
}


def generate_rule_candidates(circuit: Circuit, component_mapping: dict[str, dict[str, Any]],
                             spice_text: str, observations: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Generate only modes compatible with explicit existing SPICE mappings."""
    observation_nodes = {str(row.get("node", "")) for row in observations or []}
    result = []
    for reference, mapping in sorted(component_mapping.items()):
        if reference not in circuit.components or not isinstance(mapping, dict):
            continue
        model_type = mapping.get("model_type")
        component = circuit.components[reference]
        for fault_type in sorted(SUPPORTED_FAULT_TYPES):
            if FAULT_MODELS[fault_type] != model_type:
                continue
            try:
                parameter_options = _parameter_options(fault_type, mapping, spice_text)
            except (ValueError, KeyError):
                continue
            for parameters in parameter_options:
                suffix = "_".join(f"{k}_{v}" for k, v in sorted(parameters.items())) or "default"
                scenario = create_scenario(reference, fault_type, parameters,
                                           scenario_id=f"FAULT_{reference}_{fault_type.upper()}_{suffix}")
                nets = sorted(set(circuit.component_nets.get(reference, ())) & observation_nodes) if observation_nodes else []
                result.append({
                    "hypothesis_id": scenario.scenario_id,
                    "component_ref": reference,
                    "component_value": component.value,
                    "library_id": component.library_id,
                    "nets": list(circuit.component_nets.get(reference, ())),
                    "fault_type": fault_type,
                    "parameters": parameters,
                    "reason": "Catalog candidate from an explicit compatible SPICE mapping; connectivity is context only.",
                    "supporting_observations": sorted(observation_nodes),
                    "generation_method": "catalog_rule",
                    "status": "proposed",
                    "simulation_status": "SIMULATION_NOT_RUN",
                    "simulation_supported": None,
                    "preflight_status": "NOT_CHECKED",
                })
    return result


def _parameter_options(fault_type: str, mapping: dict[str, Any], spice_text: str) -> list[dict[str, Any]]:
    if fault_type in {"resistor_open", "resistor_short", "regulator_output_loss"}:
        return [{}]
    if fault_type in {"stuck_open", "stuck_on"}:
        required = ("control_source", "switch_threshold", "open_control_voltage", "closed_control_voltage")
        if any(key not in mapping for key in required):
            raise ValueError("Switch mapping is incomplete")
        return [{}]
    element = re.escape(str(mapping.get("spice_element", "")))
    line = next((line for line in spice_text.splitlines()
                 if re.match(rf"\s*{element}\s+", line, flags=re.IGNORECASE)), None)
    if not line:
        raise ValueError("Mapped SPICE element is missing")
    tokens = line.split()
    if fault_type == "resistance_drift":
        if len(tokens) < 4:
            raise ValueError("Malformed resistor")
        baseline = _spice_number(tokens[3])
        return [{"fault_resistance": _format_spice(baseline * factor)} for factor in (2.0, 0.5, 10.0, 0.1)]
    if len(tokens) < 5 or tokens[3].casefold() != "dc":
        raise ValueError("Regulator requires an explicit DC setpoint")
    baseline = _spice_number(tokens[4])
    if fault_type == "regulator_output_rise":
        return [{"output_voltage": _format_spice(baseline + delta)} for delta in (1.0, 3.0)]
    if fault_type == "regulator_output_drop":
        if baseline <= 0.1:
            raise ValueError("No positive regulator drop setpoint is available")
        return [{"output_voltage": _format_spice(max(0.1, baseline - delta))} for delta in (1.0, 2.0)]
    return [{}]


def _spice_number(value: str) -> float:
    match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]*)\s*", value)
    if not match:
        raise ValueError(f"Unsupported SPICE numeric value {value!r}")
    suffix = match.group(2).upper()
    scales = {"": 1.0, "T": 1e12, "G": 1e9, "MEG": 1e6, "K": 1e3,
              "M": 1e-3, "U": 1e-6, "N": 1e-9, "P": 1e-12, "F": 1e-15}
    if suffix not in scales:
        raise ValueError(f"Unsupported SPICE suffix {suffix!r}")
    return float(match.group(1)) * scales[suffix]


def _format_spice(value: float) -> str:
    if value <= 0:
        raise ValueError("Fault parameter must be positive")
    return f"{value:.12g}"
