"""Compute supported per-component electrical values from measured SPICE data."""

import csv
import json
from pathlib import Path
from typing import Any

from .fault_injector import _spice_number
from .simulation_results import align_time_axes, interpolate
from .spice_runner import SimulationRun


def analyze_component_stress(
    baseline: SimulationRun,
    fault: SimulationRun,
    baseline_circuit: str | Path,
    fault_circuit: str | Path,
    probes: dict[str, Any],
    component_mapping: dict[str, dict[str, Any]],
    output_dir: str | Path,
    *,
    fault_type: str,
) -> dict[str, Any]:
    """Calculate resistor and mapped ideal-voltage-source values; mark other models unavailable."""
    baseline_text = Path(baseline_circuit).read_text(encoding="utf-8")
    fault_text = Path(fault_circuit).read_text(encoding="utf-8")
    times = align_time_axes(baseline, fault)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    series: dict[str, dict[str, list[float] | None]] = {}

    for reference in probes.get("components", []):
        mapping = component_mapping.get(reference)
        if not isinstance(mapping, dict):
            records.append(_unavailable(reference, "No explicit KiCad-to-SPICE mapping is available."))
            continue
        element = str(mapping.get("spice_element", "")).strip()
        model_type = mapping.get("model_type")
        if model_type == "resistor":
            record, values = _analyze_resistor(reference, element, baseline_text, fault_text, baseline, fault, times)
        elif model_type == "capacitor":
            record, values = _analyze_capacitor(reference, element, baseline_text, fault_text, baseline, fault, times)
        elif model_type == "regulator_output":
            record, values = _analyze_voltage_source(
                reference, element, baseline_text, fault_text, baseline, fault, times, fault_type
            )
        elif model_type == "switch":
            record, values = _analyze_switch(reference, element, baseline_text, fault_text, baseline, fault, times)
        else:
            records.append(_unavailable(reference, f"No stress calculation is supported for model type {model_type!r}."))
            continue
        records.append(record)
        series[reference] = values

    csv_path = directory / "component_stress.csv"
    json_path = directory / "component_stress.json"
    fieldnames = ["time_s"]
    for reference in series:
        for metric in ("voltage_v", "current_a", "power_w"):
            fieldnames.extend([
                f"{reference}_{metric}_baseline",
                f"{reference}_{metric}_fault",
                f"{reference}_{metric}_delta",
            ])
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, time in enumerate(times):
            row: dict[str, Any] = {"time_s": time}
            for reference, metrics in series.items():
                for metric, values in metrics.items():
                    baseline_values = values["baseline"]
                    fault_values = values["fault"]
                    row[f"{reference}_{metric}_baseline"] = None if baseline_values is None else baseline_values[index]
                    row[f"{reference}_{metric}_fault"] = None if fault_values is None else fault_values[index]
                    row[f"{reference}_{metric}_delta"] = (
                        None if baseline_values is None or fault_values is None
                        else fault_values[index] - baseline_values[index]
                    )
            writer.writerow(row)

    result = {
        "fault_type": fault_type,
        "simulation_success": True,
        "power_sign_convention": "Positive power is absorbed by the component; voltage-source delivery is negative.",
        "components": records,
        "time_series_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    result["component_stress_json"] = str(json_path)
    return result


def _analyze_resistor(reference, element, baseline_text, fault_text, baseline, fault, times):
    before = _element_tokens(baseline_text, element)
    after = _element_tokens(fault_text, element)
    if len(before) < 4 or len(after) < 4 or before[0][0].upper() != "R" or after[0][0].upper() != "R":
        return _unavailable(reference, f"Mapped resistor element {element!r} is missing or unsupported."), _empty_series()
    if [node.casefold() for node in before[1:3]] != [node.casefold() for node in after[1:3]]:
        return _unavailable(reference, "The mapped resistor terminals differ between baseline and fault circuits."), _empty_series()
    try:
        baseline_resistance = _spice_number(before[3])
        fault_resistance = _spice_number(after[3])
        if baseline_resistance <= 0 or fault_resistance <= 0:
            raise ValueError("resistance must be positive")
    except ValueError as exc:
        return _unavailable(reference, f"Resistor value cannot be evaluated: {exc}"), _empty_series()

    baseline_voltage = _terminal_voltage(before[1], before[2], baseline, times)
    fault_voltage = _terminal_voltage(after[1], after[2], fault, times)
    baseline_current = [value / baseline_resistance for value in baseline_voltage]
    fault_current = [value / fault_resistance for value in fault_voltage]
    baseline_power = [value * current for value, current in zip(baseline_voltage, baseline_current)]
    fault_power = [value * current for value, current in zip(fault_voltage, fault_current)]
    values = {
        "voltage_v": {"baseline": baseline_voltage, "fault": fault_voltage},
        "current_a": {"baseline": baseline_current, "fault": fault_current},
        "power_w": {"baseline": baseline_power, "fault": fault_power},
    }
    record = _record(reference, "simulated", values, f"V=V(node1)-V(node2); I=V/R; P=V²/R. R1={before[3]}, R2={after[3]}.")
    record["current_direction"] = f"{before[1]} to {before[2]}"
    return record, values


def _analyze_capacitor(reference, element, baseline_text, fault_text, baseline, fault, times):
    before = _element_tokens(baseline_text, element)
    after = _element_tokens(fault_text, element)
    if len(before) < 4 or len(after) < 4 or before[0][0].upper() != "C" or after[0][0].upper() != "C":
        return _unavailable(reference, f"Mapped capacitor element {element!r} is missing or unsupported."), _empty_series()
    if [node.casefold() for node in before[1:3]] != [node.casefold() for node in after[1:3]]:
        return _unavailable(reference, "The mapped capacitor terminals differ between baseline and fault circuits."), _empty_series()
    baseline_voltage = _terminal_voltage(before[1], before[2], baseline, times)
    fault_voltage = _terminal_voltage(after[1], after[2], fault, times)
    values = {
        "voltage_v": {"baseline": baseline_voltage, "fault": fault_voltage},
        "current_a": {"baseline": None, "fault": None},
        "power_w": {"baseline": None, "fault": None},
    }
    record = _record(
        reference,
        "voltage_only",
        values,
        f"V=V({before[1]})-V({before[2]}); capacitor current, ESR loss, and heating are not modeled.",
    )
    record["unsupported_metrics"] = {
        "current_a": "Capacitor current is not calculated in this phase.",
        "power_w": "Capacitor ESR loss and heating are not modeled.",
    }
    return record, values


def _analyze_voltage_source(reference, element, baseline_text, fault_text, baseline, fault, times, fault_type):
    before = _element_tokens(baseline_text, element)
    if len(before) < 5 or before[0][0].upper() != "V":
        return _unavailable(reference, f"Mapped voltage source {element!r} is missing or unsupported."), _empty_series()
    baseline_voltage = _terminal_voltage(before[1], before[2], baseline, times)
    baseline_current = baseline.columns.get(f"current:{reference}")
    if baseline_current is None:
        return _unavailable(reference, f"ngspice did not return the current for {element}."), _empty_series()
    baseline_current = _resample(baseline, baseline_current, times)
    baseline_power = [voltage * current for voltage, current in zip(baseline_voltage, baseline_current)]

    after = _element_tokens(fault_text, element)
    if after and after[0][0].upper() == "V":
        fault_voltage = _terminal_voltage(after[1], after[2], fault, times)
        fault_current = fault.columns.get(f"current:{reference}")
        if fault_current is None:
            return _unavailable(reference, f"ngspice did not return the fault current for {element}."), _empty_series()
        fault_current = _resample(fault, fault_current, times)
        fault_power = [voltage * current for voltage, current in zip(fault_voltage, fault_current)]
        method = "V=V(node1)-V(node2); I from ngspice source branch current; P=V×I (positive absorbed)."
    elif fault_type == "regulator_output_loss":
        replacement = _element_tokens(fault_text, f"RFAULT_{element}")
        if len(replacement) < 4 or replacement[0][0].upper() != "R":
            return _unavailable(reference, "Output-loss replacement resistor is missing."), _empty_series()
        fault_voltage = _terminal_voltage(replacement[1], replacement[2], fault, times)
        try:
            resistance = _spice_number(replacement[3])
        except ValueError as exc:
            return _unavailable(reference, f"Output-loss resistance is invalid: {exc}"), _empty_series()
        fault_current = [value / resistance for value in fault_voltage]
        fault_power = [value * current for value, current in zip(fault_voltage, fault_current)]
        method = "Baseline I from ngspice; fault I=V/R for the explicit high-resistance output-loss approximation."
    else:
        return _unavailable(reference, f"Mapped voltage source {element} is absent from the fault circuit."), _empty_series()

    values = {
        "voltage_v": {"baseline": baseline_voltage, "fault": fault_voltage},
        "current_a": {"baseline": baseline_current, "fault": fault_current},
        "power_w": {"baseline": baseline_power, "fault": fault_power},
    }
    return _record(reference, "simulated", values, method), values


def _analyze_switch(reference, element, baseline_text, fault_text, baseline, fault, times):
    before = _element_tokens(baseline_text, element)
    after = _element_tokens(fault_text, element)
    if len(before) < 6 or len(after) < 6 or before[0][0].upper() != "S" or after[0][0].upper() != "S":
        return _unavailable(reference, f"Mapped switch {element!r} is missing or unsupported."), _empty_series()
    baseline_voltage = _terminal_voltage(before[1], before[2], baseline, times)
    fault_voltage = _terminal_voltage(after[1], after[2], fault, times)
    values = {
        "voltage_v": {"baseline": baseline_voltage, "fault": fault_voltage},
        "current_a": {"baseline": None, "fault": None},
        "power_w": {"baseline": None, "fault": None},
    }
    record = _record(reference, "measured_voltage_only", values, "Switch voltage is measured; switch current/power are not supported by this analyzer.")
    return record, values


def _terminal_voltage(node_plus, node_minus, run, times):
    plus = [0.0] * len(times) if node_plus == "0" else _resample(run, run.columns[f"voltage:{node_plus}"], times)
    minus = [0.0] * len(times) if node_minus == "0" else _resample(run, run.columns[f"voltage:{node_minus}"], times)
    return [positive - negative for positive, negative in zip(plus, minus)]


def _resample(run: SimulationRun, values: list[float], times: list[float]) -> list[float]:
    return [interpolate(run.columns["time_s"], values, time) for time in times]


def _element_tokens(netlist: str, element: str) -> list[str]:
    for line in netlist.splitlines():
        tokens = line.strip().split()
        if tokens and not tokens[0].startswith("*") and tokens[0].casefold() == element.casefold():
            return tokens
    return []


def _record(reference, status, values, method):
    baseline = {metric: _metric_summary(pair["baseline"]) for metric, pair in values.items()}
    fault = {metric: _metric_summary(pair["fault"]) for metric, pair in values.items()}
    deltas = {}
    for metric, pair in values.items():
        before, after = pair["baseline"], pair["fault"]
        if before is None or after is None:
            deltas[metric] = None
        else:
            difference = [right - left for left, right in zip(before, after)]
            deltas[metric] = _metric_summary(difference)
    return {
        "component_ref": reference,
        "status": status,
        "baseline": baseline,
        "fault": fault,
        "delta_fault_minus_baseline": deltas,
        "calculation_method": method,
    }


def _metric_summary(values):
    if values is None:
        return {"first": None, "last": None, "minimum": None, "maximum": None}
    return {"first": values[0], "last": values[-1], "minimum": min(values), "maximum": max(values)}


def _unavailable(reference, reason):
    empty = {metric: {"first": None, "last": None, "minimum": None, "maximum": None}
             for metric in ("voltage_v", "current_a", "power_w")}
    return {
        "component_ref": reference,
        "status": "unavailable",
        "baseline": empty,
        "fault": empty,
        "delta_fault_minus_baseline": {metric: None for metric in empty},
        "calculation_method": None,
        "reason": reason,
    }


def _empty_series():
    return {
        "voltage_v": {"baseline": None, "fault": None},
        "current_a": {"baseline": None, "fault": None},
        "power_w": {"baseline": None, "fault": None},
    }
