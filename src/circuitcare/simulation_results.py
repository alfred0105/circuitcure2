"""Align and compare successful baseline and fault transient simulations."""

import csv
import json
from bisect import bisect_left
from pathlib import Path
from typing import Any

from .spice_runner import SimulationRun


def align_time_axes(baseline: SimulationRun, fault: SimulationRun) -> list[float]:
    """Build a shared grid over only the time interval covered by both runs."""
    baseline_time = baseline.columns["time_s"]
    fault_time = fault.columns["time_s"]
    start = max(baseline_time[0], fault_time[0])
    stop = min(baseline_time[-1], fault_time[-1])
    if stop < start:
        raise ValueError("Baseline and fault simulations have no overlapping time interval")
    return sorted({
        value
        for value in (*baseline_time, *fault_time)
        if start <= value <= stop
    })


def interpolate(times: list[float], values: list[float], target: float) -> float:
    """Linearly interpolate one value on a strictly increasing time axis."""
    index = bisect_left(times, target)
    if index < len(times) and times[index] == target:
        return values[index]
    if index == 0 or index >= len(times):
        raise ValueError(f"Cannot interpolate outside simulated time range at {target:g} s")
    left_time, right_time = times[index - 1], times[index]
    left_value, right_value = values[index - 1], values[index]
    fraction = (target - left_time) / (right_time - left_time)
    return left_value + fraction * (right_value - left_value)


def compare_runs(
    baseline: SimulationRun,
    fault: SimulationRun,
    voltage_nodes: list[str],
    output_dir: str | Path,
    *,
    scenario_id: str,
) -> dict[str, Any]:
    """Write synchronized node-voltage traces and compact comparison metrics."""
    times = align_time_axes(baseline, fault)
    if len(times) < 2:
        raise ValueError("The simulations have fewer than two comparable time samples")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "comparison.csv"
    json_path = directory / "comparison.json"
    baseline_times = baseline.columns["time_s"]
    fault_times = fault.columns["time_s"]
    traces: dict[str, dict[str, list[float]]] = {}
    fieldnames = ["time_s"]
    for node in voltage_nodes:
        key = f"voltage:{node}"
        if key not in baseline.columns or key not in fault.columns:
            raise ValueError(f"Node {node!r} was not measured in both simulations")
        baseline_values = [interpolate(baseline_times, baseline.columns[key], time) for time in times]
        fault_values = [interpolate(fault_times, fault.columns[key], time) for time in times]
        delta_values = [fault_value - baseline_value for baseline_value, fault_value in zip(baseline_values, fault_values)]
        traces[node] = {"baseline": baseline_values, "fault": fault_values, "delta": delta_values}
        fieldnames.extend([f"baseline_V({node})_V", f"fault_V({node})_V", f"delta_V({node})_V"])

    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, time in enumerate(times):
            row = {"time_s": time}
            for node, values in traces.items():
                row[f"baseline_V({node})_V"] = values["baseline"][index]
                row[f"fault_V({node})_V"] = values["fault"][index]
                row[f"delta_V({node})_V"] = values["delta"][index]
            writer.writerow(row)

    node_summary = {}
    for node, values in traces.items():
        node_summary[node] = {
            "baseline": _series_summary(values["baseline"]),
            "fault": _series_summary(values["fault"]),
            "delta_fault_minus_baseline": _series_summary(values["delta"]),
        }
    result = {
        "scenario_id": scenario_id,
        "simulation_success": True,
        "analysis_stage": "transient_voltage_comparison",
        "evidence_level": "simulated_electrical_change",
        "damage_assessment": "not_performed",
        "baseline": {"circuit": baseline.circuit_path, "log": baseline.log_path, "csv": baseline.csv_path},
        "fault": {"circuit": fault.circuit_path, "log": fault.log_path, "csv": fault.csv_path},
        "time_range_s": {"start": times[0], "stop": times[-1], "samples": len(times)},
        "voltage_nodes": node_summary,
        "time_series_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    result["comparison_json"] = str(json_path)
    return result


def _series_summary(values: list[float]) -> dict[str, float]:
    return {"minimum": min(values), "maximum": max(values), "last": values[-1]}
