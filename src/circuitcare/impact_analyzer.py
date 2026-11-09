"""Run baseline/fault simulations and summarize measured electrical changes."""

import json
from pathlib import Path
from typing import Any

from .graph import direct_candidates
from .models import Circuit
from .simulation_results import compare_runs
from .spice_runner import run_ngspice
from .stress_analyzer import analyze_component_stress


def analyze_fault_impact(
    circuit: Circuit,
    scenario: dict[str, Any],
    baseline_circuit: str | Path,
    fault_circuit: str | Path,
    component_mapping: dict[str, dict[str, Any]],
    probes: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run both decks and write node, component, and impact JSON/CSV artifacts."""
    fault_ref = scenario["component_ref"]
    connected = direct_candidates(circuit, fault_ref)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    baseline_run = run_ngspice(
        baseline_circuit, probes, component_mapping, directory, "baseline"
    )
    fault_run = run_ngspice(
        fault_circuit, probes, component_mapping, directory, "fault"
    )
    comparison = compare_runs(
        baseline_run,
        fault_run,
        list(probes.get("voltage_nodes", [])),
        directory,
        scenario_id=scenario["scenario_id"],
    )
    stress = analyze_component_stress(
        baseline_run,
        fault_run,
        baseline_circuit,
        fault_circuit,
        probes,
        component_mapping,
        directory,
        fault_type=scenario["fault_type"],
    )

    stress_by_ref = {entry["component_ref"]: entry for entry in stress["components"]}
    candidate_reports = []
    for candidate in connected:
        measured = stress_by_ref.get(candidate.target_ref)
        delta = measured.get("delta_fault_minus_baseline", {}) if measured else {}
        is_measured = bool(measured) and measured.get("status") not in {"unavailable", "not_probed"}
        observed = any(
            values is not None
            and any(
                isinstance(values.get(metric), (int, float))
                and abs(values[metric]) > 1e-9
                for metric in ("minimum", "maximum", "last")
            )
            for values in delta.values()
        ) if is_measured else None
        candidate_reports.append({
            "component_ref": candidate.target_ref,
            "shared_nets": list(candidate.shared_nets),
            "probe_status": measured["status"] if measured else "not_probed",
            "electrical_change_observed": observed,
            "stress_record": "component_stress.json" if measured else None,
        })

    report_path = directory / "impact_report.json"
    report = {
        "fault_component": fault_ref,
        "fault_type": scenario["fault_type"],
        "scenario_id": scenario["scenario_id"],
        "analysis_stage": "fault_impact_simulated",
        "simulation_status": "success",
        "simulation_engine": "ngspice",
        "simulation_run_status": {"baseline": "success", "fault": "success"},
        "connected_candidates": candidate_reports,
        "measured_voltage_nodes": list(probes.get("voltage_nodes", [])),
        "comparison_json": comparison["comparison_json"],
        "comparison_csv": comparison["time_series_csv"],
        "component_stress_json": stress["component_stress_json"],
        "component_stress_csv": stress["time_series_csv"],
        "damage_assessment": "not_performed",
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    report["impact_report_json"] = str(report_path)
    return {"report": report, "comparison": comparison, "component_stress": stress}
