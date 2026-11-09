"""Run allowlisted fault scenarios through the existing injector and ngspice runner."""

import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from circuitcare.fault_injector import inject_fault
from circuitcare.fault_models import create_scenario
from circuitcare.spice_runner import SimulationError, ngspice_available, run_ngspice

from .observation_matcher import compare_observations


def preflight_candidates(candidates: list[dict[str, Any]], *, component_refs: set[str],
                         component_mapping: dict[str, dict[str, Any]], spice_text: str) -> list[dict[str, Any]]:
    from .hypothesis_validator import validate_hypotheses
    return validate_hypotheses(candidates, component_refs=component_refs,
                               component_mapping=component_mapping, spice_text=spice_text)


def simulate_hypothesis(hypothesis: dict[str, Any], *, base_netlist: str | Path,
                        component_mapping: dict[str, dict[str, Any]], probes: dict[str, Any],
                        observations: list[dict[str, Any]], timeout_seconds: int = 60,
                        executable: str = "ngspice") -> dict[str, Any]:
    row = dict(hypothesis)
    if row.get("validation_status") != "VALIDATED_EXECUTABLE":
        row.update(simulation_status="UNSUPPORTED_HYPOTHESIS", simulation_error="Hypothesis did not pass preflight.")
        return row
    if not ngspice_available(executable):
        row.update(simulation_status="SIMULATION_UNAVAILABLE", simulation_error=f"ngspice {executable!r} is unavailable.")
        return row
    base_path = Path(base_netlist)
    baseline = base_path.read_text(encoding="utf-8")
    scenario = create_scenario(row["component_ref"], row["fault_type"], row.get("parameters", {}))
    injected = inject_fault(baseline, scenario, component_mapping)
    key_source = json.dumps({"baseline": hashlib.sha256(baseline.encode()).hexdigest(),
                             "fault": scenario.to_dict(), "probes": probes}, sort_keys=True)
    result_key = hashlib.sha256(key_source.encode()).hexdigest()
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="circuitcare-fault-ai-") as directory:
            target = Path(directory)
            faulted_path = target / "faulted.cir"
            faulted_path.write_text(injected.faulted_netlist, encoding="utf-8")
            run = run_ngspice(faulted_path, probes, component_mapping, target / "spice", "hypothesis",
                              executable=executable, timeout_seconds=timeout_seconds)
            predictions = {}
            for item in observations:
                if item.get("measurement_mode", "final_sample") != "final_sample":
                    continue
                column = f"voltage:{item.get('node')}"
                values = run.columns.get(column)
                if values:
                    predictions[str(item["node"])] = float(values[-1])
            match = compare_observations(predictions, observations)
            row.update(simulation_status="SIMULATION_SUCCESS", simulation_result_key=result_key,
                       simulation_log_summary="Transient ngspice run succeeded; data was kept in a temporary directory.",
                       prediction_method="last transient sample for final_sample observations",
                       predicted_measurements=predictions, unsupported_probes=run.unsupported_currents,
                       observation_match=match, simulation_seconds=round(time.perf_counter() - started, 6))
    except (OSError, ValueError, SimulationError) as exc:
        row.update(simulation_status="SIMULATION_FAILED", simulation_error=str(exc)[:800],
                   simulation_seconds=round(time.perf_counter() - started, 6),
                   simulation_result_key=result_key)
    return row


def simulate_selected(hypotheses: list[dict[str, Any]], *, selected_ids: list[str], max_runs: int = 5,
                      **context: Any) -> list[dict[str, Any]]:
    selected = set(selected_ids[:max_runs])
    output = []
    for row in hypotheses:
        if row.get("hypothesis_id") in selected:
            output.append(simulate_hypothesis(row, **context))
        else:
            output.append({**row, "simulation_status": "SIMULATION_NOT_RUN"})
    return output
