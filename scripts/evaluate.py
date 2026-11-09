"""Small reproducible Phase 09 evaluation for independent circuit topologies."""

from __future__ import annotations

import json
import platform
import re
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from circuitcare.diagnosis_ai import run_diagnosis
from circuitcare.graph import direct_candidates, find_paths
from circuitcare.local_llm import LocalLLMError, OllamaClient
from circuitcare.netlist import load_netlist

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    {"name": "resistor_divider", "netlist": "tests/fixtures/divider_netlist.xml", "fault": "R1", "type": "resistance_drift", "value": "10k", "base": "data/circuits/divider.cir", "mapping": "data/circuits/divider_mapping.json", "probes": "data/circuits/divider_probes.json", "ratings": "data/component_specs/divider_ratings.json", "output": "data/evaluations/divider"},
    {"name": "rc_charging", "netlist": "tests/fixtures/rc_netlist.xml", "fault": "RCHARGE", "type": "resistance_drift", "value": "100k", "base": "data/circuits/rc_charge.cir", "mapping": "data/circuits/rc_charge_mapping.json", "probes": "data/circuits/rc_charge_probes.json", "ratings": "data/component_specs/rc_charge_ratings.json", "output": "data/evaluations/rc_charge"},
]


def timed(function):
    started = time.perf_counter()
    result = function()
    return result, round(time.perf_counter() - started, 6)


def main() -> None:
    records = []
    for case in CASES:
        output = ROOT / case["output"]
        command = [sys.executable, "-m", "circuitcare.cli", "--netlist", case["netlist"], "--fault", case["fault"], "--fault-type", case["type"], "--fault-value", case["value"], "--spice-base", case["base"], "--spice-mapping", case["mapping"], "--output", case["output"], "--simulate", "--probes", case["probes"], "--ratings", case["ratings"], "--assess-risk"]
        cli_started = time.perf_counter()
        proc = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        cli_wall = round(time.perf_counter() - cli_started, 6)
        result = json.loads(proc.stdout)

        circuit, parse_seconds = timed(lambda: load_netlist(ROOT / case["netlist"]))
        candidates, graph_seconds = timed(lambda: find_paths(circuit, case["fault"], max_hops=3))
        direct = direct_candidates(circuit, case["fault"])
        graph_refs = sorted({item.target_ref for item in candidates})
        report = result["fault_scenario"]["simulation_analysis"]
        comparison_path = ROOT / report["comparison_json"]
        risk_path = ROOT / result["fault_scenario"]["risk_assessment"]["risk_assessment_json"]
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        risk = json.loads(risk_path.read_text(encoding="utf-8"))
        physics_rank = _physics_rank(ROOT / report["component_stress_json"], risk)
        expected = _independent_reference(case["name"])
        records.append({
            "circuit": case["name"],
            "topology": "resistor divider" if case["name"] == "resistor_divider" else "first-order RC charge network",
            "distinct_topology": True,
            "fault_injection": {"component": case["fault"], "type": case["type"], "parameter": case["value"]},
            "simulation_engine": "ngspice",
            "simulation_success": comparison.get("simulation_success") is True,
            "graph_only": {"direct_candidate_count": len(direct), "path_candidate_count": len(graph_refs), "candidate_refs": graph_refs, "evidence": "connectivity_only; not physical stress or damage"},
            "physics_only": {"rated_exceeded_refs": [item["component_ref"] for item in risk.get("secondary_candidates", []) if item.get("risk_level") == "EXCEEDED"], "measured_change_rank": physics_rank, "reference_affected_components": expected["affected_components"], "reference_method": expected["method"], "damage_detection_claim": False},
            "measurements": comparison.get("voltage_nodes", {}),
            "timings_seconds": {"netlist_parse": parse_seconds, "graph_paths": graph_seconds, "fault_injection_ngspice_risk_cli_end_to_end": cli_wall},
            "end_to_end_wall_seconds": cli_wall,
            "result_files": {"comparison": str(comparison_path.relative_to(ROOT)), "risk_assessment": str(risk_path.relative_to(ROOT))},
        })

    diagnosis_seconds_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="circuitcare-eval-") as temp_dir:
        bayesian = run_diagnosis(ROOT / "data/diagnosis/demo_hypotheses.json", ROOT / "data/diagnosis/demo_measurements.json", Path(temp_dir) / "posterior")
    bayesian_seconds = time.perf_counter() - diagnosis_seconds_started
    post = bayesian["diagnosis"]
    bayesian_summary = {"circuit": "regulator+divider demo (not counted as an independent held-out topology)", "measurement_source": post["used_measurements"][0].get("source") if post["used_measurements"] else None, "measurement_node": post["used_measurements"][0].get("node") if post["used_measurements"] else None, "posterior_order": [{"hypothesis_id": row["hypothesis_id"], "posterior": row["posterior"]} for row in post["posterior"]], "model_mismatch": post.get("model_mismatch"), "time_seconds": round(bayesian_seconds, 6), "interpretation": "Conditional on configured scenarios/measurement model; not failure probability. Simulated demo measurement is not an instrument measurement."}

    # The persisted Phase 08 responses include Ollama inference durations; we
    # report those recorded values without adding a potentially minute-long call.
    llm = []
    for folder in ("demo_c1", "demo_probability"):
        item = json.loads((ROOT / f"data/repair_advice/{folder}/llm_explanation.json").read_text(encoding="utf-8"))
        inference = item.get("inference", {})
        llm.append({"artifact": f"data/repair_advice/{folder}", "model": item.get("model"), "total_duration_ns": inference.get("total_duration_ns"), "eval_count": inference.get("eval_count"), "wall_time_seconds": _read_wall_time(item)})

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_memory_mb = usage / (1024 * 1024) if platform.system() == "Darwin" else usage / 1024
    summary = {
        "phase": "09",
        "evaluation_type": "small local engineering evaluation; not generalization or damage-accuracy validation",
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "cpu": platform.processor(), "ngspice": _ngspice_version(), "local_ollama_models_detected": _ollama_models(), "peak_process_memory_mb": round(peak_memory_mb, 2), "memory_scope": "evaluation script process peak RSS via resource.getrusage; excludes Ollama/ngspice child process peaks"},
        "independent_circuits": records,
        "graph_only_vs_physics_only": "Graph ranks topological neighbors; physics adds ngspice stress and configured rating comparisons. Candidate sets/targets differ by topology, and neither output is physical damage ground truth.",
        "physics_plus_bayesian": bayesian_summary,
        "local_llm_recorded_runs": llm,
        "limitations": ["Only two small topologies were evaluated.", "Rating limits are explicit demo assumptions, not datasheet verified.", "No real damaged-component labels or field measurements were available; Top-k damage coverage and avoidable inspections are not calculated.", "Bayesian demo is a separate third circuit scenario and uses simulated_demo input.", "No cross-topology accuracy or calibrated failure probability is claimed."],
    }
    destination = ROOT / "data/evaluations/phase09_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(destination.relative_to(ROOT)), "circuits": len(records), "max_memory_mb": summary["environment"]["peak_process_memory_mb"], "diagnosis_seconds": bayesian_summary["time_seconds"]}, indent=2))


def _physics_rank(stress_path: Path, risk: dict) -> list[dict]:
    stress = json.loads(stress_path.read_text(encoding="utf-8"))
    risk_rank = {item.get("component_ref"): item.get("priority_rank") for item in risk.get("secondary_candidates", [])}
    ranked = []
    for component in stress.get("components", []):
        ref = component.get("component_ref")
        deltas = component.get("delta_fault_minus_baseline", {})
        changed = any(isinstance(metric.get("last"), (int, float)) and abs(metric["last"]) > 1e-9 for metric in deltas.values() if metric)
        if changed and ref in risk_rank:
            ranked.append({"component_ref": ref, "risk_rank": risk_rank[ref], "status": component.get("status")})
    return sorted(ranked, key=lambda row: row["risk_rank"])


def _independent_reference(name: str) -> dict:
    if name == "resistor_divider":
        baseline = 5 * 1000 / (1000 + 1000)
        fault = 5 * 1000 / (10000 + 1000)
        return {"affected_components": ["R2"], "method": f"Independent Ohm's law: Vout 5×R2/(R1+R2) changes {baseline:.4f} V → {fault:.4f} V; not inferred from CircuitCare ranking."}
    return {"affected_components": ["C1", "RBLEED"], "method": "Independent first-order RC circuit equation predicts a changed VC transient and bleed branch response after RCHARGE changes from 10 kΩ to 100 kΩ; no real damage labels."}


def _ngspice_version() -> str | None:
    try:
        result = subprocess.run(["ngspice", "--version"], capture_output=True, text=True, timeout=5)
        output = result.stdout or result.stderr
        match = re.search(r"ngspice[- ]\d+(?:\.\d+)*", output, re.IGNORECASE)
        return match.group(0) if match else output.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _ollama_models() -> list[dict]:
    try:
        models = OllamaClient(timeout_seconds=2).list_models()
    except (LocalLLMError, ValueError):
        return []
    return [
        {
            "name": item.get("name"),
            "size_bytes": item.get("size"),
            "runner": (item.get("details") or {}).get("runner"),
            "quantization_level": (item.get("details") or {}).get("quantization_level"),
        }
        for item in models
    ]


def _read_wall_time(item: dict) -> float | None:
    return item.get("wall_time_seconds")


if __name__ == "__main__":
    main()
