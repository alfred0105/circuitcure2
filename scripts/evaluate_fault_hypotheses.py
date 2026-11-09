#!/usr/bin/env python3
"""Run blinded proposal and separate diagnostic-ranking development evaluations."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from circuitcare.fault_ai.evaluation import (
    build_blind_diagnostic_input,
    score_fault_proposals,
    score_fault_rankings,
)
from circuitcare.fault_ai.fault_catalog import generate_rule_candidates
from circuitcare.fault_ai.hypothesis_validator import validate_hypotheses
from circuitcare.fault_ai.llm_hypothesis import PROMPT_VERSION, propose_hypotheses
from circuitcare.fault_ai.hypothesis_ranking import rank_hypotheses
from circuitcare.fault_ai.simulation_bridge import preflight_candidates, simulate_hypothesis
from circuitcare.local_llm import OllamaClient
from circuitcare.netlist import load_netlist

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="evaluation/fault_hypothesis/manifest.json")
    parser.add_argument("--output", default="evaluation/fault_hypothesis/results")
    parser.add_argument("--run-llm", action="store_true", help="Run the installed local Ollama model.")
    parser.add_argument("--model", default="gemma4:12b")
    parser.add_argument("--llm-runs", type=int, default=1, help="Repeat the blinded proposal call for variability measurement.")
    parser.add_argument("--simulate", action="store_true", help="Explicitly run ngspice for up to 40 candidates per case and run.")
    args = parser.parse_args()
    if args.llm_runs < 1:
        parser.error("--llm-runs must be at least 1")
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    cases = manifest.get("cases", [])
    run_records = []
    model_metadata = None
    local = OllamaClient() if args.run_llm else None
    repeat_count = args.llm_runs if args.run_llm else 1

    for run_index in range(1, repeat_count + 1):
        rule_predictions: dict[str, list[dict]] = {}
        llm_predictions: dict[str, list[dict]] = {}
        combined_predictions: dict[str, list[dict]] = {}
        case_results = []
        for case in cases:
            paths = {key: ROOT / value for key, value in case["circuit"].items()}
            circuit = load_netlist(paths["netlist"])
            mapping = json.loads(paths["mapping"].read_text(encoding="utf-8"))
            probes = json.loads(paths["probes"].read_text(encoding="utf-8"))
            spice_text = paths["spice"].read_text(encoding="utf-8")
            observations = case.get("observations", [])
            rule_raw = generate_rule_candidates(circuit, mapping, spice_text, observations)
            rule_rows = preflight_candidates(rule_raw, component_refs=set(circuit.components),
                                             component_mapping=mapping, spice_text=spice_text)
            llm_rows, llm_info = [], None
            llm_error = None
            if args.run_llm:
                components = [{"ref": item.ref, "value": item.value, "library_id": item.library_id,
                               "nets": list(circuit.component_nets.get(item.ref, ())),
                               "mapping_available": item.ref in mapping,
                               "model_type": mapping.get(item.ref, {}).get("model_type")}
                              for item in circuit.components.values()]
                public_input = build_blind_diagnostic_input(case, components=components,
                    observations=observations, executable_catalog=rule_rows)
                try:
                    raw = propose_hypotheses(local, args.model,
                        circuit_id=public_input["circuit_id"], components=public_input["components"],
                        observations=public_input["observations"], symptoms=public_input["symptoms"],
                        executable_catalog=rule_rows, include_candidate_parameters=False)
                    llm_rows = validate_hypotheses(raw["hypotheses"], component_refs=set(circuit.components),
                        component_mapping=mapping, spice_text=spice_text, allowed_catalog=rule_rows)
                    llm_info = {key: raw.get(key) for key in ("model", "model_info", "prompt_version", "latency_seconds", "json_mode")}
                    model_metadata = raw.get("model_info")
                except (ValueError, RuntimeError, OSError) as exc:
                    llm_error = str(exc)[:500]
                    llm_info = {"status": "FAILED", "error": llm_error,
                                "prompt_version": PROMPT_VERSION}

            combined_rows = _dedupe([*rule_rows, *llm_rows])
            if args.simulate:
                for row in combined_rows:
                    if row.get("validation_status") == "VALIDATED_EXECUTABLE":
                        if sum(candidate.get("simulation_status") in {"SIMULATION_SUCCESS", "SIMULATION_FAILED"}
                               for candidate in combined_rows) >= 40:
                            row.update({"simulation_status": "SIMULATION_NOT_RUN",
                                        "simulation_error": "Per-case simulation cap of 40 candidates reached."})
                            continue
                        row.update(simulate_hypothesis(row, base_netlist=paths["spice"], component_mapping=mapping,
                            probes=probes, observations=observations, timeout_seconds=60))

            rule_keys = {_identity(row) for row in rule_rows}
            llm_keys = {_identity(row) for row in llm_rows}
            rule_ranked = rank_hypotheses([dict(row) for row in combined_rows if _identity(row) in rule_keys])
            llm_ranked = rank_hypotheses([dict(row) for row in combined_rows if _identity(row) in llm_keys])
            combined_ranked = rank_hypotheses([dict(row) for row in combined_rows])
            case_id = case["case_id"]  # Scoring-only: never passed to the proposal call.
            rule_predictions[case_id] = rule_ranked
            llm_predictions[case_id] = llm_ranked
            combined_predictions[case_id] = combined_ranked
            case_results.append({"case_id": case_id, "llm_case_id": case.get("llm_case_id"),
                "truth_source": case["truth_source"], "split": case["split"],
                "llm_input_audit": "ALLOWLISTED_ANSWER_FREE_PAYLOAD" if args.run_llm else "NOT_RUN",
                "rule_candidate_count": len(rule_rows), "llm_candidate_count": len(llm_rows),
                "llm_generation": llm_info, "llm_error": llm_error,
                "rule_candidates": rule_ranked, "llm_candidates": llm_ranked,
                "combined_ranked": combined_ranked})

        methods = {
            "A_rule_based": _method_metrics(cases, rule_predictions),
            "B_local_llm_only": _method_metrics(cases, llm_predictions),
            "C_rules_llm_validator": _method_metrics(cases, combined_predictions),
        }
        llm_times = [item["llm_generation"]["latency_seconds"] for item in case_results
                     if isinstance(item.get("llm_generation", {}).get("latency_seconds"), (int, float))]
        for name in ("B_local_llm_only", "C_rules_llm_validator"):
            methods[name]["inference_latency_seconds"] = llm_times
            methods[name]["mean_inference_latency_seconds"] = sum(llm_times) / len(llm_times) if llm_times else None
        run_records.append({"run_id": f"run_{run_index:02d}", "methods": methods,
                            "case_results": case_results,
                            "llm_failures": sum(row.get("llm_error") is not None for row in case_results)})

    result = {
        "evaluation_status": "LEAKAGE_FREE_DEVELOPMENT_RESULT" if args.run_llm else "RULE_BASED_DEVELOPMENT_RESULT",
        "status": "MEASURED_SYNTHETIC" if args.simulate else "PROPOSAL_ONLY",
        "input_leakage_audit": {"status": "PASSED_BY_ALLOWLIST_CONSTRUCTION" if args.run_llm else "NOT_RUN",
            "tests": "tests/test_evaluation_integrity.py", "private_ground_truth_passed_to_llm": False,
            "internal_case_id_passed_to_llm": False, "case_paths_passed_to_llm": False,
            "fault_parameter_candidates_passed_to_llm": False,
            "observations_retained": True},
        "dataset_id": manifest.get("dataset_id"), "dataset_version": manifest.get("version"),
        "split": manifest.get("split"), "independent_holdout": manifest.get("independent_holdout", False),
        "run_metadata": {"git_commit_sha": _git_sha(), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "os": platform.platform(), "llm": model_metadata,
            "model_requested": args.model if args.run_llm else None,
            "prompt_version": PROMPT_VERSION if args.run_llm else None,
            "seed": None, "generation_options": None,
            "simulator": _ngspice_version() if args.simulate else None,
            "simulate": args.simulate, "llm_enabled": args.run_llm,
            "repeat_count": repeat_count},
        "methods": _aggregate_methods(run_records, run_llm=args.run_llm),
        "runs": run_records,
        "holdout_status": "INDEPENDENT_HOLDOUT_NOT_EVALUATED",
        "interpretation": "All labeled cases are previously used, simulation-injected development examples. Blinding removes prompt leakage but does not make this dataset independent or representative of real repairs.",
    }
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    _archive_leaked_results(output)
    output_name = "leakage_free_development_metrics.json" if args.run_llm else "rule_only_development_metrics.json"
    target = output / output_name
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.run_llm:
        (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_method_outputs(output, result)
    print(json.dumps({"output": str(target.relative_to(ROOT)), "evaluation_status": result["evaluation_status"],
                      "repeat_count": repeat_count, "methods": result["methods"]}, ensure_ascii=False, indent=2))
    return 0


def _method_metrics(cases, predictions):
    rows = [item for case in cases for item in predictions.get(case["case_id"], [])]
    attempted = [item for item in rows if item.get("simulation_status") in {"SIMULATION_SUCCESS", "SIMULATION_FAILED"}]
    eligible = [item for item in rows if item.get("validation_status") == "VALIDATED_EXECUTABLE"]
    successes = sum(item.get("simulation_status") == "SIMULATION_SUCCESS" for item in attempted)
    return {"proposal_quality": score_fault_proposals(cases, predictions),
            "diagnostic_ranking": score_fault_rankings(cases, predictions),
            "simulation": {"candidate_count": len(rows), "eligible_count": len(eligible),
                "unsupported_candidate_count": len(rows) - len(eligible), "attempted_count": len(attempted),
                "success_count": successes, "failed_count": len(attempted) - successes,
                "eligible_not_run_count": len(eligible) - len(attempted),
                "success_rate_of_attempted": successes / len(attempted) if attempted else None,
                "status": "MEASURED" if attempted else "NOT_EVALUATED"}}


def _aggregate_methods(run_records, *, run_llm: bool):
    names = ("A_rule_based", "B_local_llm_only", "C_rules_llm_validator")
    result = {}
    for name in names:
        records = [row["methods"][name] for row in run_records]
        result[name] = {"run_count": len(records), "runs": records}
        latencies = [value for row in records for value in row.get("inference_latency_seconds", [])]
        result[name]["inference_latency_seconds"] = latencies
        result[name]["mean_inference_latency_seconds"] = sum(latencies) / len(latencies) if latencies else None
        for section in ("proposal_quality", "diagnostic_ranking"):
            metrics = [row[section] for row in records]
            for key in ("candidate_count", "candidate_precision", "candidate_recall", "top_k_accuracy", "precision_at_k", "ranking_coverage"):
                vals = [metric.get(key) for metric in metrics if isinstance(metric.get(key), (int, float))]
                if vals:
                    result[name].setdefault("variability", {}).setdefault(section, {})[key] = {
                        "min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals)}
        sim_rows = [row["simulation"] for row in records]
        attempted = sum(row["attempted_count"] for row in sim_rows)
        successes = sum(row["success_count"] for row in sim_rows)
        result[name]["simulation"] = {"candidate_count": sum(row["candidate_count"] for row in sim_rows),
            "eligible_count": sum(row["eligible_count"] for row in sim_rows),
            "unsupported_candidate_count": sum(row["unsupported_candidate_count"] for row in sim_rows),
            "attempted_count": attempted, "success_count": successes,
            "failed_count": sum(row["failed_count"] for row in sim_rows),
            "eligible_not_run_count": sum(row["eligible_not_run_count"] for row in sim_rows),
            "success_rate_of_attempted": successes / attempted if attempted else None,
            "status": "MEASURED" if attempted else "NOT_EVALUATED"}
        if name == "A_rule_based" and run_llm and len(records) > 1:
            result[name]["note"] = "Rule inputs are invariant across paired LLM repetitions; metrics repeated as a run-level pairing record."
    return result


def _write_method_outputs(output: Path, result: dict):
    last = result["runs"][-1] if result["runs"] else None
    if not last:
        return
    for filename, method in (("rule_baseline.json", "A_rule_based"),
                             ("llm_candidates.json", "B_local_llm_only"),
                             ("validated_candidates.json", "C_rules_llm_validator")):
        run_method = last["methods"][method]
        cases = {row["case_id"]: row["rule_candidates" if method == "A_rule_based" else "llm_candidates" if method == "B_local_llm_only" else "combined_ranked"]
                 for row in last["case_results"]}
        (output / filename).write_text(json.dumps({"evaluation_status": result["evaluation_status"],
            "run_metadata": result["run_metadata"], "method": method, "metrics": run_method, "cases": cases},
            ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "simulation_verification.json").write_text(json.dumps({
        "evaluation_status": result["evaluation_status"], "run_metadata": result["run_metadata"],
        "runs": [{"run_id": run["run_id"], "cases": {row["case_id"]: [
            {"hypothesis_id": candidate.get("hypothesis_id"), "simulation_status": candidate.get("simulation_status"),
             "simulation_seconds": candidate.get("simulation_seconds"),
             "predicted_measurements": candidate.get("predicted_measurements"),
             "observation_match": candidate.get("observation_match"),
             "simulation_error": candidate.get("simulation_error")}
            for candidate in row["combined_ranked"]] for row in run["case_results"]}}
            for run in result["runs"]]}, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _archive_leaked_results(output: Path) -> None:
    metrics = output / "metrics.json"
    archive = output / "leaked_development_result"
    if not metrics.is_file() or archive.exists():
        return
    prior = json.loads(metrics.read_text(encoding="utf-8"))
    if prior.get("evaluation_status") == "LEAKAGE_FREE_DEVELOPMENT_RESULT":
        return
    archive.mkdir()
    for name in ("metrics.json", "rule_baseline.json", "llm_candidates.json",
                 "validated_candidates.json", "simulation_verification.json"):
        source = output / name
        if not source.is_file():
            continue
        item = json.loads(source.read_text(encoding="utf-8"))
        item["evaluation_status"] = "LEAKED_DEVELOPMENT_RESULT"
        item["integrity_note"] = ("The prior prompt included an answer-bearing case ID and a 'known injected fault' symptom. "
                                  "Retained for audit only; not valid blinded proposal or ranking performance.")
        (archive / name).write_text(json.dumps(item, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _dedupe(rows):
    result, seen = [], set()
    for row in rows:
        key = _identity(row)
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(row))
    return result


def _identity(row):
    normalized = {}
    for key, value in row.get("parameters", {}).items():
        try:
            normalized[key] = f"{float(value):.12g}"
        except (TypeError, ValueError):
            normalized[key] = str(value).strip().casefold()
    return (row.get("component_ref"), row.get("fault_type"), json.dumps(normalized, sort_keys=True))


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _ngspice_version():
    try:
        return subprocess.run(["ngspice", "--version"], check=False, capture_output=True, text=True, timeout=5).stdout[-500:]
    except (OSError, subprocess.SubprocessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
