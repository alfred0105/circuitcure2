#!/usr/bin/env python3
"""Run explicit rules/LLM/hybrid and optional SPICE comparisons.

The proposal stage receives only the public diagnostic payload. Private truth is
loaded for scoring metadata after each proposal is generated.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import time
from pathlib import Path

from circuitcare.benchmark.dataset import load_public_dataset, load_private_ground_truth
from circuitcare.benchmark.performance import efficiency_metrics
from circuitcare.benchmark.freeze import benchmark_code_fingerprint, benchmark_sources_match_commit
from circuitcare.benchmark.integrity import audit_public_payload
from circuitcare.benchmark.performance import measure_callable
from circuitcare.benchmark.runner import reproducibility_metadata, save_result
from circuitcare.benchmark.scoring import score_hypothesis_levels, score_ranked_hypotheses
from circuitcare.benchmark.statistics import cluster_bootstrap_interval
from circuitcare.fault_ai.evaluation import build_blind_diagnostic_input
from circuitcare.fault_ai.fault_catalog import generate_rule_candidates
from circuitcare.fault_ai.hypothesis_ranking import rank_hypotheses
from circuitcare.fault_ai.hypothesis_validator import validate_hypotheses, validate_hypothesis
from circuitcare.fault_ai.llm_hypothesis import PROMPT_VERSION, propose_hypotheses
from circuitcare.fault_ai.simulation_bridge import preflight_candidates, simulate_hypothesis
from circuitcare.fault_ai.observation_matcher import compare_observations
from circuitcare.local_llm import OllamaClient
from circuitcare.netlist import load_netlist
from circuitcare.spice_runner import ngspice_available

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evaluation/phase17/development")
    parser.add_argument("--results", default="evaluation/phase17/results")
    parser.add_argument("--run-llm", action="store_true")
    parser.add_argument("--model", default="gemma4:12b")
    parser.add_argument("--llm-runs", type=int, default=1)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--spice-budget", type=int, default=40, help="Maximum runs per method per case.")
    parser.add_argument("--reuse-existing-holdout", action="store_true",
                        help="Explicitly reevaluate an immutable old holdout with new metrics; never claim a fresh independent run.")
    args = parser.parse_args()
    benchmark_started = time.perf_counter()
    if args.llm_runs < 1 or args.spice_budget < 0:
        parser.error("--llm-runs must be >= 1 and --spice-budget must be >= 0")
    bundle = load_public_dataset(ROOT / args.dataset)
    reevaluating_existing_holdout = bool(bundle.manifest.get("independent_holdout") and args.reuse_existing_holdout)
    if bundle.manifest.get("independent_holdout") and not args.reuse_existing_holdout:
        expected_fingerprint = bundle.manifest.get("benchmark_code_fingerprint", {}).get("sha256")
        actual_fingerprint = benchmark_code_fingerprint(ROOT)["sha256"]
        if not expected_fingerprint or expected_fingerprint != actual_fingerprint:
            raise SystemExit("Benchmark code changed since holdout freeze; refusing an independent-holdout claim.")
        if bundle.manifest.get("code_freeze_sha") != _git_sha():
            raise SystemExit("Git SHA differs from the holdout code-freeze revision; create a fresh holdout.")
        if not benchmark_sources_match_commit(ROOT, str(bundle.manifest.get("code_freeze_sha"))):
            raise SystemExit("Benchmark sources have changed since the holdout was frozen; refusing evaluation.")
    elif bundle.manifest.get("code_freeze_sha") and not args.reuse_existing_holdout:
        if bundle.manifest["code_freeze_sha"] != _git_sha():
            raise SystemExit("Dataset code-freeze SHA differs from HEAD; refuse benchmark execution.")
        expected_fingerprint = bundle.manifest.get("benchmark_code_fingerprint", {}).get("sha256")
        if expected_fingerprint and expected_fingerprint != benchmark_code_fingerprint(ROOT)["sha256"]:
            raise SystemExit("Dataset benchmark fingerprint differs from the committed code.")
        if not benchmark_sources_match_commit(ROOT, str(bundle.manifest["code_freeze_sha"])):
            raise SystemExit("Benchmark sources changed after dataset freeze.")
    cases = list(bundle.public_cases)
    public_rows = []
    llm_client = OllamaClient() if args.run_llm else None
    model_info = None
    model_repetitions = args.llm_runs if args.run_llm else 1
    raw_method_runs = {name: [] for name in ("Rules Only", "LLM Only", "Hybrid")}
    raw_simulation_runs = {name: [] for name in ("Rules + SPICE", "Hybrid + SPICE")}

    for repetition in range(model_repetitions):
        method_cases = {name: [] for name in raw_method_runs}
        spice_cases = {name: [] for name in raw_simulation_runs}
        for public in cases:
            payload = public["diagnostic_input"]
            audit_public_payload(payload)
            paths = {key: ROOT / value for key, value in public["circuit"].items()}
            circuit = load_netlist(paths["netlist"])
            mapping = json.loads(paths["mapping"].read_text(encoding="utf-8"))
            spice_text = paths["spice"].read_text(encoding="utf-8")
            probes = json.loads(paths["probes"].read_text(encoding="utf-8"))
            observations = payload["observations"]

            started = time.perf_counter()
            rule_raw = generate_rule_candidates(circuit, mapping, spice_text, observations)
            rule_rows = preflight_candidates(rule_raw, component_refs=set(circuit.components),
                                             component_mapping=mapping, spice_text=spice_text)
            rule_seconds = time.perf_counter() - started
            llm_rows, llm_seconds, llm_model_reported_seconds, llm_error = [], None, None, None
            prompt_audit = {"status": "NOT_RUN", "payload_sha256": None}
            if args.run_llm:
                llm_started = time.perf_counter()
                captured_audit = []
                blind = build_blind_diagnostic_input({"llm_case_id": public["llm_case_id"]},
                    components=payload["components"], observations=observations,
                    executable_catalog=rule_rows)
                try:
                    generated = propose_hypotheses(llm_client, args.model,
                        circuit_id=blind["circuit_id"], components=blind["components"],
                        observations=blind["observations"], symptoms=blind["symptoms"],
                        executable_catalog=rule_rows, include_candidate_parameters=False,
                        pre_call_audit=lambda prompt_payload: captured_audit.append(audit_public_payload(prompt_payload)))
                    prompt_audit = captured_audit[-1] if captured_audit else {"status": "FAILED", "payload_sha256": None}
                    llm_model_reported_seconds = generated.get("latency_seconds")
                    model_info = generated.get("model_info") or model_info
                    llm_rows = validate_hypotheses(generated["hypotheses"],
                        component_refs=set(circuit.components), component_mapping=mapping,
                        spice_text=spice_text, allowed_catalog=rule_rows)
                except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
                    llm_error = f"{type(exc).__name__}: {str(exc)[:300]}"
                finally:
                    llm_seconds = time.perf_counter() - llm_started
                    if captured_audit:
                        prompt_audit = captured_audit[-1]

            merged = _dedupe([*rule_rows, *llm_rows])
            score_case = public
            method_cases["Rules Only"].append({"case": score_case, "rows": rule_rows,
                                                "seconds": rule_seconds, "error": None})
            method_cases["LLM Only"].append({"case": score_case, "rows": llm_rows,
                "seconds": llm_seconds, "error": llm_error})
            method_cases["Hybrid"].append({"case": score_case, "rows": merged,
                "seconds": rule_seconds + (llm_seconds or 0), "error": llm_error})

            if args.simulate and ngspice_available():
                for method_name, rows in (("Rules + SPICE", rule_rows), ("Hybrid + SPICE", merged)):
                    began = time.perf_counter()
                    attempted = 0
                    simulated = []
                    for row in rows:
                        if row.get("validation_status") != "VALIDATED_EXECUTABLE":
                            simulated.append({**row, "simulation_status": "UNSUPPORTED_HYPOTHESIS"})
                        elif attempted >= args.spice_budget:
                            simulated.append({**row, "simulation_status": "SIMULATION_BUDGET_EXHAUSTED"})
                        else:
                            attempted += 1
                            simulated.append(simulate_hypothesis(row, base_netlist=paths["spice"],
                                component_mapping=mapping, probes=probes, observations=observations))
                    ranked = rank_hypotheses(simulated)
                    spice_cases[method_name].append({"case": score_case, "rows": ranked,
                        "seconds": time.perf_counter() - began, "spice_runs": attempted,
                        "simulation_successes": sum(row.get("simulation_status") == "SIMULATION_SUCCESS" for row in simulated),
                        "simulation_failures": sum(row.get("simulation_status") == "SIMULATION_FAILED" for row in simulated)})

            public_rows.append({"case_id": public["case_id"], "repetition": repetition + 1,
                "diagnostic_input_sha256": prompt_audit.get("payload_sha256") or audit_public_payload(payload)["payload_sha256"],
                "prompt_audit": {"status": prompt_audit.get("status"), "payload_sha256": prompt_audit.get("payload_sha256")},
                "candidate_counts": {"rules": len(rule_rows), "llm": len(llm_rows), "hybrid": len(merged)},
                "llm_attempted": bool(args.run_llm), "llm_success": bool(args.run_llm and llm_error is None),
                "llm_error": llm_error, "llm_seconds": llm_seconds,
                "llm_wall_seconds": llm_seconds, "llm_model_reported_seconds": llm_model_reported_seconds,
                "rule_generation_seconds": rule_seconds})

        for name, rows in method_cases.items():
            raw_method_runs[name].append(rows)
        for name, rows in spice_cases.items():
            raw_simulation_runs[name].append(rows)

    # Ground truth is first opened after all proposal and optional SPICE work is complete.
    truths, ground_truth_hash = load_private_ground_truth(bundle.directory, bundle.manifest,
        expected_case_ids={case["case_id"] for case in cases})
    truths = {key: list(value) for key, value in truths.items()}
    supported = {}
    for public in cases:
        paths = {key: ROOT / value for key, value in public["circuit"].items()}
        circuit = load_netlist(paths["netlist"])
        mapping = json.loads(paths["mapping"].read_text(encoding="utf-8"))
        spice_text = paths["spice"].read_text(encoding="utf-8")
        supported[public["case_id"]] = all(validate_hypothesis(truth,
            component_refs=set(circuit.components), component_mapping=mapping,
            spice_text=spice_text).get("validation_status") == "VALIDATED_EXECUTABLE"
            for truth in truths[public["case_id"]])
    supported_cases = [{**case, "ground_truth_supported": supported[case["case_id"]]} for case in cases]
    method_runs = {}
    for name, all_runs in raw_method_runs.items():
        method_runs[name] = []
        for rows in all_runs:
            scored_rows = [{**row, "case": {**row["case"], "ground_truth_supported": supported[row["case"]["case_id"]]}}
                           for row in rows]
            method_runs[name].append(_score_method(scored_rows, truths))
    simulation_runs = {}
    for name, all_runs in raw_simulation_runs.items():
        simulation_runs[name] = []
        for rows in all_runs:
            scored_rows = [{**row, "case": {**row["case"], "ground_truth_supported": supported[row["case"]["case_id"]]}}
                           for row in rows]
            simulation_runs[name].append(_score_ranked_method(scored_rows, truths))

    methods = {}
    evaluation_status = _evaluation_status(bundle.manifest, reevaluating_existing_holdout)
    for name, runs in method_runs.items():
        available = not (name in {"LLM Only", "Hybrid"} and not args.run_llm)
        status = evaluation_status if available else "NOT_EVALUATED"
        methods[name] = {"status": status, "runs": runs if available else [],
                         "run_count": len(runs) if available else 0,
                         "candidate_seconds": _summarize([run["candidate_seconds"] for run in runs]) if available else {"status": "NOT_EVALUATED", "mean": None},
                         "generation_metrics": _mean_metrics(runs) if status != "NOT_EVALUATED" else {"status": "NOT_EVALUATED"},
                         "repeat_stability": _repeat_stability(runs) if status != "NOT_EVALUATED" else {"status": "NOT_EVALUATED"}}
    for name, runs in simulation_runs.items():
        available = args.simulate and bool(runs) and (name != "Hybrid + SPICE" or args.run_llm)
        methods[name] = {"status": evaluation_status if available else "NOT_EVALUATED",
            "runs": runs if available else [], "run_count": len(runs) if available else 0,
            "ranking": _mean_rankings(runs) if available else {"status": "NOT_EVALUATED", "top_k_accuracy": {"1": None, "3": None}},
            "conditional_ranking": _mean_rankings(runs, metric="conditional_ranking") if available else {"status": "NOT_EVALUATED", "top_k_accuracy": {"1": None, "3": None}}}
        if available:
            generation_method = "Rules Only" if name == "Rules + SPICE" else "Hybrid"
            proposal_time = methods[generation_method]["candidate_seconds"].get("mean") or 0
            spice_time = sum(run["seconds"] for run in runs)
            supported_n = sum(run["ranking"]["case_count"] for run in runs)
            top1 = methods[name]["ranking"]["top_k_accuracy"].get("1")
            correct = round((top1 or 0) * supported_n)
            spice_count = sum(run["simulation_runs"] for run in runs)
            total_elapsed = proposal_time * len(runs) + spice_time
            methods[name]["efficiency"] = {**efficiency_metrics(correct_cases=correct,
                elapsed_seconds=total_elapsed, spice_runs=spice_count),
                "proposal_wall_seconds": proposal_time * len(runs),
                "spice_wall_seconds": spice_time,
                "total_wall_seconds": total_elapsed}
            methods[name]["ranking_by_measurement_node_count"] = _mean_node_rankings(runs)
    paired_comparisons = _paired_method_comparisons(method_runs, simulation_runs)
    cluster_summary = {name: _cluster_interval(runs, supported_cases, iterations=2000, seed=1701 + index)
                       for index, (name, runs) in enumerate(method_runs.items()) if runs}
    metadata = reproducibility_metadata(dataset_id=bundle.manifest.get("dataset_id", "unknown"),
        public_hash=bundle.public_hash, ground_truth_hash=ground_truth_hash,
        split=bundle.manifest.get("split", "unknown"), prompt_version=PROMPT_VERSION if args.run_llm else None,
        model_name=args.model if args.run_llm else None,
        model_digest=model_info.get("digest") if isinstance(model_info, dict) else None,
        simulator_version=_simulator_version() if args.simulate else None, seed=None)
    metadata["benchmark_code_fingerprint"] = benchmark_code_fingerprint(ROOT)["sha256"]
    independent_synthetic_holdout = bool(bundle.manifest.get("independent_holdout")) and not reevaluating_existing_holdout
    grid_analysis = _candidate_grid_analysis(cases, truths, raw_method_runs.get("Rules Only", []), supported)
    node_counts = {}
    for case in cases:
        node_counts[str(len(case.get("diagnostic_input", {}).get("observations", [])))] = node_counts.get(str(len(case.get("diagnostic_input", {}).get("observations", []))), 0) + 1
    llm_attempts = [row for row in public_rows if row.get("llm_attempted")]
    llm_successful_times = [row["llm_wall_seconds"] for row in llm_attempts
                            if row.get("llm_success") and isinstance(row.get("llm_wall_seconds"), (int, float))]
    llm_attempt_times = [row["llm_wall_seconds"] for row in llm_attempts
                         if isinstance(row.get("llm_wall_seconds"), (int, float))]
    method_spice_failures = sum(run.get("simulation_failures", 0) for values in simulation_runs.values() for run in values)
    record = {"status": evaluation_status,
        "metric_schema_version": "phase17-metrics-v2",
        "dataset_id": bundle.manifest.get("dataset_id"), "split": bundle.manifest.get("split"),
        "independent_holdout": independent_synthetic_holdout,
        "source_dataset_was_independent_holdout": bool(bundle.manifest.get("independent_holdout")),
        "reevaluation_of_existing_holdout": reevaluating_existing_holdout,
        "holdout_status": ("EVALUATED_SYNTHETIC_NOT_PHYSICAL" if independent_synthetic_holdout else
                           "REUSED_SYNTHETIC_DATASET_REEVALUATED_NOT_FRESH" if reevaluating_existing_holdout else
                           "SYNTHETIC_OFF_GRID_REUSED_TOPOLOGY" if bundle.manifest.get("off_grid") else
                           "INDEPENDENT_SYNTHETIC_HOLDOUT_NOT_EVALUATED"),
        "real_repair_status": "NOT_EVALUATED", "supported_case_count": sum(supported.values()),
        "unsupported_ground_truth_case_count": len(supported) - sum(supported.values()),
        "case_count": len(cases), "independent_case_count": len(cases),
        "topology_group_count": len({case["topology_group"] for case in cases}),
        "repeated_run_count": model_repetitions, "llm_attempt_count": len(llm_attempts),
        "llm_success_count": sum(row.get("llm_success", False) for row in llm_attempts),
        "llm_failure_count": sum(not row.get("llm_success", False) for row in llm_attempts),
        "simulation_attempt_count": sum(run.get("simulation_runs", 0) for runs in simulation_runs.values() for run in runs),
        "simulation_failure_count": method_spice_failures,
        "candidate_grid_analysis": grid_analysis,
        "paired_method_comparisons": paired_comparisons,
        "llm_enabled": args.run_llm,
        "llm_repeat_count": args.llm_runs if args.run_llm else 0,
        "spice_enabled": args.simulate, "spice_run_budget_per_method_case": args.spice_budget,
        "methods": methods, "input_audit": {"status": "PASSED",
            "prompt_hash_count": len({row["diagnostic_input_sha256"] for row in public_rows}),
            "private_ground_truth_passed_to_proposal_method": False,
            "private_ground_truth_loaded_after_proposals_and_simulation": True},
        "case_runs": public_rows, "metadata": metadata,
        "performance_environment": {"process_peak_rss_bytes": _max_rss_bytes(),
            "benchmark_command_wall_seconds": time.perf_counter() - benchmark_started,
            "llm_latency": {"status": "MEASURED" if llm_attempts else "NOT_EVALUATED",
                "attempt_count": len(llm_attempts), "success_count": len(llm_successful_times),
                "failure_count": len(llm_attempts) - len(llm_successful_times),
                "mean_successful_seconds": _average(llm_successful_times),
                "mean_attempted_seconds": _average(llm_attempt_times),
                "total_attempted_wall_seconds": sum(llm_attempt_times) if llm_attempt_times else None},
            "memory_note": "Process high-water RSS during this benchmark command; not method-isolated.",
            "llm_token_counts": "NOT_AVAILABLE_FROM_OLLAMA_ADAPTER",
            "warmup_runs": 0, "cache_policy": "Normal local Ollama/ngspice cache state; not controlled or separated."},
        "generation_settings": {"ollama_temperature": 0.1 if args.run_llm else None,
            "ollama_num_predict": 1300 if args.run_llm else None, "seed": None,
            "json_mode_requested": True if args.run_llm else None,
            "simulation_timeout_seconds": 60 if args.simulate else None,
            "spice_budget_per_method_per_case": args.spice_budget if args.simulate else None},
        "statistics": {"exact_recall_cluster_bootstrap_95ci": cluster_summary,
            "note": "Exploratory cluster bootstrap; intervals are unstable with few topology groups."},
        "measurement_node_count_comparison": {"observed_case_counts": node_counts,
            "intervention_status": "NOT_EVALUATED",
            "note": "Observed strata are descriptive; this is not a same-case node-count intervention."},
        "phase16_prior_status": "REAL_REPAIR_PRIOR_PERFORMANCE_NOT_EVALUATED",
        "bayesian_posterior_runtime": "NOT_EVALUATED",
        "datasheet_independent_holdout_status": "NOT_EVALUATED",
        "pcb_vision_reference_match_status": "NOT_EVALUATED",
        "limitations": (["Results are synthetic simulation evidence, not real repair accuracy."] +
            ([] if independent_synthetic_holdout else ["Cases and circuit topologies were previously used in Phase 15 development."])) + [
            "Only one observed node per case is currently evaluated.",
            "LLM prompt candidates are restricted to the existing preflighted catalog and parameter grid."]}
    target = save_result(ROOT / args.results, record)
    try:
        result_label = str(target.relative_to(ROOT))
    except ValueError:
        result_label = str(target)
    print(json.dumps({"result": result_label, "run_id": target.stem,
        "case_count": len(cases), "methods": {key: value["status"] for key, value in methods.items()},
        "holdout": record["holdout_status"]}, indent=2))
    return 0


def _score_method(rows, truths):
    cases = [row["case"] for row in rows]
    pred = {row["case"]["case_id"]: row["rows"] for row in rows}
    truth = {case["case_id"]: truths[case["case_id"]] for case in cases}
    levels = score_hypothesis_levels(cases, truth, pred)
    return {"levels": levels, "candidate_seconds": sum(row["seconds"] or 0 for row in rows),
            "candidate_count": sum(len(row["rows"]) for row in rows),
            "error_count": sum(row["error"] is not None for row in rows),
            "candidate_details": [{"case_id": row["case"]["case_id"], "candidates": [
                {key: candidate.get(key) for key in ("hypothesis_id", "component_ref", "fault_type",
                 "parameters", "validation_status", "simulation_status")} for candidate in row["rows"]]}
                for row in rows],
            "per_case": levels["per_case"]}


def _score_ranked_method(rows, truths):
    cases = [row["case"] for row in rows]
    ranked = {row["case"]["case_id"]: row["rows"] for row in rows}
    truth = {case["case_id"]: truths[case["case_id"]] for case in cases}
    rank = score_ranked_hypotheses(cases, truth, ranked)
    selected_predictions = {case["case_id"]: [
        {**row, "selected_for_diagnosis": row.get("evidence_rank") == 1}
        for row in ranked.get(case["case_id"], [])] for case in cases}
    parameter_scores = score_hypothesis_levels(cases, truth, selected_predictions)
    oracle_scores = score_hypothesis_levels(cases, truth, ranked)
    conditional_cases = []
    conditional_ranked = {}
    for case in cases:
        case_id = case["case_id"]
        ground = truth[case_id]
        rows_for_case = ranked.get(case_id, [])
        truth_in_successful = any(row.get("simulation_status") == "SIMULATION_SUCCESS"
            and any(row.get("component_ref") == label.get("component_ref")
                    and row.get("fault_type") == label.get("fault_type")
                    and _parameters_equal(row.get("parameters", {}), label.get("parameters", {}))
                    for label in ground) for row in rows_for_case)
        if truth_in_successful:
            conditional_cases.append(case)
            conditional_ranked[case_id] = rows_for_case
    conditional_truth = {case["case_id"]: truth[case["case_id"]] for case in conditional_cases}
    conditional = score_ranked_hypotheses(conditional_cases, conditional_truth, conditional_ranked)
    node_groups: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        count = str(len(case.get("diagnostic_input", {}).get("observations", [])))
        node_groups.setdefault(count, []).append(case)
    by_node_count = {}
    for count, group in node_groups.items():
        group_truth = {case["case_id"]: truth[case["case_id"]] for case in group}
        group_ranked = {case["case_id"]: ranked.get(case["case_id"], []) for case in group}
        by_node_count[count] = score_ranked_hypotheses(group, group_truth, group_ranked)
    return {"ranking": rank, "seconds": sum(row["seconds"] for row in rows),
            "conditional_ranking": conditional,
            "end_to_end_ranking": rank,
            "selected_parameter_error": parameter_scores["parameter_error"]["selected_candidate"],
            "oracle_best_candidate_parameter_error": oracle_scores["parameter_error"]["oracle_best_candidate"],
            "candidate_details": [{"case_id": case["case_id"], "candidates": [
                {key: candidate.get(key) for key in ("hypothesis_id", "component_ref", "fault_type",
                 "parameters", "validation_status", "simulation_status", "evidence_rank", "observation_match")}
                for candidate in ranked.get(case["case_id"], [])]} for case in cases],
            "ranking_by_measurement_node_count": by_node_count,
            "simulation_runs": sum(row["spice_runs"] for row in rows),
            "simulation_successes": sum(row["simulation_successes"] for row in rows),
            "simulation_failures": sum(row["simulation_failures"] for row in rows)}


def _evaluation_status(manifest, reusing_existing):
    if reusing_existing:
        return "MEASURED_EXISTING_HOLDOUT_REEVALUATION"
    if manifest.get("independent_holdout"):
        return "MEASURED_INDEPENDENT_SYNTHETIC_HOLDOUT"
    if manifest.get("off_grid"):
        return "MEASURED_SYNTHETIC_OFF_GRID_REUSED_TOPOLOGY"
    return "MEASURED_SYNTHETIC_DEVELOPMENT"


def _candidate_grid_analysis(cases, truths, rule_runs, supported):
    by_case = {}
    unique_status = {}
    for repetition, rows in enumerate(rule_runs, 1):
        generated = {row["case"]["case_id"]: row["rows"] for row in rows}
        for case in cases:
            cid = case["case_id"]
            labels = []
            for truth in truths.get(cid, []):
                included = any(_hypothesis_matches(truth, candidate) for candidate in generated.get(cid, []))
                status = ("UNSUPPORTED_FAULT_MODEL" if not supported.get(cid, False) else
                          "IN_CATALOG_GRID" if included else "OUT_OF_CATALOG_GRID")
                label = {"component_ref": truth.get("component_ref"), "fault_type": truth.get("fault_type"),
                         "status": status}
                labels.append(label)
                unique_status.setdefault(cid, label["status"])
            by_case[f"run_{repetition}:{cid}"] = labels
    values = [label["status"] for labels in by_case.values() for label in labels]
    independent_values = list(unique_status.values())
    return {"status": "MEASURED" if values else "NOT_EVALUATED", "case_level_truth_count": len(values),
            "in_catalog_grid_count": values.count("IN_CATALOG_GRID"),
            "out_of_catalog_grid_count": values.count("OUT_OF_CATALOG_GRID"),
            "independent_case_count": len(independent_values),
            "independent_in_catalog_grid_count": independent_values.count("IN_CATALOG_GRID"),
            "independent_out_of_catalog_grid_count": independent_values.count("OUT_OF_CATALOG_GRID"),
            "unsupported_fault_model_count": values.count("UNSUPPORTED_FAULT_MODEL"),
            "per_case_repetition": by_case,
            "note": "IN_CATALOG_GRID means an exact generated Rules candidate matched the private label."}


def _hypothesis_matches(left, right):
    if left.get("component_ref") != right.get("component_ref") or left.get("fault_type") != right.get("fault_type"):
        return False
    left_params, right_params = left.get("parameters", {}), right.get("parameters", {})
    if set(left_params) != set(right_params):
        return False
    for key, value in left_params.items():
        try:
            a, b = float(value), float(right_params[key])
            if abs(a - b) > 1e-9 * max(abs(a), abs(b), 1.0):
                return False
        except (TypeError, ValueError):
            if str(value).casefold() != str(right_params[key]).casefold():
                return False
    return True


def _parameters_equal(left, right):
    if set(left) != set(right):
        return False
    for key, value in left.items():
        try:
            if float(value) != float(right[key]):
                return False
        except (TypeError, ValueError):
            if str(value).casefold() != str(right[key]).casefold():
                return False
    return True


def _mean_metrics(runs):
    if not runs:
        return {"status": "NOT_EVALUATED"}
    levels = {}
    for level in ("component", "fault_mode", "exact_hypothesis"):
        levels[level] = {}
        for metric in ("precision", "recall", "f1"):
            values = [run["levels"]["levels"][level][metric] for run in runs
                      if run["levels"]["levels"][level][metric] is not None]
            levels[level][metric] = sum(values) / len(values) if values else None
    candidate_fields = ("raw_candidate_count", "unique_component_count", "unique_fault_mode_count",
                        "unique_exact_hypothesis_count", "duplicate_component_candidate_count",
                        "duplicate_fault_mode_candidate_count", "duplicate_exact_candidate_count")
    candidate_counts = {field: _summarize([run["levels"].get(field, 0) for run in runs])
                        for field in candidate_fields}
    raw_values = [run["levels"].get("raw_candidate_levels", {}).get("exact_hypothesis", {}).get("precision")
                  for run in runs]
    raw_values = [value for value in raw_values if value is not None]
    raw_exact = sum(raw_values) / len(raw_values) if raw_values else None
    return {"status": "MEASURED", "recall": levels, "candidate_counts": candidate_counts,
            "raw_candidate_precision": raw_exact}


def _repeat_stability(runs):
    if not runs:
        return {"status": "NOT_EVALUATED"}
    fields = {"candidate_count": [run["candidate_count"] for run in runs],
              "candidate_generation_seconds": [run["candidate_seconds"] for run in runs],
              "error_count": [run["error_count"] for run in runs]}
    for key in ("raw_candidate_count", "unique_component_count", "unique_fault_mode_count",
                "unique_exact_hypothesis_count", "duplicate_component_candidate_count",
                "duplicate_fault_mode_candidate_count", "duplicate_exact_candidate_count"):
        fields[key] = [run["levels"].get(key, 0) for run in runs]
    for level in ("component", "fault_mode", "exact_hypothesis"):
        for metric in ("precision", "recall", "f1"):
            fields[f"{level}_{metric}"] = [run["levels"]["levels"][level][metric]
                for run in runs if run["levels"]["levels"][level][metric] is not None]
    return {"status": "MEASURED" if len(runs) > 1 else "SINGLE_RUN",
            "independent_case_count": len(runs[0].get("per_case", [])),
            "repetitions_are_not_independent_cases": True,
            "metrics": {key: _summarize(values) for key, values in fields.items()}}


def _cluster_interval(runs, cases, *, iterations, seed):
    if not runs:
        return {"status": "NOT_EVALUATED", "lower": None, "upper": None, "cluster_count": 0}
    by_case = {}
    for run in runs:
        for row in run.get("per_case", []):
            if row.get("status") == "SCORED":
                by_case.setdefault(row["case_id"], []).append(
                    float(row.get("correct", {}).get("exact_hypothesis", 0) > 0))
    rows = [{"case_id": case["case_id"], "topology_group": case["topology_group"],
             "value": sum(by_case[case["case_id"]]) / len(by_case[case["case_id"]])}
            for case in cases if by_case.get(case["case_id"])]
    return cluster_bootstrap_interval(rows, value_key="value", iterations=iterations, seed=seed)


def _mean_rankings(runs, *, metric="ranking"):
    if not runs:
        return {"status": "NOT_EVALUATED", "top_k_accuracy": {"1": None, "3": None}}
    if not sum(run.get(metric, {}).get("case_count", 0) for run in runs):
        return {"status": "NOT_EVALUATED", "case_count": 0,
                "top_k_accuracy": {"1": None, "3": None}, "mean_reciprocal_rank": None}
    result = {}
    for key in ("1", "3"):
        values = [run[metric]["top_k_accuracy"].get(key) for run in runs
                  if run.get(metric, {}).get("top_k_accuracy", {}).get(key) is not None]
        result.setdefault("top_k_accuracy", {})[key] = sum(values) / len(values) if values else None
    mrrs = [run.get(metric, {}).get("mean_reciprocal_rank") for run in runs
            if run.get(metric, {}).get("mean_reciprocal_rank") is not None]
    for field in ("strict_top1", "tie_aware_top1"):
        values = [run[metric].get(field) for run in runs if run.get(metric, {}).get(field) is not None]
        result[field] = sum(values) / len(values) if values else None
    result.update(status="MEASURED", case_count=sum(run[metric].get("case_count", 0) for run in runs),
                  mean_reciprocal_rank=sum(mrrs) / len(mrrs) if mrrs else None,
                  tie_case_count=sum(run[metric].get("tie_case_count", 0) for run in runs),
                  ambiguous_case_count=sum(run[metric].get("ambiguous_case_count", 0) for run in runs),
                  independent_case_count=max((run[metric].get("independent_case_count", 0) for run in runs), default=0),
                  repeated_run_count=len(runs))
    return result


def _paired_method_comparisons(method_runs, simulation_runs):
    pairs = (("Rules Only", "LLM Only", "exact_hypothesis"),
             ("Rules Only", "Hybrid", "exact_hypothesis"),
             ("Rules + SPICE", "Hybrid + SPICE", "strict_top1"))
    report = {}
    for left, right, metric in pairs:
        left_runs = method_runs.get(left, []) if "SPICE" not in left else simulation_runs.get(left, [])
        right_runs = method_runs.get(right, []) if "SPICE" not in right else simulation_runs.get(right, [])
        if not left_runs or not right_runs:
            report[f"{left} vs {right}"] = {"status": "NOT_EVALUATED"}
            continue
        per_case = {}
        for run_index, (a, b) in enumerate(zip(left_runs, right_runs), 1):
            if metric == "exact_hypothesis":
                a_rows, b_rows = a.get("per_case", []), b.get("per_case", [])
                a_values = {row["case_id"]: int(row.get("correct", {}).get(metric, 0) > 0)
                            for row in a_rows if row.get("status") == "SCORED"}
                b_values = {row["case_id"]: int(row.get("correct", {}).get(metric, 0) > 0)
                            for row in b_rows if row.get("status") == "SCORED"}
            else:
                a_values = {row["case_id"]: int(row.get("strict_top1", False))
                            for row in a.get("ranking", {}).get("per_case", []) if row.get("status") == "SCORED"}
                b_values = {row["case_id"]: int(row.get("strict_top1", False))
                            for row in b.get("ranking", {}).get("per_case", []) if row.get("status") == "SCORED"}
            for case_id in sorted(set(a_values) & set(b_values)):
                per_case.setdefault(case_id, []).append({"run": run_index,
                    "left": a_values[case_id], "right": b_values[case_id],
                    "difference_right_minus_left": b_values[case_id] - a_values[case_id]})
        paired = [item for rows in per_case.values() for item in rows]
        delta = [item["difference_right_minus_left"] for item in paired]
        report[f"{left} vs {right}"] = {
            "status": "MEASURED_DESCRIPTIVE_PAIRED",
            "metric": metric, "independent_case_count": len(per_case),
            "repeated_run_count": max((len(rows) for rows in per_case.values()), default=0),
            "paired_case_run_count": len(paired),
            "mean_difference_right_minus_left": _average(delta),
            "right_wins": sum(value > 0 for value in delta),
            "ties": sum(value == 0 for value in delta),
            "left_wins": sum(value < 0 for value in delta),
            "significance_claim": "NONE; small synthetic sample, descriptive only",
            "per_case_repetitions": per_case}
    return report


def _mean_node_rankings(runs):
    strata = {}
    for run in runs:
        for count, values in run.get("ranking_by_measurement_node_count", {}).items():
            strata.setdefault(count, []).append(values)
    result = {}
    for count, items in strata.items():
        measured = [item for item in items if item.get("status") == "MEASURED"]
        if not measured:
            result[count] = {"status": "NOT_EVALUATED", "independent_case_count": 0}
            continue
        result[count] = {"status": "MEASURED",
            "independent_case_count": max(item.get("case_count", 0) for item in measured),
            "repeated_case_count": sum(item.get("case_count", 0) for item in measured),
            "top_k_accuracy": {key: _average([item.get("top_k_accuracy", {}).get(key) for item in measured])
                               for key in ("1", "3")},
            "mean_reciprocal_rank": _average([item.get("mean_reciprocal_rank") for item in measured]),
            "strict_top1": _average([item.get("strict_top1") for item in measured]),
            "tie_aware_top1": _average([item.get("tie_aware_top1") for item in measured])}
    return result


def _average(values):
    values = [value for value in values if isinstance(value, (int, float))]
    return sum(values) / len(values) if values else None


def _summarize(values):
    if not values:
        return {"status": "NOT_EVALUATED", "mean": None, "median": None, "standard_deviation": None}
    values = sorted(values)
    n = len(values)
    return {"status": "MEASURED", "mean": sum(values) / n,
            "median": values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2,
            "standard_deviation": (sum((value - sum(values) / n) ** 2 for value in values) / (n - 1)) ** 0.5 if n > 1 else None,
            "min": min(values), "max": max(values)}


def _dedupe(rows):
    unique, seen = [], set()
    for row in rows:
        normalized = {}
        for name, value in row.get("parameters", {}).items():
            try:
                normalized[name] = f"{float(value):.12g}"
            except (TypeError, ValueError):
                normalized[name] = str(value).strip().casefold()
        key = (row.get("component_ref"), row.get("fault_type"), json.dumps(normalized, sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _max_rss_bytes():
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if platform.system() == "Darwin" else value * 1024)
    except (AttributeError, OSError, ValueError):
        return None


def _simulator_version():
    try:
        return subprocess.run(["ngspice", "--version"], capture_output=True, text=True, timeout=5).stdout.splitlines()[1].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
