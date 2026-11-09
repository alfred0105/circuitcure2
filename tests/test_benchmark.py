import json
import subprocess
import sys
from pathlib import Path

import pytest

from circuitcare.benchmark.dataset import (DatasetBundle, load_public_dataset,
    validate_dataset_bundle)
from circuitcare.benchmark.integrity import audit_holdout_splits, audit_public_payload
from circuitcare.benchmark.methods import apply_simulation_budget, merge_hypothesis_proposals, run_input_method
from circuitcare.benchmark.performance import efficiency_metrics, measure_callable
from circuitcare.benchmark.runner import save_result
from circuitcare.benchmark_ui import render_benchmark_tab
from circuitcare.benchmark.scoring import score_hypothesis_levels, score_ranked_hypotheses
from circuitcare.benchmark.statistics import cluster_bootstrap_interval, summarize
from circuitcare.fault_ai.llm_hypothesis import propose_hypotheses


def _case(case_id="p17_001", group="topology_a", split="development"):
    return {"case_id": case_id, "topology_group": group, "split": split,
            "diagnostic_input": {"components": [{"ref": "R1", "value": "1k"}],
                                 "observations": [{"node": "VOUT", "value": 2.5, "unit": "V"}]}}


def test_public_payload_rejects_answers_nested_in_prompt():
    with pytest.raises(ValueError, match="fault_type"):
        audit_public_payload({"observations": [{"fault_type": "resistor_open"}]})


def test_public_method_only_receives_audited_input():
    result = run_input_method(lambda payload: [{"component_ref": payload["components"][0]["ref"]}], _case())
    assert result["predictions"][0]["component_ref"] == "R1"
    assert len(result["diagnostic_input_hash"]) == 64


def test_actual_llm_payload_is_captured_and_audited_before_call():
    events = []
    class FakeClient:
        def chat_json(self, model, messages):
            events.append("llm_call")
            payload = json.loads(messages[-1]["content"])
            assert "ground_truth" not in json.dumps(payload)
            assert "parameters" not in payload["executable_catalog"][0]
            return {"content": {"hypotheses": []}, "model_info": {"digest": "test"}}
    result = propose_hypotheses(FakeClient(), "test", circuit_id="case_0001", components=[],
        observations=[], symptoms="unexpected output",
        executable_catalog=[{"component_ref": "R1", "fault_type": "resistance_drift",
                             "parameters": {"fault_resistance": 10000}, "nets": ["VOUT"],
                             "validation_status": "VALIDATED_EXECUTABLE"}],
        include_candidate_parameters=False,
        pre_call_audit=lambda payload: (events.append("audit"), audit_public_payload(payload)))
    assert events == ["audit", "llm_call"]
    assert result["prompt_version"]


def test_dataset_rejects_topology_group_split_leakage():
    cases = (_case(group="same", split="development"), _case("p17_002", "same", "holdout"))
    bundle = DatasetBundle(cases, {"p17_001": ({"component_ref": "R1"},),
                                   "p17_002": ({"component_ref": "R2"},)}, {"case_ids": ["p17_001", "p17_002"]},
                          "x", "y", "z")
    with pytest.raises(ValueError, match="cross dataset splits"):
        validate_dataset_bundle(bundle)


def test_public_loader_does_not_require_or_open_private_truth(tmp_path):
    (tmp_path / "public_cases.json").write_text(json.dumps({"cases": [_case()]}), encoding="utf-8")
    (tmp_path / "split_manifest.json").write_text(json.dumps({"case_ids": ["p17_001"]}), encoding="utf-8")
    public = load_public_dataset(tmp_path)
    assert public.public_cases[0]["case_id"] == "p17_001"
    assert not hasattr(public, "ground_truth")


def test_hierarchical_accuracy_separates_component_mode_and_exact_parameters():
    cases = [{"case_id": "a"}]
    truth = {"a": [{"component_ref": "R1", "fault_type": "resistance_drift", "parameters": {"fault_resistance": 10000}}]}
    predictions = {"a": [{"component_ref": "R1", "fault_type": "resistance_drift", "parameters": {"fault_resistance": 5000}}]}
    result = score_hypothesis_levels(cases, truth, predictions)
    assert result["levels"]["component"]["recall"] == 1
    assert result["levels"]["fault_mode"]["recall"] == 1
    assert result["levels"]["exact_hypothesis"]["recall"] == 0
    assert result["parameter_error"]["mean"] == 0.5


def test_parameter_exact_zero_policy_is_stable():
    rows = [{"case_id": "a"}]
    truth = {"a": [{"component_ref": "R1", "fault_type": "resistance_drift", "parameters": {"x": 0}}]}
    result = score_hypothesis_levels(rows, truth, {"a": [{"component_ref": "R1", "fault_type": "resistance_drift", "parameters": {"x": 0}}]})
    assert result["levels"]["exact_hypothesis"]["recall"] == 1
    assert result["parameter_error"]["mean"] == 0


def test_metric_v2_deduplicates_predictions_at_each_label_level():
    case = [{"case_id": "a"}]
    truth = {"a": [{"component_ref": "R1", "fault_type": "resistance_drift",
                    "parameters": {"fault_resistance": 2000}}]}
    predictions = {"a": [
        {"component_ref": "R1", "fault_type": "resistance_drift", "parameters": {"fault_resistance": 2000}},
        {"component_ref": "R1", "fault_type": "resistance_drift", "parameters": {"fault_resistance": 5000}},
        {"component_ref": "R1", "fault_type": "resistor_open", "parameters": {}},
    ]}
    result = score_hypothesis_levels(case, truth, predictions)
    assert result["levels"]["component"]["precision"] == 1.0
    assert result["levels"]["fault_mode"]["precision"] == 0.5
    assert result["levels"]["exact_hypothesis"]["precision"] == pytest.approx(1 / 3)
    assert result["raw_candidate_count"] == 3
    assert result["duplicate_component_candidate_count"] == 2
    assert result["duplicate_fault_mode_candidate_count"] == 1


def test_metric_v2_counts_false_component_and_fault_once():
    case = [{"case_id": "a"}]
    truth = {"a": [{"component_ref": "R1", "fault_type": "resistance_drift",
                    "parameters": {"fault_resistance": 2000}}]}
    predictions = {"a": [
        {"component_ref": "R1", "fault_type": "resistance_drift", "parameters": {"fault_resistance": 2000}},
        {"component_ref": "R2", "fault_type": "resistance_drift", "parameters": {"fault_resistance": 2000}},
    ]}
    result = score_hypothesis_levels(case, truth, predictions)
    assert result["levels"]["component"]["precision"] == 0.5
    assert result["levels"]["fault_mode"]["precision"] == 0.5
    assert result["levels"]["exact_hypothesis"]["precision"] == 0.5


def test_metric_v2_duplicate_exact_candidate_does_not_change_unique_precision():
    case = [{"case_id": "a"}]
    row = {"component_ref": "R1", "fault_type": "resistance_drift", "parameters": {"fault_resistance": "2k"}}
    result = score_hypothesis_levels(case, {"a": [row]}, {"a": [row, dict(row)]})
    assert result["raw_candidate_count"] == 2
    assert result["duplicate_exact_candidate_count"] == 1
    assert result["levels"]["exact_hypothesis"]["precision"] == 1
    assert result["raw_candidate_levels"]["exact_hypothesis"]["precision"] == 0.5


def test_metric_v2_multilabel_matching_is_one_to_one():
    case = [{"case_id": "a"}]
    truths = {"a": [
        {"component_ref": "R1", "fault_type": "resistance_drift", "parameters": {"x": 1}},
        {"component_ref": "R1", "fault_type": "resistor_open", "parameters": {}},
    ]}
    predictions = {"a": [
        {"component_ref": "R1", "fault_type": "resistance_drift", "parameters": {"x": 1}},
        {"component_ref": "R1", "fault_type": "resistance_drift", "parameters": {"x": 1}},
    ]}
    result = score_hypothesis_levels(case, truths, predictions)
    assert result["levels"]["component"]["tp"] == 1
    assert result["levels"]["fault_mode"]["tp"] == 1
    assert result["levels"]["exact_hypothesis"]["tp"] == 1
    assert result["levels"]["exact_hypothesis"]["fn"] == 1


def test_parameter_v2_separates_oracle_from_unselected_candidate_and_parses_units():
    case = [{"case_id": "a"}]
    truths = {"a": [{"component_ref": "R1", "fault_type": "resistance_drift",
                     "parameters": {"fault_resistance": 2000}}]}
    predictions = {"a": [
        {"component_ref": "R1", "fault_type": "resistance_drift", "parameters": {"fault_resistance": "5k"}},
        {"component_ref": "R1", "fault_type": "resistance_drift", "parameters": {"fault_resistance": "2K"}},
    ]}
    result = score_hypothesis_levels(case, truths, predictions)
    assert result["parameter_error"]["legacy_first_candidate"]["mean"] == 1.5
    assert result["parameter_error"]["oracle_best_candidate"]["mean"] == 0
    assert result["parameter_error"]["selected_candidate"]["status"] == "NOT_EVALUATED"


def test_parameter_v2_selected_error_requires_explicit_selection():
    case = [{"case_id": "a"}]
    truth = {"a": [{"component_ref": "R1", "fault_type": "resistance_drift", "parameters": {"x": 10}}]}
    rows = {"a": [
        {"component_ref": "R1", "fault_type": "resistance_drift", "parameters": {"x": 8},
         "selected_for_diagnosis": True},
        {"component_ref": "R1", "fault_type": "resistance_drift", "parameters": {"x": 10}},
    ]}
    result = score_hypothesis_levels(case, truth, rows)
    assert result["parameter_error"]["selected_candidate"]["mean"] == pytest.approx(0.2)
    assert result["parameter_error"]["oracle_best_candidate"]["mean"] == 0


def test_spice_tie_aware_rank_keeps_strict_top1_and_marks_ambiguity():
    case = [{"case_id": "a"}]
    truth = {"a": [{"component_ref": "R2", "fault_type": "resistor_open", "parameters": {}}]}
    rows = {"a": [
        {"component_ref": "R1", "fault_type": "resistor_open", "parameters": {},
         "simulation_status": "SIMULATION_SUCCESS", "evidence_rank": 1,
         "observation_match": {"max_absolute_normalized_residual": 0.2}},
        {"component_ref": "R2", "fault_type": "resistor_open", "parameters": {},
         "simulation_status": "SIMULATION_SUCCESS", "evidence_rank": 2,
         "observation_match": {"max_absolute_normalized_residual": 0.25}},
    ]}
    result = score_ranked_hypotheses(case, truth, rows, residual_tie_tolerance=0.1)
    assert result["strict_top1"] == 0
    assert result["tie_aware_top1"] == 1
    assert result["ambiguous_case_count"] == 1


def test_rank_metrics_count_misses_when_truth_is_not_ranked():
    result = score_ranked_hypotheses([{"case_id": "a"}], {"a": [{"component_ref": "R1"}]}, {"a": []})
    assert result["top_k_accuracy"]["1"] == 0
    assert result["coverage"] == 0
    assert result["mean_reciprocal_rank"] == 0


def test_holdout_audit_finds_shared_topology():
    result = audit_holdout_splits([_case(split="development"), _case("p17_002", split="holdout")])
    assert result["status"] == "FAILED"
    assert result["topology_group_leaks"] == ["topology_a"]


def test_cluster_bootstrap_requires_independent_groups():
    assert cluster_bootstrap_interval([{"topology_group": "a", "score": 1}], value_key="score")["status"] == "NOT_EVALUATED"
    result = cluster_bootstrap_interval([{"topology_group": "a", "score": 1},
                                         {"topology_group": "b", "score": 0}], value_key="score", iterations=100)
    assert result["status"] == "MEASURED_EXPLORATORY"


def test_bounded_simulation_budget_does_not_overrun():
    rows = [{"validation_status": "VALIDATED_EXECUTABLE", "id": 1},
            {"validation_status": "VALIDATED_EXECUTABLE", "id": 2},
            {"validation_status": "INVALID_REFERENCE", "id": 3}]
    calls = []
    result = apply_simulation_budget(rows, max_runs=1, simulate=lambda row: (calls.append(row["id"]) or {**row, "simulation_status": "SIMULATION_SUCCESS"}))
    assert calls == [1]
    assert result[1]["simulation_status"] == "SIMULATION_BUDGET_EXHAUSTED"
    assert result[2]["simulation_status"] == "UNSUPPORTED_HYPOTHESIS"


def test_methods_dedupe_and_efficiency_zero_cases_are_safe():
    row = {"component_ref": "R1", "fault_type": "resistor_open", "parameters": {}}
    assert len(merge_hypothesis_proposals([row], [row])) == 1
    assert efficiency_metrics(correct_cases=0, elapsed_seconds=1, spice_runs=0)["spice_runs_per_correct_diagnosis"] is None
    assert summarize([])["status"] == "NOT_EVALUATED"


def test_performance_wrapper_measures_without_assuming_native_memory():
    result = measure_callable(lambda: sum(range(10)), repeats=2)
    assert result["repeats"] == 2
    assert "Python-managed" in result["memory_note"]


def test_result_store_never_overwrites_existing_run(tmp_path):
    first = save_result(tmp_path, {"run_id": "fixed", "status": "MEASURED"})
    assert json.loads(first.read_text())["result_hash"]
    with pytest.raises(FileExistsError):
        save_result(tmp_path, {"run_id": "fixed"})


def test_benchmark_dashboard_empty_state_does_not_run_benchmark(tmp_path):
    from streamlit.testing.v1 import AppTest
    script = tmp_path / "benchmark_ui_app.py"
    script.write_text("import streamlit as st\nfrom pathlib import Path\n"
        "from circuitcare.benchmark_ui import render_benchmark_tab\n"
        f"render_benchmark_tab(st, root=Path({str(tmp_path / 'empty')!r}))\n", encoding="utf-8")
    app = AppTest.from_file(str(script), default_timeout=20).run()
    assert not app.exception
    assert any("평가 결과가 없습니다" in item.value for item in app.info)


def test_phase17_development_runner_scores_only_after_proposal_phase(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "phase17-results"
    completed = subprocess.run([sys.executable, "scripts/run_phase17_benchmark.py",
        "--dataset", "evaluation/phase17/development", "--results", str(output)],
        cwd=repository_root, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    result_file = next(output.glob("*.json"))
    result = json.loads(result_file.read_text(encoding="utf-8"))
    assert result["status"] == "MEASURED_SYNTHETIC_DEVELOPMENT"
    assert result["input_audit"]["private_ground_truth_loaded_after_proposals_and_simulation"] is True
    assert result["methods"]["LLM Only"]["status"] == "NOT_EVALUATED"
    assert result["methods"]["Rules Only"]["status"] == result["status"]
    assert result["metric_schema_version"] == "phase17-metrics-v2"
    assert result["methods"]["Rules Only"]["runs"][0]["levels"]["unique_component_count"] > 0


def test_existing_holdout_rerun_is_labeled_non_fresh_and_preserves_method_state(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "reevaluated-holdout"
    completed = subprocess.run([sys.executable, "scripts/run_phase17_benchmark.py",
        "--dataset", "evaluation/phase17/holdout", "--results", str(output),
        "--reuse-existing-holdout"], cwd=repository_root, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(next(output.glob("*.json")).read_text(encoding="utf-8"))
    assert result["status"] == "MEASURED_EXISTING_HOLDOUT_REEVALUATION"
    assert result["independent_holdout"] is False
    assert result["source_dataset_was_independent_holdout"] is True
    assert result["methods"]["Rules Only"]["status"] == result["status"]
    assert result["methods"]["LLM Only"]["status"] == "NOT_EVALUATED"
    assert result["candidate_grid_analysis"]["independent_in_catalog_grid_count"] == 10
    assert result["candidate_grid_analysis"]["independent_out_of_catalog_grid_count"] == 0
