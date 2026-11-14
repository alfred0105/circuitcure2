from copy import deepcopy

import pytest

from circuitcare.benchmark.pcb_validation import (
    evaluate_runs, run_case, score_run, validate_case,
)
from circuitcare.frontend.board_component_registry import (
    build_board_component_registry, current_source_observation_fingerprint,
    resolve_measurement_board_component,
)
from circuitcare.frontend.measurement_models import create_measurement_record
from circuitcare.dashboard.views.measurement_panel import _measurement_verification_gaps
from circuitcare.symptom_diagnosis.measurement_evaluator import evaluate_component_measurements
from circuitcare.symptom_diagnosis.measurement_validation import measurement_verification_gaps
from circuitcare.symptom_diagnosis.evidence_strategy import (
    optimize_safe_tests, rank_by_verified_evidence,
)


def _profile():
    return {"profiles": {"resistor": {"tests": {"resistance": {
        "measurement_type": "resistance", "target_keywords": ["A-B"],
        "condition_keywords": ["power off"], "fault_relations": {
            "R_FAIL": {"TEST_NORMAL": "supports", "TEST_ABNORMAL": "refutes"}}}}}}}


def _measurement(value=1.5):
    return {"measurement_id": f"m-{value}", "component_id": "bc-r1", "board_component_id": "bc-r1",
            "identity_resolution_status": "USER_RECONFIRMED_VERIFIED", "component_identity_status": "USER_CONFIRMED",
            "fault_hypothesis": "R_FAIL", "test_id": "resistance", "measurement_target": "A-B",
            "measurement_location": "A-B", "operating_conditions": "power off", "reference_conditions": "power off",
            "location_verified": True, "source": "user_measurement", "verification_status": "USER_REPORTED",
            "uncertainty": 0.01, "uncertainty_source": "meter specification", "reference_min": 1,
            "reference_max": 2, "reference_source": "datasheet", "reference_source_id": "ds-r1",
            "reference_verification_status": "DATASHEET_VERIFIED", "component_type": "resistor",
            "measurement_type": "resistance", "value": value}


def test_ui_and_engine_share_measurement_requirements_without_node_mapping():
    row = create_measurement_record(project_id="p", board_id="b", component_id="c",
        board_component_id="bc-c", component_identity_status="USER_CONFIRMED", measurement_type="voltage",
        value="2", unit="V", source="user_measurement")
    gaps_ui = _measurement_verification_gaps(row)
    gaps_engine = measurement_verification_gaps(row, profiles=_profile())
    assert gaps_ui == gaps_engine
    assert "전기적 측정 지점 연결 미확인" not in gaps_ui
    assert row.mapping_status == "NODE_MAPPING_UNAVAILABLE"


def test_conflicting_valid_measurements_are_conflicted_but_missing_requirements_abstain():
    readings = [_measurement(1.5), _measurement(9.0)]
    result = evaluate_component_measurements(readings, profiles=_profile())
    assert next(iter(result.values()))["status"] == "CONFLICTED"
    invalid = _measurement(9.0)
    invalid["location_verified"] = False
    outcome = evaluate_component_measurements([invalid], profiles=_profile())
    assert next(iter(outcome.values()))["status"] == "INCONCLUSIVE"
    assert "측정 지점 확인 전" in next(iter(outcome.values()))["reason"]


def test_structured_power_state_rejects_negated_or_unconfirmed_safety_text():
    profiles = {"profiles": {"resistor": {"tests": {"resistance": {
        "measurement_type": "resistance", "target_keywords": ["A-B"],
        "required_conditions": {"power_state": "POWER_OFF_CONFIRMED",
                                "stored_energy_status": "SAFE_CONFIRMED",
                                "component_isolation_status": "ISOLATED_CONFIRMED"},
        "safety_protocol": "DEENERGIZED_ISOLATED_PASSIVE",
        "fault_relations": {"R_FAIL": {"TEST_NORMAL": "refutes", "TEST_ABNORMAL": "supports"}},
    }}}}}
    unsafe = _measurement()
    unsafe["measurement_conditions"] = {
        "power_state": "POWER_ON_CONFIRMED",
        "stored_energy_status": "UNKNOWN",
        "component_isolation_status": "NOT_ISOLATED",
    }
    unsafe["operating_conditions"] = "전원 차단 안 함"
    assert "전원 차단 상태 확인 전" in measurement_verification_gaps(unsafe, profiles=profiles)
    assert "잔류 에너지 안전 확인 전" in measurement_verification_gaps(unsafe, profiles=profiles)
    assert "측정 대상 분리 상태 확인 전" in measurement_verification_gaps(unsafe, profiles=profiles)
    unsafe["measurement_conditions"] = {
        "power_state": "POWER_OFF_CONFIRMED",
        "stored_energy_status": "SAFE_CONFIRMED",
        "component_isolation_status": "ISOLATED_CONFIRMED",
    }
    safe_gaps = measurement_verification_gaps(unsafe, profiles=profiles)
    assert not any("안전 확인 전" in gap or "전원 차단 상태 확인 전" in gap for gap in safe_gaps)


def test_observation_fingerprint_blocks_same_photo_reused_id_after_redetection():
    base = {"version": "v1", "image_fingerprint": "photo-sha", "review_version": "review1",
            "registry_components": [{"component_id": "R1", "component_type": "resistor",
              "bbox_xyxy": [1, 2, 20, 30], "identification_status": "USER_CONFIRMED",
              "presence_status": "CONFIRMED", "component_type_status": "USER_CONFIRMED"}]}
    registry1 = build_board_component_registry({}, project_id="p", board_id="b", photo_snapshots={"photo": base})
    fp1 = current_source_observation_fingerprint(registry1, photo_id="photo", component_id="R1")
    current, status = resolve_measurement_board_component(registry1, photo_id="photo", component_id="R1",
        photo_fingerprint="photo-sha", observation_fingerprint=fp1)
    assert current and status == "VERIFIED_SOURCE_IDENTITY"
    changed = deepcopy(base)
    changed["version"] = "v2"
    changed["registry_components"][0]["bbox_xyxy"] = [30, 40, 50, 60]
    registry2 = build_board_component_registry({}, project_id="p", board_id="b", photo_snapshots={"photo": changed})
    fp2 = current_source_observation_fingerprint(registry2, photo_id="photo", component_id="R1")
    assert fp2 != fp1
    current, status = resolve_measurement_board_component(registry2, photo_id="photo", component_id="R1",
        photo_fingerprint="photo-sha", observation_fingerprint=fp1)
    assert current is None and status == "RECONFIRMATION_REQUIRED"


def _case():
    return {"case_id": "fixture-mock-1", "board_id": "board-1", "case_kind": "MOCK",
            "photo_paths": [], "inference_input": {"project_id": "project-1", "symptoms": ["NO_POWER"],
                "components": [], "measurements": []},
            "reference": {"adjudication_status": "UNKNOWN", "root_cause_ids": [], "evidence_sources": []}}


def test_harness_uses_production_diagnose_and_separates_truth():
    case = _case()
    validate_case(case)
    run = run_case(case)
    assert run["status"].startswith(("CANDIDATES", "ABSTAIN"))
    assert "reference" not in run
    assert score_run(case, run)["status"] == "NOT_EVALUATED"
    metrics = evaluate_runs([case], [run])
    assert metrics["top1_accuracy"]["status"] == "NOT_EVALUATED"
    leaked = deepcopy(case)
    leaked["inference_input"]["fault_type"] = "DO_NOT_LEAK"
    with pytest.raises(ValueError, match="Gold-answer"):
        validate_case(leaked)


def test_mock_model_stages_are_separate_and_llm_failure_keeps_rule_diagnosis():
    case = _case()
    run = run_case(case, detector=lambda _inputs: {"detections": [], "status": "MOCK"},
        reviewer=lambda _inputs, result: {"components": [], "status": "MOCK_REVIEW"},
        llm=lambda _result: (_ for _ in ()).throw(RuntimeError("offline")))
    assert run["status"] != "PIPELINE_FAILED"
    assert run["llm"]["status"] == "FAILED_RULES_RETAINED"
    assert [step["stage"].split(":", 1)[0] for step in run["stage_trace"]][:2] == ["yolo_detection", "component_review"]


def test_block_recall_at_three_uses_only_adjudicated_block_ids_and_reports_fraction():
    case = _case()
    case["reference"] = {
        "adjudication_status": "CONFIRMED", "fault_type_ids": ["F1"],
        "faulty_functional_block_ids": ["power", "control"],
        "evidence_sources": ["independent_repair_record"],
    }
    run = {"case_id": case["case_id"], "execution_mode": "MOCK", "status": "CANDIDATES_AVAILABLE",
           "hypotheses": [{"fault_id": "F1", "score": 1}],
           "functional_block_hypotheses": [
               {"functional_block_id": "power", "score": 2},
               {"functional_block_id": "driver", "score": 1},
               {"functional_block_id": "input", "score": .5},
           ]}
    scored = score_run(case, run)
    assert scored["block_recall_at_3"] == .5
    assert scored["block_recall_status"] == "EVALUATED"
    aggregate = evaluate_runs([case], [run])
    assert aggregate["block_recall_at_3"]["value"] == .5
    assert aggregate["block_recall_at_3"]["n"] == 1


def test_active_plan_recommendations_are_scored_and_dataset_kinds_are_separated():
    mock_case = _case()
    mock_case["reference"] = {"adjudication_status": "CONFIRMED", "fault_type_ids": ["F1"],
        "useful_test_ids": ["T1"], "recommendation_labels_status": "CONFIRMED",
        "evidence_sources": ["fixture-label"]}
    synthetic_case = deepcopy(mock_case)
    synthetic_case["case_id"] = "fixture-synthetic-1"
    synthetic_case["case_kind"] = "SYNTHETIC"
    mock_run = {"case_id": mock_case["case_id"], "case_kind": "MOCK", "execution_mode": "MOCK",
        "strategy": "active_inspection_v1", "status": "CANDIDATES_AVAILABLE",
        "hypotheses": [{"fault_id": "F1", "score": 1}], "top_candidates": [],
        "all_component_candidates": [], "next_tests": [], "measurement_evidence": [],
        "active_inspection_plan": {"candidates": [{"test_id": "T1", "separation_count": 1}]},
        "measurement_comparison": {"reassessment_duration_ms": 12.5}}
    synthetic_run = {**mock_run, "case_id": synthetic_case["case_id"], "case_kind": "SYNTHETIC",
                     "active_inspection_plan": {"candidates": [{"test_id": "T2", "separation_count": 0}]}}
    mock_score = score_run(mock_case, mock_run)
    assert mock_score["useful_test_recommended"] is True
    assert mock_score["reassessment_duration_ms"] == 12.5
    report = evaluate_runs([mock_case, synthetic_case], [mock_run, synthetic_run])
    assert set(report["dataset_breakdown"]) == {"MOCK", "SYNTHETIC"}
    assert report["dataset_breakdown"]["MOCK"]["cases"][0]["useful_test_recommended"] is True


def test_top3_scoring_and_detection_metrics_require_adjudicated_labels():
    case = _case()
    case["reference"] = {"adjudication_status": "CONFIRMED", "root_cause_ids": ["F1", "F2"],
        "gold_components_status": "CONFIRMED", "gold_components": [{"bbox_xyxy": [0, 0, 10, 10],
        "component_type": "resistor"}], "evidence_sources": ["repair_record"]}
    run = {"case_id": case["case_id"], "status": "CANDIDATES_AVAILABLE",
        "hypotheses": [{"fault_id": "F3"}, {"fault_id": "F2"}], "top_candidates": [],
        "detections": [{"bbox_xyxy": [0, 0, 10, 10], "class_name": "resistor"}],
        "measurement_evidence": []}
    score = score_run(case, run)
    assert score["top1_hit"] is False and score["top3_hit"] is True
    metrics = evaluate_runs([case], [run])
    assert metrics["detection_precision_recall_f1"]["f1"] == 1
    assert metrics["top3_hit_rate"]["n"] == 1
    assert metrics["top1_accuracy"]["confidence_interval_95_wilson"] is not None


def test_fault_type_component_and_joint_topk_are_scored_independently_with_ties_and_dedup():
    case = _case()
    case["case_kind"] = "REAL_CONFIRMED"
    case["photo_paths"] = ["unused-in-test.png"]
    case["reference"] = {"adjudication_status": "CONFIRMED", "fault_type_ids": ["F_SHORT"],
        "faulty_board_component_ids": ["bc-R7"], "fault_pairs": [
            {"fault_type_id": "F_SHORT", "board_component_id": "bc-R7"}],
        "evidence_sources": ["repair_record"]}
    run = {"case_id": case["case_id"], "execution_mode": "REAL_MODEL", "status": "CANDIDATES_AVAILABLE",
        "hypotheses": [{"fault_id": "F_OPEN", "score": 2}, {"fault_id": "F_SHORT", "score": 1}],
        "top_candidates": [
            {"fault_hypothesis": "F_OPEN", "component_id": "bc-R7", "score": 4},
            {"fault_hypothesis": "F_SHORT", "component_id": "bc-R7", "score": 3},
            {"fault_hypothesis": "F_SHORT", "component_id": "bc-R7", "score": 2}],
        "detections": [], "measurement_evidence": []}
    score = score_run(case, run)
    assert score["fault_type_top1_hit"] is False and score["fault_type_top3_hit"] is True
    assert score["component_top1_hit"] is True
    assert score["joint_top1_hit"] is False and score["joint_top3_hit"] is True
    report = evaluate_runs([case], [run])
    assert report["fault_type_top1"]["n"] == 1
    assert report["joint_top3"]["value"] == 1


def test_topk_cutoff_includes_all_equal_score_ties_without_duplicate_ids():
    case = _case()
    case["reference"] = {"adjudication_status": "CONFIRMED", "fault_type_ids": ["F2"],
                         "evidence_sources": ["repair_record"]}
    run = {"execution_mode": "MOCK", "hypotheses": [
        {"fault_id": "F1", "score": 1}, {"fault_id": "F2", "score": 1},
        {"fault_id": "F2", "score": 1}], "top_candidates": [], "detections": []}
    score = score_run(case, run)
    assert score["fault_type_top1_hit"] is True
    assert score["fault_type_top3_hit"] is True


def test_non_component_fault_does_not_score_component_location():
    case = _case()
    case["case_kind"] = "REAL_NON_COMPONENT"
    case["photo_paths"] = ["unused-in-test.png"]
    case["reference"] = {"adjudication_status": "CONFIRMED", "fault_type_ids": ["SOLDER_OPEN"],
                         "faulty_board_component_ids": ["must-not-be-scored"],
                         "evidence_sources": ["repair_record"]}
    run = {"execution_mode": "REAL_MODEL", "hypotheses": [{"fault_id": "SOLDER_OPEN"}],
           "top_candidates": [{"component_id": "must-not-be-scored", "score": 99}], "detections": []}
    score = score_run(case, run)
    assert score["fault_type_top1_hit"] is True
    assert score["component_metrics_status"] == "NOT_APPLICABLE"
    assert score["component_top1_hit"] is None


def test_topk_evaluation_uses_full_rankings_before_display_top3_slice():
    case = _case()
    case["case_kind"] = "REAL_CONFIRMED"
    case["photo_paths"] = ["unused-in-test.png"]
    case["reference"] = {"adjudication_status": "CONFIRMED", "fault_type_ids": ["F4"],
        "faulty_board_component_ids": ["bc4"],
        "fault_pairs": [{"fault_type_id": "F4", "board_component_id": "bc4"}],
        "evidence_sources": ["repair_record"]}
    pairs = [{"fault_hypothesis": f"F{i}", "component_id": f"local{i}",
              "board_component_id": f"bc{i}", "score": score}
             for i, score in enumerate((5, 4, 3, 3), start=1)]
    run = {"execution_mode": "REAL_MODEL", "hypotheses": [{"fault_id": f"F{i}", "score": score}
                                                               for i, score in enumerate((5, 4, 3), start=1)],
        "all_hypotheses": [{"fault_id": f"F{i}", "score": score}
                           for i, score in enumerate((5, 4, 3, 3), start=1)],
        "top_candidates": pairs[:3], "all_component_candidates": pairs, "detections": []}
    score = score_run(case, run)
    assert score["fault_type_top3_hit"] is True
    assert score["component_top3_hit"] is True
    assert score["joint_top3_hit"] is True


def test_evidence_strategy_reranks_full_pairs_before_selecting_unique_display_top3():
    baseline_candidates = [
        {"component_id": f"bc{i}", "fault_hypothesis": "F1", "score": float(7-i),
         "score_breakdown": {"validated_measurement": 0.0}, "measurement_verdict": "non_diagnostic"}
        for i in range(1, 5)
    ]
    baseline = {"hypotheses": [{"fault_id": "F1", "score": 4, "evidence_details": []}],
        "component_candidates": baseline_candidates[:3], "all_component_candidates": baseline_candidates,
        "measurement_evidence": [
            {"measurement_id": f"m{i}", "component_id": f"bc{i}", "fault_hypothesis": "F1",
             "verdict": "refutes" if i < 4 else "supports", "assessment": "SUPPORTED",
             "source": "user_measurement"}
            for i in range(1, 5)], "functional_blocks": [], "observations": {}}
    result = rank_by_verified_evidence(baseline, components=[], measurements=[], observations=None)
    assert len(result["all_component_candidates"]) == 4
    assert result["component_candidates"][0]["component_id"] == "bc4"
    assert len(result["component_candidates"]) == 3


def test_real_cases_are_not_scored_from_mock_or_synthetic_execution():
    case = _case()
    case["case_kind"] = "REAL_CONFIRMED"
    case["photo_paths"] = ["unused-in-test.png"]
    case["reference"] = {"adjudication_status": "CONFIRMED", "fault_type_ids": ["F1"],
                         "evidence_sources": ["repair_record"]}
    assert score_run(case, {"execution_mode": "MOCK"})["status"] == "NOT_EVALUATED"


def test_evidence_strategy_deduplicates_measurement_and_ignores_conflict_and_llm_guess():
    baseline = {"hypotheses": [{"fault_id": "F1", "score": 8, "evidence_details": [
        {"code": "NO_POWER", "source": "user_report", "status": "observed", "weight": 2},
        {"code": "NO_POWER", "source": "user_report", "status": "observed", "weight": 2},
        {"code": "LLM_FAULT_GUESS", "source": "qwen_suggestion", "status": "observed", "weight": 99}]}],
        "component_candidates": [], "measurement_evidence": [
            {"measurement_id": "m1", "fault_hypothesis": "F1", "component_id": "bc1",
             "verdict": "supports", "assessment": "SUPPORTED"},
            {"measurement_id": "m1", "fault_hypothesis": "F1", "component_id": "bc1",
             "verdict": "supports", "assessment": "SUPPORTED"},
            {"measurement_id": "m2", "fault_hypothesis": "F1", "component_id": "bc1",
             "verdict": "conflicted", "assessment": "CONFLICTED"}],
        "functional_blocks": []}
    result = rank_by_verified_evidence(baseline, components=[], measurements=[], observations=None)
    assert result["hypotheses"][0]["score"] == 1.9
    assert result["hypotheses"][0]["baseline_score"] == 8
    assert result["score_semantics"].endswith("not a probability")
    assert len(result["hypotheses"][0]["evidence_details"]) == 2


def test_safe_test_order_excludes_unsafe_completed_duplicate_and_unmet_prerequisite():
    tests = [
        {"test_id": "safe", "risk_level": "none", "answer_score_simulation": {
            "yes": [("F1", 2)], "no": [("F2", 1)]}, "expected_duration_seconds": 30},
        {"test_id": "unsafe", "risk_level": "high", "answer_score_simulation": {}},
        {"test_id": "poweroff", "risk_level": "low", "requires_power_off": True,
         "answer_score_simulation": {"yes": [("F1", 2)], "no": [("F2", 1)]}},
        {"test_id": "done", "risk_level": "none", "answer_score_simulation": {"yes": [], "no": []}},
        {"test_id": "safe", "risk_level": "none", "answer_score_simulation": {"yes": [], "no": []}},
    ]
    ranked = optimize_safe_tests(tests, observations={"POWER_OFF_CONFIRMED": False},
                                 measurements=[{"test_id": "done"}])
    assert [row["test_id"] for row in ranked] == ["safe"]
    assert ranked[0]["separation_score"] == 1


def test_shared_registry_projection_keeps_only_matching_photo_observation_active():
    snapshot = {"version": "map-v1", "image_fingerprint": "photo-sha", "review_version": "r1",
        "registry_components": [{"component_id": "R1", "component_type": "resistor",
            "bbox_xyxy": [1, 2, 20, 30], "identification_status": "USER_CONFIRMED",
            "presence_status": "CONFIRMED", "component_type_status": "USER_CONFIRMED"}]}
    registry = build_board_component_registry({}, project_id="p", board_id="b", photo_snapshots={"front": snapshot})
    fp = current_source_observation_fingerprint(registry, photo_id="front", component_id="R1")
    row = {**_measurement(), "component_id": "R1", "source_component_id": "R1",
        "source_photo_id": "front", "source_photo_fingerprint": "photo-sha",
        "source_observation_fingerprint": fp, "component_map_version": "map-v1"}
    from circuitcare.frontend.board_component_registry import project_measurements_to_registry
    history, current = project_measurements_to_registry([row], registry)
    assert len(history) == len(current) == 1
    assert current[0]["board_component_id"] == registry["components"][0]["board_component_id"]
    stale = {**row, "source_observation_fingerprint": "old-fingerprint"}
    archived, active = project_measurements_to_registry([stale], registry)
    assert len(archived) == 1 and active == []


def test_multi_photo_harness_builds_registry_from_each_photo_snapshot(tmp_path):
    photos = [tmp_path / "front.jpg", tmp_path / "detail.jpg"]
    for path in photos:
        path.write_bytes(path.name.encode())
    case = _case()
    case["photo_paths"] = [{"photo_id": "front", "path": str(photos[0])},
                           {"photo_id": "detail", "path": str(photos[1])}]
    case["inference_input"]["components"] = [{"component_id": "R1", "component_type": "resistor",
        "bbox_xyxy": [1, 2, 20, 30], "presence_status": "CONFIRMED"}]
    run = run_case(case, execution_mode="MOCK", detector=lambda inputs: {
        "detections": [{"component_id": "R1", "class_name": "resistor", "bbox_xyxy": [1, 2, 20, 30]}]})
    detection_stages = [row for row in run["stage_trace"] if row["stage"].startswith("yolo_detection:")]
    assert len(detection_stages) == 2
    assert run["registry_component_count"] == 2
