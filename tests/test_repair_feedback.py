import json
import sqlite3

import pytest

from circuitcare.diagnosis_models import DiagnosisHypothesis, Measurement
from circuitcare.fault_ai.hypothesis_ranking import conditional_bayesian_update
from circuitcare.repair_feedback.case_repository import RepairCaseRepository
from circuitcare.repair_feedback.case_similarity import case_fingerprint, find_duplicate_cases
from circuitcare.repair_feedback.case_validator import validate_case
from circuitcare.repair_feedback.diagnosis_feedback import apply_prior_map, compare_prior_diagnoses, create_diagnosis_snapshot
from circuitcare.repair_feedback.evidence_review import review_case
from circuitcare.repair_feedback.evaluation import evaluate_prior_predictions
from circuitcare.repair_feedback.models import new_repair_case
from circuitcare.repair_feedback.outcome_tracker import compare_before_after
from circuitcare.repair_feedback.prior_estimator import estimate_prior, smooth_prior
from circuitcare.repair_feedback.prior_registry import (activate_prior, active_prior, approve_prior,
    deactivate_prior, propose_prior, rollback_prior)
from circuitcare.repair_feedback.training_eligibility import (assess_training_eligibility,
    find_training_evaluation_leakage)


HYPOTHESES = {"U1-rise": 0.5, "U1-drop": 0.5}


def _case(case_id="case-1", *, index=1, source="real_repair"):
    snapshot = {"snapshot_id": f"snapshot-{case_id}", "hypothesis_set_id": "regulator-set-v1",
        "prior_weights": dict(HYPOTHESES),
        "hypotheses": [{"hypothesis_id": key, "prior_weight": value} for key, value in HYPOTHESES.items()]}
    before = {"phase": "before", "node": "VOUT", "value": 7.8 + index * 0.01, "unit": "V",
        "uncertainty": 0.1, "source": "user_measurement", "conditions": {"load_ma": 100}}
    after_measurement = {"phase": "after", "node": "VOUT", "value": 5.0, "unit": "V",
        "uncertainty": 0.1, "source": "user_measurement", "conditions": {"load_ma": 100}}
    case = new_repair_case(case_id=case_id, circuit_id="board-family", circuit_family="regulator-family",
        circuit_hash=f"circuit-hash-{index}", board_instance_id=f"board-instance-{index}",
        data_source=source, component_refs=["U1", "R1"], observations=[before],
        diagnosis_snapshot=snapshot, interventions=[{"component_ref": "U1", "action": "component_replaced"}],
        post_repair_tests=[{"passed": True, "source": "user_measurement", "observations": [after_measurement]}],
        split="development")
    case.update({"verification_status": "ROOT_CAUSE_REVIEWED", "outcome": "RESOLVED", "case_status": "CLOSED",
        "root_cause": {"component_ref": "U1", "fault_type": "regulator_output_rise",
            "hypothesis_id": "U1-rise", "verified": True},
        "training_review": {"status": "APPROVED", "approved": True, "reviewer": "reviewer"}})
    return case


def test_empty_database_does_not_create_a_prior(tmp_path):
    repository = RepairCaseRepository(tmp_path / "local.sqlite3")
    proposal = estimate_prior([], group_id="regulator-family", hypothesis_set_id="regulator-set-v1",
        base_prior=HYPOTHESES)
    assert repository.list_cases() == []
    assert repository.list_prior_models() == []
    assert proposal["status"] == "INSUFFICIENT_VERIFIED_REPAIR_DATA"
    assert proposal["prior_weights"] is None


def test_case_model_defaults_to_unverified_and_prediction_is_not_truth():
    case = new_repair_case(circuit_id="c", circuit_hash="hash", diagnosis_snapshot={"hypotheses": [{"id": "U1"}]})
    assert case["verification_status"] == "UNVERIFIED"
    assert case["training_eligibility"]["status"] == "NOT_ELIGIBLE"
    assert assess_training_eligibility(case, [case])["status"] == "REVIEW_REQUIRED"


def test_case_rejects_bad_reference_units_and_nonfinite_measurements():
    case = _case()
    case["interventions"][0]["component_ref"] = "U99"
    case["observations"][0]["unit"] = "parsec"
    case["observations"][0]["value"] = float("nan")
    assert any("Reference absent" in issue for issue in validate_case(case))
    assert any("unsupported unit" in issue for issue in validate_case(case))
    assert any("finite numeric" in issue for issue in validate_case(case))


def test_sqlite_create_list_events_backup_and_delete_are_local_and_transactional(tmp_path):
    repository = RepairCaseRepository(tmp_path / "private" / "repair.sqlite3")
    case = new_repair_case(case_id="saved", circuit_id="board", circuit_hash="hash")
    repository.create_case(case)
    assert repository.get_case("saved")["case_id"] == "saved"
    repository.add_event("saved", "INSPECTION_RECORDED", {"component_ref": "U1"})
    assert [event["event_type"] for event in repository.list_events("saved")] == ["CASE_CREATED", "INSPECTION_RECORDED"]
    backup_path = tmp_path / "backup.sqlite3"
    repository.backup(backup_path)
    assert backup_path.exists()
    anonymized = repository.export_json(anonymize=True)["cases"][0]
    assert anonymized["case_id"] == "anonymous_00001"
    assert repository.delete_case("saved") is True
    assert repository.get_case("saved") is None
    with sqlite3.connect(backup_path) as db:
        assert db.execute("SELECT count(*) FROM repair_cases").fetchone()[0] == 1


def test_duplicate_case_id_is_rejected(tmp_path):
    repository = RepairCaseRepository(tmp_path / "repair.sqlite3")
    repository.create_case(new_repair_case(case_id="same", circuit_id="board", circuit_hash="hash"))
    with pytest.raises(sqlite3.IntegrityError):
        repository.create_case(new_repair_case(case_id="same", circuit_id="board", circuit_hash="hash"))


def test_repository_rejects_exact_event_and_flags_suspected_duplicate(tmp_path):
    repository = RepairCaseRepository(tmp_path / "repair.sqlite3")
    first = _case("physical-1", index=1)
    repository.create_case(first)
    exact = dict(first, case_id="physical-duplicate")
    with pytest.raises(ValueError, match="Duplicate repair event"):
        repository.create_case(exact)
    suspected = _case("physical-suspected", index=2)
    suspected["board_instance_id"] = first["board_instance_id"]
    suspected["circuit_hash"] = first["circuit_hash"]
    suspected["created_at"] = first["created_at"]
    repository.create_case(suspected)
    stored = repository.get_case("physical-suspected")
    assert stored["duplicate_review"]["status"] == "REVIEW_REQUIRED"
    assert stored["training_eligibility"]["status"] == "REVIEW_REQUIRED"


def test_failed_multi_table_insert_rolls_back_case_and_events(tmp_path):
    repository = RepairCaseRepository(tmp_path / "repair.sqlite3")
    with sqlite3.connect(repository.path) as db:
        db.execute("CREATE TRIGGER fail_intervention BEFORE INSERT ON repair_interventions BEGIN SELECT RAISE(ABORT, 'forced rollback'); END")
    case = new_repair_case(case_id="rollback", circuit_id="board", circuit_hash="hash",
        interventions=[{"component_ref": "U1", "action": "component_replaced"}])
    with pytest.raises(sqlite3.IntegrityError):
        repository.create_case(case)
    assert repository.get_case("rollback") is None
    with sqlite3.connect(repository.path) as db:
        assert db.execute("SELECT count(*) FROM case_events").fetchone()[0] == 0


def test_unsupported_future_database_schema_is_rejected(tmp_path):
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=42")
    with pytest.raises(RuntimeError, match="newer than supported"):
        RepairCaseRepository(path)


def test_original_diagnosis_snapshot_cannot_be_overwritten(tmp_path):
    repository = RepairCaseRepository(tmp_path / "repair.sqlite3")
    case = new_repair_case(case_id="immutable", circuit_id="board", circuit_hash="hash",
        diagnosis_snapshot={"snapshot_id": "before", "hypotheses": []})
    repository.create_case(case)
    case["diagnosis_snapshot"] = {"snapshot_id": "rewritten", "hypotheses": []}
    with pytest.raises(ValueError, match="immutable"):
        repository.update_case(case, event_type="TEST", event_payload={})


def test_prior_reanalysis_is_saved_as_a_separate_snapshot(tmp_path):
    repository = RepairCaseRepository(tmp_path / "repair.sqlite3")
    case = new_repair_case(case_id="snapshot-case", circuit_id="board", circuit_hash="hash",
        diagnosis_snapshot={"snapshot_id": "original", "hypotheses": [{"hypothesis_id": "h1"}]})
    repository.create_case(case)
    later = {"snapshot_id": "later", "snapshot_type": "PRIOR_REANALYSIS", "source_snapshot_id": "original",
        "prior_model_version": "prior-1", "hypotheses": [{"hypothesis_id": "h1", "prior_weight": 0.7}]}
    repository.add_diagnosis_snapshot("snapshot-case", later)
    assert repository.get_case("snapshot-case")["diagnosis_snapshot"]["snapshot_id"] == "original"
    assert [item["snapshot_id"] for item in repository.list_diagnosis_snapshots("snapshot-case")] == ["original", "later"]


def test_human_review_separates_repair_outcome_from_root_cause(tmp_path):
    repository = RepairCaseRepository(tmp_path / "repair.sqlite3")
    repository.create_case(new_repair_case(case_id="review", circuit_id="board", circuit_hash="hash"))
    reviewed = review_case(repository, "review", verification_status="REPAIR_OUTCOME_VERIFIED",
        outcome="RESOLVED", reviewer="A", note="Measured after repair", root_cause=None)
    assert reviewed["outcome"] == "RESOLVED"
    assert reviewed["root_cause"] is None
    assert reviewed["verification_status"] == "REPAIR_OUTCOME_VERIFIED"
    assert repository.list_events("review")[-1]["event_type"] == "EVIDENCE_REVIEWED"


def test_root_cause_review_requires_explicit_component_fault_and_verification(tmp_path):
    repository = RepairCaseRepository(tmp_path / "repair.sqlite3")
    repository.create_case(new_repair_case(case_id="review", circuit_id="board", circuit_hash="hash"))
    with pytest.raises(ValueError, match="verified root_cause"):
        review_case(repository, "review", verification_status="ROOT_CAUSE_REVIEWED",
            outcome="RESOLVED", reviewer="A", root_cause=None)


def test_before_after_requires_same_conditions_and_physical_sources():
    before = [{"node": "VOUT", "value": 7.8, "unit": "V", "uncertainty": 0.1,
        "source": "user_measurement", "conditions": {"load": 1}}]
    after = [{"node": "VOUT", "value": 5.0, "unit": "V", "uncertainty": 0.1,
        "source": "user_measurement", "conditions": {"load": 2}}]
    result = compare_before_after(before, after, expected={"VOUT": {"minimum": 4.9, "maximum": 5.1}})
    assert result["status"] == "NOT_COMPARABLE"
    assert result["comparisons"][0]["delta_after_minus_before"] is None
    after[0]["conditions"] = {"load": 1}
    assert compare_before_after(before, after)["comparisons"][0]["delta_after_minus_before"] == pytest.approx(-2.8)
    before[0]["source"] = "simulated_demo"
    assert compare_before_after(before, after)["status"] == "NOT_COMPARABLE"


def test_training_eligibility_excludes_synthetic_unverified_and_holdout_records():
    assert assess_training_eligibility(_case(source="synthetic_demo"), []) ["status"] == "NOT_ELIGIBLE"
    assert assess_training_eligibility(_case(source="user_reported_repair"), []) ["status"] == "REVIEW_REQUIRED"
    holdout = _case()
    holdout["split"] = "independent_holdout"
    assert assess_training_eligibility(holdout, []) ["status"] == "NOT_ELIGIBLE"
    unverified = _case()
    unverified["verification_status"] = "REPAIR_OUTCOME_VERIFIED"
    assert assess_training_eligibility(unverified, []) ["status"] == "INSUFFICIENT_EVIDENCE"


def test_training_eligibility_requires_physical_matched_before_after_and_review():
    case = _case()
    assert assess_training_eligibility(case, [case], supported_hypothesis_ids=set(HYPOTHESES))["status"] == "ELIGIBLE"
    case["training_review"]["approved"] = False
    assert assess_training_eligibility(case, [case])["status"] == "REVIEW_REQUIRED"
    case["training_review"]["approved"] = True
    case["post_repair_tests"][0]["observations"][0]["conditions"] = {"load_ma": 200}
    assert assess_training_eligibility(case, [case])["status"] == "INSUFFICIENT_EVIDENCE"


def test_multiple_interventions_need_isolating_followup_test():
    case = _case()
    case["component_refs"] = ["U1", "R1"]
    case["interventions"].append({"component_ref": "R1", "action": "component_replaced"})
    assert assess_training_eligibility(case, [case])["status"] == "REVIEW_REQUIRED"
    case["root_cause"]["isolated_by_followup_test"] = True
    assert assess_training_eligibility(case, [case])["status"] == "ELIGIBLE"


def test_duplicate_detection_separates_exact_and_suspected_cases():
    first = _case("first", index=1)
    exact = dict(first, case_id="second")
    assert case_fingerprint(first) == case_fingerprint(exact)
    assert find_duplicate_cases(exact, [first])["status"] == "DUPLICATE_CASE"
    suspected = _case("third", index=2)
    suspected["board_instance_id"] = first["board_instance_id"]
    suspected["circuit_hash"] = first["circuit_hash"]
    suspected["created_at"] = first["created_at"]
    assert find_duplicate_cases(suspected, [first])["status"] == "REVIEW_REQUIRED"


def test_eligibility_blocks_duplicate_and_unsupported_hypothesis():
    case = _case("one")
    duplicate = _case("two", index=2)
    duplicate["board_instance_id"] = case["board_instance_id"]
    duplicate["circuit_hash"] = case["circuit_hash"]
    duplicate["created_at"] = case["created_at"]
    assert assess_training_eligibility(duplicate, [case])["status"] == "REVIEW_REQUIRED"
    assert assess_training_eligibility(case, [case], supported_hypothesis_ids={"not-supported"})["status"] == "UNSUPPORTED_FAULT_CLASS"


def test_smoothing_matches_configured_dirichlet_shrinkage_and_positive_support():
    result = smooth_prior({"h1": 0.5, "h2": 0.5}, {"h1": 2, "h2": 0}, alpha=2)
    assert result == pytest.approx({"h1": 0.75, "h2": 0.25})
    assert all(value > 0 for value in result.values())
    assert sum(result.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        smooth_prior({"h1": 1.0, "h2": 0.0}, {}, alpha=1)


def test_prior_estimator_uses_only_eligible_real_cases_and_requires_minimum():
    cases = [_case(f"case-{index}", index=index) for index in range(1, 5)]
    result = estimate_prior(cases, group_id="regulator-family", hypothesis_set_id="regulator-set-v1",
        base_prior=HYPOTHESES, alpha=5, minimum_cases=5)
    assert result["status"] == "INSUFFICIENT_VERIFIED_REPAIR_DATA"
    assert result["prior_weights"] is None
    cases.append(_case("case-5", index=5))
    cases.append(_case("synthetic", index=6, source="synthetic_demo"))
    result = estimate_prior(cases, group_id="regulator-family", hypothesis_set_id="regulator-set-v1",
        base_prior=HYPOTHESES, alpha=5, minimum_cases=5)
    assert result["status"] == "PROPOSAL_READY"
    assert result["eligible_case_count"] == 5
    assert result["counts"] == {"U1-rise": 5, "U1-drop": 0}
    assert result["prior_weights"] == pytest.approx({"U1-rise": 0.75, "U1-drop": 0.25})
    assert result["contains_synthetic_records"] is False
    assert "synthetic" in {item["case_id"] for item in result["excluded_cases"]}


def test_prior_estimator_deduplicates_case_ids_and_requires_matching_group():
    case = _case("one")
    result = estimate_prior([case, case], group_id="regulator-family", hypothesis_set_id="regulator-set-v1",
        base_prior=HYPOTHESES, minimum_cases=1)
    assert result["eligible_case_count"] == 0
    assert result["duplicate_case_ids"] == ["one"]
    other_group = estimate_prior([case], group_id="other", hypothesis_set_id="regulator-set-v1",
        base_prior=HYPOTHESES, minimum_cases=1)
    assert other_group["candidate_case_count"] == 0


def test_prior_registry_requires_approval_and_supports_activate_deactivate_rollback(tmp_path):
    repository = RepairCaseRepository(tmp_path / "repair.sqlite3")
    cases = [_case(f"case-{index}", index=index) for index in range(1, 6)]
    for case in cases:
        repository.create_case(case)
    first = estimate_prior(cases, group_id="regulator-family", hypothesis_set_id="regulator-set-v1",
        base_prior=HYPOTHESES, minimum_cases=5)
    saved = propose_prior(repository, first)
    with pytest.raises(ValueError, match="human-approved"):
        activate_prior(repository, saved["version"], reviewer="A")
    approve_prior(repository, saved["version"], reviewer="A", note="verified")
    activate_prior(repository, saved["version"], reviewer="A")
    assert active_prior(repository, "regulator-family")["version"] == saved["version"]
    assert deactivate_prior(repository, saved["version"], reviewer="A")["status"] == "APPROVED"
    cases.append(_case("case-6", index=6))
    repository.create_case(cases[-1])
    second = estimate_prior(cases, group_id="regulator-family", hypothesis_set_id="regulator-set-v1",
        base_prior=HYPOTHESES, minimum_cases=5)
    propose_prior(repository, second)
    approve_prior(repository, second["version"], reviewer="A")
    activate_prior(repository, second["version"], reviewer="A")
    assert rollback_prior(repository, group_id="regulator-family", target_version=saved["version"], reviewer="A")["version"] == saved["version"]
    assert active_prior(repository, "regulator-family")["version"] == saved["version"]
    with repository._connect() as db:
        db.execute("CREATE TRIGGER fail_prior_event BEFORE INSERT ON prior_activation_events BEGIN SELECT RAISE(ABORT, 'forced failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match="forced failure"):
        rollback_prior(repository, group_id="regulator-family", target_version=second["version"], reviewer="A")
    assert active_prior(repository, "regulator-family")["version"] == saved["version"]
    assert [event["action"] for event in repository.list_prior_events()]


def test_prior_registry_rejects_operational_threshold_below_five(tmp_path):
    repository = RepairCaseRepository(tmp_path / "repair.sqlite3")
    one_case = [_case("single")]
    proposal = estimate_prior(one_case, group_id="regulator-family", hypothesis_set_id="regulator-set-v1",
        base_prior=HYPOTHESES, minimum_cases=1)
    assert proposal["status"] == "PROPOSAL_READY"
    with pytest.raises(ValueError, match="minimum of 5"):
        propose_prior(repository, proposal)


def test_deleting_a_training_case_invalidates_and_deactivates_its_prior(tmp_path):
    repository = RepairCaseRepository(tmp_path / "repair.sqlite3")
    cases = [_case(f"case-{index}", index=index) for index in range(1, 6)]
    for case in cases:
        repository.create_case(case)
    proposal = estimate_prior(repository.list_cases(), group_id="regulator-family",
        hypothesis_set_id="regulator-set-v1", base_prior=HYPOTHESES, minimum_cases=5)
    saved = propose_prior(repository, proposal)
    approve_prior(repository, saved["version"], reviewer="A")
    activate_prior(repository, saved["version"], reviewer="A")
    assert repository.delete_case("case-1") is True
    invalidated = repository.get_prior_model(saved["version"])
    assert invalidated["status"] == "INVALIDATED_SOURCE_CHANGE"
    assert "case-1" not in json.dumps(invalidated)
    assert invalidated["prior_weights"] is None
    assert active_prior(repository, "regulator-family") is None
    with pytest.raises(ValueError, match="Invalidated prior"):
        rollback_prior(repository, group_id="regulator-family", target_version=saved["version"], reviewer="A")
    assert active_prior(repository, "regulator-family") is None
    assert repository.get_case("case-1") is None
    assert repository.list_prior_events()[-1]["action"] == "INVALIDATED_SOURCE_CHANGE"


def test_anonymized_export_removes_free_text_and_identifiers(tmp_path):
    repository = RepairCaseRepository(tmp_path / "repair.sqlite3")
    case = new_repair_case(case_id="private-id", circuit_id="customer-board", circuit_hash="board-fingerprint",
        board_instance_id="serial-123", source_case_id="ticket-45", reported_symptoms=["personal detail"],
        private_notes="private note", created_by="operator")
    repository.create_case(case)
    exported = repository.export_json(anonymize=True)["cases"][0]
    as_text = json.dumps(exported)
    for private in ("customer-board", "board-fingerprint", "serial-123", "ticket-45", "personal detail", "private note", "operator"):
        assert private not in as_text
    assert exported["case_id"] == "anonymous_00001"


def test_snapshot_is_copy_with_version_and_prior_provenance():
    diagnosis = {"posterior": [{"hypothesis_id": "h1", "prior_weight": 1.0}], "used_measurements": [{"node": "VOUT"}], "model_mismatch": False}
    snapshot = create_diagnosis_snapshot(circuit_id="board", circuit_hash="hash", diagnosis=diagnosis,
        hypothesis_rows=[{"hypothesis_id": "h1", "predicted_measurements": {"VOUT": 5.0}}], model_version="local-1", prompt_version="prompt-2")
    diagnosis["posterior"][0]["prior_weight"] = 0.0
    assert snapshot["prior_weights"] == {"h1": 1.0}
    assert snapshot["hypothesis_definitions"][0]["predicted_measurements"]["VOUT"] == 5.0
    assert snapshot["snapshot_status"] == "ORIGINAL_IMMUTABLE"


def _diagnosis_hypotheses():
    return [DiagnosisHypothesis("h1", "U1", "rise", {}, 1.0, {"VOUT": 5.0}, "spice-a", "", "s1", "board"),
        DiagnosisHypothesis("h2", "U1", "drop", {}, 1.0, {"VOUT": 8.0}, "spice-b", "", "s2", "board")]


def test_prior_map_validation_and_same_evidence_bayesian_comparison():
    hypotheses = _diagnosis_hypotheses()
    measurement = [Measurement("VOUT", 7.9, "V", 10.0, "user_measurement")]
    with pytest.raises(ValueError, match="exactly match"):
        apply_prior_map(hypotheses, {"h1": 1.0})
    with pytest.raises(ValueError, match="strictly positive"):
        apply_prior_map(hypotheses, {"h1": 1.0, "h2": 0.0})
    result = compare_prior_diagnoses(hypotheses, measurement, learned_prior_weights={"h1": 0.9, "h2": 0.1}, learned_prior_version="prior-v1")
    assert result["status"] == "COMPARED"
    assert result["learned_prior_version"] == "prior-v1"
    assert result["default"]["used_measurements"] == result["learned"]["used_measurements"]
    assert result["posterior_comparison"][0]["default_posterior"] != result["posterior_comparison"][0]["learned_posterior"]
    assert result["improvement_claim"].startswith("NOT_EVALUATED")


def test_fault_bayesian_adapter_defaults_and_optionally_applies_approved_prior():
    rows = [{"hypothesis_id": "h1", "component_ref": "U1", "fault_type": "rise", "parameters": {},
        "simulation_status": "SIMULATION_SUCCESS", "predicted_measurements": {"VOUT": 5.0}},
        {"hypothesis_id": "h2", "component_ref": "U1", "fault_type": "drop", "parameters": {},
        "simulation_status": "SIMULATION_SUCCESS", "predicted_measurements": {"VOUT": 8.0}}]
    observations = [{"node": "VOUT", "value": 7.9, "unit": "V", "uncertainty": 10.0, "source": "user_measurement"}]
    default = conditional_bayesian_update(rows, observations)
    learned = conditional_bayesian_update(rows, observations, prior_weights={"h1": 0.9, "h2": 0.1}, prior_model_version="prior-1")
    assert default["prior_source"] == "DEFAULT_EQUAL"
    assert learned["prior_source"] == "APPROVED_LEARNED"
    assert learned["prior_model_version"] == "prior-1"
    assert learned["posterior"] != default["posterior"]
    invalid = conditional_bayesian_update(rows, observations, prior_weights={"h1": 1.0})
    assert invalid["status"] == "INACTIVE"


def test_prior_evaluation_empty_and_board_leakage_are_not_reported_as_accuracy():
    assert evaluate_prior_predictions([], [])["status"] == "REAL_REPAIR_PRIOR_PERFORMANCE_NOT_EVALUATED"
    training = [{"case_id": "train", "board_instance_id": "board-1"}]
    evaluation = [{"case_id": "test", "board_instance_id": "board-1", "split": "independent_holdout"}]
    report = evaluate_prior_predictions(evaluation, training)
    assert report["status"] == "INVALID_EVALUATION"
    assert report["integrity_audit"]["overlapping_board_instance_ids"] == ["board-1"]


def test_independent_synthetic_fixture_is_not_real_repair_performance():
    record = {"case_id": "synthetic-test", "split": "independent_holdout", "data_source": "synthetic_demo",
        "true_hypothesis_id": "h1", "default_posterior": {"h1": 0.8, "h2": 0.2},
        "learned_posterior": {"h1": 0.7, "h2": 0.3}}
    report = evaluate_prior_predictions([record], [])
    assert report["status"] == "REAL_REPAIR_PRIOR_PERFORMANCE_NOT_EVALUATED"
    assert report["default_prior"]["top1_accuracy"]["value"] == 1.0
    assert report["default_prior"]["calibration_error"]["value"] is None


def test_training_evaluation_leakage_checks_case_ids_and_board_ids():
    result = find_training_evaluation_leakage([{"case_id": "x", "board_instance_id": "board"}],
        [{"case_id": "x", "board_instance_id": "board"}])
    assert result["status"] == "INVALID_EVALUATION"
    assert result["overlapping_case_ids"] == ["x"]
    assert result["overlapping_board_instance_ids"] == ["board"]

