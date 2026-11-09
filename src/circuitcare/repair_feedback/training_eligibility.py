"""Evidence-only training eligibility and leakage checks."""

from __future__ import annotations

from typing import Any

from .case_similarity import find_duplicate_cases
from .outcome_tracker import compare_before_after

_PHYSICAL_SOURCES = {"user_measurement", "instrument_recorded"}


def assess_training_eligibility(case: dict[str, Any], existing_cases: list[dict[str, Any]], *,
                                supported_hypothesis_ids: set[str] | None = None) -> dict[str, Any]:
    supported_hypothesis_ids = supported_hypothesis_ids or set()
    if case.get("data_source") in {"simulated_fault", "synthetic_demo", "llm_prediction", "unverified_observation"}:
        return _result("NOT_ELIGIBLE", "Synthetic, simulated, predicted, or unverified records cannot train real-repair priors.")
    if case.get("data_source") != "real_repair":
        return _result("REVIEW_REQUIRED", "A user-reported repair needs independent verification before use.")
    if case.get("split") in {"validation", "holdout", "independent_holdout", "test"}:
        return _result("NOT_ELIGIBLE", "Validation and holdout records cannot enter prior training.")
    duplicate = find_duplicate_cases(case, [item for item in existing_cases if item.get("case_id") != case.get("case_id")])
    if duplicate["status"] == "DUPLICATE_CASE":
        return {**duplicate, "reasons": ["This physical repair event is already represented."]}
    if duplicate["status"] == "REVIEW_REQUIRED":
        return {**duplicate, "reasons": ["Possible repeated event on the same board needs human review."]}
    if case.get("verification_status") != "ROOT_CAUSE_REVIEWED":
        return _result("INSUFFICIENT_EVIDENCE", "Root cause has not been reviewed and confirmed.")
    if case.get("outcome") not in {"RESOLVED", "PARTIALLY_RESOLVED"}:
        return _result("INSUFFICIENT_EVIDENCE", "A documented post-repair outcome is required.")
    if not (case.get("training_review") or {}).get("approved"):
        return _result("REVIEW_REQUIRED", "A reviewer must explicitly approve use for prior estimation.")
    cause = case.get("root_cause") or {}
    if not cause.get("verified") or not cause.get("hypothesis_id"):
        return _result("INSUFFICIENT_EVIDENCE", "A verified and comparable root-cause hypothesis ID is required.")
    if supported_hypothesis_ids and cause["hypothesis_id"] not in supported_hypothesis_ids:
        return _result("UNSUPPORTED_FAULT_CLASS", "The verified fault class is not in the selected hypothesis set.")
    if not case.get("diagnosis_snapshot", {}).get("hypotheses"):
        return _result("INSUFFICIENT_EVIDENCE", "The immutable diagnosis snapshot must include its original hypothesis set.")
    if not any(row.get("source") in _PHYSICAL_SOURCES for row in case.get("observations", [])):
        return _result("INSUFFICIENT_EVIDENCE", "At least one actual physical measurement is required.")
    tests = case.get("post_repair_tests", [])
    if not any(row.get("passed") is True and row.get("source") in _PHYSICAL_SOURCES for row in tests):
        return _result("INSUFFICIENT_EVIDENCE", "A passed post-repair test from a physical measurement is required.")
    before = [row for row in case.get("observations", []) if row.get("phase", "before") == "before"]
    after = [measurement for test in tests if test.get("passed") is True
             for measurement in test.get("observations", []) if isinstance(measurement, dict)]
    if compare_before_after(before, after)["comparable_count"] == 0:
        return _result("INSUFFICIENT_EVIDENCE", "Comparable physical before/after measurements under matching conditions are required.")
    interventions = case.get("interventions", [])
    replaced = {row.get("component_ref") for row in interventions
                if row.get("action") in {"component_replaced", "component_removed", "solder_rework", "wiring_modified"}}
    if len(replaced) > 1 and not cause.get("isolated_by_followup_test"):
        return _result("REVIEW_REQUIRED", "Multiple possible causal components were changed without an isolating follow-up test.")
    return {"status": "ELIGIBLE", "reasons": ["Physical evidence, reviewed root cause, repair outcome, and explicit training approval are present."],
            "duplicate_check": duplicate}


def find_training_evaluation_leakage(training_cases: list[dict[str, Any]], evaluation_cases: list[dict[str, Any]]) -> dict[str, Any]:
    train_ids = {case.get("case_id") for case in training_cases}
    eval_ids = {case.get("case_id") for case in evaluation_cases}
    train_boards = {case.get("board_instance_id") for case in training_cases if case.get("board_instance_id")}
    eval_boards = {case.get("board_instance_id") for case in evaluation_cases if case.get("board_instance_id")}
    overlap_ids, overlap_boards = sorted(train_ids & eval_ids), sorted(train_boards & eval_boards)
    return {"status": "INVALID_EVALUATION" if overlap_ids or overlap_boards else "PASSED",
            "overlapping_case_ids": overlap_ids, "overlapping_board_instance_ids": overlap_boards,
            "reason": "Training and evaluation cases or board instances overlap." if overlap_ids or overlap_boards else None}


def _result(status: str, reason: str) -> dict[str, Any]:
    return {"status": status, "reasons": [reason]}
