"""Human review transitions; repair success and root cause remain separate."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .case_repository import RepairCaseRepository
from .case_validator import assert_valid_case
from .models import utc_now


def review_case(repository: RepairCaseRepository, case_id: str, *, verification_status: str,
                outcome: str, reviewer: str, note: str = "", root_cause: dict[str, Any] | None = None,
                approve_for_training: bool = False) -> dict[str, Any]:
    case = repository.get_case(case_id)
    if case is None:
        raise KeyError(case_id)
    reviewed = deepcopy(case)
    reviewed["verification_status"] = verification_status
    reviewed["outcome"] = outcome
    reviewed["case_status"] = "CLOSED" if outcome in {"RESOLVED", "PARTIALLY_RESOLVED", "NOT_RESOLVED", "RECURRENCE"} else "POST_REPAIR_TESTED" if reviewed.get("post_repair_tests") else "INSPECTED"
    reviewed["root_cause"] = root_cause
    reviewed["training_review"] = {"status": "APPROVED" if approve_for_training else "NOT_APPROVED",
        "approved": bool(approve_for_training), "reviewer": reviewer, "note": note, "reviewed_at": utc_now()}
    reviewed["training_eligibility"] = {"status": "REVIEW_REQUIRED", "reasons": ["Eligibility must be recalculated against the full case set."]}
    event = {"verification_status": verification_status, "outcome": outcome, "reviewer": reviewer,
             "note": note, "root_cause": root_cause, "training_approved": bool(approve_for_training)}
    assert_valid_case(reviewed)
    repository.update_case(reviewed, event_type="EVIDENCE_REVIEWED", event_payload=event)
    return reviewed
