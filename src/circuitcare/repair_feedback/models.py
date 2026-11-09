"""Data vocabulary for local repair feedback records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

CASE_SOURCES = {"real_repair", "user_reported_repair", "simulated_fault", "synthetic_demo", "llm_prediction", "unverified_observation"}
CASE_STATUSES = {"REPORTED", "INSPECTED", "REPAIR_ATTEMPTED", "POST_REPAIR_TESTED", "CLOSED", "REOPENED"}
VERIFICATION_STATUSES = {"UNVERIFIED", "MEASUREMENT_SUPPORTED", "REPAIR_OUTCOME_VERIFIED", "ROOT_CAUSE_REVIEWED", "INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"}
OUTCOMES = {"RESOLVED", "PARTIALLY_RESOLVED", "NOT_RESOLVED", "RECURRENCE", "UNKNOWN"}
ELIGIBILITY = {"ELIGIBLE", "NOT_ELIGIBLE", "REVIEW_REQUIRED", "DUPLICATE_CASE", "INSUFFICIENT_EVIDENCE", "UNSUPPORTED_FAULT_CLASS"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_repair_case(*, circuit_id: str, circuit_hash: str, data_source: str = "user_reported_repair",
                    circuit_family: str | None = None, **fields: Any) -> dict[str, Any]:
    """Build a new unverified case. No supplied prediction is treated as truth."""
    optional = {key: fields[key] for key in ("circuit_version", "component_refs", "hypothesis_set_id", "private_notes", "photos",
        "created_by", "inspection_events", "split", "expected_ranges", "duplicate_of") if key in fields}
    return {
        "case_id": fields.pop("case_id", f"repair_{uuid4().hex}"),
        "schema_version": 1,
        "data_source": data_source,
        "circuit_id": circuit_id,
        "circuit_family": circuit_family or circuit_id,
        "circuit_hash": circuit_hash,
        "board_instance_id": fields.pop("board_instance_id", None),
        "source_case_id": fields.pop("source_case_id", None),
        "reported_symptoms": fields.pop("reported_symptoms", []),
        "observations": fields.pop("observations", []),
        "diagnosis_snapshot": fields.pop("diagnosis_snapshot", {}),
        "interventions": fields.pop("interventions", []),
        "post_repair_tests": fields.pop("post_repair_tests", []),
        "verification_status": "UNVERIFIED",
        "outcome": "UNKNOWN",
        "case_status": "REPORTED",
        "root_cause": None,
        "training_review": {"status": "NOT_REVIEWED", "approved": False},
        "training_eligibility": {"status": "NOT_ELIGIBLE", "reasons": ["Case has not been reviewed."]},
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "privacy": {"local_only": True, "include_photos": False},
        **optional,
    }
