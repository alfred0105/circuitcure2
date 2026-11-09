"""Human-approved, versioned prior lifecycle with explicit activation and rollback."""

from __future__ import annotations

import sqlite3
import math

from .case_repository import RepairCaseRepository
from .models import utc_now

MIN_OPERATIONAL_PRIOR_CASES = 5


def propose_prior(repository: RepairCaseRepository, proposal: dict) -> dict:
    if proposal.get("status") != "PROPOSAL_READY" or not proposal.get("prior_weights"):
        raise ValueError("Only a fully estimated proposal can be registered.")
    if proposal.get("contains_synthetic_records") is not False:
        raise ValueError("Prior proposal provenance must explicitly exclude synthetic records.")
    ids = proposal.get("training_case_ids", [])
    if proposal.get("minimum_case_count", 0) < MIN_OPERATIONAL_PRIOR_CASES:
        raise ValueError(f"Operational prior proposals require a minimum of {MIN_OPERATIONAL_PRIOR_CASES} distinct verified cases.")
    if len(ids) != len(set(ids)) or len(ids) != proposal.get("eligible_case_count") or not ids:
        raise ValueError("Prior proposal requires distinct case IDs matching its eligible case count.")
    if len(ids) < proposal.get("minimum_case_count", 1):
        raise ValueError("Prior proposal does not meet its configured minimum case count.")
    weights = proposal["prior_weights"]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-8:
        raise ValueError("Prior weights must be finite, strictly positive, and normalized.")
    model = {**proposal, "status": "PROPOSED"}
    try:
        repository.save_prior_model(model)
    except sqlite3.IntegrityError:
        existing = repository.get_prior_model(proposal["version"])
        if existing is None or existing.get("prior_weights") != proposal.get("prior_weights") or existing.get("training_case_ids") != proposal.get("training_case_ids"):
            raise ValueError("Prior version collision with different proposal contents.")
        repository.record_prior_event(version=proposal["version"], action="PROPOSAL_ALREADY_EXISTS", payload={})
        return existing
    repository.record_prior_event(version=proposal["version"], action="PROPOSED", payload={"eligible_case_count": proposal["eligible_case_count"]})
    return repository.get_prior_model(proposal["version"])


def approve_prior(repository: RepairCaseRepository, version: str, *, reviewer: str, note: str = "") -> dict:
    model = repository.get_prior_model(version)
    if not model or model.get("status") not in {"PROPOSED", "APPROVED"}:
        raise ValueError("Prior must exist as a proposal before approval.")
    model["approval"] = {"reviewer": reviewer, "note": note, "approved_at": utc_now(), "approved": True}
    model["status"] = "APPROVED"
    repository.update_prior_model(version, status="APPROVED")
    # Persist approval details in the version payload through an append-only event.
    repository.record_prior_event(version=version, action="APPROVED", payload=model["approval"])
    _replace_payload(repository, model)
    return model


def activate_prior(repository: RepairCaseRepository, version: str, *, reviewer: str) -> dict:
    with repository._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        target = _load_and_validate_activation(db, repository, version, None)
        _activate_in_transaction(db, repository, target, reviewer, "ACTIVATED")
    return repository.get_prior_model(version)


def deactivate_prior(repository: RepairCaseRepository, version: str, *, reviewer: str) -> dict:
    model = repository.get_prior_model(version)
    if not model or model.get("status") != "ACTIVE":
        raise ValueError("Only an active prior can be deactivated.")
    updated = repository.update_prior_model(version, status="APPROVED")
    repository.record_prior_event(version=version, action="DEACTIVATED", payload={"reviewer": reviewer})
    return updated


def rollback_prior(repository: RepairCaseRepository, *, group_id: str, target_version: str, reviewer: str) -> dict:
    with repository._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        target = _load_and_validate_activation(db, repository, target_version, group_id)
        _activate_in_transaction(db, repository, target, reviewer, "ROLLBACK_ACTIVATED")
    return repository.get_prior_model(target_version)


def active_prior(repository: RepairCaseRepository, group_id: str) -> dict | None:
    return next((model for model in repository.list_prior_models(group_id=group_id) if model.get("status") == "ACTIVE"), None)


def _replace_payload(repository: RepairCaseRepository, model: dict) -> None:
    # A narrow repository update keeps the version immutable while appending review events.
    import json
    with repository._connect() as db:
        db.execute("UPDATE prior_models SET payload_json=? WHERE version=?",
                   (json.dumps(model, ensure_ascii=False, sort_keys=True, allow_nan=False), model["version"]))


def _load_and_validate_activation(db: sqlite3.Connection, repository: RepairCaseRepository,
                                  version: str, group_id: str | None) -> dict:
    import json

    row = db.execute("SELECT payload_json,status,group_id FROM prior_models WHERE version=?", (version,)).fetchone()
    if not row:
        raise ValueError("Prior activation target does not exist.")
    model = json.loads(row[0])
    status = row[1]
    if group_id is not None and row[2] != group_id:
        raise ValueError("Rollback target must belong to the requested group.")
    if status == "INVALIDATED_SOURCE_CHANGE" or model.get("status") == "INVALIDATED_SOURCE_CHANGE":
        raise ValueError("Invalidated prior cannot be activated or rolled back.")
    if status != "APPROVED" or not model.get("approval", {}).get("approved"):
        raise ValueError("A human-approved prior is required before activation.")
    if model.get("contains_synthetic_records") is not False:
        raise ValueError("Synthetic or unknown training provenance cannot be activated.")
    ids = model.get("training_case_ids")
    weights = model.get("prior_weights")
    if not isinstance(ids, list) or len(ids) < MIN_OPERATIONAL_PRIOR_CASES or len(ids) != len(set(ids)):
        raise ValueError(f"Activation requires at least {MIN_OPERATIONAL_PRIOR_CASES} distinct verified real-repair cases.")
    if not isinstance(weights, dict) or not weights or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-8:
        raise ValueError("Prior weights must be finite, strictly positive, and normalized.")
    rows = db.execute("SELECT case_id,payload_json FROM repair_cases").fetchall()
    cases_by_id = {item[0]: json.loads(item[1]) for item in rows}
    if any(case_id not in cases_by_id for case_id in ids):
        raise ValueError("Prior references a missing training case.")
    from .training_eligibility import assess_training_eligibility
    all_cases = list(cases_by_id.values())
    for case_id in ids:
        eligibility = assess_training_eligibility(cases_by_id[case_id], all_cases)
        if eligibility.get("status") != "ELIGIBLE":
            raise ValueError(f"Prior training case is no longer eligible: {case_id} ({eligibility.get('status')}).")
    return model


def _activate_in_transaction(db: sqlite3.Connection, repository: RepairCaseRepository,
                             target: dict, reviewer: str, action: str) -> None:
    import json

    now = utc_now()
    for row in db.execute("SELECT version,payload_json FROM prior_models WHERE group_id=?", (target["group_id"],)).fetchall():
        version, payload = row[0], json.loads(row[1])
        if payload.get("status") == "ACTIVE" and version != target["version"]:
            payload["status"] = "APPROVED"
            payload["updated_at"] = now
            db.execute("UPDATE prior_models SET status='APPROVED',payload_json=? WHERE version=?",
                       (repository._json(payload), version))
            db.execute("INSERT INTO prior_activation_events VALUES (?,?,?,?,?)",
                       (f"prior_event_{__import__('uuid').uuid4().hex}", version, "DEACTIVATED_FOR_REPLACEMENT", now,
                        repository._json({"replaced_by": target["version"], "reviewer": reviewer})))
    target["status"] = "ACTIVE"
    target["updated_at"] = now
    db.execute("UPDATE prior_models SET status='ACTIVE',payload_json=? WHERE version=?",
               (repository._json(target), target["version"]))
    db.execute("INSERT INTO prior_activation_events VALUES (?,?,?,?,?)",
               (f"prior_event_{__import__('uuid').uuid4().hex}", target["version"], action, now,
                repository._json({"reviewer": reviewer})))
