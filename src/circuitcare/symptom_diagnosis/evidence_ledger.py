"""Normalized, provenance-preserving ledger for diagnosis evidence."""

from __future__ import annotations

import hashlib
import json
from typing import Any


_LEDGER_STATUSES = {"VERIFIED", "USER_REPORTED", "INFERRED", "AI_SUGGESTED", "UNKNOWN", "CONFLICTED"}


def build_evidence_ledger(
    *, observations: list[dict[str, Any]] | None = None,
    conflicts: list[dict[str, Any]] | None = None,
    components: list[dict[str, Any]] | None = None,
    functional_blocks: list[dict[str, Any]] | None = None,
    functional_roles: list[dict[str, Any]] | None = None,
    measurements: list[dict[str, Any]] | None = None,
    ai_suggestions: list[dict[str, Any]] | None = None,
    source_version: str | None = None,
) -> list[dict[str, Any]]:
    """Map existing records into typed ledger rows without elevating suggestions."""
    rows: list[dict[str, Any]] = []

    def append(kind: str, subject_id: str | None, value: Any, source: str,
               status: str, record_id: str | None = None, version: str | None = None) -> None:
        status = status if status in _LEDGER_STATUSES else "UNKNOWN"
        stable = {"kind": kind, "subject_id": subject_id, "value": value,
                  "source": source, "record_id": record_id, "version": version or source_version}
        evidence_id = hashlib.sha256(json.dumps(stable, sort_keys=True, ensure_ascii=False,
                                                  default=str).encode()).hexdigest()[:20]
        rows.append({"evidence_id": f"ev-{evidence_id}", "kind": kind,
                     "subject_id": subject_id, "component_id": subject_id,
                     "value": value, "source": source, "source_version": version or source_version,
                     "status": status, "record_id": record_id})

    for item in observations or []:
        if not isinstance(item, dict):
            continue
        raw_status = str(item.get("status", "observed")).casefold()
        source = str(item.get("source") or "unknown")
        status = ("VERIFIED" if raw_status == "measured" and item.get("verification_status") in
                 {"VERIFIED", "verified", "VALIDATED"} else
                 "USER_REPORTED" if source in {"user_report", "user_report_or_confirmed_symptom", "manual"}
                 or raw_status == "observed" else
                 "INFERRED" if raw_status == "inferred" else "UNKNOWN")
        append("observation", str(item.get("component_id") or "") or None,
               {"code": item.get("code"), "value": item.get("value")}, source, status,
               str(item.get("id") or item.get("observation_id") or "") or None,
               str(item.get("version") or item.get("timestamp") or "") or None)

    for item in measurements or []:
        if not isinstance(item, dict):
            continue
        verified = str(item.get("verification_status") or "").upper() in {"VERIFIED", "VALIDATED"}
        append("measurement", str(item.get("board_component_id") or item.get("component_id") or "") or None,
               {key: item.get(key) for key in ("value", "unit", "measurement_type", "test_id",
                                                "measurement_target", "operating_conditions",
                                                "fault_hypothesis") if key in item},
               str(item.get("source") or "measurement_record"), "VERIFIED" if verified else "UNKNOWN",
               str(item.get("measurement_id") or "") or None,
               str(item.get("source_observation_fingerprint") or item.get("version")
                   or item.get("measured_at") or item.get("created_at") or "") or None)

    for item in components or []:
        if not isinstance(item, dict):
            continue
        component_id = str(item.get("board_component_id") or item.get("component_id") or "") or None
        type_status = str(item.get("component_type_status") or "").upper()
        presence = str(item.get("presence_status") or "").upper()
        status = "VERIFIED" if type_status == "USER_CONFIRMED" and presence in {"CONFIRMED", "USER_CONFIRMED"} else "UNKNOWN"
        append("component_identity", component_id,
               {"type": item.get("component_type"), "subtype": item.get("component_subtype")},
               str(item.get("source") or "component_review"), status,
               str(item.get("detection_id") or item.get("observation_id") or component_id or "") or None,
               str(item.get("observation_version") or item.get("version") or "") or None)
        if item.get("ocr_marking"):
            ocr_status = str(item.get("ocr_evidence_status") or item.get("ocr_status") or "").upper()
            append("ocr_marking", component_id, item.get("ocr_marking"),
                   str(item.get("ocr_source") or "rapidocr"),
                   "VERIFIED" if ocr_status in {"USER_CONFIRMED", "VERIFIED", "CONFIRMED"} else "UNKNOWN",
                   str(item.get("ocr_observation_id") or "") or None,
                   str(item.get("ocr_version") or item.get("observation_version") or "") or None)
        for candidate in item.get("ocr_candidates", []) if isinstance(item.get("ocr_candidates"), list) else []:
            append("ocr_candidate", component_id, candidate, "rapidocr", "UNKNOWN",
                   str(item.get("ocr_version") or "") or None)

    for item in functional_roles or []:
        if not isinstance(item, dict):
            continue
        component_id = str(item.get("board_component_id") or item.get("component_id") or "")
        if not component_id:
            continue
        confirmed_roles = {str(value) for value in item.get("confirmed_functional_roles", [])}
        suggested_roles = {str(value) for value in item.get("ai_suggested_functional_roles", [])}
        evidence_by_role = item.get("role_evidence", {})
        for role in item.get("candidate_blocks", []):
            role = str(role)
            status = ("VERIFIED" if role in confirmed_roles else
                      "AI_SUGGESTED" if role in suggested_roles else "INFERRED")
            append("component_functional_role", component_id,
                   {"role": role, "role_status": status,
                    "evidence": list(evidence_by_role.get(role, [])) if isinstance(evidence_by_role, dict) else []},
                   "user" if status == "VERIFIED" else "qwen" if status == "AI_SUGGESTED" else "rule_and_layout_inference",
                   status, str(item.get("role_observation_id") or component_id) or None,
                   str(item.get("role_version") or item.get("observation_version") or "") or None)

    for item in functional_blocks or []:
        if not isinstance(item, dict):
            continue
        block_id = str(item.get("block_id") or item.get("functional_block_id") or "")
        if item.get("component_ids"):
            membership_status = ("VERIFIED" if item.get("verified") is True else
                                 "AI_SUGGESTED" if item.get("verification_status") == "AI_SUGGESTED"
                                 or item.get("status") == "AI_SUGGESTED" else "INFERRED")
            for component_id in item.get("component_ids", []):
                append("functional_block_membership", str(component_id),
                       {"block_id": block_id, "role": item.get("role"),
                        "partition_status": item.get("partition_status"),
                        "connectivity_status": "NOT_INFERRED"},
                       "qwen" if membership_status == "AI_SUGGESTED" else "functional_block_builder",
                       membership_status,
                       block_id or None, str(item.get("version") or "") or None)

    for item in ai_suggestions or []:
        if isinstance(item, dict):
            append("ai_suggestion", str(item.get("board_component_id") or item.get("component_id") or "") or None,
                   {key: item.get(key) for key in ("suspected_fault_id", "functional_block", "reason") if key in item},
                   str(item.get("source") or "qwen"), "AI_SUGGESTED",
                   str(item.get("proposal_id") or "") or None)

    for item in conflicts or []:
        if not isinstance(item, dict):
            continue
        append("conflict", str(item.get("component_id") or "") or None,
               {"code": item.get("code"), "values": item.get("values"), "evidence": item.get("evidence")},
               "conflicting_observations", "CONFLICTED", str(item.get("code") or "") or None)

    seen: set[str] = set()
    return [row for row in rows if not (row["evidence_id"] in seen or seen.add(row["evidence_id"]))]
