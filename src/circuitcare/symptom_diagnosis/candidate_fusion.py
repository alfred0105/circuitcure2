"""Experimental, evidence-neutral fusion of rule and Qwen inspection candidates."""

from __future__ import annotations

from typing import Any


_EXCLUDED_STATES = {"EXCLUDED", "STALE", "STALE_OR_UNRESOLVED", "UNRESOLVED", "RETIRED"}


def fuse_inspection_candidates(
    *, rule_candidates: list[dict[str, Any]], qwen_proposals: list[dict[str, Any]],
    components: list[dict[str, Any]], suspected_components: list[dict[str, Any]] | None = None,
    allowed_fault_ids: set[str], top_k: int = 3,
) -> dict[str, Any]:
    """Join candidates by (board_component_id, fault_id), without scoring Qwen text.

    Rule order and verified measurement verdicts remain authoritative. Qwen-only
    proposals and user suspicions are appended as review tasks with no score.
    This is an opt-in comparison strategy and never mutates the baseline result.
    """
    if not isinstance(rule_candidates, list):
        rule_candidates = []
    if not isinstance(qwen_proposals, list):
        qwen_proposals = []
    component_by_id: dict[str, dict[str, Any]] = {}
    for component in components:
        if not isinstance(component, dict):
            continue
        for alias in (component.get("component_id"), component.get("board_component_id")):
            if alias:
                component_by_id[str(alias)] = component

    suspects: dict[tuple[str, str], dict[str, Any]] = {}
    for row in suspected_components or []:
        if isinstance(row, dict):
            cid = str(row.get("board_component_id") or row.get("component_id") or "")
            fault = str(row.get("suspected_fault_mode") or "")
            if cid:
                suspects[(cid, fault)] = row

    fused: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(rule_candidates):
        if not isinstance(row, dict):
            continue
        cid = str(row.get("board_component_id") or row.get("component_id") or "")
        fault = str(row.get("fault_hypothesis") or row.get("fault_id") or "")
        if not cid or not fault:
            continue
        component = component_by_id.get(cid)
        if not cid or not fault or not component:
            continue
        presence = _presence(component)
        if presence in _EXCLUDED_STATES:
            continue
        key = (cid, fault)
        fused[key] = {
            **row,
            "component_id": cid,
            "board_component_id": cid,
            "fault_id": fault,
            "candidate_source": ["RULE_BASELINE"],
            "evidence_status": ("MEASUREMENT_VERIFIED" if row.get("measurement_verdict") in {"supports", "refutes"}
                                else "UNVERIFIED"),
            "verification_status": row.get("visual_status", row.get("status", "SUSPECTED")),
            "rank_strategy": "candidate_fusion_v1",
            "supporting_evidence_ids": list(dict.fromkeys(str(value) for value in row.get("evidence_ids", []))),
            "opposing_evidence_ids": [],
            "_rule_order": index,
        }

    for proposal in qwen_proposals:
        if not isinstance(proposal, dict):
            continue
        cid = str(proposal.get("board_component_id") or proposal.get("component_id") or "")
        fault = str(proposal.get("suspected_fault_id") or proposal.get("fault_id") or "")
        component = component_by_id.get(cid)
        presence = _presence(component or {}, proposal.get("component_review_status"))
        if (not cid or not fault or fault not in allowed_fault_ids or not component
                or presence in _EXCLUDED_STATES):
            continue
        key = (cid, fault)
        candidate = fused.get(key)
        if candidate is None:
            candidate = {
                "component_id": cid, "board_component_id": cid,
                "component_type": proposal.get("component_type", component.get("component_type", component.get("type", "unknown"))),
                "fault_id": fault,
                "fault_hypothesis": fault,
                "fault_title": proposal.get("fault_title", fault),
                "score": None,
                "status": "SUSPECTED", "visual_status": "SUSPECTED",
                "measurement_verdict": "non_diagnostic",
                "measurement_status": "INCONCLUSIVE",
                "evidence": [], "supporting_evidence": [], "opposing_evidence": [],
                "candidate_source": [], "evidence_status": "UNVERIFIED",
                "verification_status": "NEEDS_COMPONENT_REVIEW" if presence != "USER_CONFIRMED"
                    else "NEEDS_MEASUREMENT_OR_INSPECTION",
                "score_semantics": "none; review ordering only",
                "rank_strategy": "candidate_fusion_v1",
                "supporting_evidence_ids": [], "opposing_evidence_ids": [],
                "recommended_tests": [], "requires_measurement": True,
                "_rule_order": 10**9,
            }
            fused[key] = candidate
        if "QWEN_UNVERIFIED" not in candidate["candidate_source"]:
            candidate["candidate_source"].append("QWEN_UNVERIFIED")
        # Reason text is an explanation, never an independent evidence item.
        candidate["qwen_reason"] = str(proposal.get("reason") or "")[:500]
        candidate["functional_block_proposal"] = proposal.get("functional_block", "UNKNOWN")
        candidate["missing_information"] = list(proposal.get("missing_information", []))[:5]
        candidate["qwen_next_action"] = str(proposal.get("recommended_next_action") or "")[:240]
        candidate["supporting_evidence_ids"] = list(dict.fromkeys([
            *candidate["supporting_evidence_ids"], *map(str, proposal.get("supporting_evidence_ids", []))
        ]))
        candidate["opposing_evidence_ids"] = list(dict.fromkeys([
            *candidate["opposing_evidence_ids"], *map(str, proposal.get("opposing_evidence_ids", []))
        ]))
        candidate["evidence_status"] = ("MEASUREMENT_VERIFIED" if candidate.get("measurement_verdict") in {"supports", "refutes"}
                                        else "UNVERIFIED")
        # A proposal can never upgrade a measurement/verdict state.
        if candidate.get("measurement_verdict") not in {"supports", "refutes", "conflicted"}:
            candidate["verification_status"] = ("NEEDS_COMPONENT_REVIEW" if presence != "USER_CONFIRMED"
                                                  else "NEEDS_MEASUREMENT_OR_INSPECTION")

    # User suspicion creates a review task only for an explicitly selected, KB fault.
    # It changes review priority metadata, never the candidate score or verdict.
    for (cid, fault), suspect in suspects.items():
        component = component_by_id.get(cid)
        if not component or not fault:
            continue
        if fault not in allowed_fault_ids:
            continue
        key = (cid, fault)
        candidate = fused.get(key)
        if candidate is None:
            candidate = {
                "component_id": cid, "board_component_id": cid,
                "component_type": component.get("component_type", "unknown"),
                "fault_id": fault, "fault_hypothesis": fault,
                "fault_title": fault, "score": None,
                "status": "SUSPECTED", "visual_status": "SUSPECTED",
                "measurement_verdict": "non_diagnostic", "measurement_status": "INCONCLUSIVE",
                "candidate_source": [], "evidence_status": "UNVERIFIED",
                "verification_status": "NEEDS_COMPONENT_REVIEW" if _presence(component) != "USER_CONFIRMED"
                    else "NEEDS_MEASUREMENT_OR_INSPECTION",
                "rank_strategy": "candidate_fusion_v1", "score_semantics": "none; review ordering only",
                "supporting_evidence_ids": [], "opposing_evidence_ids": [],
                "evidence": [], "recommended_tests": [], "requires_measurement": True,
                "_rule_order": 10**9,
            }
            fused[key] = candidate
        if "USER_SUSPICION" not in candidate["candidate_source"]:
            candidate["candidate_source"].append("USER_SUSPICION")
        candidate["user_suspected"] = True
        candidate["user_suspicion_status"] = suspect.get("evidence_status", "UNVERIFIED")

    rows = list(fused.values())
    rows.sort(key=lambda row: (
        0 if row.get("measurement_verdict") == "supports" else
        2 if "RULE_BASELINE" in row.get("candidate_source", [])
          and row.get("measurement_verdict") not in {"refutes", "conflicted"} else
        3 if row.get("user_suspected") else 4,
        row.get("_rule_order", 10**9), row["component_id"], row["fault_id"],
    ))
    for rank, row in enumerate(rows, start=1):
        row["candidate_rank"] = rank
        row.pop("_rule_order", None)
    # The Top 3 is an inspection queue, not an assertion ranking. A refuted or
    # conflicted measurement stays in the audit trail but cannot occupy a slot
    # labelled as a leading fault candidate. Qwen-only rows remain explicitly
    # available even when rule rows fill the queue.
    primary_rows = [row for row in rows if row.get("measurement_verdict") not in {"refutes", "conflicted"}]
    top_candidates = primary_rows[:max(0, int(top_k))]
    top_keys = {(row["component_id"], row["fault_id"]) for row in top_candidates}
    additional_qwen = [row for row in rows
                       if "QWEN_UNVERIFIED" in row.get("candidate_source", [])
                       and "RULE_BASELINE" not in row.get("candidate_source", [])
                       and (row["component_id"], row["fault_id"]) not in top_keys]
    return {
        "strategy": "candidate_fusion_v1",
        "status": "AVAILABLE" if rows else "NO_CANDIDATES",
        "baseline_preserved": True,
        "score_semantics": "uncalibrated rule scores are preserved; Qwen and user suspicion add no score",
        "candidates": rows,
        "top_candidates": top_candidates,
        "additional_qwen_candidates": additional_qwen,
    }


def _presence(component: dict[str, Any], fallback: Any = None) -> str:
    # Registry state is authoritative. Proposal metadata may not upgrade a
    # detected/unresolved component into a user-confirmed one.
    labels = {str(component.get("presence_status") or "").upper(),
              str(component.get("review_status") or "").upper()}
    labels.discard("")
    if not labels and fallback:
        labels.add(str(fallback).upper())
    if labels & _EXCLUDED_STATES:
        return "EXCLUDED"
    if labels & {"CONFIRMED", "USER_CONFIRMED"}:
        return "USER_CONFIRMED"
    if labels & {"UNCONFIRMED", "DETECTED_UNVERIFIED", "UNVERIFIED"}:
        return "DETECTED_UNVERIFIED"
    return "UNKNOWN"
