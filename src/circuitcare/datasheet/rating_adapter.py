"""Gate human-reviewed datasheet limits into the existing deterministic rating engine."""

from __future__ import annotations

import copy
from typing import Any

from circuitcare.rating_checker import check_component_ratings
from circuitcare.risk_engine import assess_component_risk


class RatingApplicationError(ValueError):
    """A datasheet rating is not eligible for application to the selected component."""


def apply_approved_rating(specifications: dict[str, Any], candidate: dict[str, Any], *,
                          component_ref: str, selected_reference: str, mpn_match_status: str,
                          conditions_compatible: bool) -> dict[str, Any]:
    if component_ref != selected_reference:
        raise RatingApplicationError("PART_UNVERIFIED: Reference does not match the current selection.")
    if mpn_match_status not in {"MPN_CONFIRMED", "USER_ASSERTED_PART_MATCH"}:
        raise RatingApplicationError("PART_UNVERIFIED: exact MPN or user-confirmed part match is required.")
    if candidate.get("human_approved") is not True or candidate.get("review_decision") != "approved":
        raise RatingApplicationError("Only a user-approved rating can be applied.")
    if not str(candidate.get("manufacturer") or "").strip() or not str(candidate.get("mpn") or "").strip():
        raise RatingApplicationError("PART_UNVERIFIED: manufacturer and actual MPN are required.")
    evidence_status = candidate.get("evidence_status")
    if evidence_status not in {"SOURCE_MATCHED", "CONDITION_INCOMPLETE"} or not candidate.get("source_document_sha256"):
        raise RatingApplicationError("Source evidence is not verified; the rating remains UNKNOWN.")
    if not conditions_compatible:
        raise RatingApplicationError("MISSING_CONDITION: rating conditions have not been confirmed for this comparison.")
    if evidence_status == "CONDITION_INCOMPLETE" and not candidate.get("condition_review_note", "").strip():
        raise RatingApplicationError("CONDITION_INCOMPLETE: reviewer must record how the missing condition was resolved.")
    key = candidate.get("rating_key")
    if key not in {"max_abs_voltage_v", "max_current_a", "max_power_w"}:
        raise RatingApplicationError("UNSUPPORTED_RATING_TYPE: existing Rating Engine has no matching metric.")
    value = candidate.get("normalized_value")
    expected_type = {"max_abs_voltage_v": "voltage_rating", "max_current_a": "current_rating", "max_power_w": "power_rating"}[key]
    if candidate.get("rating_type") != expected_type:
        raise RatingApplicationError("UNSUPPORTED_RATING_TYPE: candidate class does not match the physical metric.")
    unit_for_key = {"max_abs_voltage_v": "V", "max_current_a": "A", "max_power_w": "W"}[key]
    if candidate.get("normalized_unit") != unit_for_key or isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise RatingApplicationError("Rating value or physical unit is invalid.")
    # Absolute maximum is intentionally not translated to continuous or peak.
    kind = candidate.get("rating_kind")
    rating_type = "continuous" if kind == "continuous" else ("peak" if kind == "peak" else "unknown")
    output = copy.deepcopy(specifications)
    component = output.setdefault("components", {}).setdefault(component_ref, {"ratings": {}, "rating_types": {}})
    component.setdefault("ratings", {})[key] = float(value)
    component.setdefault("rating_types", {})[key] = rating_type
    source = {
        "type": "datasheet_user_approved",
        "note": f"User-approved {candidate.get('rating_type')} from {candidate.get('manufacturer')} {candidate.get('mpn')}; page {candidate.get('pdf_page_index') + 1}; SHA-256 {candidate['source_document_sha256']}.",
        "document_sha256": candidate["source_document_sha256"],
        "manufacturer": candidate.get("manufacturer"), "mpn": candidate.get("mpn"),
        "page_index": candidate.get("pdf_page_index"), "source_text": candidate.get("source_text"),
        "conditions": candidate.get("conditions"), "match_status": mpn_match_status,
        "conditions_compatible_user_confirmed": conditions_compatible,
        "condition_review_note": candidate.get("condition_review_note"),
        "rating_kind": kind, "rating_kind_source": candidate.get("rating_kind_source", "extraction"),
    }
    component.setdefault("rating_sources", {})[key] = source
    return output


def reassess_existing_risk(risk_report: dict[str, Any], component_stress: dict[str, Any],
                           specifications: dict[str, Any], component_mapping: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Re-run only rating checks against existing SPICE stress and preserve route evidence."""
    ratings = check_component_ratings(component_stress, specifications, component_mapping)
    assessments = assess_component_risk(ratings, component_stress, component_mapping,
                                       failed_component=risk_report["failed_component"])
    by_ref = {item["component_ref"]: item for item in assessments}
    output = copy.deepcopy(risk_report)
    output["rating_checks"] = ratings
    for row in output.get("secondary_candidates", []):
        updated = by_ref.get(row["component_ref"])
        if updated is None:
            continue
        updated["connectivity_paths"] = row.get("connectivity_paths", [])
        updated["shared_nets"] = row.get("shared_nets", [])
        if updated.get("risk_level") != "UNKNOWN" and updated["connectivity_paths"]:
            updated["priority_factors"]["connectivity_path_evidence"] = 2.0
            updated["priority_score"] = min(100.0, round(updated["priority_score"] + 2.0, 2))
        row.clear()
        row.update(updated)
    output["secondary_candidates"].sort(key=lambda row: (-row.get("priority_score", 0), row["component_ref"]))
    for rank, row in enumerate(output["secondary_candidates"], 1):
        row["priority_rank"] = rank
    output["reassessment_source"] = "approved_datasheet_rating_existing_spice_stress"
    output["simulation_rerun"] = False
    return output
