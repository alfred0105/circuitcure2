"""Combine hypothesis posteriors with existing scenario-conditional risk reports."""

import json
from pathlib import Path
from typing import Any

from .diagnosis_models import DiagnosisHypothesis


def rank_inspection_candidates(
    hypotheses: list[DiagnosisHypothesis],
    prior_rows: list[dict[str, Any]],
    posterior_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute prior and posterior weighted inspection indices, retaining all scenario evidence."""
    reports = {
        hypothesis.hypothesis_id: json.loads(
            (Path(hypothesis.risk_assessment_source) if Path(hypothesis.risk_assessment_source).is_absolute()
             else Path(hypothesis.source_base) / hypothesis.risk_assessment_source).read_text(encoding="utf-8")
        )
        for hypothesis in hypotheses
    }
    prior_order = _weighted_order(hypotheses, prior_rows, reports)
    posterior_order = _weighted_order(hypotheses, posterior_rows, reports)
    prior_ranks = {item["component_ref"]: item["priority_rank"] for item in prior_order}
    changes = []
    for item in posterior_order:
        old_rank = prior_ranks.get(item["component_ref"])
        changes.append({
            "component_ref": item["component_ref"],
            "prior_rank": old_rank,
            "posterior_rank": item["priority_rank"],
            "rank_change": None if old_rank is None else old_rank - item["priority_rank"],
            "prior_weighted_priority_score": next((row["posterior_weighted_priority_score"] for row in prior_order if row["component_ref"] == item["component_ref"]), None),
            "posterior_weighted_priority_score": item["posterior_weighted_priority_score"],
            "priority_score_change": next((item["posterior_weighted_priority_score"] - row["posterior_weighted_priority_score"] for row in prior_order if row["component_ref"] == item["component_ref"]), None),
        })
    excluded = {}
    for hypothesis in hypotheses:
        for item in reports[hypothesis.hypothesis_id].get("excluded_candidates", []):
            excluded.setdefault(item["component_ref"], set()).add(item.get("reason", "Excluded from physical assessment."))
    return {
        "score_interpretation": "Posterior-weighted deterministic inspection-priority index, not damage probability.",
        "prior_weighted_inspection_order": prior_order,
        "posterior_weighted_inspection_order": posterior_order,
        "rank_changes_after_measurements": changes,
        "scenario_conditional_exceeded_evidence": [
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "posterior": next((row["posterior"] for row in posterior_rows if row["hypothesis_id"] == hypothesis.hypothesis_id), 0.0),
                "components": [
                    {"component_ref": item["component_ref"], "risk_level": item["risk_level"], "reason": item["reason"]}
                    for item in reports[hypothesis.hypothesis_id].get("secondary_candidates", [])
                    if item.get("risk_level") == "EXCEEDED"
                ],
            }
            for hypothesis in hypotheses
        ],
        "unassessed_or_model_mismatch_components": [
            {"component_ref": reference, "reasons": sorted(reasons), "risk_level": "UNKNOWN_OR_EXCLUDED"}
            for reference, reasons in sorted(excluded.items())
        ],
        "damage_confirmed": False,
        "damage_probability": None,
    }


def normalized_prior_rows(hypotheses: list[DiagnosisHypothesis]) -> list[dict[str, Any]]:
    total = sum(item.prior_weight for item in hypotheses)
    return [
        {"hypothesis_id": item.hypothesis_id, "posterior": item.prior_weight / total}
        for item in hypotheses
    ]


def _weighted_order(hypotheses, probability_rows, reports):
    probabilities = {item["hypothesis_id"]: item["posterior"] for item in probability_rows}
    by_component = {}
    for hypothesis in hypotheses:
        weight = probabilities.get(hypothesis.hypothesis_id, 0.0)
        report = reports[hypothesis.hypothesis_id]
        for item in report.get("secondary_candidates", []):
            reference = item["component_ref"]
            combined = by_component.setdefault(reference, {
                "component_ref": reference,
                "posterior_weighted_score_sum": 0.0,
                "posterior_evidence_coverage": 0.0,
                "scenario_evidence": [],
                "any_hypothesis_exceeded": False,
                "exceeded_hypotheses": [],
            })
            combined["posterior_weighted_score_sum"] += weight * float(item.get("priority_score", 0.0))
            combined["posterior_evidence_coverage"] += weight
            combined["scenario_evidence"].append({
                "hypothesis_id": hypothesis.hypothesis_id,
                "posterior": weight,
                "risk_level": item.get("risk_level", "UNKNOWN"),
                "priority_score": item.get("priority_score", 0.0),
                "dominant_metric": item.get("dominant_metric"),
                "reason": item.get("reason"),
                "metrics": _compact_metrics(item.get("metrics", {})),
            })
            if item.get("risk_level") == "EXCEEDED":
                combined["any_hypothesis_exceeded"] = True
                combined["exceeded_hypotheses"].append(hypothesis.hypothesis_id)
    for item in by_component.values():
        item["posterior_weighted_score_sum"] = round(item["posterior_weighted_score_sum"], 4)
        item["posterior_evidence_coverage"] = round(item["posterior_evidence_coverage"], 8)
        item["unassessed_posterior_weight"] = round(max(0.0, 1.0 - item["posterior_evidence_coverage"]), 8)
        item["posterior_weighted_priority_score"] = round(
            item["posterior_weighted_score_sum"] / item["posterior_evidence_coverage"], 4
        ) if item["posterior_evidence_coverage"] > 0 else None
        item["exceeded_hypotheses"] = sorted(set(item["exceeded_hypotheses"]))
    ranked = sorted(by_component.values(), key=lambda item: (-item["posterior_weighted_priority_score"], item["component_ref"]))
    for rank, item in enumerate(ranked, start=1):
        item["priority_rank"] = rank
    return ranked


def _compact_metrics(metrics):
    fields = (
        "measured_stress", "baseline_stress", "stress_increase", "rated_limit", "unit",
        "utilization_ratio", "rating_exceeded", "rating_type", "status", "reason",
    )
    return {
        name: {
            **{field: value.get(field) for field in fields},
            "rating_source_type": (value.get("rating_source") or {}).get("type"),
        }
        for name, value in metrics.items()
    }
