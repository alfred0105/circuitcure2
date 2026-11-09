"""Leakage-checked, longitudinal evaluation for approved prior proposals."""

from __future__ import annotations

import math
from typing import Any

from .training_eligibility import find_training_evaluation_leakage


def evaluate_prior_predictions(records: list[dict[str, Any]], training_cases: list[dict[str, Any]], *,
                               calibration_min_cases: int = 20) -> dict[str, Any]:
    leakage = find_training_evaluation_leakage(training_cases, records)
    if leakage["status"] != "PASSED":
        return {"status": "INVALID_EVALUATION", "integrity_audit": leakage,
                "reason": "Evaluation case or board instance was used for prior training."}
    if not records:
        return {"status": "REAL_REPAIR_PRIOR_PERFORMANCE_NOT_EVALUATED", "integrity_audit": leakage,
                "case_count": 0, "default_prior": _not_evaluated("No independent evaluation records."),
                "learned_prior": _not_evaluated("No independent evaluation records.")}
    if any(row.get("split") != "independent_holdout" for row in records):
        return {"status": "PARTIALLY_EVALUATED", "integrity_audit": leakage,
                "reason": "Only explicitly independent_holdout records can support the primary held-out metrics."}
    outputs = {}
    for method, posterior_key in (("default_prior", "default_posterior"), ("learned_prior", "learned_posterior")):
        hits1, hits3, reciprocal_ranks, nlls, briers, calibration = [], [], [], [], [], []
        unsupported = 0
        for row in records:
            posterior = row.get(posterior_key) or {}
            target = row.get("true_hypothesis_id")
            if (target not in posterior or not posterior or
                any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in posterior.values()) or
                abs(sum(posterior.values()) - 1.0) > 1e-6):
                unsupported += 1
                continue
            ranked = sorted(posterior, key=posterior.get, reverse=True)
            rank = ranked.index(target) + 1
            probability = max(1e-15, float(posterior[target]))
            hits1.append(float(rank == 1)); hits3.append(float(rank <= 3)); reciprocal_ranks.append(1 / rank)
            nlls.append(-math.log(probability))
            briers.append(sum((float(value) - (1.0 if hypothesis == target else 0.0)) ** 2
                              for hypothesis, value in posterior.items()))
            top_confidence = max(float(value) for value in posterior.values())
            calibration.append((top_confidence, float(ranked[0] == target)))
        denominator = len(records)
        scored = len(hits1)
        outputs[method] = {"top1_accuracy": _metric(sum(hits1) / denominator if scored else None, denominator),
            "top3_accuracy": _metric(sum(hits3) / denominator if scored else None, denominator),
            "mean_reciprocal_rank": _metric(sum(reciprocal_ranks) / denominator if scored else None, denominator),
            "negative_log_likelihood": _metric(sum(nlls) / scored if scored else None, scored),
            "brier_score": _metric(sum(briers) / scored if scored else None, scored),
            "calibration_error": _calibration_error(calibration, calibration_min_cases),
            "diagnostic_coverage": _metric(scored / denominator, denominator),
            "unsupported_case_count": unsupported, "scored_case_count": scored}
    real_count = sum(row.get("data_source") == "real_repair" for row in records)
    status = "MEASURED_INDEPENDENT_HOLDOUT" if real_count == len(records) else "SYNTHETIC_OR_MIXED_HOLDOUT_ONLY"
    return {"status": status if real_count else "REAL_REPAIR_PRIOR_PERFORMANCE_NOT_EVALUATED",
        "integrity_audit": leakage, "case_count": len(records), "real_repair_case_count": real_count,
        "independent_holdout": True, "default_prior": outputs["default_prior"],
        "learned_prior": outputs["learned_prior"],
        "interpretation": "Conditional ranking metrics apply to the supplied hypothesis sets; they do not estimate general component failure probabilities."}


def _metric(value: float | None, denominator: int, *, status: str = "MEASURED", reason: str | None = None) -> dict[str, Any]:
    return {"value": value, "status": status if value is not None else "NOT_EVALUATED",
            "denominator": denominator, "reason": reason}


def _not_evaluated(reason: str) -> dict[str, Any]:
    return {"value": None, "status": "NOT_EVALUATED", "denominator": 0, "reason": reason}


def _calibration_error(rows: list[tuple[float, float]], minimum: int) -> dict[str, Any]:
    if len(rows) < minimum:
        return _metric(None, len(rows), status="NOT_EVALUATED", reason=f"At least {minimum} independent scored cases are required.")
    bins = [[] for _ in range(10)]
    for confidence, correct in rows:
        bins[min(9, int(confidence * 10))].append((confidence, correct))
    value = sum(len(items) / len(rows) * abs(sum(c for c, _ in items) / len(items) - sum(y for _, y in items) / len(items))
                for items in bins if items)
    return _metric(value, len(rows))
