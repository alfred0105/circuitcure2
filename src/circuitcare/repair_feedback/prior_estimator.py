"""Smoothed, explainable prior proposals from distinct eligible repair cases."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .training_eligibility import assess_training_eligibility


def smooth_prior(base_prior: dict[str, float], counts: dict[str, int], *, alpha: float) -> dict[str, float]:
    """Dirichlet-style shrinkage toward a validated base distribution."""
    _validate_distribution(base_prior, "base prior")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if set(counts) - set(base_prior):
        raise ValueError("counts contain hypothesis IDs absent from the base prior")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()):
        raise ValueError("counts must be non-negative integers")
    total = sum(counts.values())
    denominator = total + float(alpha)
    return {key: (counts.get(key, 0) + float(alpha) * prior) / denominator
            for key, prior in base_prior.items()}


def estimate_prior(cases: list[dict[str, Any]], *, group_id: str, hypothesis_set_id: str,
                   base_prior: dict[str, float], alpha: float = 5.0, minimum_cases: int = 5,
                   supported_hypothesis_ids: set[str] | None = None) -> dict[str, Any]:
    """Return a human-reviewable proposal; insufficient data returns no prior."""
    _validate_distribution(base_prior, "base prior")
    if isinstance(minimum_cases, bool) or not isinstance(minimum_cases, int) or minimum_cases < 1:
        raise ValueError("minimum_cases must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    supported = supported_hypothesis_ids or set(base_prior)
    in_group: list[dict[str, Any]] = []
    for case in cases:
        snapshot = case.get("diagnosis_snapshot") or {}
        if case.get("circuit_family") != group_id or snapshot.get("hypothesis_set_id") != hypothesis_set_id:
            continue
        in_group.append(case)
    id_counts = Counter(case.get("case_id") for case in in_group)
    duplicate_ids = sorted(str(case_id or "MISSING_CASE_ID") for case_id, count in id_counts.items() if count > 1 or not case_id)
    unique_cases = {case["case_id"]: case for case in in_group
                    if case.get("case_id") and id_counts[case["case_id"]] == 1}
    counts = {hypothesis_id: 0 for hypothesis_id in base_prior}
    eligible_ids: list[str] = []
    exclusions: list[dict[str, Any]] = []
    for case in unique_cases.values():
        status = assess_training_eligibility(case, cases, supported_hypothesis_ids=supported)["status"]
        cause = case.get("root_cause") or {}
        if status != "ELIGIBLE":
            exclusions.append({"case_id": case.get("case_id"), "status": status})
        elif cause.get("hypothesis_id") not in counts:
            exclusions.append({"case_id": case.get("case_id"), "status": "UNSUPPORTED_FAULT_CLASS"})
        else:
            counts[cause["hypothesis_id"]] += 1
            eligible_ids.append(case["case_id"])
    if duplicate_ids:
        exclusions.extend({"case_id": item, "status": "DUPLICATE_CASE_ID"} for item in duplicate_ids)
    n = len(eligible_ids)
    if n < minimum_cases:
        return {"status": "INSUFFICIENT_VERIFIED_REPAIR_DATA", "prior_weights": None,
            "eligible_case_count": n, "minimum_case_count": minimum_cases,
            "candidate_case_count": len(in_group), "counts": counts,
            "excluded_cases": exclusions, "duplicate_case_ids": duplicate_ids,
            "reason": "No prior proposal is generated until the configured minimum number of distinct verified real-repair cases is met.",
            "configuration": {"alpha": alpha, "minimum_cases": minimum_cases,
                              "method": "Dirichlet-style shrinkage toward base prior"}}
    prior_weights = smooth_prior(base_prior, counts, alpha=alpha)
    digest_payload = {"group_id": group_id, "hypothesis_set_id": hypothesis_set_id,
        "case_ids": sorted(eligible_ids), "base_prior": base_prior, "counts": counts,
        "alpha": alpha, "weights": prior_weights}
    digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True).encode()).hexdigest()[:12]
    return {"status": "PROPOSAL_READY", "version": f"prior-v1-{digest}",
        "group_id": group_id, "hypothesis_set_id": hypothesis_set_id,
        "base_prior": dict(base_prior), "prior_weights": prior_weights, "counts": counts,
        "eligible_case_count": n, "minimum_case_count": minimum_cases,
        "training_case_ids": sorted(eligible_ids), "excluded_cases": exclusions,
        "duplicate_case_ids": duplicate_ids,
        "configuration": {"alpha": float(alpha), "minimum_cases": minimum_cases,
            "method": "Dirichlet-style shrinkage", "claim": "Configured smoothing value; not empirically optimized."},
        "status_model": "PROPOSED", "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_split": "development", "contains_synthetic_records": False,
        "evidence_note": "Only explicitly approved, reviewed real-repair cases are counted."}


def _validate_distribution(values: dict[str, float], label: str) -> None:
    if not isinstance(values, dict) or not values:
        raise ValueError(f"{label} must be a non-empty mapping")
    if any(not isinstance(key, str) or not key for key in values):
        raise ValueError(f"{label} hypothesis IDs must be non-empty strings")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError(f"{label} weights must be finite and strictly positive to prevent zero-probability hypotheses")
    total = sum(values.values())
    if total <= 0 or abs(total - 1.0) > 1e-8:
        raise ValueError(f"{label} weights must sum to 1.0")
