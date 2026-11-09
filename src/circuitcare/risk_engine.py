"""Interpretable ordinal risk classification and inspection priority."""

from typing import Any

NEAR_LIMIT_RATIO = 0.8


def assess_component_risk(
    rating_results: list[dict[str, Any]],
    component_stress: dict[str, Any],
    component_mapping: dict[str, dict[str, Any]],
    *,
    failed_component: str,
) -> list[dict[str, Any]]:
    """Classify components from measured ratios; scores are not damage probabilities."""
    stress_by_ref = {row["component_ref"]: row for row in component_stress.get("components", [])}
    assessments = []
    for item in rating_results:
        reference = item["component_ref"]
        if reference == failed_component:
            continue
        mapping = component_mapping.get(reference, {})
        if mapping.get("risk_assessment_supported") is False:
            continue
        evaluated = [m for m in item["metrics"].values() if m["status"] == "evaluated"]
        ratios = [m["utilization_ratio"] for m in evaluated]
        max_ratio = max(ratios, default=None)
        if any(m["rating_exceeded"] for m in evaluated):
            level = "EXCEEDED"
        elif any(ratio >= NEAR_LIMIT_RATIO for ratio in ratios):
            level = "NEAR_LIMIT"
        elif evaluated:
            level = "WITHIN_LIMIT"
        else:
            level = "UNKNOWN"
        dominant = max(
            (name for name, metric in item["metrics"].items() if metric["status"] == "evaluated"),
            key=lambda name: item["metrics"][name]["utilization_ratio"],
            default=None,
        )
        dominant_label = {"max_abs_voltage_v": "voltage", "max_current_a": "current", "max_power_w": "power"}.get(dominant)
        reasons = []
        if level == "EXCEEDED":
            reasons.append(f"Simulated {dominant_label} stress exceeded an explicitly configured rating.")
        elif level == "NEAR_LIMIT":
            reasons.append(f"Simulated {dominant_label} stress reached at least 80% of its configured rating.")
        elif level == "WITHIN_LIMIT":
            reasons.append("All configured and measurable ratings evaluated for this component were within the near-limit threshold.")
        else:
            reasons.append("No applicable explicit rating and simulated metric pair was available for evaluation.")
        stress = stress_by_ref.get(reference, {})
        if stress.get("status") == "not_probed":
            reasons.append("Component was not probed; its condition remains unknown.")
        elif stress.get("status") == "voltage_only":
            reasons.append("Only voltage was measured; unsupported current and power remain unknown.")
        source_types = sorted({
            metric["rating_source"].get("type", "unknown")
            for metric in evaluated
            if isinstance(metric.get("rating_source"), dict)
        })
        factors = _priority_factors(level, max_ratio, item["metrics"], has_rating_evidence=bool(evaluated))
        score = min(100.0, sum(factors.values()))
        assessments.append({
            "component_ref": reference,
            "risk_level": level,
            "priority_score": score,
            "priority_factors": {**factors, "connectivity_path_evidence": 0.0},
            "priority_rank": None,
            "maximum_utilization_ratio": max_ratio,
            "dominant_metric": dominant_label,
            "metrics": item["metrics"],
            "rating_source_types": source_types,
            "evidence_level": "simulated_electrical_stress_and_configured_rating" if evaluated else "insufficient_rating_or_measurement_data",
            "reason": " ".join(reasons),
            "damage_confirmed": False,
            "damage_probability": None,
        })
    assessments.sort(key=lambda row: (-row["priority_score"], row["component_ref"]))
    for rank, row in enumerate(assessments, start=1):
        row["priority_rank"] = rank
    return assessments


def _priority_factors(level: str, ratio: float | None, metrics: dict[str, Any], *, has_rating_evidence: bool) -> dict[str, float]:
    level_points = {"EXCEEDED": 70, "NEAR_LIMIT": 45, "WITHIN_LIMIT": 20, "UNKNOWN": 0}[level]
    utilization_points = min(20.0, 20.0 * min(ratio or 0.0, 1.5) / 1.5)
    delta_points = 5.0 if _has_increased_stress(metrics) else 0.0
    rating_evidence_points = 5.0 if has_rating_evidence else 0.0
    return {
        "risk_class_weight": level_points,
        "utilization_contribution": round(utilization_points, 2),
        "stress_change_contribution": delta_points,
        "measured_rating_evidence": rating_evidence_points,
    }


def _has_increased_stress(rating_metrics: dict[str, Any]) -> bool:
    """Return true only when comparable physical stress magnitude rose."""
    for values in rating_metrics.values():
        increase = values.get("stress_increase") if isinstance(values, dict) else None
        baseline = values.get("baseline_stress") if isinstance(values, dict) else None
        if isinstance(increase, (int, float)) and isinstance(baseline, (int, float)):
            tolerance = max(1e-12, abs(baseline) * 1e-9)
            if increase > tolerance:
                return True
    return False
