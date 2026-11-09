"""Compare measured fault-case stress with explicitly configured ratings."""

from typing import Any

_METRICS = {
    "voltage_v": ("max_abs_voltage_v", "V", "absolute_peak"),
    "current_a": ("max_current_a", "A", "absolute_peak"),
    "power_w": ("max_power_w", "W", "dissipated_peak"),
}


def check_component_ratings(
    component_stress: dict[str, Any],
    specifications: dict[str, Any],
    component_mapping: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return per-metric utilization for the simulated fault condition."""
    specs = specifications.get("components", {})
    results = []
    for component in component_stress.get("components", []):
        reference = component["component_ref"]
        spec = specs.get(reference, {})
        ratings = spec.get("ratings", {})
        model_type = component_mapping.get(reference, {}).get("model_type")
        metrics = {}
        for measured_key, (rating_key, unit, measure_method) in _METRICS.items():
            limit = ratings.get(rating_key)
            fault_summary = component.get("fault", {}).get(measured_key)
            baseline_summary = component.get("baseline", {}).get(measured_key)
            status = component.get("status")
            reason = None
            if measured_key == "power_w" and model_type != "resistor":
                reason = "Dissipated-power rating comparison is only supported for resistors; source delivery power is not dissipation."
            elif status in {"unavailable", "not_probed"} or not fault_summary or fault_summary.get("maximum") is None:
                reason = component.get("reason", "Required simulated measurement is unavailable.")
            elif limit is None:
                reason = "No explicit rating is configured for this metric."
            elif spec.get("rating_types", {}).get(rating_key, "unknown") == "unknown":
                reason = "Rating duration/type is unknown; utilization is not classified against this limit."

            stress = None
            baseline_stress = None
            utilization = None
            exceeded = None
            rating_kind = spec.get("rating_types", {}).get(rating_key, "unknown")
            source = spec.get("rating_sources", {}).get(rating_key, spec.get("source"))
            if fault_summary and fault_summary.get("maximum") is not None:
                raw_fault_peak = fault_summary["maximum"]
                stress = max(abs(fault_summary.get("minimum", raw_fault_peak)), abs(raw_fault_peak)) if measure_method == "absolute_peak" else max(0.0, raw_fault_peak)
                if baseline_summary and baseline_summary.get("maximum") is not None:
                    raw_base_peak = baseline_summary["maximum"]
                    baseline_stress = max(abs(baseline_summary.get("minimum", raw_base_peak)), abs(raw_base_peak)) if measure_method == "absolute_peak" else max(0.0, raw_base_peak)
            if reason is None:
                utilization = stress / limit
                exceeded = utilization > 1.0
            metrics[rating_key] = {
                "measured_stress": stress,
                "baseline_stress": baseline_stress,
                "stress_increase": None if stress is None or baseline_stress is None else stress - baseline_stress,
                "rated_limit": limit,
                "unit": unit,
                "utilization_ratio": utilization,
                "rating_exceeded": exceeded,
                "rating_type": rating_kind,
                "rating_source": source,
                "status": "evaluated" if reason is None else "unknown",
                "reason": reason,
            }
        results.append({
            "component_ref": reference,
            "measurement_status": component.get("status"),
            "metrics": metrics,
        })
    return results
