"""Combine graph paths, simulation measurements, and ratings into inspection ranks."""

import json
from pathlib import Path
from typing import Any

from .graph import find_paths_result
from .models import Circuit
from .rating_checker import check_component_ratings
from .risk_engine import assess_component_risk


def rank_impacts(
    circuit: Circuit,
    failed_component: str,
    fault_type: str,
    scenario_id: str,
    component_stress: dict[str, Any],
    specifications: dict[str, Any],
    component_mapping: dict[str, dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write a risk report while keeping graph evidence distinct from stress evidence."""
    path_search = find_paths_result(circuit, failed_component, max_hops=max(1, len(circuit.components) - 1))
    paths = path_search.paths
    paths_by_ref: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        paths_by_ref.setdefault(path.target_ref, []).append({
            "connection_type": path.connection_type,
            "hops": path.hops,
            "nets": list(path.nets),
            "net_types": dict(path.net_types),
            "ground_only_path": path.ground_only_path,
            "evidence_level": "connectivity_only",
        })

    ratings = check_component_ratings(component_stress, specifications, component_mapping)
    assessments = assess_component_risk(
        ratings, component_stress, component_mapping, failed_component=failed_component
    )
    assessments_by_ref = {item["component_ref"]: item for item in assessments}
    stress_refs = {item["component_ref"] for item in component_stress.get("components", [])}
    excluded = []
    for reference, mapping in component_mapping.items():
        if mapping.get("risk_assessment_supported") is False and (reference in paths_by_ref or reference in stress_refs):
            excluded.append({
                "component_ref": reference,
                "reason": mapping.get("risk_exclusion_reason", "SPICE-to-KiCad model correspondence is not verified."),
                "risk_assessment": "excluded",
            })

    secondary = []
    for reference in sorted(set(paths_by_ref) | stress_refs):
        if reference == failed_component or reference in {item["component_ref"] for item in excluded}:
            continue
        assessment = assessments_by_ref.get(reference)
        if assessment is None:
            assessment = {
                "component_ref": reference,
                "risk_level": "UNKNOWN",
                "priority_score": 0.0,
                "priority_factors": {
                    "risk_class_weight": 0.0,
                    "utilization_contribution": 0.0,
                    "stress_change_contribution": 0.0,
                    "measured_rating_evidence": 0.0,
                    "connectivity_path_evidence": 0.0,
                },
                "maximum_utilization_ratio": None,
                "dominant_metric": None,
                "metrics": {},
                "rating_source_types": [],
                "evidence_level": "connectivity_only_or_insufficient_measurement",
                "reason": "Connected candidate has no sufficient measured stress and explicit rating pair; safety is unknown.",
                "damage_confirmed": False,
                "damage_probability": None,
            }
        item = dict(assessment)
        item["connectivity_paths"] = paths_by_ref.get(reference, [])
        item["shared_nets"] = sorted({net for route in paths_by_ref.get(reference, []) for net in route["nets"]})
        if item.get("risk_level") != "UNKNOWN" and item["connectivity_paths"]:
            item["priority_factors"]["connectivity_path_evidence"] = 2.0
            item["priority_score"] = min(100.0, round(item["priority_score"] + 2.0, 2))
        secondary.append(item)
    secondary.sort(key=lambda item: (-item["priority_score"], item["component_ref"]))
    for rank, item in enumerate(secondary, start=1):
        item["priority_rank"] = rank

    graph_only = [
        {"component_ref": reference, "minimum_hops": min(route["hops"] for route in routes), "evidence_level": "connectivity_only"}
        for reference, routes in paths_by_ref.items()
        if reference != failed_component and reference not in {item["component_ref"] for item in excluded}
    ]
    graph_only.sort(key=lambda item: (item["minimum_hops"], item["component_ref"]))
    for rank, item in enumerate(graph_only, start=1):
        item["connectivity_rank"] = rank

    report = {
        "failed_component": failed_component,
        "fault_type": fault_type,
        "scenario_id": scenario_id,
        "assessment_type": "simulated_secondary_stress",
        "simulation_status": "success",
        "priority_score_meaning": "Deterministic inspection-priority index (0-100); factors are risk class, utilization, measured stress change, rating/measurement evidence, and a small connectivity-path evidence contribution. It is not damage probability.",
        "risk_thresholds": {"near_limit_utilization_ratio": 0.8, "exceeded_utilization_ratio": 1.0},
        "secondary_candidates": secondary,
        "excluded_candidates": excluded,
        "graph_only_connectivity_order": graph_only,
        "path_search": {"status": path_search.status, "expanded_states": path_search.expanded_states,
                        "peak_queue_size": path_search.peak_queue_size, "total_paths": path_search.total_paths,
                        "limit_reason": path_search.limit_reason},
        "rating_checks": ratings,
        "damage_confirmed": False,
        "damage_probability": None,
        "limitations": [
            "Connectivity paths are candidate evidence only; sharing GND alone does not establish stress propagation.",
            "Only configured ratings and supported measured metrics are evaluated.",
            "Short transient maxima are compared with configured rating types; exposure duration is retained in simulation data but no pulse derating is applied.",
        ],
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / "risk_assessment.json"
    report["risk_assessment_json"] = str(output_path)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report
