"""Evidence-gated, display-only overlays for verified schematic assets."""

from __future__ import annotations

from typing import Any

from circuitcare.graph import direct_candidates, find_paths_result
from circuitcare.models import Circuit
from circuitcare.schematic.geometry import GeometryResult
from circuitcare.schematic.measurement_geometry import MeasurementGeometryResult

RISK_LEVELS = {"EXCEEDED", "NEAR_LIMIT", "WITHIN_LIMIT", "UNKNOWN"}


def _blocked(mode: str, context_id: str, asset_key: str, reason: str,
            geometry: GeometryResult, equivalence: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "mode": mode,
        "status": "BLOCKED",
        "reason": reason,
        "context_id": context_id,
        "asset_key": asset_key,
        "geometry_status": geometry.status,
        "electrical_equivalence": equivalence,
        "markers": [],
    }


def _base_gate(*, mode: str, circuit: Circuit, failed_component: str,
               geometry: GeometryResult, equivalence: str, context_id: str,
               asset_key: str) -> dict[str, Any] | None:
    if not context_id or not asset_key:
        return _blocked(mode, context_id, asset_key, "현재 회로 Context 또는 SVG 자산 식별자가 없습니다.",
                        geometry, equivalence)
    if equivalence != "VERIFIED_EQUIVALENT":
        return _blocked(mode, context_id, asset_key,
                        f"전기적 정합성이 검증되지 않았습니다: {equivalence}.", geometry, equivalence)
    if geometry.status != "VERIFIED_GEOMETRY":
        return _blocked(mode, context_id, asset_key,
                        f"심벌 Geometry가 검증되지 않았습니다: {geometry.status}.", geometry, equivalence)
    if failed_component not in circuit.components:
        return _blocked(mode, context_id, asset_key,
                        f"현재 회로에 고장 가정 Reference {failed_component!r}가 없습니다.", geometry, equivalence)
    geometry_refs = {item.component_ref for item in geometry.regions}
    if not set(circuit.components).issubset(geometry_refs):
        return _blocked(mode, context_id, asset_key,
                        "회로 부품 전체의 심벌 Reference 매핑이 검증되지 않았습니다.", geometry, equivalence)
    return None


def build_fault_overlay(*, circuit: Circuit, failed_component: str,
                        geometry: GeometryResult, electrical_equivalence: str,
                        context_id: str, asset_key: str, max_hops: int = 4) -> dict[str, Any]:
    """Build source/direct/indirect markers using graph evidence only."""
    blocked = _base_gate(mode="fault_influence", circuit=circuit,
                         failed_component=failed_component, geometry=geometry,
                         equivalence=electrical_equivalence, context_id=context_id,
                         asset_key=asset_key)
    if blocked:
        return blocked

    direct = {item.target_ref: set(item.shared_nets)
              for item in direct_candidates(circuit, failed_component)}
    search = find_paths_result(
        circuit, failed_component, max_hops=max(1, min(max_hops, max(1, len(circuit.components) - 1))),
        max_paths_per_candidate=2, max_expanded_states=10_000,
        max_queue_size=2_000, max_total_paths=2_000,
    )
    paths_by_ref: dict[str, list[Any]] = {}
    for path in search.paths:
        paths_by_ref.setdefault(path.target_ref, []).append(path)

    markers: list[dict[str, Any]] = [{
        "component_ref": failed_component,
        "kind": "FAULT_SOURCE",
        "label": "선택한 고장 가정",
        "evidence_level": "user_selected_fault_hypothesis",
        "damage_confirmed": False,
        "shared_nets": [],
        "paths": [],
    }]
    for ref in sorted(set(direct) | set(paths_by_ref)):
        paths = sorted(paths_by_ref.get(ref, []), key=lambda item: (item.hops, item.nets))
        shortest = paths[0] if paths else None
        if ref in direct:
            kind, hop_count, nets = "DIRECT_CANDIDATE", 1, sorted(direct[ref])
        elif shortest is not None:
            kind, hop_count, nets = "INDIRECT_CANDIDATE", shortest.hops, list(shortest.nets)
        else:
            continue
        markers.append({
            "component_ref": ref,
            "kind": kind,
            "label": "직접 연결 후보" if kind == "DIRECT_CANDIDATE" else "간접 연결 후보",
            "evidence_level": "connectivity_only",
            "damage_confirmed": False,
            "hop_count": hop_count,
            "shared_nets": nets,
            "paths": [{"hops": path.hops, "nets": list(path.nets),
                       "path": list(path.path), "evidence_level": "connectivity_only"}
                      for path in paths],
        })
    return {
        "schema_version": "1.0",
        "mode": "fault_influence",
        "status": "READY",
        "reason": None,
        "context_id": context_id,
        "asset_key": asset_key,
        "geometry_status": geometry.status,
        "electrical_equivalence": electrical_equivalence,
        "path_search": {"status": search.status, "expanded_states": search.expanded_states,
                         "peak_queue_size": search.peak_queue_size,
                         "total_paths": search.total_paths, "limit_reason": search.limit_reason},
        "markers": markers,
        "wire_overlay": False,
    }


def build_risk_overlay(*, circuit: Circuit, failed_component: str, fault_type: str,
                       fault_value: str | None,
                       risk: dict[str, Any], comparison: dict[str, Any],
                       simulation_status: str, simulation_integrity: str,
                       manifest: dict[str, Any], current_fingerprint: dict[str, Any],
                       geometry: GeometryResult,
                       electrical_equivalence: str, context_id: str,
                       asset_key: str) -> dict[str, Any]:
    """Build markers only from the already verified current AnalysisSnapshot."""
    blocked = _base_gate(mode="electrical_risk", circuit=circuit,
                         failed_component=failed_component, geometry=geometry,
                         equivalence=electrical_equivalence, context_id=context_id,
                         asset_key=asset_key)
    if blocked:
        return blocked
    if simulation_status not in {"SUCCESS", "CACHED_SIMULATION"} or simulation_integrity != "VERIFIED":
        return _blocked("electrical_risk", context_id, asset_key,
                        f"현재 시뮬레이션 출처를 확인할 수 없습니다: {simulation_status}/{simulation_integrity}.",
                        geometry, electrical_equivalence)
    scenario = risk.get("scenario_id")
    fingerprint = manifest.get("fingerprint") if isinstance(manifest, dict) else None
    fault_context = fingerprint.get("fault_scenario") if isinstance(fingerprint, dict) else None
    if (not isinstance(fault_context, dict)
            or fault_context.get("component_ref") != failed_component
            or fault_context.get("fault_type") != fault_type
            or fault_context.get("parameter") != fault_value
            or not isinstance(fingerprint, dict)
            or fingerprint.get("fingerprint_sha256") != current_fingerprint.get("fingerprint_sha256")
            or risk.get("failed_component") != failed_component
            or risk.get("fault_type") != fault_type
            or risk.get("simulation_status") != "success"
            or comparison.get("simulation_success") is not True
            or comparison.get("scenario_id") != scenario
            or scenario != f"FAULT_{failed_component}_{fault_type}".upper()
            or not isinstance(scenario, str) or not scenario):
        return _blocked("electrical_risk", context_id, asset_key,
                        "Risk, 비교 결과, 현재 고장 조건의 Scenario가 일치하지 않습니다.",
                        geometry, electrical_equivalence)
    rows = risk.get("secondary_candidates")
    if not isinstance(rows, list):
        return _blocked("electrical_risk", context_id, asset_key,
                        "현재 Risk Snapshot에 부품별 평가 목록이 없습니다.", geometry, electrical_equivalence)
    geometry_refs = {item.component_ref for item in geometry.regions}
    markers: list[dict[str, Any]] = [{
        "component_ref": failed_component,
        "kind": "FAULT_SOURCE",
        "label": "고장 가정 (실제 손상 미확정)",
        "risk_level": None,
        "damage_confirmed": False,
        "evidence_level": "user_selected_fault_hypothesis",
    }]
    for row in rows:
        if not isinstance(row, dict):
            return _blocked("electrical_risk", context_id, asset_key,
                            "Risk Snapshot 행 형식이 올바르지 않습니다.", geometry, electrical_equivalence)
        ref, level = row.get("component_ref"), row.get("risk_level")
        if not isinstance(ref, str) or ref not in circuit.components or ref not in geometry_refs:
            return _blocked("electrical_risk", context_id, asset_key,
                            "Risk Snapshot에 현재 회로/Geometry에 없는 Reference가 있습니다.",
                            geometry, electrical_equivalence)
        if not isinstance(level, str) or level not in RISK_LEVELS:
            return _blocked("electrical_risk", context_id, asset_key,
                            f"{ref} Risk Level을 지원하지 않습니다.", geometry, electrical_equivalence)
        metrics = row.get("metrics", {})
        if (not isinstance(metrics, dict)
                or any(not isinstance(metric, dict) for metric in metrics.values())):
            return _blocked("electrical_risk", context_id, asset_key,
                            f"{ref} Metric 근거 형식이 올바르지 않습니다.", geometry, electrical_equivalence)
        rating_source_types = row.get("rating_source_types", [])
        if (not isinstance(rating_source_types, list)
                or any(not isinstance(source, str) for source in rating_source_types)):
            return _blocked("electrical_risk", context_id, asset_key,
                            f"{ref} 정격 출처 형식이 올바르지 않습니다.", geometry, electrical_equivalence)
        markers.append({
            "component_ref": ref,
            "kind": "RISK",
            "label": level,
            "risk_level": level,
            "damage_confirmed": False,
            "priority_rank": row.get("priority_rank"),
            "priority_score": row.get("priority_score"),
            "dominant_metric": row.get("dominant_metric"),
            "maximum_utilization_ratio": row.get("maximum_utilization_ratio"),
            "metrics": metrics,
            "rating_source_types": rating_source_types,
            "evidence_level": row.get("evidence_level", "unknown"),
            "reason": row.get("reason"),
        })
    return {
        "schema_version": "1.0",
        "mode": "electrical_risk",
        "status": "READY",
        "reason": None,
        "context_id": context_id,
        "asset_key": asset_key,
        "geometry_status": geometry.status,
        "electrical_equivalence": electrical_equivalence,
        "simulation_status": simulation_status,
        "simulation_integrity": simulation_integrity,
        "scenario_id": scenario,
        "rating_reassessment_source": risk.get("reassessment_source"),
        "markers": markers,
        "damage_confirmed": False,
    }


def build_measurement_overlay(*, circuit: Circuit, failed_component: str,
                              geometry: GeometryResult, electrical_equivalence: str,
                              context_id: str, asset_key: str, support_status: str,
                              source_class: str, measurement_set_id: str | None,
                              recommendation: dict[str, Any] | None,
                              anchors: MeasurementGeometryResult | None,
                              inspection_ranking: dict[str, Any] | None,
                              model_mismatch: bool = False) -> dict[str, Any]:
    """Build source-labelled measurement/inspection markers from scoped data only."""
    blocked = _base_gate(mode="measurement_inspection", circuit=circuit,
                         failed_component=failed_component, geometry=geometry,
                         equivalence=electrical_equivalence, context_id=context_id,
                         asset_key=asset_key)
    if blocked:
        return blocked
    if support_status != "VERIFIED_CURRENT_CONTEXT":
        return _blocked("measurement_inspection", context_id, asset_key,
                        f"현재 회로에 대한 Bayesian 근거가 지원되지 않습니다: {support_status}.",
                        geometry, electrical_equivalence)
    if source_class not in {"prior_only", "user_measurement", "simulated_demo"}:
        return _blocked("measurement_inspection", context_id, asset_key,
                        f"측정 근거 출처를 지원하지 않습니다: {source_class}.",
                        geometry, electrical_equivalence)
    if model_mismatch:
        return _blocked("measurement_inspection", context_id, asset_key,
                        "Model mismatch 상태에서는 최상위 가설/추천을 강조하지 않습니다.",
                        geometry, electrical_equivalence)
    if recommendation and (anchors is None or anchors.status != "VERIFIED_LOGICAL_NET_CANDIDATES"
                           or anchors.node != recommendation.get("node")):
        return _blocked("measurement_inspection", context_id, asset_key,
                        (anchors.message if anchors else "측정 노드 핀 Geometry를 확인할 수 없습니다."),
                        geometry, electrical_equivalence)

    geometry_refs = {item.component_ref for item in geometry.regions}
    markers: list[dict[str, Any]] = []
    if recommendation and anchors:
        for anchor in anchors.anchors:
            ref = str(anchor["component_ref"])
            if ref not in circuit.components or ref not in geometry_refs:
                return _blocked("measurement_inspection", context_id, asset_key,
                                "측정 Anchor가 현재 회로 Geometry에 없습니다.",
                                geometry, electrical_equivalence)
            markers.append({
                "kind": "MEASUREMENT_NODE_CANDIDATE",
                "component_ref": ref,
                "pin": str(anchor["pin"]),
                "node": recommendation.get("node"),
                "x": float(anchor["x"]),
                "y": float(anchor["y"]),
                "measurement_type": recommendation.get("measurement_type"),
                "unit": recommendation.get("unit"),
                "rank": recommendation.get("rank", 1),
                "expected_information_gain_nats": recommendation.get("expected_information_gain_nats"),
                "source_class": source_class,
                "uncertainty_source": recommendation.get("uncertainty_source"),
                "evidence_level": "verified_logical_pin_net_mapping",
                "physical_probe_point_confirmed": False,
            })

    ranking = inspection_ranking or {}
    weighted_key = "prior_weighted_inspection_order" if source_class == "prior_only" else "posterior_weighted_inspection_order"
    rank_rows = ranking.get(weighted_key, [])
    if not isinstance(rank_rows, list):
        return _blocked("measurement_inspection", context_id, asset_key,
                        "Inspection Ranking 목록 형식이 올바르지 않습니다.", geometry, electrical_equivalence)
    for ranking_row in rank_rows:
        if not isinstance(ranking_row, dict):
            return _blocked("measurement_inspection", context_id, asset_key,
                            "Inspection Ranking 행 형식이 올바르지 않습니다.", geometry, electrical_equivalence)
        priority_ref = ranking_row.get("component_ref")
        if not isinstance(priority_ref, str) or priority_ref not in circuit.components or priority_ref not in geometry_refs:
            return _blocked("measurement_inspection", context_id, asset_key,
                            "Inspection Ranking에 현재 회로에 없는 Reference가 있습니다.",
                            geometry, electrical_equivalence)
        markers.append({
            "component_ref": priority_ref,
            "kind": "INSPECTION_PRIORITY",
            "label": f"점검 순위 {ranking_row.get('priority_rank', '—')}",
            "priority_rank": ranking_row.get("priority_rank"),
            "priority_score": ranking_row.get("posterior_weighted_priority_score",
                                               ranking_row.get("prior_weighted_priority_score")),
            "evidence_coverage": ranking_row.get("posterior_evidence_coverage"),
            "source_class": source_class,
            "damage_probability": None,
        })
    return {
        "schema_version": "1.0",
        "mode": "measurement_inspection",
        "status": "READY" if markers or not recommendation else "UNVERIFIED_MEASUREMENT_LOCATION",
        "reason": ("Measurement nodes are logical schematic candidates, not safe PCB probe points."
                   if markers else "No verified recommendation or inspection markers are available."),
        "context_id": context_id,
        "asset_key": asset_key,
        "geometry_status": geometry.status,
        "electrical_equivalence": electrical_equivalence,
        "source_class": source_class,
        "measurement_set_id": measurement_set_id,
        "markers": markers,
        "model_mismatch": False,
        "physical_probe_point_confirmed": False,
    }
