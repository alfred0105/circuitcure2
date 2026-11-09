"""Fault influence map and evidence-grounded map exports."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from circuitcare.dashboard.case_loader import PROJECT_ROOT
from circuitcare.visualization.fault_map import (
    build_fault_map,
    fault_map_json,
    render_fault_map_dot,
)


def render_fault_map(st: Any, *, circuit: Any, case: dict, context_key: Any,
                     failed_component: str, fault_type: str, fault_value: str | None,
                     risk: dict, status: str, risk_rows: list[dict], what_if_preview: Any) -> tuple[dict, dict | None]:
    what_if_result = what_if_preview.result if what_if_preview else None
    node = what_if_preview.node if what_if_preview else None
    voltage = what_if_preview.voltage if what_if_preview else None
    sigma = what_if_preview.sigma if what_if_preview else None
    context_id = what_if_preview.context_id if what_if_preview else "none"
    ranking = what_if_result.get("inspection_ranking") if what_if_result else None
    view_mode = st.radio("지도 보기", ["Connection View", "Electrical Stress View", "Inspection Priority View"],
                         horizontal=True, key=f"fault_map_mode_{hashlib.sha256(repr(context_key).encode()).hexdigest()[:10]}")
    refs = list(circuit.components)
    selected = st.selectbox("영향 경로를 강조할 부품", refs, index=refs.index(failed_component), key=f"fault_map_component_{hashlib.sha256(repr(context_key).encode()).hexdigest()[:10]}")
    payload = build_fault_map(circuit, failed_component, risk_assessment=risk,
                              inspection_ranking=ranking, selected_component=selected,
                              view_mode=view_mode, max_hops=max(1, len(circuit.components) - 1))
    payload["analysis_context"] = {"circuit": case["netlist"], "failed_component": failed_component,
                                   "fault_type": fault_type, "fault_parameter": fault_value,
                                   "simulation_status": status,
                                   "scenario_id": (risk or {}).get("scenario_id"),
                                   "context_id": hashlib.sha256(repr(context_key).encode()).hexdigest()[:16]}
    payload["visualization_version"] = "phase13-v1"
    st.graphviz_chart(render_fault_map_dot(payload), use_container_width=True)
    if view_mode == "Electrical Stress View":
        st.caption("색상: 보라 Known Failed · 빨강 EXCEEDED · 주황 NEAR_LIMIT · 파랑 WITHIN_LIMIT · 회색 UNKNOWN. 정격 초과는 설정 정격과 시뮬레이션 스트레스의 비교이며 실제 손상 확정이 아닙니다.")
    elif view_mode == "Inspection Priority View":
        st.caption("색상은 posterior 가중 점검 지수 구간, #숫자는 순위입니다. Bayesian 조정 후에도 실제 순위는 그대로일 수 있습니다. SPICE 정격 위험과 별개이며 손상 확률이 아닙니다.")
    else:
        st.caption("보라색은 사용자가 선택한 고장 부품, 파랑은 연결 후보, 회색은 선택 경로 밖 부품입니다. Net 연결은 전력 흐름 방향을 뜻하지 않습니다.")
    detail = next(row for row in payload["nodes"] if row["component_ref"] == selected)
    st.markdown(f"#### {selected} 상세 근거")
    st.json({**detail, "connection_paths": [row for row in payload["paths"] if row["target_ref"] == selected],
             "risk_metrics": next((row.get("metrics", {}) for row in risk_rows if row.get("component_ref") == selected), {}),
             "stress_changes_with_what_if": False,
             "unknowns": ["실제 손상 여부", "미모델링 전기 변수", "구성되지 않은 정격"]})
    st.download_button("영향 지도 JSON 다운로드", fault_map_json(payload), "fault_map_data.json", "application/json",
                       key=f"fault_map_json_{context_id if what_if_preview is not None else 'none'}")
    st.download_button("영향 지도 DOT 다운로드", render_fault_map_dot(payload), "fault_influence_map.dot", "text/vnd.graphviz",
                       key=f"fault_map_dot_{hashlib.sha256(repr(context_key).encode()).hexdigest()[:12]}")
    if what_if_result:
        export_diagnosis = {key: value for key, value in what_if_result["diagnosis"].items()
                            if key not in {"diagnosis_file", "inspection_ranking_file", "recommendation_file", "measurements_file"}}
        what_if_export = {"analysis_context": payload["analysis_context"], "measurement_source": "simulated_demo",
                          "what_if_measurement": {"node": node, "value_v": voltage, "sigma_v": sigma},
                          "diagnosis": export_diagnosis,
                          "inspection_ranking": what_if_result["inspection_ranking"],
                          "recommendation": what_if_result["recommendations"],
                          "electrical_stress_recomputed": False, "damage_confirmed": False}
        st.download_button("What-if 결과 JSON 다운로드", json.dumps(what_if_export, ensure_ascii=False, indent=2),
                           "what_if_diagnosis.json", "application/json", key=f"what_if_download_{context_id}")
    return payload, what_if_export if what_if_result else None


def render_fault_map_save(st: Any, *, payload: dict, what_if_export: dict | None,
                          context_key: Any) -> None:
    context_id = hashlib.sha256(repr(context_key).encode()).hexdigest()[:12]
    if not st.button("현재 지도와 What-if 결과를 로컬 저장", key=f"fault_map_save_{context_id}"):
        return
    target = PROJECT_ROOT / "data" / "visualizations"
    target.mkdir(parents=True, exist_ok=True)
    (target / "fault_map_data.json").write_text(fault_map_json(payload), encoding="utf-8")
    (target / "fault_influence_map.dot").write_text(render_fault_map_dot(payload), encoding="utf-8")
    if what_if_export:
        (target / "what_if_diagnosis.json").write_text(
            json.dumps(what_if_export, ensure_ascii=False, indent=2), encoding="utf-8")
    st.success("분석 지도와 가상 진단 결과를 로컬에 저장했습니다.")
