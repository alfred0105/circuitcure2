"""Circuit connectivity and selected-component view."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from circuitcare.graph import build_graph, direct_candidates, find_paths_result, to_dot


def render_circuit_view(st: Any, *, circuit: Any, case_name: str, case: dict,
                        selected_fault: str, fault_type: str, fault_value: str | None,
                        root: Path, comparison_path: Path) -> None:
    st.subheader("회로 및 고장 설정")
    parameter = fault_value if fault_value is not None else "해당 없음"
    st.write(f"**선택 회로:** {case_name}  ·  **고장 후보:** {selected_fault} / {fault_type} = {parameter}")
    candidates = direct_candidates(circuit, selected_fault)
    path_search = find_paths_result(circuit, selected_fault, max_hops=2)
    graph = build_graph(circuit)
    st.graphviz_chart(to_dot(graph, selected_fault, path_search.paths), use_container_width=True)
    ref = st.selectbox("부품 연결 정보", list(circuit.components), key="component_detail")
    st.json({"component": next(vars(item) for item in circuit.components.values() if item.ref == ref),
             "connections": [{"pin": item.pin, "net": item.net} for item in circuit.connections if item.ref == ref],
             "direct_candidates": [{"reference": item.target_ref, "shared_nets": item.shared_nets,
                                    "evidence": item.evidence_level} for item in candidates],
             "path_candidate_count": len(path_search.paths),
             "path_search": {"status": path_search.status, "expanded_states": path_search.expanded_states,
                             "peak_queue_size": path_search.peak_queue_size,
                             "limit_reason": path_search.limit_reason},
             "graph_note": "연결 기반 후보 경로이며 손상 전파/실제 손상을 증명하지 않습니다."})
    if path_search.status != "COMPLETE":
        st.warning(f"경로 탐색은 {path_search.status} 상태로 제한되었습니다. 표시 경로는 부분 결과이며 전체 경로 목록이 아닙니다. 사유: {path_search.limit_reason}")
    if not case.get("mapping") or not (root / case["mapping"]).is_file():
        st.warning("KiCad XML만 확인 가능합니다. 명시적 SPICE mapping/model이 없어 물리 시뮬레이션을 지원하지 않습니다.")
    source = comparison_path.relative_to(root) if comparison_path.exists() else case.get("cached")
    st.caption(f"넷리스트: `{case['netlist']}` · 파형/위험 데이터 출처: `{source}`")
