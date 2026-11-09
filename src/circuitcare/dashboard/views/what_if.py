"""Bayesian simulated-measurement preview renderer."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from circuitcare.dashboard.case_loader import DEMO_HYPOTHESES
from circuitcare.services.diagnosis_service import (
    load_saved_diagnosis,
    run_diagnosis_for_measurements,
)


@dataclass(frozen=True)
class WhatIfPreview:
    result: dict[str, Any]
    node: str
    voltage: float
    sigma: float
    context_id: str


def render_what_if(st: Any, *, case: dict, context_key: Any, bayesian_supported: bool) -> WhatIfPreview | None:
    context_id = hashlib.sha256(repr(context_key).encode()).hexdigest()[:16]
    node = None
    voltage = None
    sigma = None
    what_if_result = None
    if bayesian_supported:
        st.markdown("### Bayesian What-if — 가상 측정 미리보기")
        st.warning("이 컨트롤은 `simulated_demo` 가상 측정만 계산합니다. 기존 실측 기록, SPICE 결과, 정격 위험도, 수리 이력은 변경하지 않습니다. 실제 측정값으로 저장할 수 없습니다.")
        context_id = hashlib.sha256(repr(context_key).encode()).hexdigest()[:16]
        node_options = load_saved_diagnosis(case).get("diagnosis", {}).get("available_measurement_nodes", ["VOUT"])
        if "VOUT" in node_options:
            node_options = ["VOUT", *[item for item in node_options if item != "VOUT"]]
        controls = st.columns([2, 2, 2, 1])
        node = controls[0].selectbox("측정 노드", node_options, key="what_if_node", index=0)
        if node == "VOUT":
            voltage_min, voltage_max, voltage_default = 2.0, 9.0, 7.9
        else:
            from circuitcare.diagnosis_data import load_diagnosis_data
            hypotheses, _ = load_diagnosis_data(DEMO_HYPOTHESES)
            prediction_values = [hypothesis.predicted_measurements[node] for hypothesis in hypotheses
                                 if node in hypothesis.predicted_measurements]
            voltage_min = max(0.0, float(int(min(prediction_values) - 1)))
            voltage_max = float(int(max(prediction_values) + 2))
            voltage_default = min(voltage_max, max(voltage_min, (min(prediction_values) + max(prediction_values)) / 2))
        voltage_key = f"what_if_voltage_{node}"
        voltage = controls[1].slider("가상 전압 (V)", min_value=voltage_min, max_value=voltage_max, step=0.1,
                                     value=float(st.session_state.get(voltage_key, voltage_default)), key=voltage_key)
        sigma = controls[2].number_input("불확실도 σ (V)", min_value=0.01, max_value=5.0, step=0.1,
                                        value=float(st.session_state.get("what_if_sigma", 0.2)), key="what_if_sigma")
        if controls[3].button("초기화", key=f"what_if_reset_{context_id}"):
            for key in list(st.session_state):
                if key.startswith("what_if_voltage_"):
                    st.session_state.pop(key, None)
            for key in ("what_if_result", "what_if_result_key", "what_if_measurements", "what_if_context_key",
                        "what_if_sigma", "what_if_node"):
                st.session_state.pop(key, None)
            st.rerun()
        state_key = (context_key, node, float(voltage), float(sigma))
        if st.session_state.get("what_if_result_key") != state_key:
            started = time.perf_counter()
            virtual_measurement = {"node": node, "value": float(voltage), "unit": "V",
                                   "uncertainty": float(sigma), "source": "simulated_demo"}
            result = run_diagnosis_for_measurements([virtual_measurement])
            st.session_state.what_if_measurements = [virtual_measurement]
            st.session_state.what_if_context_key = context_key
            st.session_state.what_if_result = result
            st.session_state.what_if_result_key = state_key
            st.session_state.what_if_elapsed_seconds = round(time.perf_counter() - started, 4)
        what_if_result = st.session_state.what_if_result
        diagnosis = what_if_result["diagnosis"]
        if diagnosis.get("model_mismatch"):
            st.error("MODEL MISMATCH · 설정된 가설이 이 측정값을 잘 설명하지 못합니다. 상위 가설을 진단 결론으로 강조하지 않습니다.")
        posterior = diagnosis.get("posterior", [])
        prior_rows = load_saved_diagnosis(case).get("diagnosis", {}).get("posterior", [])
        prior_by_id = {row["hypothesis_id"]: row.get("prior_weight", 0.0) for row in prior_rows}
        chart_rows = [{"가설": row["hypothesis_id"], "Prior": prior_by_id.get(row["hypothesis_id"], 0.0),
                       "What-if posterior": row["posterior"]} for row in posterior]
        if chart_rows:
            import pandas as pd
            st.bar_chart(pd.DataFrame(chart_rows).set_index("가설"), y_label="조건부 가중치")
            if diagnosis.get("model_mismatch"):
                st.caption(f"가상 입력: {node}={voltage:.1f} V, σ={sigma:.2f} V · 모델 불일치로 최상위 가설을 강조하지 않습니다. · 계산 {st.session_state.get('what_if_elapsed_seconds', 0):.4f} s")
            else:
                top = max(posterior, key=lambda row: row["posterior"])
                st.caption(f"가상 입력: {node}={voltage:.1f} V, σ={sigma:.2f} V · 상위 가설: {top['hypothesis_id']} ({top['posterior']:.4f}, 설정 가설 집합에 조건부) · 계산 {st.session_state.get('what_if_elapsed_seconds', 0):.4f} s")
        st.caption(diagnosis.get("probability_interpretation", "Posterior는 설정 가설 집합에 조건부이며 실제 고장 확률이 아닙니다."))
        changes = what_if_result["inspection_ranking"].get("rank_changes_after_measurements", [])
        if changes:
            st.markdown("**점검 우선순위 변화** (Bayesian 가중 점검 지수; 전기적 위험 등급과 별개)")
            st.dataframe([{"부품": row["component_ref"], "Prior 순위": row["prior_rank"],
                           "What-if 순위": row["posterior_rank"], "순위 변화 (+ 상승)": row["rank_change"],
                           "Prior 점검 지수": row["prior_weighted_priority_score"],
                           "What-if 점검 지수": row["posterior_weighted_priority_score"]} for row in changes],
                         hide_index=True, use_container_width=True)
        recommendation = what_if_result["recommendations"].get("recommended_measurement")
        if recommendation:
            st.write(f"**다음 정보량 기반 측정 후보:** {recommendation['node']} · 정보 이득 {recommendation['expected_information_gain_nats']:.4f} nats")
        else:
            st.write("**다음 정보량 기반 측정 후보:** 설정된 미측정 노드 중 의미 있는 후보 없음")
        st.caption("추천 후보가 없다는 것은 점검이 불필요하다는 의미가 아닙니다.")
    else:
        st.info("What-if Bayesian 데모는 저장된 U1 3/6/8 V 가설이 연결된 기본 회로·고장 조건에서만 사용할 수 있습니다.")

    if not bayesian_supported:
        return None
    return WhatIfPreview(what_if_result, node, float(voltage), float(sigma), context_id)
