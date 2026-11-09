"""Bayesian diagnosis and explicit measurement update controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from circuitcare.dashboard.context import _upsert_measurement
from circuitcare.services.diagnosis_service import (
    load_saved_diagnosis,
    run_diagnosis_for_measurements,
)
from circuitcare.workspace.measurement_provenance import measurement_set_sha256


def render_bayesian_view(st: Any, *, case: dict, supported: bool,
                         measurement_example: Path, context_key: Any) -> None:
    st.subheader("Bayesian 진단")
    if not supported:
        st.info("이 회로에 연결된 Bayesian 가설 모델이 없습니다. U1 3/6/8 V 시나리오가 준비된 기본 데모에서만 Bayesian 갱신을 활성화합니다.")
        return
    st.caption("저장된 U1 3/6/8 V 시나리오 Bayesian 데모입니다. 현재 ngspice 실행에서 계산한 진단 결과와는 별도로 표시합니다. 가상 측정은 simulated_demo로 구분됩니다.")
    measurements = st.session_state.setdefault("measurements", [])
    context_id = hashlib.sha256(repr(context_key).encode()).hexdigest()[:16]
    current = st.session_state.get("diagnosis_result") or (
        run_diagnosis_for_measurements(measurements) if measurements else load_saved_diagnosis(case)
    )
    if st.button("저장된 7.9 V simulated_demo 측정으로 예시 갱신"):
        with measurement_example.open(encoding="utf-8") as handle:
            sample = json.load(handle)["measurements"][0]
        measurements = _upsert_measurement(measurements, sample)
        st.session_state.measurements = measurements
        st.session_state.diagnosis_result = run_diagnosis_for_measurements(measurements)
        st.session_state.diagnosis_result.update(
            analysis_context_id=context_id,
            measurement_set_sha256=measurement_set_sha256(measurements),
        )
        st.session_state.pop("last_advisor", None)
        st.rerun()
    with st.form("measurement_form"):
        node = st.selectbox("측정 노드", current["diagnosis"].get("available_measurement_nodes", []), index=0)
        value = st.number_input("측정 전압 (V)", value=7.9, format="%.4f")
        uncertainty = st.number_input("불확실도 σ (V)", min_value=0.0001, value=0.2, format="%.4f")
        source = st.selectbox("출처", ["user_measurement", "simulated_demo"],
            format_func=lambda item: {"user_measurement": "실제 측정 (사용자 신고)", "simulated_demo": "가상 데모"}[item],
            help="user_measurement는 사용자가 실제 계측값이라고 신고한 값이며 독립 검증을 뜻하지 않습니다.")
        submitted = st.form_submit_button("측정 추가 후 재계산")
    if submitted:
        measurements = _upsert_measurement(measurements, {"node": node, "value": value, "unit": "V",
                                                           "uncertainty": uncertainty, "source": source})
        st.session_state.measurements = measurements
        st.session_state.diagnosis_result = run_diagnosis_for_measurements(measurements)
        st.session_state.diagnosis_result.update(
            analysis_context_id=context_id,
            measurement_set_sha256=measurement_set_sha256(measurements),
        )
        st.session_state.pop("last_advisor", None)
        st.rerun()
    if measurements:
        st.caption("같은 노드를 다시 입력하면 기존 값을 교체합니다.")
        st.dataframe(measurements, use_container_width=True, hide_index=True)
    if st.button("측정값 초기화"):
        st.session_state.measurements = []
        st.session_state.diagnosis_result = run_diagnosis_for_measurements([])
        st.session_state.diagnosis_result.update(
            analysis_context_id=context_id,
            measurement_set_sha256=measurement_set_sha256([]),
        )
        st.session_state.pop("last_advisor", None)
        st.rerun()
    diagnosis = current["diagnosis"]
    if diagnosis.get("model_mismatch"):
        st.error("Model mismatch: 현재 측정값을 설정된 시뮬레이션 가설이 설명하지 못합니다. 최상위 posterior를 확정 진단으로 해석하지 마세요.")
    source_label = ", ".join(sorted({item.get("source", "") for item in diagnosis.get("used_measurements", [])})) or "측정 전"
    st.dataframe([{"가설": row["hypothesis_id"], "부품": row["component_ref"],
                   "Prior weight": row["prior_weight"], "Posterior (conditional)": row["posterior"],
                   "측정 출처": source_label} for row in diagnosis.get("posterior", [])],
                 use_container_width=True, hide_index=True)
    if diagnosis.get("top_hypothesis", {}).get("posterior") == 1.0:
        st.warning("Posterior 1.0은 현재 가설 집합과 측정 모델에 조건부인 값이며 실제 고장 확률 100%를 의미하지 않습니다.")
    recommendation = current["recommendations"].get("recommended_measurement")
    st.write("**다음 정보량 기반 측정 후보:**", recommendation.get("node") if recommendation else
             current["recommendations"].get("recommendation_status", "없음"))
    st.caption("추천 없음은 추가 점검이 불필요하다는 뜻이 아닙니다. 측정 안전성 승인이 아닙니다.")
    with st.expander("진단 JSON"):
        st.json(diagnosis)
