"""Explainable gate for board-specific diagnosis; never calls demo-circuit APIs."""

from __future__ import annotations

from typing import Any

from circuitcare.frontend.diagnosis_adapter import (
    DiagnosisContext, DiagnosisStatus, assess_diagnosis_readiness,
    current_diagnosis_context, is_diagnosis_result_current, refresh_diagnosis,
)
from circuitcare.frontend.measurement_models import MeasurementRecord, measurement_set_sha256
from circuitcare.frontend.measurement_state import (
    DIAGNOSIS_STATE_KEY, diagnosis_state, owner_key,
)
from circuitcare.frontend.models import ProjectContext

STATUS_LABELS = {
    DiagnosisStatus.NO_MEASUREMENT: "측정값 없음",
    DiagnosisStatus.MEASUREMENT_RECORDED: "측정 기록 · 진단 준비 전",
    DiagnosisStatus.NODE_MAPPING_UNAVAILABLE: "측정 노드 연결 필요",
    DiagnosisStatus.HYPOTHESIS_MODEL_UNAVAILABLE: "회로 모델 미연결",
    DiagnosisStatus.DIAGNOSIS_READY: "진단 준비 완료",
    DiagnosisStatus.DIAGNOSIS_RUNNING: "진단 중",
    DiagnosisStatus.DIAGNOSIS_COMPLETE: "진단 완료",
    DiagnosisStatus.DIAGNOSIS_FAILED: "진단 실행 실패",
    DiagnosisStatus.MODEL_MISMATCH: "모델과 측정값 불일치",
    DiagnosisStatus.STALE_RESULT: "측정 변경 · 재검증 필요",
}


def render_diagnosis_panel(
    st: Any, project: ProjectContext, *, measurements: list[MeasurementRecord],
    analysis_context_id: str | None = None,
    component_map_version: str | None = None,
) -> None:
    st.markdown("### 진단")
    board_id = project.board_id or ""
    context = current_diagnosis_context(
        st.session_state, project_id=project.project_id, board_id=board_id,
        analysis_context_id=analysis_context_id, component_map_version=component_map_version,
    )
    provider = st.session_state.get("circuitcare_v5_diagnosis_provider")
    readiness = assess_diagnosis_readiness(
        context=context, measurements=measurements, provider_available=provider is not None,
    )
    stored = diagnosis_state(st.session_state, project_id=project.project_id, board_id=board_id)
    current_set = measurement_set_sha256(measurements, project_id=project.project_id, board_id=board_id)
    result_is_current = is_diagnosis_result_current(
        stored_result=stored, current_context=context, current_measurement_set=current_set,
    )
    status = stored.get("status") if result_is_current else (
        DiagnosisStatus.STALE_RESULT if stored.get("result") is not None else readiness.status
    )
    st.markdown(f"**상태** · {STATUS_LABELS.get(status, status)}")
    if status == DiagnosisStatus.STALE_RESULT:
        st.warning("분석 조건이 변경되어 기존 진단 결과를 사용할 수 없습니다.")
    for blocker in readiness.blockers:
        st.caption(f"· {blocker}")
    if result_is_current and status in {DiagnosisStatus.DIAGNOSIS_COMPLETE, DiagnosisStatus.MODEL_MISMATCH}:
        result = stored.get("result")
        if isinstance(result, dict):
            if status == DiagnosisStatus.MODEL_MISMATCH:
                st.warning("현재 측정값을 가설 모델이 충분히 설명하지 못합니다. 결과를 확정하지 마세요.")
            for candidate in result.get("candidates", []):
                if not isinstance(candidate, dict):
                    continue
                st.caption(f"가설 모델 내 상대 우선도 · {candidate.get('ranking', '미제공')}")
                st.caption(f"가설 · {candidate.get('hypothesis_id', '미제공')}")
            st.caption("Bayesian 값은 설정된 가설 집합에 조건부이며 실제 고장 확률이 아닙니다.")
    submitted = st.button("진단 갱신", key=f"cc_v5_refresh_{project.project_id}_{board_id}",
                           disabled=readiness.status != DiagnosisStatus.DIAGNOSIS_READY)
    if submitted and provider is not None:
        result, updated_readiness, error = refresh_diagnosis(
            provider, context=context, measurements=measurements,
        )
        scoped = dict(st.session_state.get(DIAGNOSIS_STATE_KEY, {}))
        scoped[owner_key(project.project_id, board_id)] = {
            "status": updated_readiness.status, "result": result,
            "context": context if result is not None else None,
            "measurement_set_sha256": current_set,
        }
        st.session_state[DIAGNOSIS_STATE_KEY] = scoped
        if error:
            st.error("진단 Provider 실행 중 오류가 발생했습니다.")
        st.rerun()
