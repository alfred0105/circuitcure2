"""PCB inspection summary and selected-component details."""

from __future__ import annotations

from typing import Any

from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel, display_identity_status
from circuitcare.frontend.measurement_state import get_measurements
from circuitcare.frontend.models import ProjectContext
from circuitcare.dashboard.views.diagnosis_panel import render_diagnosis_panel
from circuitcare.dashboard.views.measurement_panel import render_measurement_panel
from circuitcare.vision.component_recognition_panel import render_selected_component_editor

MAP_VERSION_KEY = "circuitcare_v4_selection_map_key"


def _component_label(value: str) -> str:
    types = {
        "capacitor": "커패시터", "resistor": "저항", "inductor": "인덕터",
        "diode": "다이오드", "transistor": "트랜지스터", "integrated_circuit": "IC",
        "ic": "IC", "connector": "커넥터", "unknown_component": "부품 후보",
    }
    return types.get(value.lower(), value)


def render_inspection_panel(
    st: Any, project: ProjectContext, *, preview: bool = False,
    components: tuple[ComponentOverlayViewModel, ...] = (),
    selected_component: ComponentOverlayViewModel | None = None,
    detection_status: str = "UNKNOWN",
    review_image_bytes: bytes | None = None,
    review_scope: str | None = None,
) -> None:
    with st.container(border=True):
        st.markdown('<span class="cc-inspection-marker"></span>', unsafe_allow_html=True)
        st.markdown("### 전체 진단")
        st.info("미리보기 · 진단 미실행" if preview else "진단 미실행")
        st.caption(f"사진 · {len(project.image_records)}장")
        symptom = {
            "power_failure": "전원 불량", "output_anomaly": "출력 이상",
            "overheating": "과열", "intermittent_operation": "간헐적 작동",
            "unknown": "증상 불명",
        }.get(project.symptom_category or "unknown", "증상 불명")
        st.caption(f"증상 · {symptom}")
        if project.symptom_description:
            st.caption(project.symptom_description)
        st.divider()
        if preview:
            st.markdown("**부품** · 인식 결과 없음")
        elif detection_status in {"DETECTOR_NOT_CONFIGURED", "BLOCKED_MODEL_UNAVAILABLE"}:
            st.markdown("**부품 검출** · 검출 모델 미설정")
        elif detection_status not in {"INFERENCE_COMPLETE", "PARTIAL_MAP_READY"}:
            st.markdown("**부품 검출** · 인식 결과 없음")
        elif not components:
            st.markdown("**인식된 부품** · 없음")
        else:
            st.markdown(f"**인식된 부품** · {len(components)}개")
        st.markdown("### 선택 부품")
        if selected_component is None:
            st.caption("선택 없음")
        else:
            st.markdown(f"**종류** · {_component_label(selected_component.component_type)}")
            st.caption(f"Component ID · {selected_component.component_id}")
            if selected_component.display_label:
                st.caption(f"Reference · {selected_component.display_label}")
            if selected_component.ocr_marking:
                st.caption(f"OCR 표기 · {selected_component.ocr_marking}")
            if selected_component.possible_part_number:
                st.caption(f"가능 부품 번호 · {selected_component.possible_part_number}")
            st.caption(f"식별 상태 · {display_identity_status(selected_component.identification_status)}")
            if selected_component.identity_match_status:
                st.caption(f"이전 식별 연속성 · {display_identity_status(selected_component.identity_match_status)}")
            st.caption(f"관측 사진 · {len(selected_component.source_backend_photo_ids)}장")
            st.caption("정합 · 후보 · 미검증")
            if selected_component.evidence:
                with st.expander("검출 근거"):
                    st.write(" · ".join(selected_component.evidence))
            if (selected_component.component_id.startswith("auto-")
                    and review_image_bytes is not None and review_scope):
                render_selected_component_editor(
                    st, image_bytes=review_image_bytes, scope=review_scope,
                    component_id=selected_component.component_id,
                )
        measurements = get_measurements(
            st.session_state, project_id=project.project_id, board_id=project.board_id or "",
        )
        map_version = st.session_state.get(MAP_VERSION_KEY)
        render_measurement_panel(
            st, project, selected_component=selected_component, preview=preview,
            component_map_version=map_version,
        )
        selection = st.session_state.get("circuitcare_v2_selection")
        render_diagnosis_panel(
            st, project, measurements=measurements,
            analysis_context_id=getattr(selection, "analysis_context_id", None),
            component_map_version=map_version,
        )
        st.markdown("**다음 점검** · 분석 데이터 없음")
