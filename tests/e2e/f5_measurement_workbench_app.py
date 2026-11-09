"""Explicit synthetic browser fixture; never used by the normal dashboard entrypoint."""

from __future__ import annotations

import base64
import io

import streamlit as st
from PIL import Image, ImageDraw

from circuitcare.dashboard.components.pcb_workspace import render_component_viewer
from circuitcare.dashboard.views.inspection_panel import MAP_VERSION_KEY, render_inspection_panel
from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel
from circuitcare.frontend.models import ImageRecord, ProjectContext, SelectionState
from circuitcare.frontend.state import reset_intake_for_new_project
from circuitcare.frontend.workbench_state import SELECTION_KEY, enter_workbench

st.set_page_config(page_title="F5 측정 UI 브라우저 테스트", layout="wide")
project_id = st.session_state.get("f5_fixture_project_id", "f5-e2e-project-a")
project = ProjectContext(
    project_id=project_id, board_id=f"{project_id}-board",
    image_records=(ImageRecord(image_id="f5-e2e-photo", side="front", filename="synthetic-test.png"),),
)
st.session_state["circuitcare_v2_project"] = project
if not isinstance(st.session_state.get(SELECTION_KEY), SelectionState):
    enter_workbench(st.session_state, project)

st.warning("E2E TEST FIXTURE · 합성 이미지/부품 · 실제 PCB 분석 또는 진단 결과가 아닙니다")
st.session_state[MAP_VERSION_KEY] = "f5-fixture-map-v1"
if st.button("테스트 전용 · 새 프로젝트", key="f5_fixture_new_project"):
    reset_intake_for_new_project(st.session_state)
    st.session_state["f5_fixture_project_id"] = "f5-e2e-project-b"
    next_project = ProjectContext(
        project_id="f5-e2e-project-b", board_id="f5-e2e-project-b-board",
        image_records=(ImageRecord(image_id="f5-e2e-photo-b", side="front", filename="synthetic-test-b.png"),),
    )
    st.session_state["circuitcare_v2_project"] = next_project
    enter_workbench(st.session_state, next_project)
    st.rerun()
st.title("F5 측정 입력 브라우저 검증")
left, right = st.columns([3, 1.4], gap="medium")

buffer = io.BytesIO()
image = Image.new("RGB", (640, 420), "#263745")
draw = ImageDraw.Draw(image)
draw.rectangle((80, 60, 560, 360), outline="#68a080", width=8)
draw.rectangle((140, 145, 250, 245), fill="#ded8bd", outline="#d5a629", width=4)
draw.rectangle((390, 160, 500, 270), fill="#465a89", outline="#a8bfcc", width=4)
image.save(buffer, format="PNG")
image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
components = (
    ComponentOverlayViewModel(
        project_id=project.project_id, board_id=project.board_id, board_side="TOP",
        component_id="f5-fixture-cap", component_type="capacitor", display_label=None,
        canonical_frame_id="f5-fixture-frame", coordinate_frame="MOSAIC_PIXEL",
        bbox_xyxy=(140, 145, 250, 245), polygon_xy=((140,145),(250,145),(250,245),(140,245)),
        source_backend_photo_ids=("f5-photo",), observation_ids=("f5-observation-cap",),
        detection_confidence=.87, identification_status="SUGGESTED_UNCONFIRMED",
        identity_match_status="NEW_COMPONENT", ocr_marking="103", possible_part_number=None,
        evidence=("SYNTHETIC_TEST_FIXTURE",), overlay_status="DETECTION_CANDIDATE",
        diagnosis_display_status="DIAGNOSIS_NOT_RUN",
    ),
    ComponentOverlayViewModel(
        project_id=project.project_id, board_id=project.board_id, board_side="TOP",
        component_id="f5-fixture-resistor", component_type="resistor", display_label=None,
        canonical_frame_id="f5-fixture-frame", coordinate_frame="MOSAIC_PIXEL",
        bbox_xyxy=(390, 160, 500, 270), polygon_xy=((390,160),(500,160),(500,270),(390,270)),
        source_backend_photo_ids=("f5-photo",), observation_ids=("f5-observation-resistor",),
        detection_confidence=.91, identification_status="SUGGESTED_UNCONFIRMED",
        identity_match_status="NEW_COMPONENT", ocr_marking=None, possible_part_number=None,
        evidence=("SYNTHETIC_TEST_FIXTURE",), overlay_status="DETECTION_CANDIDATE",
        diagnosis_display_status="DIAGNOSIS_NOT_RUN",
    ),
)

with left:
    selection = st.session_state.get(SELECTION_KEY)
    selected_id = getattr(selection, "selected_component", None)
    render_component_viewer(
        st, image=image, source_image_bytes=buffer.getvalue(), canonical_png_bytes=buffer.getvalue(),
        image_size_px=(640, 420), components=components, project=project, board_side="TOP",
        frame_id="f5-fixture-frame", selected_component_id=selected_id,
        map_cache_key="f5-fixture-map-v1", image_mode="통합 이미지",
    )

with right:
    selected = next((item for item in components if item.component_id == selected_id), None)
    render_inspection_panel(
        st, project, components=components, selected_component=selected,
        detection_status="INFERENCE_COMPLETE",
    )
