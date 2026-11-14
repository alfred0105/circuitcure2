"""AppTest page exercising the production diagnosis and measurement panels together."""

import streamlit as st
from PIL import Image, ImageDraw

from circuitcare.dashboard.views.measurement_panel import render_measurement_panel
from circuitcare.dashboard.views.symptom_diagnosis_panel import render_symptom_diagnosis_panel
from circuitcare.dashboard.views.inspection_panel import MAP_VERSION_KEY
from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel
from circuitcare.frontend.models import ProjectContext

st.set_page_config(page_title="Measurement reanalysis AppTest", layout="wide")
project = ProjectContext("f5-reanalysis-apptest", "f5-reanalysis-board")
component = ComponentOverlayViewModel(
    project_id=project.project_id, board_id=project.board_id, board_side="TOP",
    component_id="R1", board_component_id="BC-R1", component_type="resistor",
    display_label="R1", canonical_frame_id="photo-1", coordinate_frame="SOURCE_PHOTO_PIXEL",
    bbox_xyxy=(10, 10, 30, 30), polygon_xy=(), source_backend_photo_ids=("photo-1",),
    observation_ids=(), detection_confidence=.9, identification_status="USER_CONFIRMED",
    identity_match_status="PHOTO_LOCAL", ocr_marking="103", possible_part_number=None,
    evidence=(), overlay_status="CANDIDATE",
)
st.session_state[MAP_VERSION_KEY] = "map-v1"
image = Image.new("RGB", (320, 200), "#264334")
draw = ImageDraw.Draw(image)
draw.rectangle((10, 10, 30, 30), fill="#c8b57c", outline="#f0f0f0", width=2)
st.image(image, caption="합성 AppTest PCB 이미지")
render_symptom_diagnosis_panel(st, project, components=(component,))
render_measurement_panel(
    st, project, selected_component=component, component_map_version="map-v1",
    compact=True, fault_hypothesis="RESISTOR_FAILURE", recommended_test_id="isolated_resistance",
)
