"""Small AppTest target for the production measurement panel and profile defaults."""

import streamlit as st

from circuitcare.dashboard.views.measurement_panel import render_measurement_panel
from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel
from circuitcare.frontend.models import ProjectContext

st.set_page_config(page_title="Measurement profile AppTest", layout="wide")
project = ProjectContext("f5-profile-apptest", "f5-profile-board")
component = ComponentOverlayViewModel(
    project_id=project.project_id, board_id=project.board_id, board_side="TOP",
    component_id="R1", board_component_id="BC-R1", component_type="resistor",
    display_label="R1", canonical_frame_id="photo-1", coordinate_frame="SOURCE_PHOTO_PIXEL",
    bbox_xyxy=(10, 10, 30, 30), polygon_xy=(), source_backend_photo_ids=("photo-1",),
    observation_ids=(), detection_confidence=.9, identification_status="USER_CONFIRMED",
    identity_match_status="PHOTO_LOCAL", ocr_marking="103", possible_part_number=None,
    evidence=(), overlay_status="CANDIDATE",
)

render_measurement_panel(
    st, project, selected_component=component, component_map_version="map-v1",
    compact=True, fault_hypothesis="RESISTOR_FAILURE", recommended_test_id="isolated_resistance",
)
