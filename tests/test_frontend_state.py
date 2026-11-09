from circuitcare.frontend.intake import create_project_context
from circuitcare.frontend.state import (
    INTAKE_IMAGES_KEY, INTAKE_PROJECT_KEY, clear_intake_draft, draft_is_current,
    reset_intake_for_new_project,
)
from circuitcare.frontend.component_map_adapter import COMPONENT_MAP_STATE_KEY


def test_session_draft_is_fingerprint_scoped_and_clear_keeps_other_inputs():
    image = {
        "image_id": "image-1", "filename": "board.png", "sha256": "abc123",
        "role": "unknown", "media_type": "image/png",
    }
    project = create_project_context(images=[image], symptom_category="unknown", symptom_description=None)
    state = {INTAKE_PROJECT_KEY: project, "symptom": "unknown", "photos": [image]}

    assert draft_is_current(state, project.input_fingerprint)
    assert not draft_is_current(state, "changed-inputs")
    clear_intake_draft(state)
    assert INTAKE_PROJECT_KEY not in state
    assert state["symptom"] == "unknown"
    assert state["photos"] == [image]


def test_new_project_reset_clears_multiview_owner_and_viewer_state():
    state = {
        INTAKE_IMAGES_KEY: [{"image_id": "old"}],
        COMPONENT_MAP_STATE_KEY: {"project_id": "old-project"},
        "circuitcare_v3_photo_map": ["stale"],
        "circuitcare_v3_active_side": "back",
        "circuitcare_v3_view_mode": "통합 이미지",
        "circuitcare_v2_view_transform": {"canonical_frame_id": "old-frame"},
    }
    reset_intake_for_new_project(state)
    assert state[INTAKE_IMAGES_KEY] == []
    assert COMPONENT_MAP_STATE_KEY not in state
    assert "circuitcare_v3_photo_map" not in state
    assert "circuitcare_v3_active_side" not in state
    assert "circuitcare_v3_view_mode" not in state
    assert "circuitcare_v2_view_transform" not in state


def test_new_project_reset_clears_f5_measurement_diagnosis_provider_and_selection():
    from circuitcare.frontend.measurement_state import MEASUREMENTS_KEY, DIAGNOSIS_STATE_KEY

    state = {
        "circuitcare_v2_selection": "project-a-selection",
        MEASUREMENTS_KEY: {"project-a": ["measurement"]},
        DIAGNOSIS_STATE_KEY: {"project-a": {"result": "old-result"}},
        "circuitcare_v5_diagnosis_provider": object(),
        "circuitcare_v5_diagnosis_context": {"project-a": "old-context"},
        "cc_v5_value_project-a-board-a-c1": "3.3",
    }
    reset_intake_for_new_project(state)
    for key in ("circuitcare_v2_selection", MEASUREMENTS_KEY, DIAGNOSIS_STATE_KEY,
                "circuitcare_v5_diagnosis_provider", "circuitcare_v5_diagnosis_context",
                "cc_v5_value_project-a-board-a-c1"):
        assert key not in state


def test_editing_intake_keeps_f5_measurement_history():
    from circuitcare.frontend.measurement_state import MEASUREMENTS_KEY

    state = {MEASUREMENTS_KEY: {"project-a": ["measurement"]}, "symptom": "unknown"}
    clear_intake_draft(state)
    assert state[MEASUREMENTS_KEY] == {"project-a": ["measurement"]}
