from circuitcare.dashboard.components.pcb_workspace import (
    coordinate_status_for_viewer, scale_overlay_to_display, validate_component_selection_event,
)
from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel
from circuitcare.frontend.component_overlay_models import display_identity_status


def _model():
    return ComponentOverlayViewModel(
        project_id="p1", board_id="b1", board_side="TOP", component_id="c1",
        component_type="resistor", display_label="R4", canonical_frame_id="frame1",
        coordinate_frame="EXIF_CORRECTED_IMAGE_PIXEL", bbox_xyxy=(10, 20, 30, 50),
        polygon_xy=((10, 20), (30, 20), (30, 50), (10, 50)),
        source_backend_photo_ids=("bp1",), observation_ids=("o1",),
        detection_confidence=.8, identification_status="SUGGESTED_UNCONFIRMED",
        identity_match_status="NEW_COMPONENT", ocr_marking=None,
        possible_part_number=None, evidence=("MODEL_DETECTION_CANDIDATE",),
        overlay_status="DETECTION_CANDIDATE",
    )


def test_original_coordinates_scale_with_known_exif_corrected_resize():
    scaled = scale_overlay_to_display((_model(),), source_size=(100, 100), display_size=(50, 50))[0]
    assert scaled.bbox_xyxy == (5, 10, 15, 25)
    assert scaled.polygon_xy == ((5, 10), (15, 10), (15, 25), (5, 25))


def test_click_and_clear_selection_events_are_scoped_and_monotonic():
    event = {"project_id": "p1", "board_id": "b1", "board_side": "TOP",
             "canonical_frame_id": "frame1", "component_id": "c1",
             "selection_action": "select", "event_seq": 1, "event_id": "evt-1"}
    kwargs = dict(project_id="p1", board_id="b1", board_side="TOP", frame_id="frame1",
                  allowed_ids={"c1"}, last_sequence=0)
    assert validate_component_selection_event(event, **kwargs) == (True, None)
    assert validate_component_selection_event({**event, "project_id": "other"}, **kwargs)[0] is False
    assert validate_component_selection_event({**event, "board_id": "other"}, **kwargs)[0] is False
    assert validate_component_selection_event({**event, "board_side": "BOTTOM"}, **kwargs)[0] is False
    assert validate_component_selection_event({**event, "canonical_frame_id": "old-frame"}, **kwargs)[0] is False
    assert validate_component_selection_event({**event, "component_id": "missing"}, **kwargs)[0] is False
    assert validate_component_selection_event(event, **{**kwargs, "last_sequence": 1})[0] is False
    assert validate_component_selection_event({k: v for k, v in event.items() if k != "event_id"}, **kwargs)[0] is False
    clear = {**event, "component_id": None, "selection_action": "clear", "event_seq": 2}
    assert validate_component_selection_event(clear, **{**kwargs, "last_sequence": 1}) == (True, None)


def test_hover_identity_status_is_user_facing_korean_and_unknown_is_preserved():
    assert display_identity_status("SUGGESTED_UNCONFIRMED") == "미확인 후보"
    assert display_identity_status("IDENTITY_MATCHED") == "동일 부품 후보"
    assert display_identity_status("IDENTITY_AMBIGUOUS") == "식별 모호"
    assert display_identity_status("FUTURE_STATUS") == "FUTURE_STATUS"


def test_coordinate_status_requires_client_mount_and_valid_frame():
    assert coordinate_status_for_viewer(api_available=True, client_ready=False) == "UNVERIFIED_RESPONSIVE_CSS"
    assert coordinate_status_for_viewer(api_available=True, client_ready=True) == "VERIFIED_SHARED_SVG_VIEWBOX"
    assert coordinate_status_for_viewer(api_available=False, client_ready=False) == "VIEWER_UNAVAILABLE"
    assert coordinate_status_for_viewer(api_available=True, client_ready=True, frame_valid=False) == "INVALID_COORDINATE_FRAME"
