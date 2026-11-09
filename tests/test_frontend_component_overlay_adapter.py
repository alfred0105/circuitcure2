import pytest

from circuitcare.frontend.component_overlay_adapter import (
    adapt_component_map_components, adapt_original_photo_components,
)


@pytest.fixture
def component_map():
    return {
        "board_id": "board-1",
        "sides": {
            "TOP": {
                "detection_status": "INFERENCE_COMPLETE",
                "canonical_image": {"image_id": "mosaic:board-1:TOP", "coordinate_frame": "MOSAIC_PIXEL",
                                    "image_size_px": [200, 100]},
                "components": [{
                    "component_id": "component-a", "component_type": "capacitor",
                    "bbox_xyxy": [10, 20, 40, 60],
                    "polygon_xy": [[10, 20], [40, 20], [40, 60], [10, 60]],
                    "source_image_ids": ["backend-photo"],
                    "component_observation_ids": ["observation-a"],
                    "detection_confidence": 0.87,
                    "identification_status": "SUGGESTED_UNCONFIRMED",
                    "identity_match_status": "NEW_COMPONENT", "ocr_marking": "103",
                }],
                "component_observations": [{
                    "observation_id": "observation-a", "photo_id": "backend-photo",
                    "original_bbox_xyxy": [5, 10, 20, 30],
                    "original_coordinate_frame": "EXIF_CORRECTED_IMAGE_PIXEL",
                    "registration_status": "REFERENCE", "confidence": 0.87,
                }],
            },
            "BOTTOM": {"detection_status": "DETECTOR_NOT_CONFIGURED", "components": []},
        },
    }


def test_adapts_only_verified_current_canonical_frame(component_map):
    result = adapt_component_map_components(component_map, project_id="project-1", board_id="board-1", side="TOP")
    assert result.status == "READY"
    assert result.canonical_frame_id == "mosaic:board-1:TOP"
    assert result.components[0].component_id == "component-a"
    assert result.components[0].bbox_xyxy == (10, 20, 40, 60)
    assert result.components[0].detection_confidence == 0.87
    assert result.components[0].identification_status == "SUGGESTED_UNCONFIRMED"


@pytest.mark.parametrize("bbox", [[-1, 0, 5, 5], [1, 1, 1, 5], [1, 1, float("inf"), 5], [1, 1, 201, 5]])
def test_rejects_invalid_bounds(component_map, bbox):
    component_map["sides"]["TOP"]["components"][0]["bbox_xyxy"] = bbox
    result = adapt_component_map_components(component_map, project_id="p", board_id="board-1", side="TOP")
    assert result.status == "INVALID_DATA"
    assert not result.components
    assert result.rejected_count == 1


def test_rejects_duplicate_ids_and_wrong_board_or_frame(component_map):
    row = component_map["sides"]["TOP"]["components"][0]
    component_map["sides"]["TOP"]["components"].append(dict(row))
    duplicate = adapt_component_map_components(component_map, project_id="p", board_id="board-1", side="TOP")
    assert duplicate.status == "INVALID_DATA"
    assert duplicate.reason
    wrong_board = adapt_component_map_components(component_map, project_id="p", board_id="other", side="TOP")
    assert wrong_board.status == "INVALID_DATA"
    component_map["sides"]["TOP"]["components"] = [row]
    component_map["sides"]["TOP"]["canonical_image"]["coordinate_frame"] = "IMAGE_PIXEL"
    wrong_frame = adapt_component_map_components(component_map, project_id="p", board_id="board-1", side="TOP")
    assert wrong_frame.status == "INVALID_FRAME"


def test_bottom_side_does_not_reuse_top_components(component_map):
    result = adapt_component_map_components(component_map, project_id="p", board_id="board-1", side="BOTTOM")
    assert result.status == "NO_CANONICAL_IMAGE"
    assert not result.components


def test_original_overlay_requires_same_photo_exif_observation(component_map):
    models = adapt_original_photo_components(
        component_map, project_id="p", board_id="board-1", side="TOP",
        frontend_photo_id="front-photo", backend_photo_id="backend-photo", image_size=(100, 80),
    )
    assert len(models) == 1
    assert models[0].canonical_frame_id == "front-photo"
    assert models[0].coordinate_frame == "EXIF_CORRECTED_IMAGE_PIXEL"
    assert models[0].bbox_xyxy == (5, 10, 20, 30)
    wrong_photo = adapt_original_photo_components(
        component_map, project_id="p", board_id="board-1", side="TOP",
        frontend_photo_id="front-photo", backend_photo_id="other-photo", image_size=(100, 80),
    )
    assert wrong_photo == ()
