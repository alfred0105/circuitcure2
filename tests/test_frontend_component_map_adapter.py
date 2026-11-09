import base64
import io

import pytest
from PIL import Image

from circuitcare.frontend.component_map_adapter import (
    COMPONENT_MAP_STATE_KEY, available_photo_sides, clear_component_map_state,
    component_map_cache_key, component_map_entry_is_current, decode_canonical_image,
    map_backend_photo_ids, photo_input_fingerprint, validate_component_map,
)
from circuitcare.frontend.photo_adapter import ComponentMapPhotoInput


def _png():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), "#456").save(buffer, format="PNG")
    return buffer.getvalue()


def _map():
    """Synthetic TEST FIXTURE contract; it is never used by the product UI."""
    raw = _png()
    return {"schema_version": "circuitcare.pcb-component-map.v1", "board_id": "b1", "photos": [
        {"photo_id": "backend-1", "sha256": "a" * 64, "side": "TOP", "role": "OVERVIEW"}],
        "sides": {"TOP": {"canonical_image": {
            "image_id": "frame-1", "coordinate_frame": "MOSAIC_PIXEL", "image_size_px": [8, 6],
            "reference_photo_id": "backend-1", "image_png_base64": base64.b64encode(raw).decode()},
            "registrations": [{"photo_id": "backend-1", "status": "REFERENCE"}]}},
    }


def test_schema_validation_and_canonical_decode():
    value = validate_component_map(_map(), board_id="b1")
    canonical = decode_canonical_image(value["sides"]["TOP"]["canonical_image"])
    assert (canonical.frame_id, canonical.width, canonical.height) == ("frame-1", 8, 6)
    assert canonical.registration_status == "CANDIDATE_GEOMETRY_ONLY"


def test_schema_and_base64_fail_closed():
    with pytest.raises(ValueError):
        validate_component_map({**_map(), "schema_version": "v9"}, board_id="b1")
    canonical = _map()["sides"]["TOP"]["canonical_image"]
    with pytest.raises(ValueError):
        decode_canonical_image({**canonical, "image_png_base64": "!!!"})


def test_photo_id_mapping_requires_unique_side_role_hash():
    item = ComponentMapPhotoInput("b1", "front-1", "a" * 64, "TOP", "OVERVIEW", b"x", "name")
    result = _map()
    mapped = map_backend_photo_ids(project_id="p1", board_id="b1", inputs=[item], result=result)
    assert mapped[0].backend_photo_id == "backend-1"
    assert mapped[0].registration_status == "REFERENCE"
    duplicate = map_backend_photo_ids(project_id="p1", board_id="b1", inputs=[item, item], result=result)
    assert all(row.mapping_status == "UNCONFIRMED" for row in duplicate)


def test_cache_key_is_project_board_and_input_scoped():
    item = ComponentMapPhotoInput("b1", "front-1", "a" * 64, "TOP", "OVERVIEW", b"x", "name")
    key = component_map_cache_key(project_id="p1", board_id="b1", inputs=[item])
    assert key != component_map_cache_key(project_id="p2", board_id="b1", inputs=[item])
    assert key != component_map_cache_key(project_id="p1", board_id="b2", inputs=[item])


@pytest.mark.parametrize("sides, expected", [
    (["front", "front", "back"], ["front", "back"]),
    (["front", "unknown"], ["front", "unknown"]),
    (["front", "back", "unknown"], ["front", "back", "unknown"]),
    (["unknown", "unknown", "unknown"], ["unknown"]),
])
def test_viewer_side_options_keep_all_photos_accessible(sides, expected):
    assert available_photo_sides([{"side": side} for side in sides]) == expected


def test_cache_includes_all_photo_metadata_but_not_symptoms():
    item = ComponentMapPhotoInput("b1", "p1", "a" * 64, "TOP", "OVERVIEW", b"x", "name")
    manifest = [{"image_id": "p1", "content_sha256": "a" * 64,
                 "side": "front", "capture_type": "full_view"}]
    key = component_map_cache_key(project_id="p1", board_id="b1", inputs=[item], photo_manifest=manifest)
    changed_metadata = [{**manifest[0], "side": "unknown"}]
    assert key != component_map_cache_key(project_id="p1", board_id="b1", inputs=[item], photo_manifest=changed_metadata)
    # The fingerprint only consumes photo metadata, so a symptom-only edit does not affect it.
    symptom_edit = [{**manifest[0], "symptom_category": "overheating"}]
    assert photo_input_fingerprint(manifest) == photo_input_fingerprint(symptom_edit)


def test_project_cache_ownership_and_atomic_clear():
    entry = {"project_id": "p1", "board_id": "b1", "input_fingerprint": "fp",
             "component_map_cache_key": "key", "component_map": {}, "photo_id_map": []}
    assert component_map_entry_is_current(entry, project_id="p1", board_id="b1", photo_fingerprint="fp", cache_key="key")
    assert not component_map_entry_is_current(entry, project_id="p2", board_id="b1", photo_fingerprint="fp", cache_key="key")
    state = {COMPONENT_MAP_STATE_KEY: entry, "circuitcare_v3_photo_map": [1], "circuitcare_v3_active_side": "back"}
    clear_component_map_state(state)
    assert COMPONENT_MAP_STATE_KEY not in state
    assert "circuitcare_v3_photo_map" not in state
    assert "circuitcare_v3_active_side" not in state


def test_unsupported_canonical_frame_is_an_artifact_failure_not_schema_failure():
    value = _map()
    value["sides"]["TOP"]["canonical_image"]["coordinate_frame"] = "UNKNOWN_FRAME"
    validated = validate_component_map(value, board_id="b1")
    with pytest.raises(ValueError, match="프레임"):
        decode_canonical_image(validated["sides"]["TOP"]["canonical_image"])
