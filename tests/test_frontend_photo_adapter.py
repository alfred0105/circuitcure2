from __future__ import annotations

import io
import pytest
from PIL import Image

from circuitcare.frontend.intake import add_image_batch, create_project_context
from circuitcare.frontend.photo_adapter import (
    adapt_component_map_input, adapt_photo_record, normalize_photo_metadata,
)


def photo(**overrides):
    value = {
        "image_id": "photo-record-1",
        "filename": "board.png",
        "content_sha256": "a" * 64,
        "media_type": "image/png",
        "image_dimensions": (64, 48),
        "image_format": "PNG",
        "side": "front",
        "capture_type": "full_view",
    }
    value.update(overrides)
    return value


def png_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (20, 12), "blue").save(stream, format="PNG")
    return stream.getvalue()


def test_frontend_metadata_maps_to_backend_record_with_project_ownership_and_hash():
    result = adapt_photo_record(photo(), project_id="project-a", board_id="board-a")

    assert result.status == "READY"
    assert result.photo is not None
    assert result.photo.project_id == "project-a"
    assert result.photo.board_id == "board-a"
    assert result.photo.source_image_id == "photo-record-1"
    assert result.photo.source_content_sha256 == "a" * 64
    assert result.photo.record.photo_id == "photo-record-1"
    assert result.photo.record.file_sha256 == "a" * 64
    assert result.photo.record.side == "front"
    assert result.photo.record.capture_type == "full_view"
    assert result.photo.record.image_size_px == (64, 48)


def test_unknown_capture_type_is_reported_unsupported_without_defaulting_to_full_view():
    result = adapt_photo_record(photo(capture_type="unknown"), project_id="p", board_id="b")

    assert result.status == "UNSUPPORTED"
    assert result.photo is None
    assert "미지정" in result.reason


def test_legacy_detail_side_maps_only_to_unknown_side_close_up_capture():
    legacy = photo(side="detail")
    legacy.pop("capture_type")
    result = adapt_photo_record(legacy, project_id="p", board_id="b")

    assert result.status == "READY"
    assert result.legacy_converted
    assert result.photo.record.side == "unknown"
    assert result.photo.record.capture_type == "close_up"


@pytest.mark.parametrize("field,value", [("side", "left"), ("capture_type", "macro")])
def test_invalid_photo_axes_are_rejected(field, value):
    with pytest.raises(ValueError):
        normalize_photo_metadata(photo(**{field: value}))


def test_missing_dimensions_or_format_is_unsupported_not_guessed():
    result = adapt_photo_record(
        photo(image_dimensions=None, image_format=None), project_id="p", board_id="b",
    )
    assert result.status == "UNSUPPORTED"
    assert result.photo is None


def test_unsupported_image_format_and_invalid_hash_are_safe():
    result = adapt_photo_record(photo(image_format="TIFF"), project_id="p", board_id="b")
    assert result.status == "UNSUPPORTED"
    with pytest.raises(ValueError, match="SHA-256"):
        adapt_photo_record(photo(content_sha256="nope"), project_id="p", board_id="b")


def test_legacy_media_type_and_dimensions_are_retained_when_format_field_was_absent():
    legacy = photo()
    legacy.pop("image_format")
    legacy.pop("image_dimensions")
    legacy.update(width=64, height=48)
    record, _ = normalize_photo_metadata(legacy)
    assert record.image_dimensions == (64, 48)
    assert record.image_format == "PNG"


def test_same_content_has_stable_hash_but_distinct_photo_record_ids_across_projects():
    raw = png_bytes()
    first = add_image_batch([], [("board.png", raw)])[0][0]
    second = add_image_batch([], [("board.png", raw)])[0][0]
    assert first["sha256"] == second["sha256"]
    assert first["image_id"] != second["image_id"]
    changed_metadata = {**first, "side": "back"}
    project_one = create_project_context(
        images=[first], symptom_category="unknown", symptom_description=None,
    )
    project_two = create_project_context(
        images=[changed_metadata], symptom_category="unknown", symptom_description=None,
    )
    assert first["sha256"] == changed_metadata["sha256"]
    assert project_one.input_fingerprint != project_two.input_fingerprint
    assert project_one.project_id != project_two.project_id
    assert project_one.board_id != project_two.board_id


@pytest.mark.parametrize(
    "side,capture_type,backend_side,backend_role",
    [("front", "full_view", "TOP", "OVERVIEW"), ("back", "close_up", "BOTTOM", "CLOSEUP"),
     ("front", "oblique", "TOP", "ANGLED")],
)
def test_phase21_2_contract_mapping_preserves_frontend_identity_and_hash(
    side, capture_type, backend_side, backend_role,
):
    raw = png_bytes()
    result = adapt_component_map_input(
        photo(side=side, capture_type=capture_type), board_id="board-7", data=raw,
    )
    assert result.status == "READY"
    assert result.request.board_id == "board-7"
    assert result.request.frontend_photo_id == "photo-record-1"
    assert result.request.content_sha256 == "a" * 64
    assert result.request.side == backend_side
    assert result.request.role == backend_role
    assert result.request.data == raw


@pytest.mark.parametrize(
    "side,capture_type", [("unknown", "full_view"), ("front", "unknown"),
                          ("front", "marking_detail")],
)
def test_phase21_2_adapter_withholds_unknown_and_unmapped_values(side, capture_type):
    result = adapt_component_map_input(
        photo(side=side, capture_type=capture_type), board_id="board-7", data=png_bytes(),
    )
    assert result.status == "UNSUPPORTED"
    assert result.request is None
