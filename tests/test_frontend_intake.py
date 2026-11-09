from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from circuitcare.frontend.intake import (
    CAPTURE_TYPES,
    MAX_SYMPTOM_CHARS,
    PHOTO_SIDES,
    add_image_batch,
    create_project_context,
    normalize_symptom_description,
    validate_circuit_upload,
    validate_image_upload,
)


def png_bytes(size: tuple[int, int] = (20, 12), color: str = "red") -> bytes:
    image = Image.new("RGB", size, color)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def make_image(data: bytes, filename: str = "board.png", role: str = "unknown") -> dict:
    digest, width, height = validate_image_upload(filename, data)
    return {
        "image_id": "photo-1", "sha256": digest, "filename": filename,
        "data": data, "side": role, "capture_type": "unknown", "media_type": "image/png",
        "width": width, "height": height, "image_dimensions": (width, height), "image_format": "PNG",
    }


def test_accepts_multiple_images_with_roles_and_skips_duplicate():
    first, second = png_bytes(), png_bytes(color="blue")
    images, messages = add_image_batch([], [("front.png", first), ("detail.png", second)])
    assert len(images) == 2
    assert not messages
    assert [item["side"] for item in images] == ["unknown", "unknown"]
    assert [item["capture_type"] for item in images] == ["unknown", "unknown"]
    assert images[0]["image_id"] != images[1]["image_id"]
    images, messages = add_image_batch(images, [("copy.png", first)])
    assert len(images) == 2
    assert "중복" in messages[0]


def test_rejects_unsupported_corrupt_and_oversized_images_without_losing_valid_ones():
    corrupt = b"not an image"
    too_large = b"x" * (20 * 1024 * 1024 + 1)
    oversized_pixels = png_bytes((5001, 5000))
    valid = png_bytes()
    images, messages = add_image_batch([], [
        ("bad.gif", valid), ("broken.png", corrupt), ("big.png", too_large),
        ("pixels.png", oversized_pixels), ("ok.png", valid),
    ])
    assert [item["filename"] for item in images] == ["ok.png"]
    assert len(messages) == 4


def test_image_count_and_aggregate_byte_limits_are_enforced():
    first, second, third = png_bytes(), png_bytes(color="blue"), png_bytes(color="green")
    images, messages = add_image_batch([], [("1.png", first), ("2.png", second), ("3.png", third)], max_photos=2)
    assert len(images) == 2
    assert "최대 2장" in messages[0]
    limited, messages = add_image_batch([], [("1.png", first), ("2.png", second)], max_total_bytes=len(first))
    assert len(limited) == 1
    assert "용량 한도" in messages[0]


def test_symptom_normalization_and_length_limit():
    assert normalize_symptom_description("  전원\n  LED   불량 ") == "전원 LED 불량"
    assert normalize_symptom_description("  ") is None
    with pytest.raises(ValueError):
        normalize_symptom_description("x" * (MAX_SYMPTOM_CHARS + 1))


def test_project_ids_and_fingerprints_capture_inputs_and_roles():
    image = make_image(png_bytes())
    base = create_project_context(images=[image], symptom_category="unknown", symptom_description=None)
    another_project = create_project_context(images=[image], symptom_category="unknown", symptom_description=None)
    changed_symptom = create_project_context(images=[image], symptom_category="overheating", symptom_description=None)
    changed_side = create_project_context(
        images=[{**image, "side": "front"}], symptom_category="unknown", symptom_description=None,
    )
    changed_capture = create_project_context(
        images=[{**image, "capture_type": "close_up"}], symptom_category="unknown", symptom_description=None,
    )
    assert base.project_id != another_project.project_id
    assert base.board_id != another_project.board_id
    assert base.input_fingerprint == another_project.input_fingerprint
    assert base.input_fingerprint != changed_symptom.input_fingerprint
    assert base.input_fingerprint != changed_side.input_fingerprint
    assert base.input_fingerprint != changed_capture.input_fingerprint
    assert base.image_records[0].capture_type == "unknown"
    assert base.image_records[0].side == "unknown"


def test_circuit_attachment_fingerprint_uses_content_and_type_not_display_filename():
    image = make_image(png_bytes())
    a = create_project_context(
        images=[image], symptom_category="unknown", symptom_description=None,
        circuit_source={"filename": "first.xml", "content_sha256": "a" * 64, "source_type": "kicad_xml_netlist"},
    )
    renamed = create_project_context(
        images=[image], symptom_category="unknown", symptom_description=None,
        circuit_source={"filename": "renamed.xml", "content_sha256": "a" * 64, "source_type": "kicad_xml_netlist"},
    )
    changed = create_project_context(
        images=[image], symptom_category="unknown", symptom_description=None,
        circuit_source={"filename": "first.xml", "content_sha256": "b" * 64, "source_type": "kicad_xml_netlist"},
    )
    assert a.input_fingerprint == renamed.input_fingerprint
    assert a.input_fingerprint != changed.input_fingerprint


def test_image_order_affects_fingerprint_and_empty_photo_set_is_rejected():
    first = {**make_image(png_bytes(color="red"), "a.png", "front"), "image_id": "photo-a"}
    second = {**make_image(png_bytes(color="blue"), "b.png", "back"), "image_id": "photo-b"}
    one = create_project_context(images=[first, second], symptom_category="unknown", symptom_description="")
    two = create_project_context(images=[second, first], symptom_category="unknown", symptom_description="")
    assert one.input_fingerprint != two.input_fingerprint
    with pytest.raises(ValueError, match="한 장 이상"):
        create_project_context(images=[], symptom_category="unknown", symptom_description=None)


def test_existing_supported_kicad_files_are_validated():
    root = Path(__file__).parent
    netlist = (root / "fixtures" / "divider_netlist.xml").read_bytes()
    schematic = (root / "fixtures" / "divider.kicad_sch").read_bytes()
    assert validate_circuit_upload("divider.xml", netlist)["source_type"] == "kicad_xml_netlist"
    assert validate_circuit_upload("divider.kicad_sch", schematic)["source_type"] == "kicad_schematic"
    with pytest.raises(ValueError, match="지원하는 설계 자료"):
        validate_circuit_upload("board.kicad_pcb", b"(kicad_pcb)")


def test_side_and_capture_codes_are_separate_and_default_unspecified():
    assert set(PHOTO_SIDES) == {"front", "back", "unknown"}
    assert set(CAPTURE_TYPES) == {"full_view", "close_up", "oblique", "marking_detail", "unknown"}
    image = add_image_batch([], [("board.png", png_bytes())])[0][0]
    assert image["side"] == image["capture_type"] == "unknown"
    assert image["image_dimensions"] == (20, 12)
    assert image["image_format"] == "PNG"
