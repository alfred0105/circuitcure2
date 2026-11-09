from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from circuitcare.vision.board_detector import detect_board_candidates, validate_corners
from circuitcare.vision.image_loader import ImageLoadError, load_board_image
from circuitcare.vision.overlay import render_risk_overlay, risk_by_reference
from circuitcare.vision.pcb_positions import match_netlist_references, parse_position_csv
from circuitcare.vision.perspective import rectify_board
from circuitcare.vision.registration import fit_registration, map_component_positions
from circuitcare.vision.models import ComponentPosition


def _image_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    stream = io.BytesIO()
    image.save(stream, format=fmt)
    return stream.getvalue()


def test_loader_applies_exif_orientation_and_preserves_scaling() -> None:
    image = Image.new("RGB", (100, 50), "green")
    exif = image.getexif()
    exif[274] = 6
    stream = io.BytesIO()
    image.save(stream, format="JPEG", exif=exif)

    loaded = load_board_image(stream.getvalue(), max_edge=25)

    assert loaded.format == "JPEG"
    assert loaded.original_size == (50, 100)
    assert loaded.processed_size == (12, 25)
    assert loaded.image.mode == "RGB"
    assert loaded.scale_xy[0] == pytest.approx(12 / 50)


def test_loader_rejects_non_image_and_pixel_limit() -> None:
    with pytest.raises(ImageLoadError):
        load_board_image(b"not an image")
    with pytest.raises(ImageLoadError, match="픽셀"):
        load_board_image(_image_bytes(Image.new("RGB", (20, 20))), max_pixels=100)


def test_board_detector_finds_synthetic_quad_and_manual_corners_are_checked() -> None:
    image = Image.new("RGB", (320, 240), "white")
    from PIL import ImageDraw
    ImageDraw.Draw(image).polygon([(40, 30), (280, 45), (265, 205), (55, 210)], fill=(30, 100, 70))

    result = detect_board_candidates(image)

    assert result["board_detected"] is True
    assert result["status"] == "candidate_unverified"
    assert len(result["candidates"][0]["corners_px"]) == 4
    with pytest.raises(ValueError, match="중복"):
        validate_corners([(0, 0), (0, 0), (20, 20), (0, 20)], (320, 240))
    with pytest.raises(ValueError, match="범위"):
        validate_corners([(-1, 0), (300, 0), (300, 200), (0, 200)], (320, 240))


def test_perspective_rectification_preserves_known_physical_aspect() -> None:
    image = Image.new("RGB", (300, 240), "white")
    result = rectify_board(image, [(30, 20), (250, 35), (240, 210), (40, 215)],
                          physical_size_mm=(80, 40), max_output_edge=800)

    assert result["image"].size == (800, 400)
    assert result["physical_size_mm"] == [80, 40]
    assert np.asarray(result["transform_matrix"]).shape == (3, 3)


def test_kicad_position_csv_mm_inch_side_and_reference_matching() -> None:
    text = 'Ref,Val,Package,PosX,PosY,Rot,Side\n"C1","10u","Capacitor_SMD:C_0805",25.4,12.7,90,top\n"R1","1k","Resistor_SMD:R_0805",1,2,0,bottom\n'
    rows, meta = parse_position_csv(text, units="in", board_side="both", origin_description="absolute")

    assert rows[0].x_mm == pytest.approx(645.16)
    assert rows[0].y_mm == pytest.approx(322.58)
    assert rows[0].side == "front"
    assert rows[1].side == "back"
    assert meta["coordinates_normalized_to"] == "mm"
    assert match_netlist_references(rows, ["C1", "U1"]) == {
        "matched": ["C1"], "in_netlist_without_position": ["U1"],
        "position_without_netlist_component": ["R1"],
    }


def test_kicad_position_csv_rejects_duplicate_bad_unit_and_unknown_side() -> None:
    duplicate = "Ref,Val,Package,PosX,PosY,Rot,Side\nC1,a,b,1,2,0,front\nC1,c,d,3,4,0,front\n"
    with pytest.raises(ValueError, match="중복"):
        parse_position_csv(duplicate, units="mm", board_side="front")
    with pytest.raises(ValueError, match="단위"):
        parse_position_csv(duplicate, units="auto", board_side="front")
    bad_side = "Ref,Val,Package,PosX,PosY,Rot,Side\nC1,a,b,1,2,0,inner\n"
    with pytest.raises(ValueError, match="면"):
        parse_position_csv(bad_side, units="mm", board_side="both")


def test_known_homography_maps_position_and_requires_user_confirmation() -> None:
    correspondences = [
        {"reference": "U1", "x_mm": 0, "y_mm": 0, "u_px": 100, "v_px": 50},
        {"reference": "C1", "x_mm": 100, "y_mm": 0, "u_px": 500, "v_px": 50},
        {"reference": "R1", "x_mm": 100, "y_mm": 50, "u_px": 500, "v_px": 250},
        {"reference": "R2", "x_mm": 0, "y_mm": 50, "u_px": 100, "v_px": 250},
    ]
    result = fit_registration(correspondences, board_side="front")
    position = ComponentPosition("C3", "1u", "C_0805", 50, 25, 0, "front")

    assert result.status == "pending_verification"
    assert result.reprojection_error_px == pytest.approx(0, abs=1e-7)
    with pytest.raises(ValueError, match="확인"):
        map_component_positions([position], result.homography, board_side="front", image_size=(600, 300), confirmed=False)
    mapped = map_component_positions([position], result.homography, board_side="front", image_size=(600, 300), confirmed=True)
    assert mapped["components"][0]["image_position_px"] == pytest.approx([300, 150])
    assert mapped["components"][0]["position_status"] == "registered_user_confirmed"


def test_registration_rejects_collinear_and_duplicate_points() -> None:
    points = [{"reference": f"R{i}", "x_mm": i * 10, "y_mm": 0, "u_px": i * 20, "v_px": 4} for i in range(4)]
    with pytest.raises(ValueError, match="직선"):
        fit_registration(points, board_side="front")
    spread = [
        {"reference": "A", "x_mm": 0, "y_mm": 0, "u_px": 0, "v_px": 0},
        {"reference": "A", "x_mm": 10, "y_mm": 0, "u_px": 10, "v_px": 0},
        {"reference": "C", "x_mm": 10, "y_mm": 10, "u_px": 10, "v_px": 10},
        {"reference": "D", "x_mm": 0, "y_mm": 10, "u_px": 0, "v_px": 10},
    ]
    with pytest.raises(ValueError, match="중복"):
        fit_registration(spread, board_side="front")


def test_independent_validation_point_error_is_reported() -> None:
    points = [
        {"reference": "A", "x_mm": 0, "y_mm": 0, "u_px": 100, "v_px": 50},
        {"reference": "B", "x_mm": 100, "y_mm": 0, "u_px": 500, "v_px": 50},
        {"reference": "C", "x_mm": 100, "y_mm": 50, "u_px": 500, "v_px": 250},
        {"reference": "D", "x_mm": 0, "y_mm": 50, "u_px": 100, "v_px": 250},
    ]
    result = fit_registration(points, board_side="front", validation_points=[
        {"reference": "E", "x_mm": 50, "y_mm": 25, "u_px": 310, "v_px": 155},
    ])
    assert result.validation_points[0]["error_px"] == pytest.approx(np.sqrt(125))


def test_risk_overlay_uses_only_matching_successful_scenario() -> None:
    assessment = {
        "simulation_status": "success", "failed_component": "U1", "fault_type": "regulator_output_rise",
        "secondary_candidates": [{"component_ref": "C1", "risk_level": "EXCEEDED", "priority_rank": 1,
                                   "priority_score": 98.93, "metrics": {"voltage": {"measured_stress": 8}}}],
    }
    risks, issue = risk_by_reference(assessment, expected_fault="U1", expected_fault_type="regulator_output_rise")
    assert issue is None
    assert risks["C1"]["risk_level"] == "EXCEEDED"
    assert risks["U1"]["risk_level"] == "KNOWN_FAILED"
    assert risk_by_reference(assessment, expected_fault="R1", expected_fault_type="resistance_drift")[0] == {}

    image = Image.new("RGB", (100, 100), "white")
    locations = {"components": [{"reference": "C1", "image_position_px": [50, 50]}]}
    overlay, meta = render_risk_overlay(image, locations, assessment, expected_fault="U1",
                                        expected_fault_type="regulator_output_rise", selected_reference="C1")
    assert overlay.getpixel((50, 50)) == (220, 38, 38)
    assert meta["matched"][0]["risk"]["priority_rank"] == 1


def test_synthetic_data_is_explicitly_not_real_pcb_verification() -> None:
    image = Image.new("RGB", (10, 10), "green")
    buffer = _image_bytes(image)
    loaded = load_board_image(buffer)
    manifest = {"verification_level": "Synthetic Fixture Verified", "image_sha256": loaded.sha256,
                "real_pcb_verified": False, "source_data": "generated in test"}
    assert json.loads(json.dumps(manifest))["real_pcb_verified"] is False
