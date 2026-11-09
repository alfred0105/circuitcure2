import pytest
from PIL import Image, ImageDraw

from circuitcare.vision.reconstruction.anchor_matching import DesignAnchor, match_anchors
from circuitcare.vision.reconstruction.anchor_detection import ImageAnchor
from circuitcare.vision.reconstruction.coordinate_frames import (
    CalibrationPoint, CoordinateFrame, CoordinateTransform, TransformType,
    VerificationStatus,
)
from circuitcare.vision.reconstruction.registration_service import (
    RegistrationBudget, RegistrationErrorCode, RegistrationRequest,
    RegistrationServiceError, register, require_verified_transform,
)


def _auto_fixture():
    points = [(0, 0), (10, 0), (10, 10), (0, 10), (5, 3), (15, 7), (4, 12)]
    image = Image.new("RGB", (300, 220), "white")
    draw = ImageDraw.Draw(image)
    pads = []
    for index, (x, y) in enumerate(points):
        px, py = int(x * 8 + 60), int(y * 8 + 35)
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill="black")
        pads.append({"ref": f"J{index}", "pin": str(index + 1), "x": x, "y": y,
                     "type": "thru_hole", "layers": ["F.Cu", "B.Cu"],
                     "shape": "circle", "size": [1.5, 1.5]})
    geometry = {"pads": pads, "vias": []}
    return image, geometry


def test_registration_service_auto_contract_is_explicit_and_unverified():
    image, geometry = _auto_fixture()
    result = register(RegistrationRequest("fixture", "TOP", image, geometry))
    record = result.to_dict()
    assert record["schema_version"] == "circuitcare.registration.v1"
    assert record["board_id"] == "fixture"
    assert record["board_side"] == "TOP"
    assert record["registration_mode"] == "AUTO_REGISTRATION"
    assert record["registration_status"] == "AUTO_ESTIMATED_UNVERIFIED"
    assert record["transformation"]["verification_status"] == "AUTO_ESTIMATED_UNVERIFIED"
    assert record["coordinate_frame"]["units"] == {"source": "mm", "target": "px"}
    assert record["matching"]["inlier_count"] >= 4
    assert record["independent_validation"]["status"] == "BLOCKED_NO_INDEPENDENT_VALIDATION"
    assert record["provenance"]["track"] == "DESIGN_ASSISTED_REGISTRATION"
    assert record["runtime_seconds"] >= 0
    with pytest.raises(RegistrationServiceError, match="not safe"):
        require_verified_transform(result.transform)


def test_pair_product_budget_returns_failure_not_a_transform():
    image, geometry = _auto_fixture()
    budget = RegistrationBudget(max_pair_products=1)
    result = register(RegistrationRequest("fixture", "TOP", image, geometry, budget=budget))
    record = result.to_dict()
    assert record["registration_status"] == "REGISTRATION_FAILED"
    assert record["error_code"] == RegistrationErrorCode.RESOURCE_LIMIT_PAIR_PRODUCTS.value
    assert record["transformation"] is None
    assert record["resources"]["pair_product_count"] > 1


def test_image_pixel_budget_and_invalid_side_are_rejected():
    image, geometry = _auto_fixture()
    result = register(RegistrationRequest(
        "large", "TOP", image, geometry, budget=RegistrationBudget(max_image_pixels=10),
    ))
    assert result.error_code == RegistrationErrorCode.IMAGE_TOO_LARGE.value
    with pytest.raises(RegistrationServiceError) as error:
        register(RegistrationRequest("fixture", "LEFT", image, geometry))
    assert error.value.code == RegistrationErrorCode.INVALID_INPUT


def test_manual_service_verifies_only_reviewed_disjoint_held_out_points():
    def point(point_id, xy, role, evidence, reviewed=False):
        x, y = xy
        return CalibrationPoint(
            point_id, (x, y), (2 * x + 20, 3 * y + 30),
            CoordinateFrame.KICAD_DESIGN, CoordinateFrame.IMAGE_RAW,
            "TOP", evidence, role, "REVIEWED" if reviewed else "NOT_REVIEWED",
        )

    calibration = tuple(point(f"c{i}", xy, "CALIBRATION", "separately recorded calibration")
                        for i, xy in enumerate(((0, 0), (10, 0), (10, 10), (0, 10))))
    validation = tuple(point(f"v{i}", xy, "VALIDATION", "independently reviewed point", True)
                       for i, xy in enumerate(((2, 3), (7, 2), (8, 8), (1, 9))))
    result = register(RegistrationRequest(
        "manual-fixture", "TOP", Image.new("RGB", (100, 100), "white"), {},
        registration_mode="MANUAL_CALIBRATION", calibration_points=calibration,
        validation_points=validation, acceptance_threshold_px=0.1,
    ))
    assert result.registration_status == VerificationStatus.VERIFIED.value
    assert result.independent_validation["status"] == "VERIFIED"
    assert result.independent_validation["metrics"]["max"] < 1e-5
    assert require_verified_transform(result.transform) is result.transform


def test_unstable_search_budgets_are_explicit_and_deterministic():
    points = [(0, 0), (10, 0), (10, 10), (0, 10), (5, 3), (15, 7), (4, 12)]
    design = [DesignAnchor(str(i), x, y, "PAD", f"J{i}", str(i), "F.Cu",
                           "thru_hole", "circle", (1, 1), "test")
              for i, (x, y) in enumerate(points)]
    image = [ImageAnchor(str(i), "fixture", "TOP", (x * 8 + 60, y * 8 + 35),
                         30, 0.8, 1.0) for i, (x, y) in enumerate(points)]
    limited = match_anchors(design, image, board_side="TOP", image_size_px=(300, 220),
                            max_hypotheses=1)
    assert limited.status == "REGISTRATION_FAILED"
    assert limited.reason == "RESOURCE_LIMIT_HYPOTHESES"
    assert limited.transform is None

    timed_out = match_anchors(design, image, board_side="TOP", image_size_px=(300, 220),
                              timeout_seconds=1e-12)
    assert timed_out.status == "REGISTRATION_FAILED"
    assert timed_out.reason == "RESOURCE_LIMIT_TIMEOUT"
    assert timed_out.transform is None

    first = match_anchors(design, image, board_side="TOP", image_size_px=(300, 220))
    second = match_anchors(design, image, board_side="TOP", image_size_px=(300, 220))
    first_record, second_record = first.to_dict(), second.to_dict()
    first_record.pop("runtime_seconds")
    second_record.pop("runtime_seconds")
    assert first_record == second_record


def test_registration_service_serializes_invalid_held_out_data_as_unverified():
    image, geometry = _auto_fixture()
    invalid_validation = tuple(CalibrationPoint(
        f"v{i}", (i, i + 20), (i + 30, i + 30),
        CoordinateFrame.KICAD_DESIGN, CoordinateFrame.IMAGE_RAW,
        "TOP", "test point", "VALIDATION", "NOT_REVIEWED",
    ) for i in range(4))
    result = register(RegistrationRequest(
        "fixture", "TOP", image, geometry, validation_points=invalid_validation,
        acceptance_threshold_px=1.0,
    ))
    assert result.registration_status == VerificationStatus.CALIBRATED_NOT_INDEPENDENTLY_VERIFIED.value
    assert result.error_code == RegistrationErrorCode.INDEPENDENT_VALIDATION_REQUIRED.value
    assert result.transform.verification_status == VerificationStatus.CALIBRATED_NOT_INDEPENDENTLY_VERIFIED


def test_verified_transform_guard_rejects_unknown_and_failed_states():
    transform = CoordinateTransform(
        CoordinateFrame.KICAD_DESIGN, CoordinateFrame.IMAGE_RAW,
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)), TransformType.AFFINE,
        "TOP", "synthetic", ("p1", "p2", "p3"), 0, None,
        VerificationStatus.REGISTRATION_FAILED, 3, 1.0, 2.0,
    )
    with pytest.raises(RegistrationServiceError) as error:
        require_verified_transform(transform)
    assert error.value.code == RegistrationErrorCode.UNVERIFIED_TRANSFORM

    singular = CoordinateTransform(
        CoordinateFrame.KICAD_DESIGN, CoordinateFrame.IMAGE_RAW,
        ((0, 0, 0), (0, 0, 0), (0, 0, 0)), TransformType.AFFINE,
        "TOP", "synthetic", ("p1", "p2", "p3"), 0, {"v1": 0},
        VerificationStatus.VERIFIED, 3, 1.0, 2.0,
    )
    with pytest.raises(RegistrationServiceError, match="invalid frames, side, or matrix"):
        require_verified_transform(singular)
