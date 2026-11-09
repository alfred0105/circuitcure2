import numpy as np
import pytest

from circuitcare.vision.reconstruction.coordinate_frames import (
    CalibrationPoint, CoordinateFrame, TransformType, VerificationStatus, fit_transform,
)
from circuitcare.vision.reconstruction.dataset_bundle import build_prism_bundle
from circuitcare.vision.reconstruction.ground_truth import (
    project_reference_geometry, rasterize_projected_geometry,
)


def _point(point_id, xy, *, role="CALIBRATION", side="TOP"):
    x, y = xy
    return CalibrationPoint(
        point_id, (x, y), (2 * x + 10, 3 * y + 20),
        CoordinateFrame.KICAD_DESIGN, CoordinateFrame.IMAGE_RAW,
        side, "synthetic test correspondence", role,
    )


def test_homography_status_and_inverse_consistency():
    points = [_point("a", (0, 0)), _point("b", (10, 0)),
              _point("c", (10, 10)), _point("d", (0, 10))]
    transform, report = fit_transform(points, board_side="TOP",
                                       source_frame=CoordinateFrame.KICAD_DESIGN,
                                       target_frame=CoordinateFrame.IMAGE_RAW)
    assert transform.verification_status == VerificationStatus.CALIBRATED_NOT_INDEPENDENTLY_VERIFIED
    assert report["validation_point_count"] == 0
    mapped = transform.apply([(2, 3)], source_frame=CoordinateFrame.KICAD_DESIGN,
                             target_frame=CoordinateFrame.IMAGE_RAW, board_side="TOP")
    np.testing.assert_allclose(mapped[0], (14, 29), atol=1e-5)
    inverse = transform.inverse()
    restored = inverse.apply(mapped, source_frame=CoordinateFrame.IMAGE_RAW,
                             target_frame=CoordinateFrame.KICAD_DESIGN, board_side="TOP")
    np.testing.assert_allclose(restored[0], (2, 3), atol=1e-5)
    assert report["transform_invertible"]


@pytest.mark.parametrize(
    "name, map_xy, kind, points",
    [
        ("identity", lambda x, y: (x, y), TransformType.HOMOGRAPHY,
         [(0, 0), (10, 0), (10, 10), (0, 10)]),
        ("translation", lambda x, y: (x + 4, y - 7), TransformType.HOMOGRAPHY,
         [(0, 0), (10, 0), (10, 10), (0, 10)]),
        ("rotation", lambda x, y: (-y, x), TransformType.AFFINE,
         [(0, 0), (10, 0), (0, 10)]),
        ("scale", lambda x, y: (2 * x, 3 * y), TransformType.AFFINE,
         [(0, 0), (10, 0), (0, 10)]),
        ("mirror", lambda x, y: (20 - x, y), TransformType.MIRROR,
         [(0, 0), (10, 0), (10, 10), (0, 10)]),
    ],
)
def test_explicit_transform_families_are_representable(name, map_xy, kind, points):
    correspondence = [
        CalibrationPoint(str(index), source, map_xy(*source),
                         CoordinateFrame.KICAD_DESIGN, CoordinateFrame.IMAGE_RAW,
                         "BOTTOM" if name == "mirror" else "TOP", "explicit synthetic transform")
        for index, source in enumerate(points)
    ]
    transform, _ = fit_transform(
        correspondence, board_side="BOTTOM" if name == "mirror" else "TOP",
        source_frame=CoordinateFrame.KICAD_DESIGN, target_frame=CoordinateFrame.IMAGE_RAW,
        transform_type=kind,
    )
    result = transform.apply([(3, 4)], source_frame=CoordinateFrame.KICAD_DESIGN,
                             target_frame=CoordinateFrame.IMAGE_RAW,
                             board_side="BOTTOM" if name == "mirror" else "TOP")
    np.testing.assert_allclose(result[0], map_xy(3, 4), atol=1e-5)
    assert transform.transform_type == kind


def test_independent_validation_required_for_verified_state():
    calibration = [_point("a", (0, 0)), _point("b", (10, 0)),
                   _point("c", (10, 10)), _point("d", (0, 10))]
    validation = [CalibrationPoint(
        point_id="heldout-1", source_xy=(5, 5), target_xy=(20, 35),
        source_frame=CoordinateFrame.KICAD_DESIGN, target_frame=CoordinateFrame.IMAGE_RAW,
        board_side="TOP", evidence="separate synthetic held-out point",
        role="VALIDATION", review_status="REVIEWED",
    )]
    transform, report = fit_transform(
        calibration, board_side="TOP", source_frame=CoordinateFrame.KICAD_DESIGN,
        target_frame=CoordinateFrame.IMAGE_RAW, validation_points=validation,
        acceptance_threshold_px=0.1,
    )
    assert transform.verification_status == VerificationStatus.VERIFIED
    assert report["validation_point_count"] == 1
    assert report["validation_independently_reviewed"]
    assert report["held_out_error_target_units"]["p95"] == pytest.approx(0, abs=1e-8)


def test_unreviewed_validation_cannot_mark_transform_verified():
    calibration = [_point("a", (0, 0)), _point("b", (10, 0)),
                   _point("c", (10, 10)), _point("d", (0, 10))]
    validation = [CalibrationPoint(
        "heldout-1", (5, 5), (20, 35), CoordinateFrame.KICAD_DESIGN,
        CoordinateFrame.IMAGE_RAW, "TOP", "not independently adjudicated", "VALIDATION",
    )]
    transform, report = fit_transform(
        calibration, board_side="TOP", source_frame=CoordinateFrame.KICAD_DESIGN,
        target_frame=CoordinateFrame.IMAGE_RAW, validation_points=validation,
        acceptance_threshold_px=0.1,
    )
    assert transform.verification_status == VerificationStatus.CALIBRATED_NOT_INDEPENDENTLY_VERIFIED
    assert not report["validation_independently_reviewed"]


def test_same_physical_source_location_cannot_cross_calibration_validation_split():
    calibration = [_point("a", (0, 0)), _point("b", (10, 0)),
                   _point("c", (10, 10)), _point("d", (0, 10))]
    validation = [CalibrationPoint(
        "different-id-same-pad", (0, 0), (10, 20), CoordinateFrame.KICAD_DESIGN,
        CoordinateFrame.IMAGE_RAW, "TOP", "duplicate physical point", "VALIDATION", "REVIEWED",
    )]
    with pytest.raises(ValueError, match="same physical source location"):
        fit_transform(
            calibration, board_side="TOP", source_frame=CoordinateFrame.KICAD_DESIGN,
            target_frame=CoordinateFrame.IMAGE_RAW, validation_points=validation,
        )


def test_frame_side_and_split_errors_are_rejected():
    points = [_point("a", (0, 0)), _point("b", (10, 0)),
              _point("c", (10, 10)), _point("d", (0, 10))]
    transform, _ = fit_transform(points, board_side="TOP",
                                  source_frame=CoordinateFrame.KICAD_DESIGN,
                                  target_frame=CoordinateFrame.IMAGE_RAW)
    with pytest.raises(ValueError, match="frame mismatch"):
        transform.apply([(1, 1)], source_frame=CoordinateFrame.IMAGE_RAW,
                        target_frame=CoordinateFrame.KICAD_DESIGN, board_side="TOP")
    with pytest.raises(ValueError, match="side mismatch"):
        transform.apply([(1, 1)], source_frame=CoordinateFrame.KICAD_DESIGN,
                        target_frame=CoordinateFrame.IMAGE_RAW, board_side="BOTTOM")
    with pytest.raises(ValueError, match="disjoint"):
        fit_transform(points, board_side="TOP", source_frame=CoordinateFrame.KICAD_DESIGN,
                      target_frame=CoordinateFrame.IMAGE_RAW,
                      validation_points=[_point("a", (2, 2), role="VALIDATION")])


def test_degenerate_points_and_insufficient_correspondences_fail():
    line = [_point(str(i), (i, 0)) for i in range(4)]
    with pytest.raises(ValueError, match="nearly collinear"):
        fit_transform(line, board_side="TOP", source_frame=CoordinateFrame.KICAD_DESIGN,
                      target_frame=CoordinateFrame.IMAGE_RAW)
    with pytest.raises(ValueError, match="at least 4"):
        fit_transform(line[:3], board_side="TOP", source_frame=CoordinateFrame.KICAD_DESIGN,
                      target_frame=CoordinateFrame.IMAGE_RAW)


def test_reference_projection_requires_verified_registration_and_marks_visibility_unknown():
    points = [_point("a", (0, 0)), _point("b", (10, 0)),
              _point("c", (10, 10)), _point("d", (0, 10))]
    unverified, _ = fit_transform(points, board_side="TOP",
                                  source_frame=CoordinateFrame.KICAD_DESIGN,
                                  target_frame=CoordinateFrame.IMAGE_RAW)
    ref = {"pads": [{"x": 1, "y": 1, "net": "N"}],
           "segments": [{"start": [0, 0], "end": [10, 10], "net": "N"}]}
    with pytest.raises(ValueError, match="independently verified"):
        project_reference_geometry(ref, unverified, board_side="TOP", image_size_px=(100, 100))
    auto_estimated = unverified.__class__(
        **{**unverified.__dict__,
           "verification_status": VerificationStatus.AUTO_ESTIMATED_UNVERIFIED}
    )
    with pytest.raises(ValueError, match="independently verified"):
        project_reference_geometry(ref, auto_estimated, board_side="TOP", image_size_px=(100, 100))
    validation = [CalibrationPoint(
        "v1", (5, 5), (20, 35), CoordinateFrame.KICAD_DESIGN,
        CoordinateFrame.IMAGE_RAW, "TOP", "separate reviewed synthetic point",
        "VALIDATION", "REVIEWED",
    )]
    verified, _ = fit_transform(points, board_side="TOP",
                                 source_frame=CoordinateFrame.KICAD_DESIGN,
                                 target_frame=CoordinateFrame.IMAGE_RAW,
                                 validation_points=validation, acceptance_threshold_px=0.1)
    projected = project_reference_geometry(ref, verified, board_side="TOP", image_size_px=(100, 100))
    assert projected["reference_geometry"]["pads"]["items"][0]["visual_observability"] == "UNKNOWN_OCCLUSION_NOT_MODELED"
    masks = rasterize_projected_geometry(projected)
    assert masks["segments"].shape == (100, 100)
    assert masks["segments"].max() == 255


def test_prism_bundle_records_absence_and_license_uncertainty():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "data/external/prism"
    if not root.exists():
        pytest.skip("Pinned external PRISM dataset is not installed")
    bundle = build_prism_bundle(root, "Simple")
    data = bundle.to_dict()
    assert data["source_commit"] == "9189c87bbec6995b745b809db99cef6d3269401f"
    assert data["calibration"]["status"] == "REGISTRATION_UNVERIFIED"
    assert data["design"]["position_csv"]["status"] == "NOT_AVAILABLE"
    assert data["license_metadata"]["redistribution_status"] == "REVIEW_REQUIRED"
