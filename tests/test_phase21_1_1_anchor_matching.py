import numpy as np
import pytest
from PIL import Image, ImageDraw

from circuitcare.vision.reconstruction.anchor_detection import (
    ImageAnchor, detect_dark_round_anchors,
)
from circuitcare.vision.reconstruction.anchor_matching import (
    DesignAnchor, extract_design_anchors, match_anchors,
)
from circuitcare.vision.reconstruction.coordinate_frames import (
    TransformType, VerificationStatus,
)


def test_dark_round_detector_returns_unassigned_image_candidates():
    image = Image.new("RGB", (500, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((20, 20, 34, 34), fill="black")
    draw.ellipse((75, 40, 89, 54), fill="black")
    anchors, size = detect_dark_round_anchors(image, photo_id="fixture", board_side="TOP")
    assert size == (500, 300)
    assert len(anchors) == 2
    assert all(anchor.status == "DETECTED_CANDIDATE" for anchor in anchors)
    assert not any(anchor.to_dict().get("net") for anchor in anchors)


def test_design_anchor_matching_estimates_mirror_but_never_verifies_it():
    source = [(0, 0), (10, 0), (10, 10), (0, 10), (5, 3), (15, 7), (4, 12)]
    affine = np.asarray([[-8.0, 0.0], [0.0, 8.0]])
    translation = np.asarray([150.0, 30.0])
    design = [
        DesignAnchor(str(index), x, y, "PAD", f"R{index}", str(index), "F.Cu",
                     "thru_hole", "circle", (2.0, 2.0), "synthetic test design")
        for index, (x, y) in enumerate(source)
    ]
    image = []
    for index, point in enumerate(source):
        x, y = affine @ np.asarray(point) + translation
        image.append(ImageAnchor(f"image-{index}", "fixture", "BOTTOM",
                                 (float(x), float(y)), 80, 0.8, 1.0))
    image.append(ImageAnchor("false-positive", "fixture", "BOTTOM",
                             (17.0, 84.0), 30, 0.7, 1.1))

    result = match_anchors(design, image, board_side="BOTTOM",
                           image_size_px=(250, 180), match_tolerance_px=3)

    assert len(result.matches) == 7
    assert result.candidate_count >= len(result.matches)
    assert result.status == VerificationStatus.AUTO_ESTIMATED_UNVERIFIED.value
    assert result.transform is not None
    assert result.transform.transform_type == TransformType.MIRROR
    assert result.median_residual_px < 1e-4
    assert result.reason and "not independent validation" in result.reason


def test_round_anchor_matching_rejects_collinear_correspondences():
    design = [
        DesignAnchor(str(i), float(i), 0.0, "PAD", f"R{i}", str(i), "F.Cu",
                     "thru_hole", "circle", (1.0, 1.0), "synthetic")
        for i in range(4)
    ]
    image = [ImageAnchor(f"i{i}", "p", "TOP", (i * 10.0, 20.0), 20, .8, 1.0)
             for i in range(4)]
    result = match_anchors(design, image, board_side="TOP",
                           image_size_px=(100, 80), match_tolerance_px=2)
    assert result.status == "REGISTRATION_FAILED"
    assert result.transform is None


def test_design_anchor_extraction_keeps_pad_and_via_evidence_separate_by_side():
    geometry = {
        "pads": [
            {"ref": "J1", "pin": "1", "x": 2, "y": 3, "type": "thru_hole",
             "layers": ["F.Cu", "B.Cu"], "shape": "circle", "size": [1.5, 1.5]},
            {"ref": "U1", "pin": "2", "x": 4, "y": 5, "type": "smd",
             "layers": ["F.Cu"], "shape": "rect", "size": [1.0, 0.5]},
        ],
        "vias": [{"at": [8, 9], "layers": ["F.Cu", "B.Cu"], "size": 0.8}],
    }
    anchors = extract_design_anchors(geometry, board_side="TOP")
    assert [anchor.anchor_kind for anchor in anchors] == ["PAD", "VIA"]
    assert anchors[0].component_reference == "J1"
    assert anchors[0].pin_number == "1"
    assert anchors[0].dimensions_mm == (1.5, 1.5)
    assert anchors[1].source_evidence.startswith("PRISM expected_geometry.json")


def test_duplicate_design_anchor_ids_are_rejected():
    anchor = DesignAnchor("same", 0, 0, "PAD", "J1", "1", "F.Cu",
                          "thru_hole", "circle", (1, 1), "synthetic")
    image = [ImageAnchor("i1", "p", "TOP", (0, 0), 20, .8, 1.0)]
    with pytest.raises(ValueError, match="Design anchor IDs must be unique"):
        match_anchors([anchor, anchor], image, board_side="TOP", image_size_px=(100, 80))


def test_low_support_candidate_is_retained_as_diagnostic_but_transform_is_rejected():
    points = [(0, 0), (10, 0), (2, 7)]
    design = [DesignAnchor(str(i), x, y, "PAD", f"J{i}", str(i), "F.Cu",
                           "thru_hole", "circle", (1, 1), "synthetic")
              for i, (x, y) in enumerate(points)]
    image = [ImageAnchor(str(i), "fixture", "TOP", (x * 10 + 30, y * 10 + 20),
                         20, .8, 1.0) for i, (x, y) in enumerate(points)]
    result = match_anchors(design, image, board_side="TOP",
                           image_size_px=(300, 200), match_tolerance_px=2)
    assert result.candidate_count == 3
    assert result.review_flags
    assert result.status == "REGISTRATION_FAILED"
    assert result.transform is None
