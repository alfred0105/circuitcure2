from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from circuitcare.vision.image_loader import ImageLoadError, load_board_image
from circuitcare.vision.reconstruction.copper_segmentation import (
    detect_pad_via_candidates, segment_visible_copper, vectorize_copper,
)
from circuitcare.vision.reconstruction.models import EvidenceStatus, PadViaCandidate
from circuitcare.vision.reconstruction.multiview_registration import (
    back_to_front_transform, register_manual_correspondences, register_same_side,
)
from circuitcare.vision.reconstruction.photo_collection import collect_photos
from circuitcare.vision.reconstruction.quality import inspect_photo_quality
from circuitcare.vision.reconstruction.topology import build_observed_topology


class Upload:
    def __init__(self, name: str, raw: bytes):
        self.name = name
        self._raw = raw

    def getvalue(self) -> bytes:
        return self._raw


def _image_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    stream = io.BytesIO()
    image.save(stream, format=fmt)
    return stream.getvalue()


def _textured_board() -> np.ndarray:
    rng = np.random.default_rng(2026)
    image = np.full((360, 480, 3), (28, 87, 63), dtype=np.uint8)
    for _ in range(180):
        x, y = rng.integers(20, 460), rng.integers(20, 340)
        color = (int(rng.integers(80, 240)), int(rng.integers(80, 230)), int(rng.integers(70, 220)))
        cv2.circle(image, (int(x), int(y)), int(rng.integers(2, 8)), color, -1)
    for _ in range(45):
        x1, y1 = rng.integers(20, 460), rng.integers(20, 340)
        x2, y2 = rng.integers(20, 460), rng.integers(20, 340)
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (180, 185, 178), 2)
    return image


def test_multiphoto_collection_hashes_deduplicates_and_tracks_roles() -> None:
    image = Image.new("RGB", (120, 80), "green")
    raw = _image_bytes(image)
    collected = collect_photos(
        [Upload("front.png", raw), Upload("copy.png", raw)],
        side_by_name={"0:front.png": "front", "1:copy.png": "front"},
        capture_by_name={"0:front.png": "full_view", "1:copy.png": "close_up"})
    assert collected[0][0].file_sha256 == collected[1][0].file_sha256
    assert collected[1][0].duplicate_of == collected[0][0].photo_id
    assert collected[0][0].side == "front"
    assert collected[0][2] == raw


def test_webp_is_decoded_and_unsupported_format_is_rejected() -> None:
    image = Image.new("RGB", (48, 32), "green")
    assert load_board_image(_image_bytes(image, "WEBP")).format == "WEBP"
    with pytest.raises(ImageLoadError):
        load_board_image(b"<svg></svg>")


def test_quality_returns_multiple_signals_without_rejecting_blurry_photo() -> None:
    image = Image.new("RGB", (640, 480), "white")
    quality = inspect_photo_quality(image)
    assert "POSSIBLE_BLUR" in quality["flags"]
    assert "OVEREXPOSED_AREAS" in quality["flags"]
    assert quality["status"] == "QUALITY_HINTS_ONLY"


def test_same_side_orb_ransac_registers_view_and_blank_image_fails_closed() -> None:
    reference = _textured_board()
    matrix = np.float32([[1, 0, 7], [0, 1, -5]])
    moving = cv2.warpAffine(reference, matrix, (reference.shape[1], reference.shape[0]))
    result = register_same_side(reference, moving, source_photo_id="moving",
                                reference_photo_id="reference", board_side="front",
                                min_inliers=8)
    assert result.transformation is not None
    assert result.registration_status == "REGISTERED_CANDIDATE"
    assert result.evidence_status == EvidenceStatus.INFERRED
    failed = register_same_side(np.zeros_like(reference), np.zeros_like(reference),
                                source_photo_id="blank", reference_photo_id="blank-ref",
                                board_side="back")
    assert failed.registration_status == "REGISTRATION_UNVERIFIED"
    assert failed.transformation is None


def test_back_front_transform_requires_confirmation_state_and_keeps_center() -> None:
    result = back_to_front_transform(rotation_degrees=180, mirror_x=True,
                                     board_size_px=(400, 300))
    matrix = np.asarray(result["matrix"])
    center = matrix @ np.array([199.5, 149.5, 1.0])
    assert center[:2] == pytest.approx([199.5, 149.5])
    assert (matrix @ np.array([0.0, 0.0, 1.0]))[:2] == pytest.approx([0, 299])
    assert result["status"] == "USER_CONFIRMED_ORIENTATION_REQUIRED"
    with pytest.raises(ValueError):
        back_to_front_transform(rotation_degrees=45, mirror_x=False)


def test_manual_correspondence_fallback_returns_candidate_never_verified() -> None:
    reference = [(20, 20), (420, 20), (420, 320), (20, 320)]
    moving = [(30, 15), (430, 15), (430, 315), (30, 315)]
    result = register_manual_correspondences(reference, moving, source_photo_id="m",
                                             reference_photo_id="r", board_side="front")
    assert result.registration_status == "REGISTERED_CANDIDATE"
    assert result.method == "USER_MATCHED_FOUR_POINT_HOMOGRAPHY"
    assert result.evidence_status == EvidenceStatus.INFERRED
    with pytest.raises(ValueError, match="넓게"):
        register_manual_correspondences([(0, 1), (1, 1), (2, 1), (3, 1)],
                                        [(0, 1), (1, 1), (2, 1), (3, 1)],
                                        source_photo_id="m", reference_photo_id="r", board_side="back")


def test_copper_candidate_mask_vectorizes_paths_and_does_not_claim_net() -> None:
    image = np.full((100, 140, 3), (30, 95, 64), dtype=np.uint8)
    cv2.line(image, (20, 50), (105, 50), (220, 220, 215), 5)
    cv2.line(image, (65, 50), (65, 78), (220, 220, 215), 5)
    segmented = segment_visible_copper(image, board_side="front")
    assert segmented.mask.sum() > 100
    assert segmented.status == "INFERRED_COPPER_LIKE_CANDIDATES"
    traces = vectorize_copper(segmented.mask, photo_id="p1", board_side="front")
    assert traces
    assert traces[0].evidence_status == EvidenceStatus.INFERRED
    assert traces[0].uncertainty.startswith("EXPOSED_COPPER_UNVERIFIED")
    assert traces[0].polylines_px
    assert all(len(point) == 2 for line in traces[0].polylines_px for point in line)


def test_synthetic_trace_segmentation_metrics_are_explicitly_fixture_only() -> None:
    import importlib.util
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures/reconstruction_synthetic.py"
    spec = importlib.util.spec_from_file_location("reconstruction_synthetic_fixture", fixture)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    images = module.synthetic_front_back(include_ground_truth=True)
    metrics = []
    for side, image, truth in zip(("front", "back"), (images[0], images[1]), (images[2], images[3])):
        predicted = segment_visible_copper(Image.fromarray(image), board_side=side).mask
        tp = int(np.count_nonzero(predicted & truth))
        fp = int(np.count_nonzero(predicted & ~truth))
        fn = int(np.count_nonzero(~predicted & truth))
        metrics.append({"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
                        "iou": tp / max(1, tp + fp + fn)})
    assert metrics[0]["iou"] > 0.6
    assert metrics[1]["iou"] > 0.6
    # The test fixture is algorithm smoke evidence, not a real-PCB accuracy claim.
    assert metrics[0]["recall"] > 0.9 and metrics[1]["recall"] > 0.9


def test_pad_via_candidates_and_topology_are_same_side_inferred_only() -> None:
    image = np.full((120, 160, 3), (30, 95, 64), dtype=np.uint8)
    cv2.circle(image, (50, 60), 9, (215, 215, 210), -1)
    cv2.line(image, (58, 60), (120, 60), (215, 215, 210), 5)
    segmented = segment_visible_copper(image, board_side="front")
    candidate = PadViaCandidate("pad-front", "photo-front", "front", (50, 60), 9,
                                "CIRCULAR", "PAD_OR_VIA_CANDIDATE", 0.2)
    topology = build_observed_topology(segmented.mask, photo_id="photo-front", board_side="front",
                                       pad_candidates=[candidate])
    assert topology.status == "OBSERVED_TOPOLOGY_ONLY"
    assert topology.edges
    assert topology.edges[0].same_side_contact
    assert topology.edges[0].verification_status == EvidenceStatus.INFERRED
    assert candidate.through_plated is None
    detected = detect_pad_via_candidates(image, photo_id="photo-front", board_side="front",
                                          copper_mask=segmented.mask, min_radius_px=3)
    assert all(item.through_plated is None for item in detected)
    other_side = build_observed_topology(segmented.mask, photo_id="photo-front", board_side="back",
                                         pad_candidates=[candidate])
    assert not other_side.edges
