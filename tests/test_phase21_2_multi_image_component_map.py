import base64
import io

import cv2
import numpy as np
import pytest
from PIL import Image
from PIL import ImageFilter

from circuitcare.vision.reconstruction.multi_image_component_map import (
    CaptureRole, ImageBudget, MultiImageComponentMapService, MultiImageError,
    PhotoInput, _detection_to_processed_frame, _fuse_observations, _native_rectified_image,
    save_identity_manifest, to_json,
)


def _textured_scene():
    rng = np.random.default_rng(9142)
    image = rng.integers(0, 256, (400, 520, 3), dtype=np.uint8)
    image = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(1.2))).copy()
    for x in range(20, 520, 32):
        for y in range(20, 400, 32):
            color = (255, 255, 255) if (x + y) % 2 else (0, 0, 0)
            cv2.circle(image, (x, y), 5, color, 2)
    return image


def _png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def test_intake_hashes_orients_limits_and_keeps_sides_separate():
    service = MultiImageComponentMapService()
    image = np.full((80, 120, 3), 120, dtype=np.uint8)
    session = service.ingest("board-1", [
        PhotoInput(_png(image), "TOP", CaptureRole.OVERVIEW, "top.png"),
        PhotoInput(_png(image), "BOTTOM", CaptureRole.CLOSEUP, "bottom.png"),
        PhotoInput(_png(image), "TOP", CaptureRole.ANGLED, "top-copy.png"),
    ])
    assert session.photos[0].sha256 == session.photos[2].sha256
    assert session.photos[0].photo_id != session.photos[1].photo_id
    assert session.photos[2].duplicate_of == session.photos[0].photo_id
    reordered = service.ingest("board-1", [
        PhotoInput(_png(image), "TOP", CaptureRole.ANGLED, "top-copy.png"),
        PhotoInput(_png(image), "BOTTOM", CaptureRole.CLOSEUP, "bottom.png"),
        PhotoInput(_png(image), "TOP", CaptureRole.OVERVIEW, "top.png"),
    ])
    reordered_by_name = {photo.name: photo.photo_id for photo in reordered.photos}
    original_by_name = {photo.name: photo.photo_id for photo in session.photos}
    assert reordered_by_name == original_by_name
    result = service.build_map(session)
    assert set(result["sides"]) == {"TOP", "BOTTOM"}
    assert result["design_registration"]["status"] == "NOT_RUN_SEPARATE_FROM_PHOTO_REGISTRATION"
    assert "rgb_pixels" not in result["sides"]["TOP"]["canonical_image"]
    assert to_json(result).startswith('{"board_id":"board-1"')

    limited = MultiImageComponentMapService(budget=ImageBudget(max_images=1))
    try:
        limited.ingest("board", [PhotoInput(_png(image), "TOP"), PhotoInput(_png(image), "TOP")])
    except MultiImageError as exc:
        assert exc.code == "IMAGE_COUNT_LIMIT"
    else:
        raise AssertionError("image count limit was not enforced")


def test_native_tiled_frame_restores_detection_to_processed_coordinates():
    source = np.full((160, 200, 3), 145, dtype=np.uint8)
    service = MultiImageComponentMapService(budget=ImageBudget(max_decoded_edge=80))
    session = service.ingest("native-tile", [PhotoInput(_png(source), "TOP")])
    photo = session.photos[0]
    native, native_to_processed = _native_rectified_image(photo)
    assert native.size == (200, 160)
    assert photo.image.size == (80, 64)
    mapped = _detection_to_processed_frame({"bbox_xyxy": [50, 30, 100, 70]}, native_to_processed)
    assert mapped["bbox_xyxy"] == pytest.approx([20, 12, 40, 28])
    assert mapped["source_polygon_xy"][0] == pytest.approx([20, 12])


def test_invalid_detections_are_rejected_without_losing_valid_candidates():
    image = _textured_scene()
    service = MultiImageComponentMapService()
    session = service.ingest("invalid-detection-board", [PhotoInput(_png(image), "TOP")])
    valid = [{"detection_id": f"ok-{index}", "class_name": "resistor", "confidence": 0.9,
              "bbox_xyxy": [20 + index * 45, 30, 35 + index * 45, 45]} for index in range(9)]
    invalid = [
        {"detection_id": "missing-box", "class_name": "resistor", "confidence": 0.8},
        {"detection_id": "short-box", "class_name": "resistor", "confidence": 0.8, "bbox_xyxy": [1, 2, 3]},
        {"detection_id": "reversed", "class_name": "resistor", "confidence": 0.8, "bbox_xyxy": [8, 8, 2, 9]},
        {"detection_id": "outside", "class_name": "resistor", "confidence": 0.8, "bbox_xyxy": [-1, 1, 5, 5]},
        {"detection_id": "nan", "class_name": "resistor", "confidence": 0.8, "bbox_xyxy": [1, 1, float("nan"), 5]},
        {"detection_id": "bad-confidence", "class_name": "resistor", "confidence": "high", "bbox_xyxy": [1, 1, 5, 5]},
        {"detection_id": "bad-class", "class_name": "transformer", "confidence": 0.8, "bbox_xyxy": [1, 1, 5, 5]},
        {"detection_id": "bad-polygon", "class_name": "resistor", "confidence": 0.8,
         "bbox_xyxy": [1, 1, 5, 5], "polygon_xy": [[1, 1], [2, float("inf")], [4, 4]]},
        {"detection_id": "missing-class", "confidence": 0.8, "bbox_xyxy": [1, 1, 5, 5]},
        None,
    ]
    result = service.build_map(session, detector=lambda _: {
        "status": "INFERENCE_COMPLETE", "detections": valid + invalid})
    side = result["sides"]["TOP"]
    assert len(side["components"]) == 9
    assert len(side["component_observations"]) == 9
    assert len(side["detection_rejections"]) == 10
    assert {row["error_code"] for row in side["detection_rejections"]} == {
        "INVALID_DETECTION_BOX", "INVALID_CONFIDENCE", "INVALID_DETECTION_CLASS",
        "INVALID_DETECTION_POLYGON", "MALFORMED_DETECTOR_RESPONSE"}
    assert result["detection_rejections"] == side["detection_rejections"]


@pytest.mark.parametrize("response", [None, {"status": "INFERENCE_COMPLETE", "detections": {"bad": "shape"}}])
def test_malformed_detector_response_produces_empty_map_and_structured_error(response):
    image = _textured_scene()
    service = MultiImageComponentMapService()
    session = service.ingest("malformed-response", [PhotoInput(_png(image), "TOP")])
    result = service.build_map(session, detector=lambda _: response)
    assert result["sides"]["TOP"]["components"] == []
    assert result["detection_rejections"][0]["error_code"] == "MALFORMED_DETECTOR_RESPONSE"


def test_missing_confidence_is_retained_as_low_confidence_and_duplicate_ids_are_diagnosed():
    image = _textured_scene()
    service = MultiImageComponentMapService()
    session = service.ingest("detection-confidence", [PhotoInput(_png(image), "TOP")])
    result = service.build_map(session, detector=lambda _: {"detections": [
        {"detection_id": "same", "class_name": "resistor", "bbox_xyxy": [50, 50, 70, 70]},
        {"detection_id": "same", "class_name": "resistor", "confidence": 0.8,
         "bbox_xyxy": [100, 50, 120, 70]},
    ]})
    side = result["sides"]["TOP"]
    assert len(side["components"]) == 2
    assert side["component_observations"][0]["recognition_status"] == "LOW_CONFIDENCE"
    assert side["detection_warnings"][0]["error_code"] == "DUPLICATE_DETECTION_ID"


def test_exif_orientation_is_applied_and_corrupt_images_fail_closed():
    image = Image.new("RGB", (120, 80), "navy")
    exif = Image.Exif()
    exif[274] = 6
    stream = io.BytesIO()
    image.save(stream, format="JPEG", exif=exif)
    service = MultiImageComponentMapService()
    session = service.ingest("exif-board", [PhotoInput(stream.getvalue(), "TOP")])
    assert session.photos[0].original_size_px == (80, 120)
    assert session.photos[0].processed_size_px == (80, 120)
    assert session.photos[0].coordinate_transforms["exif_orientation"] == 6
    assert np.asarray(session.photos[0].coordinate_transforms["encoded_to_exif_corrected"]).shape == (3, 3)

    result = service.build_map(session, detector=lambda _: {"status": "INFERENCE_COMPLETE", "detections": [{
        "detection_id": "rotated-u1", "class_name": "integrated_circuit", "confidence": 0.9,
        "bbox_xyxy": [10, 20, 30, 40],
    }]})
    photo_record = result["photos"][0]
    transforms = photo_record["coordinate_transforms"]
    encoded_to_canonical = np.asarray(transforms["encoded_to_canonical"])
    assert np.allclose(encoded_to_canonical, np.asarray(transforms["processed_to_canonical"])
                       @ np.asarray(transforms["decoded_to_rectified"])
                       @ np.asarray(transforms["exif_corrected_to_decoded"])
                       @ np.asarray(transforms["encoded_to_exif_corrected"]))
    observation = result["sides"]["TOP"]["component_observations"][0]
    assert np.allclose(observation["encoded_bbox_xyxy"], [20, 49, 40, 69], atol=1)
    assert np.allclose(observation["original_bbox_xyxy"], [10, 20, 30, 40], atol=1)
    assert observation["original_coordinate_frame"] == "EXIF_CORRECTED_IMAGE_PIXEL"
    try:
        service.ingest("bad-image", [PhotoInput(b"not an image", "TOP")])
    except MultiImageError as exc:
        assert exc.code == "INVALID_IMAGE"
    else:
        raise AssertionError("corrupt image was accepted")


def test_overlapping_photos_stitch_and_fuse_same_detection_with_provenance():
    scene = _textured_scene()
    first, second = scene[:, :400], scene[:, 120:520]
    service = MultiImageComponentMapService()
    session = service.ingest("fixture-board", [
        PhotoInput(_png(first), "TOP", CaptureRole.REFERENCE, "overview.png"),
        PhotoInput(_png(second), "TOP", CaptureRole.CLOSEUP, "closeup.png"),
    ])
    def detector(image):
        # Same physical component is at x=220 in canonical space.
        x = 220 if image.width == 400 and image.getpixel((10, 10)) == tuple(first[10, 10]) else 100
        return {"status": "INFERENCE_COMPLETE", "detections": [
            {"detection_id": "U1", "class_name": "integrated_circuit", "confidence": 0.91,
             "bbox_xyxy": [x - 15, 120, x + 15, 150]},
        ]}
    result = service.build_map(session, detector=detector)
    top = result["sides"]["TOP"]
    regs = {item["photo_id"]: item for item in top["registrations"]}
    moving = next(photo for photo in session.photos if photo.name == "closeup.png")
    assert regs[moving.photo_id]["status"] == "REGISTERED_CANDIDATE"
    assert regs[moving.photo_id]["inlier_ratio"] > 0.2
    repeat = service.build_map(session, detector=detector)
    repeat_regs = {item["photo_id"]: item for item in repeat["sides"]["TOP"]["registrations"]}
    assert regs[moving.photo_id]["transform_moving_to_canonical"] == repeat_regs[moving.photo_id]["transform_moving_to_canonical"]
    assert top["canonical_image"]["image_size_px"][0] >= 400
    assert len(top["components"]) == 1
    assert top["components"][0]["identity_fusion_status"] == "MULTI_OBSERVATION_FUSED"
    assert len(top["components"][0]["source_image_ids"]) == 2
    assert len({component["component_id"] for component in top["components"]}) == 1
    mosaic_png = base64.b64decode(top["canonical_image"]["image_png_base64"])
    assert Image.open(io.BytesIO(mosaic_png)).size == tuple(top["canonical_image"]["image_size_px"])
    coverage = top["canonical_image"]["coverage_metrics"]
    source_map = np.asarray(Image.open(io.BytesIO(base64.b64decode(
        top["canonical_image"]["pixel_source_map_png_base64"]))))
    assert coverage["valid_pixel_count"] == int(np.count_nonzero(source_map))
    assert coverage["unobserved_pixel_count"] == coverage["canvas_pixel_count"] - coverage["valid_pixel_count"]
    assert abs(sum(coverage["source_contribution_ratio"].values()) - 1.0) < 1e-6

    overlay = service.fault_overlay(result, "TOP", [
        {"component_id": top["components"][0]["component_id"], "status": "CONFIRMED_FAULT",
         "confirmed_by_measurement_or_inspection": True},
        {"component_id": "missing", "status": "SUSPECTED_FAULT"},
    ])
    assert overlay["candidates"][0]["fault_candidate_status"] == "CONFIRMED_FAULT"
    assert overlay["candidates"][1]["mapping_status"] == "UNMAPPED_COMPONENT_ID"


def test_nonoverlapping_photos_are_not_forced_into_mosaic():
    rng = np.random.default_rng(33)
    one = rng.integers(0, 255, (220, 240, 3), dtype=np.uint8)
    two = rng.integers(0, 255, (220, 240, 3), dtype=np.uint8)
    session = MultiImageComponentMapService().ingest("unrelated", [
        PhotoInput(_png(one), "TOP", CaptureRole.REFERENCE),
        PhotoInput(_png(two), "TOP", CaptureRole.CLOSEUP),
    ])
    result = MultiImageComponentMapService().build_map(session)
    moving = next(photo for photo in session.photos if photo.photo_id != result["sides"]["TOP"]["registrations"][0]["photo_id"])
    records = {item["photo_id"]: item for item in result["sides"]["TOP"]["registrations"]}
    assert records[moving.photo_id]["status"] == "UNREGISTERED"
    assert records[moving.photo_id]["transform_moving_to_canonical"] is None


def test_component_ids_and_measurement_refs_survive_reprocessing_and_added_views(tmp_path):
    image = _textured_scene()[:, :400]
    service = MultiImageComponentMapService()
    base = PhotoInput(_png(image), "TOP", CaptureRole.REFERENCE, "reference.png")
    session = service.ingest("identity-board", [base])

    def detector_at(x):
        def detect(image):
            local_x = x if image.width == 400 else x - 100
            return {"status": "INFERENCE_COMPLETE", "detections": [{
                "detection_id": "U7", "class_name": "integrated_circuit", "confidence": 0.88,
                "bbox_xyxy": [local_x - 16, 130, local_x + 16, 162],
            }]}
        return detect

    first = service.build_map(session, detector=detector_at(175))
    old_component = first["sides"]["TOP"]["components"][0]
    old_component["user_confirmed_identity"] = True
    old_component["measurement_history_refs"] = ["measurement-0042"]
    old_component_id = old_component["component_id"]

    # Add a close-up after the original reference. The old frame/reference is retained.
    closeup = PhotoInput(_png(image[:, 100:400]), "TOP", CaptureRole.CLOSEUP, "closeup.png")
    expanded = service.ingest("identity-board", [base, closeup])
    updated = service.build_map(expanded, detector=detector_at(177), previous_component_map=first)
    component = updated["sides"]["TOP"]["components"][0]
    assert component["component_id"] == old_component_id
    assert component["identity_match_status"] == "USER_CONFIRMED"
    assert component["measurement_history_refs"] == ["measurement-0042"]
    mapping = updated["sides"]["TOP"]["historical_component_id_mappings"][0]
    assert mapping["status"] == "PRESERVED"

    manifest = tmp_path / "identity.json"
    save_identity_manifest(manifest, updated)
    reloaded = service.build_map(expanded, detector=detector_at(177), previous_component_map=manifest)
    assert reloaded["sides"]["TOP"]["components"][0]["component_id"] == old_component_id
    assert reloaded["sides"]["TOP"]["components"][0]["measurement_history_refs"] == ["measurement-0042"]


def test_fusion_rejects_same_photo_duplicate_and_conflicting_ocr():
    image = _textured_scene()[:, :400]
    service = MultiImageComponentMapService()
    session = service.ingest("fusion-board", [PhotoInput(_png(image), "TOP", CaptureRole.REFERENCE, "ref.png")])

    def detector(_):
        return {"status": "INFERENCE_COMPLETE", "detections": [
            {"detection_id": "R1", "class_name": "resistor", "confidence": 0.8,
             "bbox_xyxy": [90, 100, 120, 130], "ocr_text": "1001"},
            {"detection_id": "R2", "class_name": "resistor", "confidence": 0.8,
             "bbox_xyxy": [92, 102, 122, 132], "ocr_text": "1001"},
        ]}
    result = service.build_map(session, detector=detector)
    assert len(result["sides"]["TOP"]["components"]) == 2

    two_views = service.ingest("ocr-conflict", [
        PhotoInput(_png(image), "TOP", CaptureRole.REFERENCE, "ref.png"),
        PhotoInput(_png(np.clip(image.astype(np.int16) + 1, 0, 255).astype(np.uint8)),
                   "TOP", CaptureRole.CLOSEUP, "duplicate-view.png"),
    ])
    def conflicting_ocr(_):
        # Homography is identity for identical input pixels, so boxes coincide.
        marker = "1001" if calls[0] == 0 else "1002"
        calls[0] += 1
        return {"status": "INFERENCE_COMPLETE", "detections": [{
            "detection_id": "R", "class_name": "resistor", "confidence": 0.9,
            "bbox_xyxy": [90, 100, 120, 130], "ocr_text": marker,
        }]}
    calls = [0]
    conflict_result = service.build_map(two_views, detector=conflicting_ocr)
    assert len(conflict_result["sides"]["TOP"]["components"]) == 2

    class_calls = [0]
    def conflicting_class(_):
        class_name = "resistor" if class_calls[0] == 0 else "capacitor"
        class_calls[0] += 1
        return {"status": "INFERENCE_COMPLETE", "detections": [{
            "detection_id": "same-position", "class_name": class_name, "confidence": 0.9,
            "bbox_xyxy": [90, 100, 120, 130],
        }]}
    class_result = service.build_map(two_views, detector=conflicting_class)
    class_components = class_result["sides"]["TOP"]["components"]
    assert len(class_components) == 2
    assert all(component["identity_match_status"] == "IDENTITY_CONFLICT" for component in class_components)
    assert all(any(evidence["method"] == "OVERLAPPING_CLASS_CONFLICT"
                   for evidence in component["identity_evidence"]) for component in class_components)


def test_missing_and_duplicate_detector_ids_are_unique_and_order_stable():
    image = _textured_scene()[:, :400]
    service = MultiImageComponentMapService()
    session = service.ingest("id-collision-board", [PhotoInput(_png(image), "TOP", CaptureRole.REFERENCE, "same.png")])
    detections = [
        {"class_name": "resistor", "bbox_xyxy": [80, 90, 100, 110], "confidence": 0.8},
        {"class_name": "resistor", "bbox_xyxy": [120, 90, 140, 110], "confidence": 0.8},
        {"detection_id": "reused", "class_name": "capacitor", "bbox_xyxy": [160, 90, 180, 110], "confidence": 0.8},
        {"detection_id": "reused", "class_name": "capacitor", "bbox_xyxy": [200, 90, 220, 110], "confidence": 0.8},
        {"class_name": "diode", "bbox_xyxy": [240, 90, 260, 110], "confidence": 0.8},
        {"class_name": "diode", "bbox_xyxy": [240, 90, 260, 110], "confidence": 0.8},
    ]

    def run(rows):
        result = service.build_map(session, detector=lambda _: {"status": "INFERENCE_COMPLETE", "detections": rows})
        observations = result["sides"]["TOP"]["component_observations"]
        return observations

    first = run(detections)
    second = run(list(reversed(detections)))
    assert len({row["observation_id"] for row in first}) == len(detections)
    assert len({row["detection_id"] for row in first}) == len(detections)
    key = lambda row: (row["class_name"], tuple(row["processed_bbox_xyxy"]))
    first_ids = {}
    second_ids = {}
    for row in first:
        first_ids.setdefault(key(row), []).append(row["observation_id"])
    for row in second:
        second_ids.setdefault(key(row), []).append(row["observation_id"])
    assert {item: sorted(ids) for item, ids in first_ids.items()} == {
        item: sorted(ids) for item, ids in second_ids.items()}


def test_complete_link_fusion_does_not_chain_merge_and_marks_ambiguity():
    def obs(name, box):
        return {"observation_id": name, "photo_id": name, "detection_id": name,
                "class_name": "resistor", "bbox_xyxy": box, "original_bbox_xyxy": box,
                "confidence": 0.9, "registration_status": "REGISTERED_CANDIDATE",
                "registration_rmse_px": 1.0, "registration_inlier_ratio": 0.8,
                "evidence": "TEST"}

    components, _ = _fuse_observations("chain-board", "TOP", [
        obs("A", [0, 0, 10, 10]), obs("B", [2, 0, 12, 10]), obs("C", [-2, 0, 8, 10]),
    ])
    assert len(components) == 2
    assert sorted(len(item["component_observation_ids"]) for item in components) == [1, 2]
    assert any(item["identity_match_status"] == "IDENTITY_AMBIGUOUS"
               and any(evidence["method"] == "PAIRWISE_FUSION_DISAGREEMENT"
                       for evidence in item["identity_evidence"]) for item in components)


def test_same_photo_overlapping_detector_boxes_are_kept_separate_and_flagged():
    def obs(name, box):
        return {"observation_id": name, "photo_id": "same-photo", "detection_id": name,
                "class_name": "capacitor", "bbox_xyxy": box, "original_bbox_xyxy": box,
                "confidence": 0.9, "registration_status": "REFERENCE",
                "evidence": "TEST"}

    components, _ = _fuse_observations("dense-board", "TOP", [
        obs("C1", [40, 40, 64, 64]), obs("C2", [41, 41, 65, 65]),
    ])
    assert len(components) == 2
    assert all(component["identity_match_status"] == "IDENTITY_AMBIGUOUS" for component in components)
    assert all(any(evidence["method"] == "SAME_PHOTO_OVERLAPPING_DETECTIONS"
                   for evidence in component["identity_evidence"]) for component in components)


def test_automatic_rectification_uses_clear_boundary_candidate_and_detector_absence_is_explicit(tmp_path):
    import cv2

    board = np.full((300, 400, 3), 230, dtype=np.uint8)
    cv2.rectangle(board, (60, 40), (340, 260), (30, 70, 30), -1)
    cv2.rectangle(board, (60, 40), (340, 260), (255, 255, 255), 5)
    service = MultiImageComponentMapService()
    session = service.ingest("rectification-board", [PhotoInput(_png(board), "TOP")])
    assert session.photos[0].rectification["status"] == "RECTIFIED_CANDIDATE_NOT_PHYSICALLY_VALIDATED"
    assert session.photos[0].rectification["confidence_diagnostics"]["method"] == "AUTO_UNIQUE_LARGE_BOUNDARY_CANDIDATE"

    missing = service.build_map(session, model_path=tmp_path / "missing.onnx")
    assert missing["sides"]["TOP"]["detection_status"] == "BLOCKED_MODEL_UNAVAILABLE"
    assert missing["sides"]["TOP"]["components"] == []

    random_pixels = np.random.default_rng(216).integers(0, 256, (420, 660, 3), dtype=np.uint8)
    frame_texture = np.asarray(Image.fromarray(random_pixels).filter(ImageFilter.GaussianBlur(0.8)))[:, :420]
    frame_like = MultiImageComponentMapService().ingest("frame-like", [PhotoInput(_png(frame_texture), "TOP")])
    assert frame_like.photos[0].rectification["status"] == "RECTIFICATION_UNVERIFIED"
    assert frame_like.photos[0].rectification["reason"] == "BOUNDARY_TOUCHES_IMAGE_FRAME"
    assert frame_like.photos[0].processed_size_px == (420, 420)


def test_canvas_budget_falls_back_to_reference_and_does_not_keep_overflow_transform():
    scene = _textured_scene()[:, :520]
    budget = ImageBudget(max_decoded_edge=400, max_canvas_edge=400, max_canvas_pixels=160_000)
    # The first view is the reference, while the second has a large translation.
    first, second = scene[:, :400], scene[:, 120:520]
    session = MultiImageComponentMapService(budget=budget).ingest("canvas-budget", [
        PhotoInput(_png(first), "TOP", CaptureRole.REFERENCE, "reference.png"),
        PhotoInput(_png(second), "TOP", CaptureRole.CLOSEUP, "closeup.png"),
    ])
    result = MultiImageComponentMapService(budget=budget).build_map(session)
    side = result["sides"]["TOP"]
    assert side["canonical_image"]["image_size_px"] == [400, 400]
    moving = next(photo for photo in session.photos if photo.name == "closeup.png")
    record = next(item for item in side["registrations"] if item["photo_id"] == moving.photo_id)
    assert record["status"] == "UNREGISTERED"
    assert record["transform_moving_to_canonical"] is None
