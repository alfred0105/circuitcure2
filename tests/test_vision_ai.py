from __future__ import annotations

import numpy as np
import pytest
import time
from PIL import Image

from circuitcare.vision import component_detector
from circuitcare.vision.component_detector import add_manual_detection, apply_detection_corrections
from circuitcare.vision.detection_evaluation import evaluate_detections
from circuitcare.vision.detection_postprocess import class_aware_nms, decode_yolov8_output, letterbox_rgb
from circuitcare.vision.detector_backend import DetectorUnavailable, model_file_info, run_detector
from circuitcare.vision.model_artifacts import CHECKPOINT_SHA256, CHECKPOINT_SIZE_BYTES
from circuitcare.vision.ocr_backend import extract_reference_candidates
from circuitcare.vision.overlay import render_detection_risk_overlay
from circuitcare.vision.reference_matcher import suggest_reference_matches
from circuitcare.vision.vision_ai_ui import default_detector_model_path


def test_detector_model_file_missing_is_explicit(tmp_path) -> None:
    result = model_file_info(tmp_path / "missing.onnx")
    assert result["status"] == "MODEL_UNAVAILABLE"
    assert len(CHECKPOINT_SHA256) == 64
    assert CHECKPOINT_SIZE_BYTES > 80_000_000
    with pytest.raises(DetectorUnavailable, match="모델이 없습니다"):
        run_detector(Image.new("RGB", (10, 10)), tmp_path / "missing.onnx")


def test_detector_ui_default_matches_preparation_script_output() -> None:
    path = default_detector_model_path()
    repository_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    assert path == repository_root / "models" / "pcb_component_detector.onnx"
    info = model_file_info(path)
    assert info["status"] == ("READY" if path.is_file() else "MODEL_UNAVAILABLE")


def test_letterbox_and_yolov8_decode_restore_source_coordinates() -> None:
    image = Image.new("RGB", (200, 100), "white")
    tensor, ratio, padding = letterbox_rgb(image, size=100)
    assert tensor.shape == (1, 3, 100, 100)
    assert ratio == pytest.approx(0.5)
    assert padding == pytest.approx((0.0, 25.0))
    raw = np.zeros((1, 14, 1), dtype=np.float32)
    raw[0, :4, 0] = [50, 50, 20, 10]
    raw[0, 4, 0] = 0.95
    detections = decode_yolov8_output(raw, image_size=image.size, ratio=ratio, padding=padding,
                                      confidence_threshold=0.2)
    assert len(detections) == 1
    assert detections[0]["class_name"] == "resistor"
    assert detections[0]["bbox_xyxy"] == pytest.approx([80, 40, 120, 60])


def test_nms_removes_same_class_duplicates_but_keeps_other_classes() -> None:
    boxes = np.asarray([[0, 0, 20, 20], [1, 1, 21, 21], [0, 0, 20, 20]], dtype=float)
    scores = np.asarray([0.9, 0.8, 0.7])
    class_ids = np.asarray([0, 0, 1])
    assert class_aware_nms(boxes, scores, class_ids, iou_threshold=0.4) == [0, 2]


def test_tiled_detector_restores_coordinates_and_fuses_tile_edge_duplicates(monkeypatch, tmp_path) -> None:
    calls = []
    def fake_detect(tile, model_path, *, confidence_threshold, iou_threshold):
        calls.append(tile.size)
        if len(calls) == 1:
            rows = [{"class_name": "resistor", "predicted_class": "resistor", "confidence": 0.9,
                     "bbox_xyxy": [700, 100, 760, 160]}]
        else:
            rows = [
                {"class_name": "resistor", "predicted_class": "resistor", "confidence": 0.8,
                 "bbox_xyxy": [568, 100, 628, 160]},
                {"class_name": "capacitor", "predicted_class": "capacitor", "confidence": 0.7,
                 "bbox_xyxy": [718, 250, 760, 300]},
            ]
        return {"status": "INFERENCE_COMPLETE", "model": {"status": "READY"}, "detections": rows}
    monkeypatch.setattr(component_detector, "detect_components", fake_detect)
    result = component_detector.detect_components_tiled(Image.new("RGB", (900, 600)),
        str(tmp_path / "model.onnx"), tile_size_px=768, tile_overlap_px=128)
    assert result["tile_count"] == 2
    assert calls == [(768, 600), (768, 600)]
    assert len(result["detections"]) == 2
    resistor = next(row for row in result["detections"] if row["class_name"] == "resistor")
    assert resistor["bbox_xyxy"] == pytest.approx([700, 100, 760, 160])
    assert resistor["source_tile"]["column"] == 0
    capacitor = next(row for row in result["detections"] if row["class_name"] == "capacitor")
    assert capacitor["bbox_xyxy"] == pytest.approx([850, 250, 892, 300])


def test_tiled_detector_enforces_resource_and_time_budgets(monkeypatch, tmp_path) -> None:
    with pytest.raises(ValueError, match="tile size"):
        component_detector.detect_components_tiled(Image.new("RGB", (100, 100)),
            str(tmp_path / "model.onnx"), tile_size_px=128, tile_overlap_px=96)
    with pytest.raises(ValueError, match="40 megapixel"):
        component_detector.detect_components_tiled(type("ImageFrame", (), {"size": (7000, 6000),
            "convert": lambda self, mode: self})(),
            str(tmp_path / "model.onnx"), tile_size_px=1024, tile_overlap_px=128)

    def slow_detect(tile, model_path, *, confidence_threshold, iou_threshold):
        time.sleep(0.01)
        return {"status": "INFERENCE_COMPLETE", "detections": []}
    monkeypatch.setattr(component_detector, "detect_components", slow_detect)
    with pytest.raises(TimeoutError, match="time budget"):
        component_detector.detect_components_tiled(Image.new("RGB", (300, 300)),
            str(tmp_path / "model.onnx"), tile_size_px=256, tile_overlap_px=64,
            timeout_seconds=0.001)


def test_low_score_class_becomes_unknown_and_outside_box_is_rejected() -> None:
    raw = np.zeros((1, 14, 2), dtype=np.float32)
    raw[0, :4, 0] = [50, 50, 20, 20]
    raw[0, 4, 0] = 0.30
    raw[0, :4, 1] = [150, 150, 20, 20]
    raw[0, 5, 1] = 0.99
    detections = decode_yolov8_output(raw, image_size=(100, 100), ratio=1.0, padding=(0, 0),
                                      confidence_threshold=0.25)
    assert len(detections) == 1
    assert detections[0]["predicted_class"] == "resistor"
    assert detections[0]["class_name"] == "unknown_component"
    assert detections[0]["identification_status"] == "unknown"


def test_ocr_reference_candidates_preserve_ambiguous_raw_characters() -> None:
    assert extract_reference_candidates("C1 R12 U1 D5 J3 O1") == ["C1", "R12", "U1", "D5", "J3"]


def test_reference_suggestions_use_position_ocr_and_type_without_confirming() -> None:
    detections = [{"detection_id": "det_1", "class_name": "capacitor", "bbox_xyxy": [90, 90, 110, 110]}]
    ocr = [{"raw_text": "C1", "reference_candidates": ["C1"], "confidence": 0.9,
            "bbox_xyxy": [112, 88, 130, 98]}]
    positions = [{"reference": "C1", "image_position_px": [100, 100]},
                 {"reference": "R1", "image_position_px": [170, 170]}]
    result = suggest_reference_matches(detections, ocr, positions, image_size=(200, 200))
    best = result["assignments"]["det_1"][0]
    assert best["reference"] == "C1"
    assert best["status"] == "suggested"
    assert result["score_kind"] == "heuristic_ordering_index_not_probability"


def test_user_corrections_keep_prediction_and_enforce_unique_references() -> None:
    original = [{"detection_id": "det_1", "class_name": "resistor", "confidence": 0.7, "reference": None},
                {"detection_id": "det_2", "class_name": "capacitor", "confidence": 0.8, "reference": None}]
    corrected = apply_detection_corrections(original, {
        "det_1": {"corrected_class": "capacitor", "confirmed_reference": "C1", "reference_confirmed": True},
    })
    assert corrected[0]["predicted_class"] == "resistor"
    assert corrected[0]["corrected_class"] == "capacitor"
    assert corrected[0]["confirmed_reference"] == "C1"
    assert corrected[0]["verification_source"] == "user_manual_confirmation"
    with pytest.raises(ValueError, match="중복"):
        apply_detection_corrections(original, {
            "det_1": {"confirmed_reference": "C1", "reference_confirmed": True},
            "det_2": {"confirmed_reference": "C1", "reference_confirmed": True},
        })


def test_manual_detection_requires_inside_image_box() -> None:
    detection = add_manual_detection([], class_name="unknown_component", bbox_xyxy=[1, 2, 8, 9], image_size=(10, 10))
    assert detection["source"] == "user_manual_addition"
    assert detection["confidence"] is None
    with pytest.raises(ValueError, match="Bounding Box"):
        add_manual_detection([], class_name="resistor", bbox_xyxy=[-1, 0, 5, 5], image_size=(10, 10))


def test_detection_risk_only_maps_confirmed_reference_and_matching_scenario() -> None:
    image = Image.new("RGB", (40, 40), "white")
    detection = [{"detection_id": "det_1", "class_name": "capacitor", "confidence": 0.8,
                  "bbox_xyxy": [5, 5, 30, 30]}]
    risk = {"simulation_status": "success", "failed_component": "U1", "fault_type": "rise",
            "secondary_candidates": [{"component_ref": "C1", "risk_level": "EXCEEDED", "priority_rank": 1}]}
    overlay, meta = render_detection_risk_overlay(image, detection, confirmed_references={}, risk_assessment=risk,
                                                  expected_fault="U1", expected_fault_type="rise")
    assert overlay.size == image.size
    assert meta["risk_mapped_count"] == 0
    _, meta = render_detection_risk_overlay(image, detection, confirmed_references={"det_1": "C1"},
                                            risk_assessment=risk, expected_fault="U1", expected_fault_type="rise")
    assert meta["risk_mapped_count"] == 1
    _, mismatched = render_detection_risk_overlay(image, detection, confirmed_references={"det_1": "C1"},
                                                   risk_assessment=risk, expected_fault="R1", expected_fault_type="other")
    assert mismatched["risk_mapped_count"] == 0


def test_metrics_are_measured_from_labeled_boxes_not_inference_confidence() -> None:
    gt = {"board.jpg": [{"class_id": 0, "bbox_xyxy": [0, 0, 10, 10]}]}
    pred = {"board.jpg": [{"class_name": "resistor", "confidence": 0.8, "bbox_xyxy": [0, 0, 10, 10]}]}
    result = evaluate_detections(gt, pred, class_names=["resistor"], confidence_threshold=0.25)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["map50"] == pytest.approx(1.0)
