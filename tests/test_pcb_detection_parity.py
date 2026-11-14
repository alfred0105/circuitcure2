"""Opt-in parity run against the existing Windows detector and local model files.

Set CIRCUITCARE_WINDOWS_REPO, CIRCUITCARE_PCB_WEIGHTS and
CIRCUITCARE_PARITY_IMAGE to run this on the same local assets. No weights are
downloaded or written by this test.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest


def _load_app_server(name: str, app_dir: Path):
    sys.path.insert(0, str(app_dir))
    spec = importlib.util.spec_from_file_location(name, app_dir / "app_server.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection + 1e-9)


@pytest.mark.integration
@pytest.mark.model_performance
def test_windows_and_mac_pipeline_outputs_match_on_same_local_assets(tmp_path):
    windows_repo = os.environ.get("CIRCUITCARE_WINDOWS_REPO")
    weights_root = os.environ.get("CIRCUITCARE_PCB_WEIGHTS")
    image_path = os.environ.get("CIRCUITCARE_PARITY_IMAGE")
    if not all((windows_repo, weights_root, image_path)):
        pytest.skip("Set CIRCUITCARE_WINDOWS_REPO, CIRCUITCARE_PCB_WEIGHTS, and CIRCUITCARE_PARITY_IMAGE")
    windows_app = Path(windows_repo) / "apps" / "pcb-component-recognition"
    local_app = Path(__file__).parents[1] / "apps" / "pcb-component-recognition"
    if not (windows_app / "app_server.py").is_file() or not Path(weights_root).is_dir() or not Path(image_path).is_file():
        pytest.skip("Windows reference, local weights, or comparison image is unavailable")
    windows_predict = windows_app / "hr" / "infer" / "predict.py"
    mac_predict = Path(__file__).parents[1] / "apps" / "pcb-component-recognition" / "hr" / "infer" / "predict.py"
    assert hashlib.sha256(windows_predict.read_bytes()).digest() == hashlib.sha256(mac_predict.read_bytes()).digest()

    os.environ["CIRCUITCARE_PCB_WEIGHTS"] = str(Path(weights_root).resolve())
    import torch  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415

    windows = _load_app_server("windows_pcb_reference", windows_app)
    mac = _load_app_server("mac_pcb_candidate", local_app)
    requested = os.environ.get("CIRCUITCARE_PARITY_DEVICES", "cpu,cpu").split(",", 1)
    windows_device, mac_device = (requested + ["cpu"])[:2]
    windows.DEVICE, mac.DEVICE = windows_device.strip(), mac_device.strip()
    weights = Path(weights_root)
    locator = weights / "any_best.pt"
    doctor = weights / "external" / "pcb-ai-doctor" / "best_v12n.pt"
    classifier = weights / "external" / "pcb-yolov8s-detection" / "best.pt"
    assert all(path.is_file() for path in (locator, doctor, classifier)), "Required local model weights are missing"

    # Use separate model instances because Ultralytics caches predictor/device state.
    for module in (windows, mac):
        module.ANY_MODEL = YOLO(str(locator))
        module.EXTERNAL_MODELS["pcb_ai_doctor_v12n"] = YOLO(str(doctor))
        module.EXTERNAL_MODELS["arshia_pcb_yolov8s"] = YOLO(str(classifier))
    image_bytes = Path(image_path).read_bytes()
    reference = windows.run_prediction(image_bytes, {"conf": .20, "tile": 1024, "overlap": .25})
    candidate = mac.run_prediction(image_bytes, {"conf": .20, "tile": 1024, "overlap": .25})
    windows_positions = windows.run_body_only_prediction(image_bytes, {"conf": .20, "tile": 1024, "overlap": .25})
    mac_positions = mac.run_body_only_prediction(image_bytes, {"conf": .20, "tile": 1024, "overlap": .25})
    reference_rows = reference["detections"]
    candidate_rows = candidate["detections"]
    # Windows drops unclassified bodies; Mac keeps them as unknown for review.
    mac_known = [row for row in candidate_rows if row["class_name"] != "unknown"]
    remaining = set(range(len(mac_known)))
    pairs = []
    for old in reference_rows:
        if not remaining:
            break
        best = max(remaining, key=lambda index: _iou(old["bbox_xyxy"], mac_known[index]["bbox_xyxy"]))
        overlap = _iou(old["bbox_xyxy"], mac_known[best]["bbox_xyxy"])
        if old["class_name"] == mac_known[best]["class_name"] and overlap >= .90:
            pairs.append((old, mac_known[best], overlap))
            remaining.remove(best)
    report = {
        "image": str(Path(image_path).resolve()), "weights_root": str(weights.resolve()),
        "devices": {"windows_pipeline": windows_device, "mac_pipeline": mac_device},
        "windows_count": len(reference_rows), "mac_count": len(candidate_rows),
        "windows_location_box_count": len(windows_positions["detections"]),
        "mac_location_box_count": len(mac_positions["detections"]),
        "mac_unknown_count": sum(row["class_name"] == "unknown" for row in candidate_rows),
        "matched_same_class_iou_ge_0_90": len(pairs),
        "class_or_location_mismatch_count": len(reference_rows) - len(pairs),
        "iou_min": min((pair[2] for pair in pairs), default=None),
        "confidence_abs_delta_max": max((abs(float(old["confidence"]) - float(new["confidence"]))
                                           for old, new, _ in pairs), default=None),
        "matched_components": [{"class_name": old["class_name"], "windows_bbox": old["bbox_xyxy"],
                                "mac_bbox": new["bbox_xyxy"], "iou": overlap,
                                "windows_confidence": old["confidence"], "mac_confidence": new["confidence"],
                                "confidence_abs_delta": abs(float(old["confidence"]) - float(new["confidence"]))}
                               for old, new, overlap in pairs],
        "class_counts_windows": reference["counts"], "class_counts_mac": candidate["counts"],
        "windows_detections": [{"class_name": row["class_name"], "bbox_xyxy": row["bbox_xyxy"],
                                "confidence": row["confidence"]} for row in reference_rows],
        "mac_detections": [{"class_name": row["class_name"], "bbox_xyxy": row["bbox_xyxy"],
                            "confidence": row["confidence"],
                            "location_confidence": row.get("location_confidence"),
                            "type_confidence": row.get("type_confidence"),
                            "match_score": row.get("match_score")} for row in candidate_rows],
        "position_boxes_windows": [row["bbox_xyxy"] for row in windows_positions["detections"]],
        "position_boxes_mac": [row["bbox_xyxy"] for row in mac_positions["detections"]],
    }
    report_path = Path(os.environ.get("CIRCUITCARE_PARITY_REPORT", tmp_path / "pcb-parity-report.json"))
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    assert len(pairs) >= max(1, round(len(reference_rows) * .95)), report
    assert len(reference_rows) - len(pairs) <= 1, report
    position_matches = sum(
        1 for old in windows_positions["detections"]
        if any(_iou(old["bbox_xyxy"], row["bbox_xyxy"]) >= .90 for row in mac_positions["detections"])
    )
    report["location_pairs_iou_ge_0_90"] = position_matches
    report["location_box_count_delta_mac_minus_windows"] = len(mac_positions["detections"]) - len(windows_positions["detections"])
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    assert len(windows_positions["detections"]) == len(mac_positions["detections"]), report
    assert position_matches == len(windows_positions["detections"]), report
