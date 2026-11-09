"""Stable public entry point for trained PCB component detection."""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np

from circuitcare.vision.detector_backend import run_detector
from circuitcare.vision.detection_postprocess import PCB_MODEL_CLASSES, class_aware_nms

CORRECTABLE_CLASSES = (*PCB_MODEL_CLASSES, "unknown_component")


def detect_components(image: Any, model_path: str, *, confidence_threshold: float = 0.25,
                      iou_threshold: float = 0.45) -> dict[str, Any]:
    """Run the configured trained PCB object detector; this is not contour detection."""
    return run_detector(image, model_path, confidence_threshold=confidence_threshold,
                         iou_threshold=iou_threshold)


def detect_components_tiled(image: Any, model_path: str, *, confidence_threshold: float = 0.25,
                            iou_threshold: float = 0.45, tile_size_px: int = 1536,
                            tile_overlap_px: int = 320,
                            timeout_seconds: float | None = None) -> dict[str, Any]:
    """Run the existing ONNX model over overlapping native-resolution tiles.

    Results are restored to full-image pixels and globally class-aware NMS
    filtered. Tiling is an opt-in candidate-recall tradeoff, not an accuracy
    guarantee; it may increase inference time and false positives.
    """
    import cv2

    if (not 256 <= tile_size_px <= 4096 or not 0 <= tile_overlap_px <= tile_size_px // 2
            or (timeout_seconds is not None and timeout_seconds <= 0)):
        raise ValueError("tile size must be 256..4096 px, overlap at most half the tile, and timeout positive")
    rgb = image.convert("RGB") if hasattr(image, "convert") else image
    width, height = rgb.size
    if width * height > 40_000_000:
        raise ValueError("Tiled detector input exceeds the 40 megapixel resource limit")
    x_starts = _tile_starts(width, tile_size_px, tile_overlap_px)
    y_starts = _tile_starts(height, tile_size_px, tile_overlap_px)
    tile_count = len(x_starts) * len(y_starts)
    if tile_count > 256:
        raise ValueError("Tiled detector input exceeds the 256 tile computation limit")
    detections = []
    model_info = None
    started = time.perf_counter()
    for tile_y, y in enumerate(y_starts):
        for tile_x, x in enumerate(x_starts):
            if timeout_seconds is not None and time.perf_counter() - started >= timeout_seconds:
                raise TimeoutError("Tiled detector exceeded its configured time budget")
            right, bottom = min(width, x + tile_size_px), min(height, y + tile_size_px)
            tile = rgb.crop((x, y, right, bottom))
            result = detect_components(tile, model_path, confidence_threshold=confidence_threshold,
                                       iou_threshold=iou_threshold)
            if timeout_seconds is not None and time.perf_counter() - started >= timeout_seconds:
                raise TimeoutError("Tiled detector exceeded its configured time budget")
            model_info = result.get("model", model_info)
            for row in result.get("detections", []):
                box = row.get("bbox_xyxy")
                if not box or len(box) != 4:
                    continue
                restored = dict(row)
                restored["bbox_xyxy"] = [
                    float(np.clip(box[0] + x, 0, width)), float(np.clip(box[1] + y, 0, height)),
                    float(np.clip(box[2] + x, 0, width)), float(np.clip(box[3] + y, 0, height)),
                ]
                restored["inference_mode"] = "overlapping_tiles"
                restored["source_tile"] = {"x": x, "y": y, "width": right - x, "height": bottom - y,
                                            "row": tile_y, "column": tile_x}
                detections.append(restored)
                if len(detections) > 50_000:
                    raise RuntimeError("Tiled detector exceeded its 50000 candidate memory limit")
    detections = _global_tile_nms(detections, iou_threshold=iou_threshold)
    return {"status": "INFERENCE_COMPLETE", "model": model_info, "runtime": "onnxruntime-cpu-tiled",
            "inference_mode": "overlapping_tiles", "tile_count": tile_count,
            "tile_size_px": tile_size_px, "tile_overlap_px": tile_overlap_px,
            "inference_latency_ms": (time.perf_counter() - started) * 1000,
            "detections": detections}


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    starts = list(range(0, max(length - tile_size, 0) + 1, step))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _global_tile_nms(detections: list[dict[str, Any]], *, iou_threshold: float) -> list[dict[str, Any]]:
    if not detections:
        return []
    valid_indices, boxes, scores, class_ids = [], [], [], []
    class_index = {name: index for index, name in enumerate(PCB_MODEL_CLASSES)}
    for index, row in enumerate(detections):
        class_name = row.get("predicted_class") or row.get("model_class_candidate") or row.get("class_name")
        if class_name not in class_index or len(row.get("bbox_xyxy", [])) != 4:
            continue
        valid_indices.append(index)
        boxes.append(row["bbox_xyxy"])
        scores.append(float(row.get("confidence") or 0.0))
        class_ids.append(class_index[class_name])
    if not valid_indices:
        return []
    keep = class_aware_nms(np.asarray(boxes, dtype=np.float32), np.asarray(scores, dtype=np.float32),
                           np.asarray(class_ids, dtype=np.int64), iou_threshold=iou_threshold,
                           max_detections=1000)
    output = [detections[valid_indices[index]] for index in keep]
    occurrences: dict[str, int] = {}
    for row in output:
        base = hashlib.sha256((str(row.get("class_name")) + ":" + ",".join(
            f"{float(value):.2f}" for value in row["bbox_xyxy"])).encode()).hexdigest()[:18]
        occurrence = occurrences.get(base, 0)
        occurrences[base] = occurrence + 1
        row["detection_id"] = f"tile_{base}" + (f"_{occurrence + 1}" if occurrence else "")
    return output


def apply_detection_corrections(detections: list[dict[str, Any]],
                                corrections: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return corrected copies while preserving model predictions as separate evidence."""
    result = []
    used_references: set[str] = set()
    for detection in detections:
        correction = corrections.get(detection["detection_id"], {})
        row = dict(detection)
        row["predicted_class"] = detection.get("predicted_class", detection["class_name"])
        corrected_class = correction.get("corrected_class")
        if corrected_class:
            if corrected_class not in CORRECTABLE_CLASSES:
                raise ValueError(f"지원하지 않는 수정 부품 종류입니다: {corrected_class}")
            row["corrected_class"] = corrected_class
        else:
            row["corrected_class"] = None
        row["predicted_reference"] = detection.get("predicted_reference", detection.get("reference"))
        confirmed_reference = str(correction.get("confirmed_reference") or "").strip().upper() or None
        if confirmed_reference and correction.get("reference_confirmed"):
            if confirmed_reference in used_references:
                raise ValueError(f"KiCad Reference {confirmed_reference}가 여러 검출에 중복 지정되었습니다.")
            used_references.add(confirmed_reference)
            row["confirmed_reference"] = confirmed_reference
            row["verification_source"] = "user_manual_confirmation"
            row["identification_status"] = "user_confirmed"
        else:
            row["confirmed_reference"] = None
            row["verification_source"] = None
            row["identification_status"] = "rejected" if correction.get("rejected") else "suggested" if correction.get("confirmed_reference") else "unmatched"
        row["is_rejected"] = bool(correction.get("rejected"))
        if not row["is_rejected"]:
            result.append(row)
    return result


def add_manual_detection(detections: list[dict[str, Any]], *, class_name: str,
                         bbox_xyxy: list[float], image_size: tuple[int, int]) -> dict[str, Any]:
    if class_name not in CORRECTABLE_CLASSES:
        raise ValueError(f"지원하지 않는 부품 종류입니다: {class_name}")
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    width, height = image_size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("수동 Bounding Box가 이미지 경계 안의 유효한 사각형이 아닙니다.")
    used = {row["detection_id"] for row in detections}
    number = len(detections) + 1
    detection_id = f"manual_{number:04d}"
    while detection_id in used:
        number += 1
        detection_id = f"manual_{number:04d}"
    return {"detection_id": detection_id, "class_name": class_name, "confidence": None,
            "bbox_xyxy": [x1, y1, x2, y2], "source": "user_manual_addition", "model_id": None,
            "reference": None, "identification_status": "class_candidate", "predicted_class": None}
