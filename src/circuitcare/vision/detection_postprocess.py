"""YOLOv8 ONNX output decoding and coordinate-safe box postprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np

PCB_MODEL_CLASSES = (
    "resistor", "capacitor", "diode", "led", "integrated_circuit", "connector",
    "switch_button", "header", "crystal", "debug_test_point",
)
PCB_MODEL_ID = "pcb-yolov8s-2026-07-11"
PCB_MODEL_IMAGE_SIZE = 960
LOW_CONFIDENCE_UNKNOWN_THRESHOLD = 0.35


def letterbox_rgb(image: Any, size: int = PCB_MODEL_IMAGE_SIZE) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize a PIL RGB image into a square YOLO canvas and return scale/padding."""
    import cv2

    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    if width < 1 or height < 1 or size < 1:
        raise ValueError("모델 입력 이미지와 해상도는 0보다 커야 합니다.")
    ratio = min(size / width, size / height)
    new_width, new_height = int(round(width * ratio)), int(round(height * ratio))
    resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_width) / 2, (size - new_height) / 2
    left, top = int(round(pad_x - 0.1)), int(round(pad_y - 0.1))
    right, bottom = size - new_width - left, size - new_height - top
    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(tensor), ratio, (float(left), float(top))


def decode_yolov8_output(output: np.ndarray, *, image_size: tuple[int, int], ratio: float,
                         padding: tuple[float, float], class_names: tuple[str, ...] = PCB_MODEL_CLASSES,
                         confidence_threshold: float = 0.25, iou_threshold: float = 0.45,
                         max_detections: int = 1000) -> list[dict[str, Any]]:
    """Decode a standard non-NMS YOLOv8 output tensor into original-image boxes."""
    prediction = np.asarray(output, dtype=np.float32)
    if prediction.ndim == 3 and prediction.shape[0] == 1:
        prediction = prediction[0]
    expected_channels = 4 + len(class_names)
    if prediction.ndim != 2:
        raise ValueError(f"지원하지 않는 ONNX 출력 shape입니다: {prediction.shape}")
    if prediction.shape[0] == expected_channels:
        prediction = prediction.T
    elif prediction.shape[1] != expected_channels:
        raise ValueError(f"모델 출력 클래스 수가 설정과 다릅니다: {prediction.shape}")
    if not np.isfinite(prediction).all():
        raise ValueError("모델 출력에 유한하지 않은 값이 있습니다.")

    class_scores = prediction[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]
    selected = scores >= confidence_threshold
    rows = prediction[selected]
    scores = scores[selected]
    class_ids = class_ids[selected]
    if not len(rows):
        return []

    cx, cy, box_w, box_h = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    boxes = np.column_stack((cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2))
    pad_x, pad_y = padding
    boxes[:, (0, 2)] = (boxes[:, (0, 2)] - pad_x) / ratio
    boxes[:, (1, 3)] = (boxes[:, (1, 3)] - pad_y) / ratio
    width, height = image_size
    # Reject boxes wholly outside the image, then clip partial boxes to its boundary.
    intersects = (boxes[:, 2] > 0) & (boxes[:, 3] > 0) & (boxes[:, 0] < width) & (boxes[:, 1] < height)
    boxes, scores, class_ids = boxes[intersects], scores[intersects], class_ids[intersects]
    boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, width)
    boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, height)
    valid = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
    boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]
    keep = class_aware_nms(boxes, scores, class_ids, iou_threshold=iou_threshold, max_detections=max_detections)
    detections = []
    for index, i in enumerate(keep):
        model_class = class_names[int(class_ids[i])]
        low_confidence = float(scores[i]) < LOW_CONFIDENCE_UNKNOWN_THRESHOLD
        detections.append({
            "detection_id": f"det_{index + 1:04d}", "class_name": "unknown_component" if low_confidence else model_class,
            "predicted_class": model_class, "model_class_candidate": model_class,
            "confidence": float(scores[i]), "bbox_xyxy": [float(value) for value in boxes[i]],
            "source": "trained_object_detector", "model_id": PCB_MODEL_ID, "reference": None,
            "identification_status": "unknown" if low_confidence else "class_candidate",
        })
    return detections


def class_aware_nms(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, *,
                    iou_threshold: float = 0.45, max_detections: int = 1000) -> list[int]:
    kept: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        indices = indices[np.argsort(scores[indices])[::-1]]
        while indices.size and len(kept) < max_detections:
            chosen = int(indices[0])
            kept.append(chosen)
            if indices.size == 1:
                break
            remaining = indices[1:]
            overlap = _box_iou(boxes[chosen], boxes[remaining])
            indices = remaining[overlap <= iou_threshold]
    return sorted(kept, key=lambda index: float(scores[index]), reverse=True)[:max_detections]


def _box_iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    left_top = np.maximum(box[:2], others[:, :2])
    right_bottom = np.minimum(box[2:], others[:, 2:])
    inter = np.prod(np.maximum(right_bottom - left_top, 0), axis=1)
    area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    other_area = np.maximum(others[:, 2] - others[:, 0], 0) * np.maximum(others[:, 3] - others[:, 1], 0)
    return inter / np.maximum(area + other_area - inter, 1e-9)
