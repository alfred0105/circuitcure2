"""OpenCV contour candidates for manual PCB-board confirmation."""

from __future__ import annotations

import hashlib
import copy
from collections import OrderedDict
from typing import Any

import cv2
import numpy as np

_DETECTION_CACHE: OrderedDict[tuple[str, int, str], dict[str, Any]] = OrderedDict()
_DETECTION_CACHE_LIMIT = 8
_DETECTION_CACHE_HITS = 0
_DETECTION_CACHE_MISSES = 0


def detection_cache_stats(*, reset: bool = False) -> dict[str, int]:
    global _DETECTION_CACHE_HITS, _DETECTION_CACHE_MISSES
    stats = {"hits": _DETECTION_CACHE_HITS, "misses": _DETECTION_CACHE_MISSES, "entries": len(_DETECTION_CACHE)}
    if reset:
        _DETECTION_CACHE_HITS = _DETECTION_CACHE_MISSES = 0
    return stats


def order_corners(points: Any) -> np.ndarray:
    """Return a convex quadrilateral as TL, TR, BR, BL."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        raise ValueError("보드 모서리는 서로 다른 네 점이어야 합니다.")
    if len(np.unique(pts, axis=0)) != 4:
        raise ValueError("보드 모서리 좌표가 중복됩니다.")
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    # Ensure the first two points are the top edge and the sequence is clockwise.
    if ordered[1, 0] < ordered[-1, 0]:
        ordered = ordered[[0, 3, 2, 1]]
    crosses = []
    for i in range(4):
        a, b, c = ordered[i], ordered[(i + 1) % 4], ordered[(i + 2) % 4]
        first, second = b - a, c - b
        crosses.append(float(first[0] * second[1] - first[1] * second[0]))
    if min(abs(value) for value in crosses) < 1e-3 or not (all(x > 0 for x in crosses) or all(x < 0 for x in crosses)):
        raise ValueError("모서리가 볼록한 사각형을 이루지 않습니다.")
    return ordered


def validate_corners(points: Any, image_size: tuple[int, int], *, min_area_ratio: float = 0.03) -> np.ndarray:
    ordered = order_corners(points)
    width, height = image_size
    if width <= 0 or height <= 0 or np.any(ordered[:, 0] < 0) or np.any(ordered[:, 0] >= width) or np.any(ordered[:, 1] < 0) or np.any(ordered[:, 1] >= height):
        raise ValueError("모서리 좌표가 이미지 범위를 벗어났습니다.")
    area = abs(float(cv2.contourArea(ordered)))
    if area < width * height * min_area_ratio:
        raise ValueError("선택한 보드 영역이 이미지에 비해 너무 작습니다.")
    return ordered


def detect_board_candidates(image: Any, *, max_candidates: int = 5) -> dict[str, Any]:
    global _DETECTION_CACHE_HITS, _DETECTION_CACHE_MISSES
    array = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("입력은 RGB 이미지여야 합니다.")
    image_hash = hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()
    cache_key = (image_hash, max_candidates, "board-contour-v1")
    if cache_key in _DETECTION_CACHE:
        _DETECTION_CACHE_HITS += 1
        _DETECTION_CACHE.move_to_end(cache_key)
        return copy.deepcopy(_DETECTION_CACHE[cache_key])
    _DETECTION_CACHE_MISSES += 1
    height, width = array.shape[:2]
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        if area < width * height * 0.03:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        try:
            points = validate_corners(polygon.reshape(4, 2), (width, height))
        except ValueError:
            continue
        candidates.append({"corners_px": points.tolist(), "area_px": area, "area_ratio": area / (width * height)})
        if len(candidates) >= max_candidates:
            break
    result = {
        "board_detected": bool(candidates),
        "detection_method": "opencv_contour",
        "status": "candidate_unverified" if candidates else "not_detected",
        "candidates": candidates,
        "image_size_px": [width, height],
    }
    _DETECTION_CACHE[cache_key] = copy.deepcopy(result)
    while len(_DETECTION_CACHE) > _DETECTION_CACHE_LIMIT:
        _DETECTION_CACHE.popitem(last=False)
    return result
