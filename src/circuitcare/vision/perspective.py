"""Perspective rectification for manually confirmed board quadrilaterals."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from circuitcare.vision.board_detector import validate_corners


def rectify_board(image: Any, corners_px: Any, *, physical_size_mm: tuple[float, float] | None = None,
                  max_output_edge: int = 3000) -> dict[str, Any]:
    if max_output_edge < 64:
        raise ValueError("max_output_edge must be at least 64 pixels")
    array = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("입력은 RGB 이미지여야 합니다.")
    source = validate_corners(corners_px, (array.shape[1], array.shape[0]))
    if physical_size_mm is not None:
        physical_w, physical_h = map(float, physical_size_mm)
        if physical_w <= 0 or physical_h <= 0:
            raise ValueError("실제 보드 폭과 높이는 양수여야 합니다.")
        aspect = physical_w / physical_h
    else:
        top = np.linalg.norm(source[1] - source[0])
        bottom = np.linalg.norm(source[2] - source[3])
        right = np.linalg.norm(source[2] - source[1])
        left = np.linalg.norm(source[3] - source[0])
        aspect = ((top + bottom) / 2) / ((left + right) / 2)
    edge = min(max_output_edge, max(64, int(max_output_edge)))
    if aspect >= 1:
        width = edge
        height = max(64, int(round(width / aspect)))
    else:
        height = edge
        width = max(64, int(round(height * aspect)))
    width, height = min(width, max_output_edge), min(height, max_output_edge)
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, destination)
    if (not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12
            or np.linalg.cond(matrix) > 1e12):
        raise ValueError("원근 변환 행렬을 계산할 수 없습니다.")
    warped = cv2.warpPerspective(array, matrix, (width, height))
    return {
        "image": Image.fromarray(warped),
        "transform_matrix": matrix.tolist(),
        "corners_px": source.tolist(),
        "output_size_px": [width, height],
        "physical_size_mm": list(physical_size_mm) if physical_size_mm else None,
        "aspect_source": "physical_dimensions" if physical_size_mm else "image_edge_estimate",
    }
