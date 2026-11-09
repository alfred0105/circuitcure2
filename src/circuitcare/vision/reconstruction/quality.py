"""Multi-signal photo quality hints; low scores never suppress observations."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def inspect_photo_quality(image: Any, *, board_corners: Any | None = None) -> dict[str, Any]:
    array = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    if array.ndim != 3 or array.shape[2] != 3 or not array.size:
        raise ValueError("품질 검사는 RGB 이미지가 필요합니다.")
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    over = float(np.mean(gray >= 248))
    under = float(np.mean(gray <= 8))
    glare = float(np.mean((hsv[:, :, 2] >= 245) & (hsv[:, :, 1] <= 35)))
    h, w = gray.shape
    flags = []
    if min(w, h) < 480:
        flags.append("LOW_RESOLUTION")
    if blur < 35:
        flags.append("POSSIBLE_BLUR")
    if over > 0.12:
        flags.append("OVEREXPOSED_AREAS")
    if glare > 0.025:
        flags.append("POSSIBLE_GLARE")
    crop = None
    perspective = None
    if board_corners is not None:
        points = np.asarray(board_corners, dtype=np.float32).reshape(-1, 2)
        if points.shape == (4, 2):
            crop = float(cv2.contourArea(points) / (w * h))
            edge_lengths = [float(np.linalg.norm(points[(i + 1) % 4] - points[i])) for i in range(4)]
            opposite_ratio = max(edge_lengths[0], edge_lengths[2]) / max(1.0, min(edge_lengths[0], edge_lengths[2]))
            perspective = abs(opposite_ratio - 1.0)
            if crop < 0.8:
                flags.append("BOARD_CROPPED_OR_SMALL")
            if perspective > 0.45:
                flags.append("PERSPECTIVE_DISTORTION")
    return {
        "resolution_px": [w, h], "blur_laplacian_variance": blur,
        "overexposed_fraction": over, "underexposed_fraction": under,
        "glare_candidate_fraction": glare, "board_area_fraction": crop,
        "perspective_asymmetry": perspective,
        "feature_matching_possible": min(w, h) >= 160 and blur >= 8,
        "occlusion": "NOT_ASSESSED", "flags": flags,
        "status": "QUALITY_HINTS_ONLY",
    }
