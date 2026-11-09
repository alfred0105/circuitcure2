"""Image-only candidate anchors for round through-hole drills in PRISM renders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageAnchor:
    anchor_id: str
    photo_id: str
    board_side: str
    center_px: tuple[float, float]
    area_px2: int
    circularity: float
    aspect_ratio: float
    status: str = "DETECTED_CANDIDATE"
    evidence: str = "DARK_CONNECTED_COMPONENT_CIRCULARITY"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_dark_round_anchors(image: np.ndarray | Image.Image | str | Path, *,
                              photo_id: str, board_side: str,
                              threshold_gray: int = 28,
                              min_area_px2: int = 8,
                              max_area_px2: int = 5000,
                              min_circularity: float = 0.35,
                              merge_distance_px: float = 5.0) -> tuple[list[ImageAnchor], tuple[int, int]]:
    """Detect dark circular candidates without assigning component/net identity.

    Tuned for the flat, rendered PRISM PNGs; it is not asserted to generalize to
    physical photographs or rectangular SMD pads.
    """
    if board_side not in {"TOP", "BOTTOM"}:
        raise ValueError("board_side must be TOP or BOTTOM")
    rgb = _load_rgb(image)
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    binary = (gray < threshold_gray).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    candidates = []
    for label in range(1, count):
        x, y, box_width, box_height, area = map(int, stats[label])
        if area < min_area_px2 or area > max_area_px2 or box_width < 3 or box_height < 3:
            continue
        aspect = box_width / box_height
        if aspect < 0.5 or aspect > 2.0 or max(box_width, box_height) > min(width, height) * 0.07:
            continue
        component = (labels[y:y + box_height, x:x + box_width] == label).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        perimeter = float(cv2.arcLength(contours[0], True))
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter)) if perimeter else 0.0
        if circularity < min_circularity:
            continue
        candidates.append((float(centroids[label][0]), float(centroids[label][1]),
                           area, circularity, aspect))
    candidates.sort(key=lambda row: (row[1], row[0], -row[2]))
    merged: list[list[float]] = []
    for candidate in candidates:
        x, y = candidate[:2]
        existing = next((row for row in merged
                         if np.hypot(x - row[0], y - row[1]) < merge_distance_px), None)
        if existing is None:
            merged.append(list(candidate))
        elif candidate[2] > existing[2]:
            existing[:] = candidate
    anchors = [ImageAnchor(f"{photo_id}:{board_side}:dark-round:{index:04d}",
                           photo_id, board_side,
                           (float(row[0]), float(row[1])), int(row[2]),
                           float(row[3]), float(row[4]))
               for index, row in enumerate(sorted(merged, key=lambda row: (row[1], row[0])))]
    return anchors, (width, height)


def _load_rgb(image: np.ndarray | Image.Image | str | Path) -> np.ndarray:
    if isinstance(image, (str, Path)):
        with Image.open(image) as loaded:
            rgb = np.asarray(loaded.convert("RGB"))
    elif isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"))
    else:
        rgb = np.asarray(image)
        if rgb.ndim == 3 and rgb.shape[2] == 3:
            # ndarray API follows RGB convention; callers can explicitly convert.
            pass
    if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.shape[0] < 2 or rgb.shape[1] < 2:
        raise ValueError("Image anchor detection requires a non-empty RGB image")
    return np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)
