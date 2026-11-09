"""Conservative, explainable visual candidates for exposed copper-like regions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from circuitcare.vision.reconstruction.models import CopperObservation, EvidenceStatus, PadViaCandidate


@dataclass(frozen=True)
class CopperSegmentation:
    mask: np.ndarray
    overlay_rgb: np.ndarray
    status: str
    unclassified_fraction: float
    candidate_fraction: float
    notes: tuple[str, ...]


def segment_visible_copper(image: Any, *, board_side: str, board_mask: np.ndarray | None = None,
                           value_threshold: int = 170, saturation_threshold: int = 105) -> CopperSegmentation:
    """Mark local bright, weakly saturated pixels as copper-like candidates.

    This intentionally does not classify hidden copper or certify conductivity.
    """
    rgb = _rgb(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    value, saturation = hsv[:, :, 2], hsv[:, :, 1]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    local = cv2.GaussianBlur(gray, (0, 0), 7)
    local_contrast = cv2.subtract(gray, local)
    candidate = ((value >= value_threshold) & (saturation <= saturation_threshold)
                 & ((local_contrast >= 8) | (value >= 225)))
    if board_mask is not None:
        mask = np.asarray(board_mask).astype(bool)
        if mask.shape != candidate.shape:
            raise ValueError("기판 마스크와 이미지 크기가 다릅니다.")
        candidate &= mask
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(bool)
    overlay = rgb.copy()
    tint = np.zeros_like(overlay)
    tint[:, :, 0] = 30
    tint[:, :, 1] = 220
    overlay[candidate] = (0.55 * overlay[candidate] + 0.45 * tint[candidate]).astype(np.uint8)
    h, w = candidate.shape
    return CopperSegmentation(candidate, overlay, "INFERRED_COPPER_LIKE_CANDIDATES",
                              1.0 - (float(candidate.sum()) / max(1, h * w)),
                              float(candidate.mean()),
                              ("마스크는 밝은 저채도 영역 후보이며 솔더마스크/실크스크린과 혼동될 수 있습니다.",
                               "가려진 배선은 NOT_OBSERVABLE이며 전기적 연결을 의미하지 않습니다.",
                               f"board_side={board_side}"))


def vectorize_copper(mask: np.ndarray, *, photo_id: str, board_side: str,
                     min_component_area: int = 4) -> list[CopperObservation]:
    binary = np.asarray(mask).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError("배선 마스크는 2차원이어야 합니다.")
    binary = (binary > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    observations = []
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if area < min_component_area:
            continue
        region = labels[y:y+h, x:x+w] == label
        area_fraction = float(area / max(1, binary.size))
        aspect_ratio = max(w, h) / max(1, min(w, h))
        kind = ("COPPER_PLANE_CANDIDATE"
                if (area_fraction >= 0.025 and aspect_ratio < 3.0) or area >= 10_000
                else "COPPER_BROAD_REGION_CANDIDATE" if area >= 600
                else "COPPER_TRACE_CANDIDATE")
        # Broad connected fills do not need medial-axis extraction. Skeletonizing
        # a board-sized plane is both wasteful and can dominate image-only runs.
        skip_skeleton = (area >= 8_000 or (area >= 2_000 and min(w, h) >= 12)
                         or (area >= 600 and max(w, h) >= 120))
        if skip_skeleton:
            local_mask = np.zeros_like(binary, dtype=np.uint8)
            local_mask[y:y+h, x:x+w] = region.astype(np.uint8)
            paths = []
            endpoints = junctions = ()
            width_estimate = None
        else:
            center = _skeletonize(region.astype(np.uint8))
            ys, xs = np.where(center)
            if not len(xs):
                continue
            degree = cv2.filter2D(center.astype(np.uint8), cv2.CV_16S,
                                  np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT) - center.astype(np.int16)
            ey, ex = np.where((center > 0) & (degree == 1))
            jy, jx = np.where((center > 0) & (degree >= 3))
            endpoints = tuple((float(x0+x), float(y0+y)) for x0, y0 in zip(ex, ey))
            junctions = tuple((float(x0+x), float(y0+y)) for x0, y0 in zip(jx, jy))
            paths = _trace_chains(center, x, y)
            width_estimate = float(area / max(1, int(center.sum())))
            local_mask = np.zeros_like(binary, dtype=np.uint8)
            local_mask[y:y+h, x:x+w] = region.astype(np.uint8)
        observations.append(CopperObservation(
            observation_id=hashlib.sha256(f"{photo_id}:{board_side}:{label}".encode()).hexdigest()[:20],
            photo_id=photo_id, board_side=board_side, kind=kind,
            mask_size_px=(binary.shape[1], binary.shape[0]),
            mask_rle=tuple(_rle(local_mask.ravel())), polylines_px=tuple(paths),
            endpoints_px=endpoints,
            junctions_px=junctions,
            width_estimate_px=width_estimate, confidence=0.25,
            uncertainty="EXPOSED_COPPER_UNVERIFIED; SILK_OR_GLARE_MAY_MATCH"))
    return observations


def detect_pad_via_candidates(image: Any, *, photo_id: str, board_side: str,
                              copper_mask: np.ndarray | None = None,
                              min_radius_px: float = 3.5, max_candidates: int = 300) -> list[PadViaCandidate]:
    """Return circular bright-feature candidates without claiming plated vias."""
    rgb = _rgb(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    if max_candidates < 1:
        raise ValueError("max_candidates는 양수여야 합니다.")
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(18, int(min_radius_px * 4)),
                               param1=110, param2=32, minRadius=max(3, int(min_radius_px)),
                               maxRadius=max(12, min(80, int(min(rgb.shape[:2]) * 0.045))) )
    candidates = []
    h, w = gray.shape
    circle_points = [] if circles is None else list(np.round(circles[0], 2))
    for index, (cx, cy, radius) in enumerate(circle_points[:max_candidates]):
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        kind = "PAD_OR_VIA_CANDIDATE"
        if copper_mask is not None:
            x, y, r = int(cx), int(cy), int(radius)
            y0, y1 = max(0, y-r), min(h, y+r+1)
            x0, x1 = max(0, x-r), min(w, x+r+1)
            if not np.asarray(copper_mask)[y0:y1, x0:x1].any():
                continue
        candidates.append(PadViaCandidate(
            candidate_id=hashlib.sha256(f"{photo_id}:{board_side}:{index}".encode()).hexdigest()[:20],
            photo_id=photo_id, board_side=board_side, center_px=(float(cx), float(cy)),
            radius_px=float(radius), shape="CIRCULAR", candidate_kind=kind,
            confidence=0.2, through_plated=None, evidence_status=EvidenceStatus.INFERRED))
    if copper_mask is not None:
        binary = (np.asarray(copper_mask) > 0).astype(np.uint8)
        if binary.shape != gray.shape:
            raise ValueError("동박 후보와 이미지 크기가 다릅니다.")
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < max(36.0, min_radius_px**2 * 4):
                continue
            x, y, box_w, box_h = cv2.boundingRect(contour)
            if min(box_w, box_h) < min_radius_px * 2 or max(box_w, box_h) / min(box_w, box_h) > 2.5:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
            if any(np.hypot(cx-item.center_px[0], cy-item.center_px[1]) < min_radius_px*2 for item in candidates):
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / max(perimeter**2, 1.0)
            polygon = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            shape = "CIRCULAR" if circularity > .72 else "RECTANGULAR" if len(polygon) == 4 else "IRREGULAR"
            candidates.append(PadViaCandidate(
                candidate_id=hashlib.sha256(f"{photo_id}:{board_side}:contour:{len(candidates)}".encode()).hexdigest()[:20],
                photo_id=photo_id, board_side=board_side, center_px=(float(cx), float(cy)),
                radius_px=float(max(box_w, box_h) / 2), shape=shape,
                candidate_kind="PAD_OR_VIA_CANDIDATE", confidence=0.2,
                through_plated=None, evidence_status=EvidenceStatus.INFERRED))
            if len(candidates) >= max_candidates:
                break
    return candidates


def _rgb(image: Any) -> np.ndarray:
    array = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("RGB 이미지를 입력하세요.")
    return np.ascontiguousarray(array)


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    current = (binary > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(current)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(current):
        eroded = cv2.erode(current, kernel)
        opened = cv2.dilate(eroded, kernel)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(current, opened))
        current = eroded
    return (skeleton > 0).astype(np.uint8)


def _rle(values: np.ndarray) -> list[int]:
    flat = (values > 0).astype(np.uint8)
    if not len(flat):
        return []
    changes = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], changes, [len(flat)]))
    return [int(bounds[i+1] - bounds[i]) for i in range(len(bounds)-1)]


def _trace_chains(skeleton: np.ndarray, offset_x: int, offset_y: int) -> list[tuple[tuple[float, float], ...]]:
    pixels = {tuple(point) for point in np.argwhere(skeleton > 0)}
    adjacency = {point: [neighbor for neighbor in
                         ((point[0]+dy, point[1]+dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                          if (dx, dy) != (0, 0)) if neighbor in pixels]
                 for point in pixels}
    used: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    paths = []

    def edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        return (a, b) if a <= b else (b, a)

    starts = sorted(point for point, neighbors in adjacency.items() if len(neighbors) != 2)
    for start in starts:
        if not adjacency[start]:
            paths.append(((float(start[1]+offset_x), float(start[0]+offset_y)),))
        for neighbor in adjacency[start]:
            if edge(start, neighbor) in used:
                continue
            chain = [start, neighbor]
            used.add(edge(start, neighbor))
            previous, current = start, neighbor
            while len(adjacency[current]) == 2:
                following = next(item for item in adjacency[current] if item != previous)
                if edge(current, following) in used:
                    break
                used.add(edge(current, following))
                chain.append(following)
                previous, current = current, following
            paths.append(_simplify_chain(chain, offset_x, offset_y))
    # Components with no endpoint/junction are loops. Start at an arbitrary pixel.
    for start in sorted(pixels):
        for neighbor in adjacency[start]:
            if edge(start, neighbor) in used:
                continue
            chain = [start, neighbor]
            used.add(edge(start, neighbor))
            previous, current = start, neighbor
            while True:
                following_options = [item for item in adjacency[current] if item != previous]
                if not following_options:
                    break
                following = following_options[0]
                if edge(current, following) in used:
                    break
                used.add(edge(current, following))
                chain.append(following)
                previous, current = current, following
            paths.append(_simplify_chain(chain, offset_x, offset_y))
    return paths


def _simplify_chain(chain: list[tuple[int, int]], offset_x: int,
                    offset_y: int) -> tuple[tuple[float, float], ...]:
    points = np.asarray([[column + offset_x, row + offset_y] for row, column in chain], dtype=np.float32)
    if len(points) > 2:
        points = cv2.approxPolyDP(points.reshape(-1, 1, 2), 1.5, False).reshape(-1, 2)
    return tuple((float(point[0]), float(point[1])) for point in points)
