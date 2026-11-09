"""Design-assisted geometric matching of round-hole image candidates to PRISM anchors."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np

from circuitcare.vision.reconstruction.anchor_detection import ImageAnchor
from circuitcare.vision.reconstruction.coordinate_frames import (
    CalibrationPoint, CoordinateFrame, CoordinateTransform, TransformType,
    VerificationStatus, estimate_affine_ransac, fit_transform,
)


@dataclass(frozen=True)
class DesignAnchor:
    anchor_id: str
    x_mm: float
    y_mm: float
    anchor_kind: str
    component_reference: str | None
    pin_number: str | None
    layer: str | None
    pad_type: str | None
    pad_shape: str | None
    dimensions_mm: tuple[float, ...]
    source_evidence: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class AnchorMatch:
    design_anchor_id: str
    image_anchor_id: str
    residual_px: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class AutomaticMatchResult:
    transform: CoordinateTransform | None
    matches: tuple[AnchorMatch, ...]
    candidate_transform_count: int
    candidate_count: int
    design_anchor_count: int
    image_anchor_count: int
    inlier_ratio: float
    mean_residual_px: float | None
    median_residual_px: float | None
    rmse_residual_px: float | None
    p95_residual_px: float | None
    max_residual_px: float | None
    spatial_coverage: float
    transform_stability_px: float | None
    orientation: str | None
    review_flags: tuple[str, ...]
    status: str
    reason: str | None
    runtime_seconds: float
    pair_product_count: int = 0
    hypotheses_examined: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform": self.transform.to_dict() if self.transform else None,
            "matches": [item.to_dict() for item in self.matches],
            "candidate_match_count": self.candidate_count,
            "candidate_transform_count": self.candidate_transform_count,
            "design_anchor_count": self.design_anchor_count,
            "image_anchor_count": self.image_anchor_count,
            "inlier_count": len(self.matches),
            "inlier_ratio": self.inlier_ratio,
            "mean_residual_px": self.mean_residual_px,
            "median_residual_px": self.median_residual_px,
            "rmse_residual_px": self.rmse_residual_px,
            "p95_residual_px": self.p95_residual_px,
            "max_residual_px": self.max_residual_px,
            "spatial_coverage": self.spatial_coverage,
            "transform_stability_px": self.transform_stability_px,
            "orientation": self.orientation,
            "review_flags": list(self.review_flags),
            "registration_status": self.status,
            "reason": self.reason,
            "runtime_seconds": self.runtime_seconds,
            "pair_product_count": self.pair_product_count,
            "estimated_pair_ratio_bytes": self.pair_product_count * 8,
            "hypotheses_examined": self.hypotheses_examined,
        }


def extract_design_anchors(reference_geometry: dict[str, Any], *, board_side: str) -> list[DesignAnchor]:
    """Extract through-hole pads and vias as design anchors; preserve their provenance."""
    if board_side not in {"TOP", "BOTTOM"}:
        raise ValueError("board_side must be TOP or BOTTOM")
    anchors: list[DesignAnchor] = []
    occupied: set[tuple[int, int]] = set()
    for index, pad in enumerate(reference_geometry.get("pads", [])):
        pad_type = str(pad.get("type", "")).lower()
        if "thru" not in pad_type:
            continue
        side = str(pad.get("side", "both")).upper()
        if side not in {"BOTH", board_side}:
            continue
        x, y = float(pad["x"]), float(pad["y"])
        key = (round(x * 1000), round(y * 1000))
        if key in occupied:
            continue
        occupied.add(key)
        anchors.append(DesignAnchor(
            f"pad:{pad.get('ref','?')}:{pad.get('pin','?')}:{index}",
            x, y, "PAD", str(pad.get("ref") or "") or None,
            str(pad.get("pin") or "") or None, str(pad.get("layers") or "") or None,
            pad_type, str(pad.get("shape") or "") or None,
            tuple(float(value) for value in pad.get("size", [])),
            "PRISM expected_geometry.json; KiCad-derived design reference",
        ))
    for index, via in enumerate(reference_geometry.get("vias", [])):
        point = via.get("at", [])
        if len(point) != 2:
            continue
        x, y = float(point[0]), float(point[1])
        key = (round(x * 1000), round(y * 1000))
        if key in occupied:
            continue
        occupied.add(key)
        anchors.append(DesignAnchor(
            f"via:{index}", x, y, "VIA", None, None,
            str(via.get("layers") or "") or None, "via", "circle",
            (float(via.get("size", 0.0)),),
            "PRISM expected_geometry.json; KiCad-derived design reference",
        ))
    return anchors


def match_anchors(design: list[DesignAnchor], image: list[ImageAnchor], *, board_side: str,
                  image_size_px: tuple[int, int], scale_range: tuple[float, float] = (1.0, 80.0),
                  scale_peak_count: int = 4, scale_tolerance: float = 0.015,
                  match_tolerance_px: float | None = None,
                  max_design_anchors: int = 128,
                  max_image_anchors: int = 128,
                  ransac_threshold_px: float = 2.0,
                  max_pair_products: int = 1_000_000,
                  max_hypotheses: int = 200_000,
                  max_retained_candidates: int = 10,
                  timeout_seconds: float = 20.0) -> AutomaticMatchResult:
    """Search pair-distance consensus, score similarity/mirror hypotheses, then affine-RANSAC refine.

    The design anchors participate in correspondence matching. This is therefore
    DESIGN_ASSISTED_REGISTRATION and cannot be called independent validation.
    """
    started = time.perf_counter()
    if board_side not in {"TOP", "BOTTOM"}:
        raise ValueError("board_side must be TOP or BOTTOM")
    if len({anchor.anchor_id for anchor in design}) != len(design):
        raise ValueError("Design anchor IDs must be unique")
    if len({anchor.anchor_id for anchor in image}) != len(image):
        raise ValueError("Image anchor IDs must be unique")
    if (max_design_anchors < 1 or max_image_anchors < 1
            or max_pair_products < 1 or max_hypotheses < 1 or max_retained_candidates < 1
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("Matching resource budgets must be positive and finite")
    if (not (0 < scale_range[0] < scale_range[1])
            or not all(math.isfinite(value) for value in scale_range)
            or scale_peak_count < 1 or not 0 < scale_tolerance < 1
            or ransac_threshold_px <= 0 or not math.isfinite(ransac_threshold_px)
            or (match_tolerance_px is not None
                and (match_tolerance_px <= 0 or not math.isfinite(match_tolerance_px)))):
        raise ValueError("Invalid matching geometry configuration")
    if len(design) > max_design_anchors:
        return _failure(design, image, started, "RESOURCE_LIMIT_DESIGN_ANCHORS")
    if len(image) > max_image_anchors:
        return _failure(design, image, started, "RESOURCE_LIMIT_IMAGE_ANCHORS")
    if len(image_size_px) != 2 or any(not isinstance(value, (int, np.integer)) or value <= 0
                                      for value in image_size_px):
        raise ValueError("image_size_px must contain positive integer width and height")
    design_pair_count = len(design) * (len(design) - 1) // 2
    image_pair_count = len(image) * (len(image) - 1) // 2
    pair_product_count = design_pair_count * image_pair_count
    if pair_product_count > max_pair_products:
        return _failure(design, image, started, "RESOURCE_LIMIT_PAIR_PRODUCTS",
                        pair_product_count=pair_product_count)
    src = np.asarray([[item.x_mm, item.y_mm] for item in design], dtype=np.float64)
    dst = np.asarray([item.center_px for item in image], dtype=np.float64)
    width, height = image_size_px
    tolerance = float(match_tolerance_px or max(3.5, min(width, height) * 0.006))
    if len(src) < 3 or len(dst) < 3:
        return _failure(design, image, started, "INSUFFICIENT_ANCHORS",
                        pair_product_count=pair_product_count)
    d_pairs, d_dist = _pairs(src)
    i_pairs, i_dist = _pairs(dst)
    if not len(d_dist) or not len(i_dist):
        return _failure(design, image, started, "INSUFFICIENT_NONZERO_PAIR_DISTANCES",
                        pair_product_count=pair_product_count)
    ratios = (i_dist[:, None] / d_dist[None, :]).ravel()
    low, high = scale_range
    ratios = ratios[(ratios >= low) & (ratios <= high) & np.isfinite(ratios)]
    if not len(ratios):
        return _failure(design, image, started, "NO_PLAUSIBLE_PIXEL_PER_MM_SCALE",
                        pair_product_count=pair_product_count)
    log_low, log_high = math.log(low), math.log(high)
    histogram, edges = np.histogram(np.log(ratios), bins=720, range=(log_low, log_high))
    peak_indices = _top_peaks(histogram, count=scale_peak_count, separation=4)
    scales = [math.exp((edges[index] + edges[index + 1]) * 0.5) for index in peak_indices]
    candidates: list[tuple[tuple[float, ...], tuple[tuple[int, float, float], np.ndarray,
                                                    list[tuple[int, int, float]], str]]] = []
    candidate_transform_count = 0
    hypotheses_examined = 0
    for scale in scales:
        for (si, sj), source_distance in zip(d_pairs, d_dist):
            if time.perf_counter() - started >= timeout_seconds:
                return _failure(design, image, started, "RESOURCE_LIMIT_TIMEOUT",
                                candidate_transform_count=candidate_transform_count,
                                pair_product_count=pair_product_count,
                                hypotheses_examined=hypotheses_examined)
            possible = np.flatnonzero(np.abs(i_dist - source_distance * scale)
                                      <= source_distance * scale * scale_tolerance)
            for target_pair_index in possible:
                ti, tj = i_pairs[int(target_pair_index)]
                source_vector = src[sj] - src[si]
                source_length = float(np.linalg.norm(source_vector))
                if source_length < 1e-9:
                    continue
                source_axis = source_vector / source_length
                source_normal = np.array([-source_axis[1], source_axis[0]])
                for first, second in ((ti, tj), (tj, ti)):
                    target_vector = dst[second] - dst[first]
                    target_length = float(np.linalg.norm(target_vector))
                    if target_length < 1e-8:
                        continue
                    target_axis = target_vector / target_length
                    target_normal = np.array([-target_axis[1], target_axis[0]])
                    for handedness in (1.0, -1.0):
                        if hypotheses_examined >= max_hypotheses:
                            return _failure(design, image, started, "RESOURCE_LIMIT_HYPOTHESES",
                                            candidate_transform_count=candidate_transform_count,
                                            pair_product_count=pair_product_count,
                                            hypotheses_examined=hypotheses_examined)
                        if time.perf_counter() - started >= timeout_seconds:
                            return _failure(design, image, started, "RESOURCE_LIMIT_TIMEOUT",
                                            candidate_transform_count=candidate_transform_count,
                                            pair_product_count=pair_product_count,
                                            hypotheses_examined=hypotheses_examined)
                        hypotheses_examined += 1
                        basis = np.column_stack((target_axis, handedness * target_normal)) @ np.column_stack((source_axis, source_normal)).T
                        matrix = basis * (target_length / source_length)
                        translation = dst[first] - matrix @ src[si]
                        affine = np.column_stack((matrix, translation))
                        pairs = _assign(src, dst, affine, tolerance)
                        if len(pairs) < 3:
                            continue
                        refined = _refine(src, dst, pairs, tolerance)
                        if refined is not None:
                            affine, pairs = refined
                        candidate_transform_count += 1
                        residuals = np.asarray([item[2] for item in pairs])
                        coverage = _spatial_coverage(src, [item[0] for item in pairs])
                        score = (len(pairs), coverage, -float(np.median(residuals)))
                        orientation = "MIRROR" if np.linalg.det(affine[:, :2]) < 0 else "ROTATION_SCALE"
                        candidate = (score, affine, pairs, orientation)
                        matrix_key = tuple(np.round(affine.reshape(-1), 6))
                        duplicate_index = next((index for index, (key, _) in enumerate(candidates)
                                                if key == matrix_key), None)
                        if duplicate_index is not None:
                            if score > candidates[duplicate_index][1][0]:
                                candidates[duplicate_index] = (matrix_key, candidate)
                                candidates.sort(key=lambda item: item[1][0], reverse=True)
                        elif len(candidates) < max_retained_candidates:
                            candidates.append((matrix_key, candidate))
                            candidates.sort(key=lambda item: item[1][0], reverse=True)
                        elif score > candidates[-1][1][0]:
                            candidates[-1] = (matrix_key, candidate)
                            candidates.sort(key=lambda item: item[1][0], reverse=True)
    if not candidates:
        return _failure(design, image, started, "NO_GEOMETRICALLY_CONSISTENT_ANCHOR_MATCHES",
                        candidate_transform_count=candidate_transform_count,
                        pair_product_count=pair_product_count,
                        hypotheses_examined=hypotheses_examined)
    candidates.sort(key=lambda item: item[1][0], reverse=True)
    unique = []
    for _, candidate in candidates:
        if all(_matrix_distance(candidate[1], selected[1], src) > 0.5 for selected in unique):
            unique.append(candidate)
        if len(unique) >= max_retained_candidates:
            break
    best_score, _, best_pairs, orientation = unique[0]
    calibration_points = [CalibrationPoint(
        f"auto:{design_index}:{image_index}",
        (float(src[design_index, 0]), float(src[design_index, 1])),
        (float(dst[image_index, 0]), float(dst[image_index, 1])),
        CoordinateFrame.KICAD_DESIGN, CoordinateFrame.IMAGE_RAW, board_side,
        "Automatically matched dark circular image candidate to PRISM design anchor",
        "CALIBRATION",
    ) for design_index, image_index, _ in best_pairs]
    try:
        fitted, _ = fit_transform(
            calibration_points, board_side=board_side,
            source_frame=CoordinateFrame.KICAD_DESIGN, target_frame=CoordinateFrame.IMAGE_RAW,
            transform_type=TransformType.AFFINE,
            ransac_threshold_px=ransac_threshold_px,
        )
        transform_type = TransformType.MIRROR if orientation == "MIRROR" else TransformType.AFFINE
        transform = replace(
            fitted, transform_type=transform_type,
            verification_status=VerificationStatus.AUTO_ESTIMATED_UNVERIFIED,
            source_evidence="DESIGN_ASSISTED: PRISM expected_geometry anchors matched to image-only dark-round candidates",
        )
    except ValueError as exc:
        return _failure(design, image, started, f"DEGENERATE_OR_INVALID_MATCH_GEOMETRY: {exc}",
                        candidate_transform_count=candidate_transform_count,
                        candidate_count=len(best_pairs), pair_product_count=pair_product_count,
                        hypotheses_examined=hypotheses_examined)
    final_matrix = np.asarray(transform.matrix, dtype=np.float64)
    matches = []
    candidate_count = len(best_pairs)
    for design_index, image_index, _ in best_pairs:
        point = np.asarray([[[*src[design_index]]]], dtype=np.float64)
        projected = cv2.perspectiveTransform(point, final_matrix)[0, 0]
        residual = float(np.linalg.norm(projected - dst[image_index]))
        if residual <= ransac_threshold_px:
            matches.append(AnchorMatch(design[design_index].anchor_id, image[image_index].anchor_id,
                                       residual, "GEOMETRICALLY_CONSISTENT"))
    if len(matches) < 3:
        return _failure(design, image, started, "RANSAC_REFINEMENT_LEFT_TOO_FEW_INLIERS",
                        candidate_transform_count=candidate_transform_count,
                        candidate_count=candidate_count, pair_product_count=pair_product_count,
                        hypotheses_examined=hypotheses_examined)
    errors = np.asarray([match.residual_px for match in matches])
    best_residual = -best_score[2]
    stable_candidates = [candidate for candidate in unique
                         if candidate[0][0] == best_score[0]
                         and candidate[0][1] >= best_score[1] - 0.02
                         and -candidate[0][2] <= best_residual + 0.5][:10]
    stability = _transform_stability(stable_candidates, src)
    review_flags = []
    if len(matches) < 4:
        review_flags.append("LOW_MATCH_SUPPORT")
    if _spatial_coverage(src, [item[0] for item in best_pairs]) < 0.10:
        review_flags.append("LOW_SPATIAL_COVERAGE")
    if stability is not None and stability > max(5.0, 0.01 * max(width, height)):
        review_flags.append("AMBIGUOUS_TRANSFORM_HYPOTHESES")
    result = AutomaticMatchResult(
        transform=transform, matches=tuple(matches),
        candidate_transform_count=candidate_transform_count,
        candidate_count=candidate_count, design_anchor_count=len(design),
        image_anchor_count=len(image),
        inlier_ratio=len(matches) / max(1, candidate_count),
        mean_residual_px=float(np.mean(errors)),
        median_residual_px=float(np.median(errors)), p95_residual_px=float(np.percentile(errors, 95)),
        rmse_residual_px=float(np.sqrt(np.mean(np.square(errors)))),
        max_residual_px=float(np.max(errors)),
        spatial_coverage=_spatial_coverage(src, [item[0] for item in best_pairs]),
        transform_stability_px=stability, orientation=orientation,
        review_flags=tuple(review_flags),
        status=VerificationStatus.AUTO_ESTIMATED_UNVERIFIED.value,
        reason="Automatic correspondences are not independent validation; rendered PRISM data only",
        runtime_seconds=time.perf_counter() - started,
        pair_product_count=pair_product_count,
        hypotheses_examined=hypotheses_examined,
    )
    blocking_flags = {"LOW_MATCH_SUPPORT", "LOW_SPATIAL_COVERAGE", "AMBIGUOUS_TRANSFORM_HYPOTHESES"}
    if blocking_flags.intersection(review_flags):
        return replace(
            result, transform=None, status=VerificationStatus.REGISTRATION_FAILED.value,
            reason="UNSTABLE_MATCH_GEOMETRY: " + ", ".join(review_flags),
        )
    return result


def _pairs(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs, distances = [], []
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            distance = float(np.linalg.norm(points[second] - points[first]))
            if distance > 1e-8:
                pairs.append((first, second))
                distances.append(distance)
    return np.asarray(pairs, dtype=np.int32), np.asarray(distances, dtype=np.float64)


def _top_peaks(histogram: np.ndarray, *, count: int, separation: int) -> list[int]:
    work = histogram.astype(np.int64).copy()
    peaks = []
    for _ in range(count):
        index = int(np.argmax(work))
        if work[index] <= 0:
            break
        peaks.append(index)
        work[max(0, index - separation):index + separation + 1] = 0
    return peaks


def _assign(source: np.ndarray, target: np.ndarray, affine: np.ndarray,
            tolerance: float) -> list[tuple[int, int, float]]:
    projected = source @ affine[:, :2].T + affine[:, 2]
    distances = np.linalg.norm(projected[:, None, :] - target[None, :, :], axis=2)
    rows, cols = np.nonzero(distances <= tolerance)
    candidates = sorted((float(distances[i, j]), int(i), int(j)) for i, j in zip(rows, cols))
    used_source: set[int] = set()
    used_target: set[int] = set()
    matches = []
    for distance, i, j in candidates:
        if i in used_source or j in used_target:
            continue
        used_source.add(i)
        used_target.add(j)
        matches.append((i, j, distance))
    return matches


def _refine(source: np.ndarray, target: np.ndarray, matches: list[tuple[int, int, float]],
            tolerance: float) -> tuple[np.ndarray, list[tuple[int, int, float]]] | None:
    current = matches
    affine = None
    for _ in range(3):
        src = np.asarray([source[i] for i, _, _ in current], dtype=np.float32)
        dst = np.asarray([target[j] for _, j, _ in current], dtype=np.float32)
        candidate, inliers = estimate_affine_ransac(src, dst, tolerance)
        if candidate is None or inliers is None or int(inliers.sum()) < 3:
            return None
        affine = candidate.astype(np.float64)
        updated = _assign(source, target, affine, tolerance)
        if len(updated) < 3:
            return None
        stable = {(i, j) for i, j, _ in updated} == {(i, j) for i, j, _ in current}
        current = updated
        if stable:
            break
    return (affine, current) if affine is not None else None


def _spatial_coverage(points: np.ndarray, indices: list[int]) -> float:
    if len(indices) < 3:
        return 0.0
    span = np.ptp(points, axis=0)
    full_area = float(span[0] * span[1])
    if full_area <= 1e-12:
        return 0.0
    hull = cv2.convexHull(points[indices].astype(np.float32))
    return min(1.0, float(cv2.contourArea(hull)) / full_area)


def _matrix_distance(first: np.ndarray, second: np.ndarray, source: np.ndarray) -> float:
    span = np.ptp(source, axis=0)
    corners = np.asarray([[source[:, 0].min(), source[:, 1].min()],
                          [source[:, 0].max(), source[:, 1].min()],
                          [source[:, 0].max(), source[:, 1].max()],
                          [source[:, 0].min(), source[:, 1].max()]], dtype=np.float64)
    one = corners @ first[:, :2].T + first[:, 2]
    two = corners @ second[:, :2].T + second[:, 2]
    return float(np.max(np.linalg.norm(one - two, axis=1))) if len(span) else float("inf")


def _transform_stability(candidates: list[tuple[Any, np.ndarray, Any, Any]],
                         source: np.ndarray) -> float | None:
    if len(candidates) < 2:
        return None
    corners = np.asarray([[source[:, 0].min(), source[:, 1].min()],
                          [source[:, 0].max(), source[:, 1].min()],
                          [source[:, 0].max(), source[:, 1].max()],
                          [source[:, 0].min(), source[:, 1].max()]], dtype=np.float64)
    best = corners @ candidates[0][1][:, :2].T + candidates[0][1][:, 2]
    errors = []
    for _, affine, _, _ in candidates[1:]:
        projected = corners @ affine[:, :2].T + affine[:, 2]
        errors.append(float(np.median(np.linalg.norm(projected - best, axis=1))))
    return float(np.median(errors)) if errors else None


def _failure(design: list[DesignAnchor], image: list[ImageAnchor], started: float,
             reason: str, *, candidate_transform_count: int = 0,
             candidate_count: int = 0, pair_product_count: int = 0,
             hypotheses_examined: int = 0) -> AutomaticMatchResult:
    return AutomaticMatchResult(None, (), candidate_transform_count, candidate_count,
                                len(design), len(image), 0.0, None, None, None, None, None,
                                0.0, None, None, (), "REGISTRATION_FAILED", reason,
                                time.perf_counter() - started, pair_product_count,
                                hypotheses_examined)
