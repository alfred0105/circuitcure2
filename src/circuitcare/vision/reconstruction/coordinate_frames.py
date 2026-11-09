"""Explicit coordinate frames and leakage-aware image/design registration."""

from __future__ import annotations

import math
import hashlib
import threading
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Iterable

import cv2
import numpy as np


_CV_RANSAC_LOCK = threading.Lock()


class CoordinateFrame(StrEnum):
    IMAGE_RAW = "IMAGE_RAW"
    IMAGE_RECTIFIED = "IMAGE_RECTIFIED"
    BOARD_CANONICAL = "BOARD_CANONICAL"
    KICAD_DESIGN = "KICAD_DESIGN"
    LAYER_LOCAL = "LAYER_LOCAL"


class TransformType(StrEnum):
    IDENTITY = "IDENTITY"
    AFFINE = "AFFINE"
    HOMOGRAPHY = "HOMOGRAPHY"
    MIRROR = "MIRROR"


class VerificationStatus(StrEnum):
    AUTO_ESTIMATED_UNVERIFIED = "AUTO_ESTIMATED_UNVERIFIED"
    CALIBRATED_NOT_INDEPENDENTLY_VERIFIED = "CALIBRATED_NOT_INDEPENDENTLY_VERIFIED"
    VERIFIED = "VERIFIED"
    REGISTRATION_UNVERIFIED = "REGISTRATION_UNVERIFIED"
    REGISTRATION_FAILED = "REGISTRATION_FAILED"


@dataclass(frozen=True)
class CalibrationPoint:
    point_id: str
    source_xy: tuple[float, float]
    target_xy: tuple[float, float]
    source_frame: CoordinateFrame
    target_frame: CoordinateFrame
    board_side: str
    evidence: str
    role: str = "CALIBRATION"
    review_status: str = "NOT_REVIEWED"

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["source_frame"] = self.source_frame.value
        row["target_frame"] = self.target_frame.value
        return row


@dataclass(frozen=True)
class CoordinateTransform:
    source_frame: CoordinateFrame
    target_frame: CoordinateFrame
    matrix: tuple[tuple[float, ...], ...]
    transform_type: TransformType
    board_side: str
    source_evidence: str
    calibration_point_ids: tuple[str, ...]
    fit_error_px: float | None
    validation_error_px: dict[str, float] | None
    verification_status: VerificationStatus
    inlier_count: int
    inlier_ratio: float
    ransac_threshold_px: float | None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("source_frame", "target_frame", "transform_type", "verification_status"):
            result[key] = getattr(self, key).value
        return result

    def apply(self, points: Iterable[tuple[float, float]], *,
              source_frame: CoordinateFrame, target_frame: CoordinateFrame,
              board_side: str) -> np.ndarray:
        if source_frame != self.source_frame or target_frame != self.target_frame:
            raise ValueError("Coordinate frame mismatch; transform direction must match exactly")
        if board_side != self.board_side:
            raise ValueError("PCB side mismatch; TOP/BOTTOM transforms cannot be interchanged")
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
            raise ValueError("Coordinate transform matrix must be finite, 3x3, and invertible")
        values = _points(points, minimum=0)
        return cv2.perspectiveTransform(values.reshape(1, -1, 2), matrix).reshape(-1, 2)

    def inverse(self) -> "CoordinateTransform":
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
            raise ValueError("Coordinate transform matrix must be finite, 3x3, and invertible")
        inverse = np.linalg.inv(matrix)
        if abs(inverse[2, 2]) > 1e-12:
            inverse /= inverse[2, 2]
        return CoordinateTransform(
            self.target_frame, self.source_frame,
            tuple(tuple(float(v) for v in row) for row in inverse),
            TransformType.MIRROR if self.transform_type == TransformType.MIRROR else self.transform_type,
            self.board_side, f"inverse:{self.source_evidence}", self.calibration_point_ids,
            self.fit_error_px, self.validation_error_px, self.verification_status,
            self.inlier_count, self.inlier_ratio, self.ransac_threshold_px,
        )


def fit_transform(calibration_points: Iterable[CalibrationPoint], *, board_side: str,
                  source_frame: CoordinateFrame, target_frame: CoordinateFrame,
                  validation_points: Iterable[CalibrationPoint] = (),
                  ransac_threshold_px: float = 2.0,
                  acceptance_threshold_px: float | None = None,
                  transform_type: TransformType = TransformType.HOMOGRAPHY) -> tuple[CoordinateTransform, dict[str, Any]]:
    """Fit with calibration points only; score independent points separately.

    Thresholds are explicit experimental configuration, not universal physical
    tolerances. No independent points means calibrated but unverified.
    """
    calibration = list(calibration_points)
    validation = list(validation_points)
    if board_side not in {"TOP", "BOTTOM", "front", "back"}:
        raise ValueError("board_side must explicitly identify TOP/BOTTOM")
    if ransac_threshold_px <= 0 or not math.isfinite(ransac_threshold_px):
        raise ValueError("RANSAC threshold must be a positive finite pixel value")
    if acceptance_threshold_px is not None and (acceptance_threshold_px <= 0 or not math.isfinite(acceptance_threshold_px)):
        raise ValueError("Acceptance threshold must be positive and finite")
    _validate_points(calibration, source_frame, target_frame, board_side, "CALIBRATION")
    _validate_points(validation, source_frame, target_frame, board_side, "VALIDATION")
    calibration_ids = {point.point_id for point in calibration}
    validation_ids = {point.point_id for point in validation}
    if calibration_ids & validation_ids:
        raise ValueError("Calibration and held-out validation point IDs must be disjoint")
    calibration_locations = np.asarray([point.source_xy for point in calibration], dtype=np.float64)
    validation_locations = np.asarray([point.source_xy for point in validation], dtype=np.float64)
    calibration_targets = np.asarray([point.target_xy for point in calibration], dtype=np.float64)
    validation_targets = np.asarray([point.target_xy for point in validation], dtype=np.float64)
    for calibration_xy, validation_xy, frame in (
        (calibration_locations, validation_locations, "source"),
        (calibration_targets, validation_targets, "target"),
    ):
        if len(calibration_xy) and len(validation_xy):
            duplicates = np.linalg.norm(calibration_xy[:, None, :] - validation_xy[None, :, :], axis=2) <= 1e-6
            if duplicates.any():
                raise ValueError(f"Calibration and validation cannot reuse the same physical {frame} location")
    required = 3 if transform_type == TransformType.AFFINE else 4
    if len(calibration) < required:
        raise ValueError(f"{transform_type.value} requires at least {required} calibration points")
    source = _points((point.source_xy for point in calibration), minimum=required)
    target = _points((point.target_xy for point in calibration), minimum=required)
    _ensure_spread(source, "source")
    _ensure_spread(target, "target")
    if transform_type == TransformType.AFFINE:
        affine, inliers = estimate_affine_ransac(source, target, ransac_threshold_px)
        if affine is None:
            raise ValueError("Affine RANSAC could not estimate a transform")
        matrix = np.vstack([affine, [0.0, 0.0, 1.0]])
        inlier_mask = np.asarray(inliers).reshape(-1).astype(bool)
    else:
        with _CV_RANSAC_LOCK:
            cv2.setRNGSeed(_ransac_seed(source, target))
            matrix, inliers = cv2.findHomography(source, target, cv2.RANSAC, ransac_threshold_px)
        if matrix is None or inliers is None:
            raise ValueError("Homography RANSAC could not estimate a transform")
        inlier_mask = np.asarray(inliers).reshape(-1).astype(bool)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Estimated transform is non-finite or singular")
    if abs(matrix[2, 2]) > 1e-12:
        matrix /= matrix[2, 2]
    predicted = cv2.perspectiveTransform(source.reshape(1, -1, 2), matrix).reshape(-1, 2)
    fit_errors = np.linalg.norm(predicted - target, axis=1)
    fit_error = float(np.sqrt(np.mean(fit_errors[inlier_mask] ** 2))) if inlier_mask.any() else None
    val_errors: dict[str, float] = {}
    if validation:
        val_source = _points((point.source_xy for point in validation), minimum=0)
        val_target = _points((point.target_xy for point in validation), minimum=0)
        val_predicted = cv2.perspectiveTransform(val_source.reshape(1, -1, 2), matrix).reshape(-1, 2)
        val_errors = {point.point_id: float(error) for point, error in
                      zip(validation, np.linalg.norm(val_predicted - val_target, axis=1))}
        reviewed = all(point.review_status == "REVIEWED" for point in validation)
        if acceptance_threshold_px is None or not reviewed:
            status = VerificationStatus.CALIBRATED_NOT_INDEPENDENTLY_VERIFIED
        elif max(val_errors.values(), default=float("inf")) <= acceptance_threshold_px:
            status = VerificationStatus.VERIFIED
        else:
            status = VerificationStatus.REGISTRATION_FAILED
    else:
        status = VerificationStatus.CALIBRATED_NOT_INDEPENDENTLY_VERIFIED
    transform = CoordinateTransform(
        source_frame, target_frame, tuple(tuple(float(v) for v in row) for row in matrix),
        transform_type, board_side,
        "manual correspondence points; role/evidence supplied per point",
        tuple(point.point_id for point in calibration), fit_error, val_errors or None,
        status, int(inlier_mask.sum()), float(inlier_mask.mean()), ransac_threshold_px,
    )
    report = registration_metrics(transform, calibration, validation)
    if acceptance_threshold_px is not None:
        report["acceptance_threshold_px"] = acceptance_threshold_px
    return transform, report


def estimate_affine_ransac(source: np.ndarray, target: np.ndarray, threshold_px: float,
                           *, max_iterations: int = 2000, confidence: float = 0.99,
                           refine_iterations: int = 10) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Run OpenCV affine RANSAC with a stable seed and serialized RNG access."""
    with _CV_RANSAC_LOCK:
        cv2.setRNGSeed(_ransac_seed(source, target))
        return cv2.estimateAffine2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=threshold_px,
            maxIters=max_iterations, confidence=confidence, refineIters=refine_iterations,
        )


def estimate_homography_ransac(source: np.ndarray, target: np.ndarray, threshold_px: float,
                               *, max_iterations: int = 5000,
                               confidence: float = 0.995) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Run homography RANSAC with the same stable, serialized RNG policy."""
    with _CV_RANSAC_LOCK:
        cv2.setRNGSeed(_ransac_seed(source, target))
        return cv2.findHomography(source, target, cv2.RANSAC, threshold_px,
                                  maxIters=max_iterations, confidence=confidence)


def _ransac_seed(source: np.ndarray, target: np.ndarray) -> int:
    digest = hashlib.sha256(
        np.asarray(source, dtype="<f8").tobytes() + np.asarray(target, dtype="<f8").tobytes(),
    ).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def registration_metrics(transform: CoordinateTransform, calibration: list[CalibrationPoint],
                         validation: list[CalibrationPoint]) -> dict[str, Any]:
    matrix = np.asarray(transform.matrix, dtype=np.float64)
    inverse = np.linalg.inv(matrix)
    source = _points((point.source_xy for point in calibration + validation), minimum=0)
    target = _points((point.target_xy for point in calibration + validation), minimum=0)
    fwd = cv2.perspectiveTransform(source.reshape(1, -1, 2), matrix).reshape(-1, 2) if len(source) else source
    back = cv2.perspectiveTransform(fwd.reshape(1, -1, 2), inverse).reshape(-1, 2) if len(fwd) else fwd
    consistency = np.linalg.norm(back - source, axis=1) if len(source) else np.asarray([])
    fit = np.asarray([np.linalg.norm(cv2.perspectiveTransform(
        np.asarray([[point.source_xy]], dtype=np.float64), matrix)[0, 0] - point.target_xy)
        for point in calibration], dtype=np.float64)
    val = np.asarray(list((transform.validation_error_px or {}).values()), dtype=np.float64)
    return {
        "source_frame": transform.source_frame.value,
        "target_frame": transform.target_frame.value,
        "board_side": transform.board_side,
        "calibration_point_count": len(calibration),
        "validation_point_count": len(validation),
        "calibration_error_px": _summary(fit),
        "held_out_error_target_units": _summary(val),
        "validation_independently_reviewed": bool(validation) and all(
            point.review_status == "REVIEWED" for point in validation),
        "forward_inverse_consistency_source_units": _summary(consistency),
        "transform_invertible": bool(abs(np.linalg.det(matrix)) > 1e-12),
        "inlier_count": transform.inlier_count,
        "inlier_ratio": transform.inlier_ratio,
        "registration_status": transform.verification_status.value,
        "error_unit_note": "forward and held-out errors use target-frame units; named px only when target is image pixels",
    }


def _validate_points(points: list[CalibrationPoint], source_frame: CoordinateFrame,
                     target_frame: CoordinateFrame, side: str, role: str) -> None:
    ids = [point.point_id for point in points]
    if len(set(ids)) != len(ids) or any(not item for item in ids):
        raise ValueError(f"{role} point IDs must be non-empty and unique")
    for point in points:
        if point.source_frame != source_frame or point.target_frame != target_frame:
            raise ValueError("Calibration point coordinate frames do not match the requested transform")
        if point.board_side != side:
            raise ValueError("TOP/BOTTOM calibration points cannot be mixed")
        if point.role != role:
            raise ValueError(f"Point {point.point_id} has role {point.role}; expected {role}")
        _points([point.source_xy, point.target_xy], minimum=0)
    for coordinates, name in (
        ([point.source_xy for point in points], "source"),
        ([point.target_xy for point in points], "target"),
    ):
        values = np.asarray(coordinates, dtype=np.float64)
        if len(values) > 1:
            distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
            np.fill_diagonal(distances, np.inf)
            if np.any(distances <= 1e-6):
                raise ValueError(f"{role} points cannot duplicate a physical {name} location")


def _points(values: Iterable[tuple[float, float]], *, minimum: int) -> np.ndarray:
    points = np.asarray(list(values), dtype=np.float64)
    if points.size == 0:
        points = np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Coordinates must be finite 2D points")
    if len(points) < minimum:
        raise ValueError("Insufficient coordinate points")
    return points


def _ensure_spread(points: np.ndarray, label: str) -> None:
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    if len(singular) < 2 or singular[0] <= 0 or singular[1] / singular[0] < 0.01:
        raise ValueError(f"Degenerate {label} point geometry: points are nearly collinear")


def _summary(values: np.ndarray) -> dict[str, float | None]:
    if not len(values):
        return {key: None for key in ("mean", "median", "rmse", "p95", "max")}
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "rmse": float(np.sqrt(np.mean(values ** 2))), "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values))}
