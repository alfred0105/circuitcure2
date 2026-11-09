"""Backend service contract for PCB image-to-design registration."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from circuitcare.vision.reconstruction.anchor_detection import ImageAnchor, detect_dark_round_anchors
from circuitcare.vision.reconstruction.anchor_matching import (
    AutomaticMatchResult, DesignAnchor, extract_design_anchors, match_anchors,
)
from circuitcare.vision.reconstruction.coordinate_frames import (
    CalibrationPoint, CoordinateFrame, CoordinateTransform, TransformType,
    VerificationStatus, fit_transform,
)


REGISTRATION_SCHEMA_VERSION = "circuitcare.registration.v1"


class RegistrationErrorCode(StrEnum):
    INVALID_INPUT = "INVALID_INPUT"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    DESIGN_ANCHOR_LIMIT = "DESIGN_ANCHOR_LIMIT"
    IMAGE_ANCHOR_LIMIT = "IMAGE_ANCHOR_LIMIT"
    RESOURCE_LIMIT_PAIR_PRODUCTS = "RESOURCE_LIMIT_PAIR_PRODUCTS"
    RESOURCE_LIMIT_HYPOTHESES = "RESOURCE_LIMIT_HYPOTHESES"
    TIMEOUT = "TIMEOUT"
    UNVERIFIED_TRANSFORM = "UNVERIFIED_TRANSFORM"
    REGISTRATION_FAILED = "REGISTRATION_FAILED"
    INDEPENDENT_VALIDATION_REQUIRED = "INDEPENDENT_VALIDATION_REQUIRED"
    INVALID_VALIDATION = "INVALID_VALIDATION"
    VALIDATION_THRESHOLD_EXCEEDED = "VALIDATION_THRESHOLD_EXCEEDED"


class RegistrationServiceError(ValueError):
    """A malformed request that cannot produce a trustworthy service record."""

    def __init__(self, code: RegistrationErrorCode, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RegistrationBudget:
    max_image_pixels: int = 4_000_000
    max_design_anchors: int = 128
    max_image_anchors: int = 128
    max_pair_products: int = 1_000_000
    max_hypotheses: int = 200_000
    max_retained_candidates: int = 10
    timeout_seconds: float = 20.0
    ransac_threshold_px: float = 2.0

    def __post_init__(self) -> None:
        integer_limits = (self.max_image_pixels, self.max_design_anchors,
                          self.max_image_anchors, self.max_pair_products,
                          self.max_hypotheses, self.max_retained_candidates)
        if any(value < 1 for value in integer_limits):
            raise ValueError("Registration integer budgets must be positive")
        if (not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0
                or not math.isfinite(self.ransac_threshold_px) or self.ransac_threshold_px <= 0):
            raise ValueError("Registration time and RANSAC budgets must be positive and finite")


@dataclass(frozen=True)
class RegistrationRequest:
    board_id: str
    board_side: str
    image: str | Path | Image.Image | np.ndarray
    design_reference_geometry: dict[str, Any]
    registration_mode: str = "AUTO_REGISTRATION"
    calibration_points: tuple[CalibrationPoint, ...] = ()
    validation_points: tuple[CalibrationPoint, ...] = ()
    acceptance_threshold_px: float | None = None
    budget: RegistrationBudget = RegistrationBudget()


@dataclass(frozen=True)
class RegistrationServiceResult:
    board_id: str
    board_side: str
    registration_mode: str
    registration_status: str
    transform: CoordinateTransform | None
    candidate_match_count: int
    inlier_count: int
    inlier_ratio: float
    independent_validation: dict[str, Any]
    provenance: dict[str, Any]
    error_code: str | None
    error_detail: str | None
    runtime_seconds: float
    image_size_px: tuple[int, int] | None = None
    image_anchors: tuple[ImageAnchor, ...] = ()
    design_anchors: tuple[DesignAnchor, ...] = ()
    automatic_match: AutomaticMatchResult | None = None
    fit_metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGISTRATION_SCHEMA_VERSION,
            "board_id": self.board_id,
            "board_side": self.board_side,
            "registration_mode": self.registration_mode,
            "registration_status": self.registration_status,
            "transformation": (self.transform.to_dict() if self.transform else None),
            "coordinate_frame": {
                "source": self.transform.source_frame.value if self.transform else "KICAD_DESIGN",
                "target": self.transform.target_frame.value if self.transform else "IMAGE_RAW",
                "units": {"source": "mm", "target": "px"},
            },
            "matching": {
                "candidate_count": self.candidate_match_count,
                "inlier_count": self.inlier_count,
                "inlier_ratio": self.inlier_ratio,
                "details": self.automatic_match.to_dict() if self.automatic_match else None,
            },
            "independent_validation": self.independent_validation,
            "provenance": self.provenance,
            "error_code": self.error_code,
            "error_detail": self.error_detail,
            "runtime_seconds": self.runtime_seconds,
            "resources": {
                "image_size_px": list(self.image_size_px) if self.image_size_px else None,
                "design_anchor_count": len(self.design_anchors),
                "image_anchor_count": len(self.image_anchors),
                "pair_product_count": (self.automatic_match.pair_product_count
                                        if self.automatic_match else 0),
                "estimated_pair_ratio_bytes": (self.automatic_match.pair_product_count * 8
                                                if self.automatic_match else 0),
                "hypotheses_examined": (self.automatic_match.hypotheses_examined
                                         if self.automatic_match else 0),
                "limits": self.provenance.get("resource_limits", {}),
            },
            "fit_metrics": self.fit_metrics,
        }


def register(request: RegistrationRequest) -> RegistrationServiceResult:
    """Run one bounded registration request and return a stable serializable record.

    Automatic matches are always design-assisted. They stay unverified unless
    separate, reviewed, disjoint validation points and a predeclared threshold
    are supplied.
    """
    started = time.perf_counter()
    _validate_request(request)
    image_size = _image_size(request.image)
    if image_size[0] * image_size[1] > request.budget.max_image_pixels:
        return _failed(request, started, RegistrationErrorCode.IMAGE_TOO_LARGE,
                       f"Image has {image_size[0] * image_size[1]} pixels; limit is {request.budget.max_image_pixels}",
                       image_size_px=image_size)
    image_anchors: list[ImageAnchor] = []
    design_anchors: list[DesignAnchor] = []
    auto: AutomaticMatchResult | None = None
    transform: CoordinateTransform | None = None
    fit_metrics: dict[str, Any] | None = None
    calibration_points = list(request.calibration_points)
    validation = {"status": "BLOCKED_NO_INDEPENDENT_VALIDATION",
                  "point_count": len(request.validation_points), "metrics": None,
                  "acceptance_threshold_px": request.acceptance_threshold_px}
    error_code: RegistrationErrorCode | None = None
    error_detail: str | None = None

    if request.registration_mode == "AUTO_REGISTRATION":
        try:
            raw_pad_count = sum("thru" in str(pad.get("type", "")).lower()
                                for pad in request.design_reference_geometry.get("pads", [])
                                if isinstance(pad, dict))
            raw_via_count = len(request.design_reference_geometry.get("vias", []))
            if raw_pad_count + raw_via_count > request.budget.max_design_anchors:
                return _failed(request, started, RegistrationErrorCode.DESIGN_ANCHOR_LIMIT,
                               "Design through-hole/via candidates exceed the configured anchor limit",
                               image_size_px=image_size)
            image_anchors, _ = detect_dark_round_anchors(
                request.image, photo_id=f"{request.board_id}-{request.board_side.lower()}",
                board_side=request.board_side,
            )
            design_anchors = extract_design_anchors(
                request.design_reference_geometry, board_side=request.board_side,
            )
        except RegistrationServiceError:
            raise
        except (OSError, ValueError, TypeError, KeyError) as exc:
            return _failed(request, started, RegistrationErrorCode.INVALID_INPUT,
                           f"{type(exc).__name__}: {exc}", image_size_px=image_size)
        if len(design_anchors) > request.budget.max_design_anchors:
            return _failed(request, started, RegistrationErrorCode.DESIGN_ANCHOR_LIMIT,
                           f"Found {len(design_anchors)} design anchors; limit is {request.budget.max_design_anchors}",
                           image_size_px=image_size, image_anchors=image_anchors,
                           design_anchors=design_anchors)
        if len(image_anchors) > request.budget.max_image_anchors:
            return _failed(request, started, RegistrationErrorCode.IMAGE_ANCHOR_LIMIT,
                           f"Found {len(image_anchors)} image anchors; limit is {request.budget.max_image_anchors}",
                           image_size_px=image_size, image_anchors=image_anchors,
                           design_anchors=design_anchors)
        if time.perf_counter() - started >= request.budget.timeout_seconds:
            return _failed(request, started, RegistrationErrorCode.TIMEOUT,
                           "Detection and design extraction exceeded the request time budget",
                           image_size_px=image_size, image_anchors=image_anchors,
                           design_anchors=design_anchors)
        auto = match_anchors(
            design_anchors, image_anchors, board_side=request.board_side,
            image_size_px=image_size,
            max_design_anchors=request.budget.max_design_anchors,
            max_image_anchors=request.budget.max_image_anchors,
            ransac_threshold_px=request.budget.ransac_threshold_px,
            max_pair_products=request.budget.max_pair_products,
            max_hypotheses=request.budget.max_hypotheses,
        max_retained_candidates=request.budget.max_retained_candidates,
            timeout_seconds=max(0.001, request.budget.timeout_seconds - (time.perf_counter() - started)),
        )
        transform = auto.transform
        if auto.status == VerificationStatus.REGISTRATION_FAILED.value:
            error_code = _error_code_for_reason(auto.reason)
            error_detail = auto.reason
        elif transform is not None:
            calibration_points = _points_from_auto_matches(auto, design_anchors, image_anchors,
                                                            request.board_side)
            if request.validation_points:
                try:
                    validation, transform = _validate_auto_transform(
                        transform, calibration_points, request.validation_points,
                        request.acceptance_threshold_px, image_size,
                    )
                    auto = replace(auto, transform=transform, status=transform.verification_status.value,
                                   reason=validation.get("reason") or auto.reason)
                    if transform.verification_status == VerificationStatus.REGISTRATION_FAILED:
                        error_code = RegistrationErrorCode.VALIDATION_THRESHOLD_EXCEEDED
                        error_detail = validation.get("reason")
                    elif transform.verification_status != VerificationStatus.VERIFIED:
                        error_code = RegistrationErrorCode.INDEPENDENT_VALIDATION_REQUIRED
                        error_detail = validation.get("reason") or validation["status"]
                except RegistrationServiceError as exc:
                    validation = {"status": "INVALID_VALIDATION", "point_count": len(request.validation_points),
                                  "metrics": None, "acceptance_threshold_px": request.acceptance_threshold_px,
                                  "reason": str(exc)}
                    error_code, error_detail = exc.code, str(exc)
                    transform = replace(transform, verification_status=VerificationStatus.AUTO_ESTIMATED_UNVERIFIED)
                    auto = replace(auto, transform=transform,
                                   status=VerificationStatus.AUTO_ESTIMATED_UNVERIFIED.value)
            if transform.verification_status == VerificationStatus.AUTO_ESTIMATED_UNVERIFIED:
                error_code = error_code or RegistrationErrorCode.INDEPENDENT_VALIDATION_REQUIRED
                error_detail = error_detail or "Automatic estimate has no independently reviewed validation set"
    else:
        try:
            if request.validation_points and len(request.validation_points) < 4:
                raise RegistrationServiceError(
                    RegistrationErrorCode.INVALID_VALIDATION,
                    "At least four independent validation points are required",
                )
            transform, fit_metrics = fit_transform(
                calibration_points, board_side=request.board_side,
                source_frame=CoordinateFrame.KICAD_DESIGN, target_frame=CoordinateFrame.IMAGE_RAW,
                validation_points=list(request.validation_points),
                ransac_threshold_px=request.budget.ransac_threshold_px,
                acceptance_threshold_px=request.acceptance_threshold_px,
                transform_type=TransformType.HOMOGRAPHY,
            )
            if request.validation_points:
                validation = {"status": ("VERIFIED" if transform.verification_status == VerificationStatus.VERIFIED
                                         else "BLOCKED_VALIDATION_NOT_REVIEWED_OR_NO_THRESHOLD"),
                              "point_count": len(request.validation_points),
                              "metrics": fit_metrics["held_out_error_target_units"],
                              "validation_independently_reviewed": fit_metrics["validation_independently_reviewed"],
                              "acceptance_threshold_px": request.acceptance_threshold_px}
            if transform.verification_status == VerificationStatus.REGISTRATION_FAILED:
                error_code = RegistrationErrorCode.VALIDATION_THRESHOLD_EXCEEDED
                error_detail = "Held-out validation error exceeded the predeclared threshold"
            elif transform.verification_status != VerificationStatus.VERIFIED:
                error_code = RegistrationErrorCode.INDEPENDENT_VALIDATION_REQUIRED
                error_detail = "Manual calibration lacks reviewed held-out validation and a predeclared threshold"
        except ValueError as exc:
            if isinstance(exc, RegistrationServiceError):
                raise
            raise RegistrationServiceError(RegistrationErrorCode.INVALID_INPUT, str(exc)) from exc

    status = transform.verification_status.value if transform else VerificationStatus.REGISTRATION_FAILED.value
    if status == VerificationStatus.REGISTRATION_FAILED.value and error_code is None:
        error_code = RegistrationErrorCode.REGISTRATION_FAILED
    provenance = {
        "track": "DESIGN_ASSISTED_REGISTRATION",
        "design_geometry_used_for_matching": request.registration_mode == "AUTO_REGISTRATION",
        "image_detector": ("dark connected components; gray < 28; circularity >= 0.35"
                           if request.registration_mode == "AUTO_REGISTRATION" else "NOT_USED_MANUAL_CORRESPONDENCES"),
        "independent_validation_data_supplied": bool(request.validation_points),
        "source_evidence": ("PRISM/KiCad-derived design reference plus rendered PRISM image"
                            if request.registration_mode == "AUTO_REGISTRATION"
                            else "User-supplied calibration/validation correspondences"),
        "resource_limits": _budget_dict(request.budget),
    }
    return RegistrationServiceResult(
        request.board_id, request.board_side, request.registration_mode, status, transform,
        auto.candidate_count if auto else len(calibration_points),
        len(auto.matches) if auto else (transform.inlier_count if transform else 0),
        auto.inlier_ratio if auto else (transform.inlier_ratio if transform else 0.0),
        validation, provenance, error_code.value if error_code else None, error_detail,
        time.perf_counter() - started, image_size, tuple(image_anchors), tuple(design_anchors),
        auto, fit_metrics,
    )


def _validate_request(request: RegistrationRequest) -> None:
    if not isinstance(request.board_id, str) or not request.board_id.strip():
        raise RegistrationServiceError(RegistrationErrorCode.INVALID_INPUT, "board_id is required")
    if request.board_side not in {"TOP", "BOTTOM"}:
        raise RegistrationServiceError(RegistrationErrorCode.INVALID_INPUT, "board_side must be TOP or BOTTOM")
    if request.registration_mode not in {"AUTO_REGISTRATION", "MANUAL_CALIBRATION"}:
        raise RegistrationServiceError(RegistrationErrorCode.INVALID_INPUT, "Unsupported registration mode")
    if not isinstance(request.design_reference_geometry, dict):
        raise RegistrationServiceError(RegistrationErrorCode.INVALID_INPUT, "design_reference_geometry must be an object")
    if request.acceptance_threshold_px is not None and (
        not math.isfinite(request.acceptance_threshold_px) or request.acceptance_threshold_px <= 0
    ):
        raise RegistrationServiceError(RegistrationErrorCode.INVALID_INPUT,
                                       "acceptance_threshold_px must be positive and finite")


def _image_size(image: str | Path | Image.Image | np.ndarray) -> tuple[int, int]:
    try:
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                size = opened.size
        elif isinstance(image, Image.Image):
            size = image.size
        elif isinstance(image, np.ndarray) and image.ndim in {2, 3}:
            size = (int(image.shape[1]), int(image.shape[0]))
        else:
            raise RegistrationServiceError(RegistrationErrorCode.INVALID_INPUT, "Unsupported image input")
    except (OSError, ValueError) as exc:
        raise RegistrationServiceError(RegistrationErrorCode.INVALID_INPUT,
                                       f"Could not inspect image: {exc}") from exc
    if len(size) != 2 or any(value <= 0 for value in size):
        raise RegistrationServiceError(RegistrationErrorCode.INVALID_INPUT, "Image dimensions must be positive")
    return int(size[0]), int(size[1])


def _points_from_auto_matches(result: AutomaticMatchResult, design: list[DesignAnchor],
                              image: list[ImageAnchor], side: str) -> list[CalibrationPoint]:
    design_by_id = {anchor.anchor_id: anchor for anchor in design}
    image_by_id = {anchor.anchor_id: anchor for anchor in image}
    points = []
    for match in result.matches:
        source = design_by_id[match.design_anchor_id]
        target = image_by_id[match.image_anchor_id]
        points.append(CalibrationPoint(
            f"{source.anchor_id}|{target.anchor_id}", (source.x_mm, source.y_mm), target.center_px,
            CoordinateFrame.KICAD_DESIGN, CoordinateFrame.IMAGE_RAW, side,
            "DESIGN_ASSISTED automatic anchor match", "CALIBRATION",
        ))
    return points


def _validate_auto_transform(transform: CoordinateTransform, calibration: list[CalibrationPoint],
                             validation: tuple[CalibrationPoint, ...], threshold: float | None,
                             image_size: tuple[int, int]) -> tuple[dict[str, Any], CoordinateTransform]:
    if len(validation) < 4:
        raise RegistrationServiceError(RegistrationErrorCode.INVALID_VALIDATION,
                                       "At least four independent validation points are required")
    ids = [point.point_id for point in validation]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise RegistrationServiceError(RegistrationErrorCode.INVALID_VALIDATION,
                                       "Validation point IDs must be non-empty and unique")
    source_positions = np.asarray([point.source_xy for point in validation], dtype=np.float64)
    target_positions = np.asarray([point.target_xy for point in validation], dtype=np.float64)
    for positions, label in ((source_positions, "source"), (target_positions, "target")):
        distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        if np.any(distances <= 1e-6):
            raise RegistrationServiceError(RegistrationErrorCode.INVALID_VALIDATION,
                                           f"Validation points duplicate a physical {label} location")
    calibration_ids = {point.point_id for point in calibration}
    if calibration_ids.intersection(ids):
        raise RegistrationServiceError(RegistrationErrorCode.INVALID_VALIDATION,
                                       "Validation point IDs overlap matched calibration IDs")
    for point in validation:
        if (point.source_frame != CoordinateFrame.KICAD_DESIGN
                or point.target_frame != CoordinateFrame.IMAGE_RAW
                or point.board_side != transform.board_side or point.role != "VALIDATION"
                or not point.evidence.strip()):
            raise RegistrationServiceError(RegistrationErrorCode.INVALID_VALIDATION,
                                           f"Validation point {point.point_id} has invalid frame, side, role, or evidence")
        if not all(math.isfinite(value) for value in (*point.source_xy, *point.target_xy)):
            raise RegistrationServiceError(RegistrationErrorCode.INVALID_VALIDATION,
                                           f"Validation point {point.point_id} has non-finite coordinates")
        if not (0 <= point.target_xy[0] < image_size[0] and 0 <= point.target_xy[1] < image_size[1]):
            raise RegistrationServiceError(RegistrationErrorCode.INVALID_VALIDATION,
                                           f"Validation point {point.point_id} falls outside the image")
        for matched in calibration:
            if (np.linalg.norm(np.asarray(point.source_xy) - matched.source_xy) <= 1e-6
                    or np.linalg.norm(np.asarray(point.target_xy) - matched.target_xy) <= 1e-6):
                raise RegistrationServiceError(RegistrationErrorCode.INVALID_VALIDATION,
                                               "Validation point reuses a calibration source or target location")
    predicted = transform.apply(
        [point.source_xy for point in validation], source_frame=CoordinateFrame.KICAD_DESIGN,
        target_frame=CoordinateFrame.IMAGE_RAW, board_side=transform.board_side,
    )
    errors = np.linalg.norm(predicted - np.asarray([point.target_xy for point in validation]), axis=1)
    metrics = {"unit": "px", "mean": float(np.mean(errors)), "median": float(np.median(errors)),
               "rmse": float(np.sqrt(np.mean(np.square(errors)))),
               "p95": float(np.percentile(errors, 95)), "max": float(np.max(errors))}
    reviewed = all(point.review_status == "REVIEWED" for point in validation)
    if not reviewed:
        status = VerificationStatus.CALIBRATED_NOT_INDEPENDENTLY_VERIFIED
        validation_status = "BLOCKED_VALIDATION_NOT_REVIEWED"
    elif threshold is None:
        status = VerificationStatus.CALIBRATED_NOT_INDEPENDENTLY_VERIFIED
        validation_status = "BLOCKED_NO_PREDECLARED_THRESHOLD"
    elif metrics["max"] <= threshold:
        status = VerificationStatus.VERIFIED
        validation_status = "VERIFIED"
    else:
        status = VerificationStatus.REGISTRATION_FAILED
        validation_status = "INDEPENDENT_VALIDATION_FAILED"
    scored_transform = replace(
        transform, verification_status=status,
        validation_error_px={point.point_id: float(error) for point, error in zip(validation, errors)},
    )
    detail = {"status": validation_status, "point_count": len(validation), "metrics": metrics,
              "validation_independently_reviewed": reviewed, "acceptance_threshold_px": threshold}
    if status == VerificationStatus.REGISTRATION_FAILED:
        detail["reason"] = "HELD_OUT_MAX_ERROR_EXCEEDS_PREDECLARED_THRESHOLD"
    return detail, scored_transform


def _error_code_for_reason(reason: str | None) -> RegistrationErrorCode:
    value = reason or ""
    if "RESOURCE_LIMIT_PAIR_PRODUCTS" in value:
        return RegistrationErrorCode.RESOURCE_LIMIT_PAIR_PRODUCTS
    if "RESOURCE_LIMIT_HYPOTHESES" in value:
        return RegistrationErrorCode.RESOURCE_LIMIT_HYPOTHESES
    if "RESOURCE_LIMIT_TIMEOUT" in value:
        return RegistrationErrorCode.TIMEOUT
    if "RESOURCE_LIMIT_DESIGN_ANCHORS" in value:
        return RegistrationErrorCode.DESIGN_ANCHOR_LIMIT
    if "RESOURCE_LIMIT_IMAGE_ANCHORS" in value:
        return RegistrationErrorCode.IMAGE_ANCHOR_LIMIT
    return RegistrationErrorCode.REGISTRATION_FAILED


def _budget_dict(budget: RegistrationBudget) -> dict[str, Any]:
    return {"max_image_pixels": budget.max_image_pixels,
            "max_design_anchors": budget.max_design_anchors,
            "max_image_anchors": budget.max_image_anchors,
            "max_pair_products": budget.max_pair_products,
            "max_hypotheses": budget.max_hypotheses,
            "max_retained_candidates": budget.max_retained_candidates,
            "timeout_seconds": budget.timeout_seconds,
            "ransac_threshold_px": budget.ransac_threshold_px}


def _failed(request: RegistrationRequest, started: float, code: RegistrationErrorCode,
            detail: str, *, image_size_px: tuple[int, int] | None = None,
            image_anchors: list[ImageAnchor] | None = None,
            design_anchors: list[DesignAnchor] | None = None) -> RegistrationServiceResult:
    return RegistrationServiceResult(
        request.board_id, request.board_side, request.registration_mode,
        VerificationStatus.REGISTRATION_FAILED.value, None, 0, 0, 0.0,
        {"status": "NOT_EVALUATED", "point_count": len(request.validation_points),
         "metrics": None, "acceptance_threshold_px": request.acceptance_threshold_px},
        {"track": "DESIGN_ASSISTED_REGISTRATION",
         "design_geometry_used_for_matching": request.registration_mode == "AUTO_REGISTRATION",
         "independent_validation_data_supplied": bool(request.validation_points),
         "resource_limits": _budget_dict(request.budget)},
        code.value, detail, time.perf_counter() - started, image_size_px,
        tuple(image_anchors or ()), tuple(design_anchors or ()), None, None,
    )


def require_verified_transform(transform: CoordinateTransform) -> CoordinateTransform:
    """Fail closed before using a transform for labels/electrical decisions."""
    if transform.verification_status != VerificationStatus.VERIFIED:
        raise RegistrationServiceError(
            RegistrationErrorCode.UNVERIFIED_TRANSFORM,
            f"Transform status {transform.verification_status.value} is not safe for downstream use",
        )
    matrix = np.asarray(transform.matrix, dtype=np.float64)
    if (transform.source_frame != CoordinateFrame.KICAD_DESIGN
            or transform.target_frame != CoordinateFrame.IMAGE_RAW
            or transform.board_side not in {"TOP", "BOTTOM"}
            or matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or abs(np.linalg.det(matrix)) < 1e-12):
        raise RegistrationServiceError(RegistrationErrorCode.UNVERIFIED_TRANSFORM,
                                       "Verified transform has invalid frames, side, or matrix")
    return transform
