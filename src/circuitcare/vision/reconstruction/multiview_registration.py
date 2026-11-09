"""Feature-based same-side registration and explicit front/back transforms."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from circuitcare.vision.reconstruction.coordinate_frames import estimate_homography_ransac
from circuitcare.vision.reconstruction.models import EvidenceStatus, RegisteredView


def register_same_side(reference: Any, moving: Any, *, source_photo_id: str,
                       reference_photo_id: str, board_side: str,
                       min_inliers: int = 12, max_reprojection_error_px: float = 4.0) -> RegisteredView:
    """Estimate moving-image to reference-image homography with ORB/RANSAC."""
    if board_side not in {"front", "back"}:
        raise ValueError("동일 면 정합은 front 또는 back으로 지정해야 합니다.")
    ref, mov = _gray(reference), _gray(moving)
    orb = cv2.ORB_create(nfeatures=2500)
    kp_ref, des_ref = orb.detectAndCompute(ref, None)
    kp_mov, des_mov = orb.detectAndCompute(mov, None)
    if des_ref is None or des_mov is None or len(kp_ref) < 4 or len(kp_mov) < 4:
        return _failed(source_photo_id, reference_photo_id, board_side, "insufficient_features")
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_mov, des_ref, k=2)
    good = [first for pair in matches if len(pair) == 2 for first, second in [pair] if first.distance < 0.72 * second.distance]
    if len(good) < 4:
        return _failed(source_photo_id, reference_photo_id, board_side, "insufficient_matches")
    src = np.float32([kp_mov[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, inliers = estimate_homography_ransac(src, dst, 4.0)
    if matrix is None or inliers is None or not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
        return _failed(source_photo_id, reference_photo_id, board_side, "homography_failed")
    mask = inliers.ravel().astype(bool)
    count = int(mask.sum())
    projected = cv2.perspectiveTransform(src[mask], matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst.reshape(-1, 2)[mask], axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None
    p95 = float(np.percentile(errors, 95)) if len(errors) else None
    inlier_ratio = count / len(good) if good else 0.0
    h, w = ref.shape
    coverage = _coverage(src.reshape(-1, 2)[mask], w, h)
    valid = (count >= min_inliers and inlier_ratio >= 0.2 and rmse is not None
             and rmse <= max_reprojection_error_px and p95 is not None
             and p95 <= max_reprojection_error_px * 2 and coverage >= 0.04)
    return RegisteredView(source_photo_id, reference_photo_id, board_side,
                          tuple(tuple(float(x) for x in row) for row in matrix),
                          "REGISTERED_CANDIDATE" if valid else "REGISTRATION_UNVERIFIED",
                          "ORB_RANSAC_HOMOGRAPHY", count, rmse, coverage,
                          EvidenceStatus.INFERRED if valid else EvidenceStatus.UNKNOWN,
                          len(good), inlier_ratio, p95)


def register_manual_correspondences(reference_points: Any, moving_points: Any, *,
                                    source_photo_id: str, reference_photo_id: str,
                                    board_side: str) -> RegisteredView:
    """Fit an image-to-image transform from four user-matched points."""
    if board_side not in {"front", "back"}:
        raise ValueError("수동 정합은 front 또는 back 면을 지정해야 합니다.")
    reference = np.asarray(reference_points, dtype=np.float32).reshape(-1, 2)
    moving = np.asarray(moving_points, dtype=np.float32).reshape(-1, 2)
    if reference.shape != moving.shape or reference.shape[0] < 4 or not np.isfinite(reference).all() or not np.isfinite(moving).all():
        raise ValueError("사진 정합에는 유한한 대응점 쌍 4개 이상이 필요합니다.")
    _require_spread(reference, "기준 사진")
    _require_spread(moving, "대상 사진")
    matrix, _ = cv2.findHomography(moving, reference, method=0)
    if matrix is None or not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError("입력한 대응점에서 사진 정합을 계산할 수 없습니다.")
    projected = cv2.perspectiveTransform(moving.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.linalg.norm(projected-reference, axis=1) ** 2)))
    coverage = _coverage(reference, int(np.ceil(reference[:, 0].max()+1)),
                         int(np.ceil(reference[:, 1].max()+1)))
    return RegisteredView(source_photo_id, reference_photo_id, board_side,
                          tuple(tuple(float(x) for x in row) for row in matrix),
                          "REGISTERED_CANDIDATE" if rmse <= 4.0 else "REGISTRATION_UNVERIFIED",
                          "USER_MATCHED_FOUR_POINT_HOMOGRAPHY", len(reference), rmse, coverage,
                          EvidenceStatus.INFERRED if rmse <= 4.0 else EvidenceStatus.UNKNOWN)


def back_to_front_transform(*, rotation_degrees: int, mirror_x: bool,
                            board_size_px: tuple[int, int] = (1000, 1000)) -> dict[str, Any]:
    """Return a user-confirmed board-frame transform; never infer it from a flip."""
    if rotation_degrees not in {0, 90, 180, 270}:
        raise ValueError("회전은 0, 90, 180, 270도 중 하나여야 합니다.")
    width, height = map(int, board_size_px)
    if width <= 0 or height <= 0:
        raise ValueError("기준 보드 크기는 양수여야 합니다.")
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    rotation = cv2.getRotationMatrix2D(center, rotation_degrees, 1.0)
    homogeneous = np.vstack([rotation, [0.0, 0.0, 1.0]])
    mirror = np.eye(3)
    if mirror_x:
        mirror[0, 0] = -1.0
        mirror[0, 2] = width - 1
    matrix = mirror @ homogeneous
    return {"matrix": matrix.tolist(), "board_size_px": [width, height], "side_mapping": "back_to_front",
            "rotation_degrees": rotation_degrees, "mirror_x": mirror_x,
            "status": "USER_CONFIRMED_ORIENTATION_REQUIRED",
            "evidence_status": EvidenceStatus.USER_CONFIRMED.value}


def _gray(image: Any) -> np.ndarray:
    array = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 16:
        raise ValueError("정합 이미지에는 16×16 이상의 RGB 이미지가 필요합니다.")
    return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)


def _coverage(points: np.ndarray, width: int, height: int) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return min(1.0, float(cv2.contourArea(hull)) / max(1.0, width * height))


def _require_spread(points: np.ndarray, label: str) -> None:
    centered = points - points.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if len(singular) < 2 or singular[0] <= 0 or singular[1] / singular[0] < 0.01:
        raise ValueError(f"{label} 대응점은 사진 전체에 넓게 배치해야 합니다.")


def _failed(source: str, reference: str, side: str, reason: str) -> RegisteredView:
    return RegisteredView(source, reference, side, None, "REGISTRATION_UNVERIFIED", reason)
