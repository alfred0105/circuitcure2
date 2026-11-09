"""Manual PCB-to-photo coordinate registration using an OpenCV homography."""

from __future__ import annotations

import math
from typing import Any, Iterable

import cv2
import numpy as np

from circuitcare.vision.models import ComponentPosition, RegistrationResult


def fit_registration(correspondences: Iterable[dict[str, Any]], *, board_side: str,
                     validation_points: Iterable[dict[str, Any]] = ()) -> RegistrationResult:
    points = list(correspondences)
    validation = list(validation_points)
    if board_side not in {"front", "back"}:
        raise ValueError("정합할 PCB 면은 front 또는 back이어야 합니다.")
    if len(points) < 4:
        raise ValueError("Homography 계산에는 기준점 4개 이상이 필요합니다.")
    references = [str(point.get("reference", "")).strip() for point in points]
    if any(not ref for ref in references) or len(set(references)) != len(references):
        raise ValueError("정합 기준 Reference가 비어 있거나 중복됩니다.")
    source, destination = _coordinate_arrays(points)
    _require_spread(source, "KiCad 설계 좌표")
    _require_spread(destination, "사진 좌표")
    matrix, mask = cv2.findHomography(source, destination, method=0)
    if matrix is None or matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Homography를 계산할 수 없습니다.")
    if abs(float(np.linalg.det(matrix))) < 1e-12 or abs(float(matrix[2, 2])) < 1e-12:
        raise ValueError("계산된 Homography가 특이 행렬입니다.")
    projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected - destination, axis=1)
    rmse = float(math.sqrt(float(np.mean(errors ** 2))))
    validation_result = []
    for point in validation:
        val_source, val_destination = _coordinate_arrays([point])
        estimated = cv2.perspectiveTransform(val_source.reshape(-1, 1, 2), matrix).reshape(-1, 2)[0]
        error = float(np.linalg.norm(estimated - val_destination[0]))
        validation_result.append({**point, "predicted_image_px": estimated.tolist(), "error_px": error})
    return RegistrationResult(
        registration_method="manual_correspondence",
        status="pending_verification",
        correspondences=tuple({**point, "reprojection_error_px": float(error)} for point, error in zip(points, errors)),
        homography=tuple(tuple(float(value) for value in row) for row in matrix),
        reprojection_error_px=rmse,
        validation_points=tuple(validation_result),
        board_side=board_side,
    )


def _coordinate_arrays(points: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    try:
        source = np.asarray([[float(point["x_mm"]), float(point["y_mm"])] for point in points], dtype=np.float64)
        destination = np.asarray([[float(point["u_px"]), float(point["v_px"])] for point in points], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("정합 좌표에 x_mm/y_mm/u_px/v_px 값이 필요합니다.") from exc
    if not np.isfinite(source).all() or not np.isfinite(destination).all():
        raise ValueError("정합 좌표는 유한한 숫자여야 합니다.")
    return source, destination


def _require_spread(points: np.ndarray, label: str) -> None:
    centered = points - points.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if len(singular) < 2 or singular[0] <= 0 or singular[1] / singular[0] < 0.01:
        raise ValueError(f"{label} 기준점이 거의 한 직선 위에 있습니다. 보드 전체에 넓게 배치하세요.")


def map_component_positions(positions: Iterable[ComponentPosition], homography: Any,
                            *, board_side: str, image_size: tuple[int, int],
                            confirmed: bool) -> dict[str, Any]:
    if not confirmed:
        raise ValueError("사용자가 정합을 확인하기 전에는 부품 위치를 활성화할 수 없습니다.")
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError("유효한 3×3 Homography 행렬이 필요합니다.")
    eligible = [item for item in positions if item.side == board_side]
    if not eligible:
        raise ValueError("선택한 앞면/뒷면에 위치한 부품이 없습니다.")
    source = np.asarray([[item.x_mm, item.y_mm] for item in eligible], dtype=np.float64).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(source, matrix).reshape(-1, 2)
    width, height = image_size
    components = []
    outside = []
    for item, (u, v) in zip(eligible, mapped):
        inside = 0 <= u < width and 0 <= v < height
        record = {
            "reference": item.reference,
            "value": item.value,
            "footprint": item.footprint,
            "position_source": item.position_source,
            "position_status": "registered_user_confirmed" if inside else "outside_image",
            "pcb_position_mm": [item.x_mm, item.y_mm],
            "image_position_px": [float(u), float(v)],
            "rotation_deg": item.rotation_deg,
            "board_side": item.side,
        }
        if inside:
            components.append(record)
        else:
            outside.append(record)
    return {
        "board_side": board_side,
        "image_size_px": [width, height],
        "components": components,
        "outside_image": outside,
        "position_source": "kicad_position_csv",
        "status": "registered_user_confirmed",
    }
