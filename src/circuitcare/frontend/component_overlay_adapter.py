"""Validate backend component geometry before it reaches an interactive viewer."""

from __future__ import annotations

import math
from typing import Any

from circuitcare.frontend.component_overlay_models import (
    ComponentOverlayResult, ComponentOverlayViewModel,
)

VALID_SIDES = {"TOP", "BOTTOM"}
VALID_CANONICAL_FRAME = "MOSAIC_PIXEL"
VALID_ORIGINAL_FRAME = "EXIF_CORRECTED_IMAGE_PIXEL"


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _box(value: Any, *, size: tuple[int, int]) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4 or not all(_number(item) for item in value):
        raise ValueError("Bounding Box 숫자 형식이 올바르지 않습니다.")
    x1, y1, x2, y2 = map(float, value)
    width, height = size
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
        raise ValueError("Bounding Box가 이미지 범위를 벗어났습니다.")
    return x1, y1, x2, y2


def _polygon(value: Any, *, box: tuple[float, float, float, float], size: tuple[int, int]) -> tuple[tuple[float, float], ...]:
    x1, y1, x2, y2 = box
    width, height = size
    if value is None:
        return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValueError("Polygon은 최소 세 점이 필요합니다.")
    points = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2 or not all(_number(v) for v in point):
            raise ValueError("Polygon 좌표 형식이 올바르지 않습니다.")
        x, y = map(float, point)
        if x < 0 or y < 0 or x > width or y > height:
            raise ValueError("Polygon이 이미지 범위를 벗어났습니다.")
        points.append((x, y))
    area = abs(sum(points[i][0] * points[(i + 1) % len(points)][1]
                   - points[(i + 1) % len(points)][0] * points[i][1]
                   for i in range(len(points)))) / 2
    if not math.isfinite(area) or area <= 0:
        raise ValueError("Polygon 면적이 유효하지 않습니다.")
    if (min(p[0] for p in points) < x1 - 1 or max(p[0] for p in points) > x2 + 1
            or min(p[1] for p in points) < y1 - 1 or max(p[1] for p in points) > y2 + 1):
        raise ValueError("Polygon과 Bounding Box 범위가 일치하지 않습니다.")
    return tuple(points)


def _view_model(
    raw: dict[str, Any], *, project_id: str, board_id: str, side: str,
    frame_id: str, coordinate_frame: str, size: tuple[int, int], bbox_key: str,
    polygon_key: str,
) -> ComponentOverlayViewModel:
    component_id = raw.get("component_id")
    component_type = raw.get("component_type", raw.get("class_name"))
    if not isinstance(component_id, str) or not component_id.strip():
        raise ValueError("Component ID가 비어 있습니다.")
    if not isinstance(component_type, str) or not component_type.strip():
        raise ValueError("부품 종류가 없습니다.")
    box = _box(raw.get(bbox_key), size=size)
    polygon = _polygon(raw.get(polygon_key), box=box, size=size)
    confidence = raw.get("detection_confidence", raw.get("confidence"))
    if confidence is not None and (not _number(confidence) or not 0 <= float(confidence) <= 1):
        raise ValueError("검출 신뢰도 범위가 올바르지 않습니다.")
    label = raw.get("reference") or raw.get("ref")
    if label is not None and (not isinstance(label, str) or not label.strip()):
        label = None
    return ComponentOverlayViewModel(
        project_id=project_id, board_id=board_id, board_side=side,
        component_id=component_id, component_type=component_type,
        display_label=label.strip() if isinstance(label, str) else None,
        canonical_frame_id=frame_id, coordinate_frame=coordinate_frame,
        bbox_xyxy=box, polygon_xy=polygon,
        source_backend_photo_ids=tuple(str(item) for item in raw.get("source_image_ids", []) if item),
        observation_ids=tuple(str(item) for item in raw.get("component_observation_ids", []) if item),
        detection_confidence=float(confidence) if confidence is not None else None,
        identification_status=str(raw["identification_status"]) if raw.get("identification_status") is not None else None,
        identity_match_status=str(raw["identity_match_status"]) if raw.get("identity_match_status") is not None else None,
        ocr_marking=str(raw["ocr_marking"]) if raw.get("ocr_marking") is not None else None,
        possible_part_number=str(raw["possible_part_number"]) if raw.get("possible_part_number") is not None else None,
        evidence=(str(raw["evidence"]),) if isinstance(raw.get("evidence"), str) else (),
        overlay_status="DETECTION_CANDIDATE",
    )


def adapt_component_map_components(
    component_map: Any, *, project_id: str, board_id: str, side: str,
) -> ComponentOverlayResult:
    """Adapt only current-side detections in the exact canonical image frame."""
    if side not in VALID_SIDES:
        return ComponentOverlayResult("INVALID_DATA", board_id, side, None, None, None, (), "UNKNOWN", reason="촬영면이 올바르지 않습니다.")
    if not isinstance(component_map, dict) or component_map.get("board_id") != board_id:
        return ComponentOverlayResult("INVALID_DATA", board_id, side, None, None, None, (), "UNKNOWN", reason="보드 식별자가 일치하지 않습니다.")
    side_data = component_map.get("sides", {}).get(side)
    if not isinstance(side_data, dict):
        return ComponentOverlayResult("NO_SIDE_DATA", board_id, side, None, None, None, (), "UNKNOWN")
    canonical = side_data.get("canonical_image")
    if not isinstance(canonical, dict):
        return ComponentOverlayResult("NO_CANONICAL_IMAGE", board_id, side, None, None, None, (), str(side_data.get("detection_status", "UNKNOWN")))
    frame_id = canonical.get("image_id")
    size = canonical.get("image_size_px")
    if (not isinstance(frame_id, str) or not frame_id or canonical.get("coordinate_frame") != VALID_CANONICAL_FRAME
            or not isinstance(size, (list, tuple)) or len(size) != 2
            or any(type(value) is not int or value < 1 for value in size)):
        return ComponentOverlayResult("INVALID_FRAME", board_id, side, None, None, None, (), str(side_data.get("detection_status", "UNKNOWN")), reason="Canonical 좌표 프레임이 유효하지 않습니다.")
    image_size = (size[0], size[1])
    raw_components = side_data.get("components", [])
    if not isinstance(raw_components, list):
        return ComponentOverlayResult("INVALID_DATA", board_id, side, frame_id, VALID_CANONICAL_FRAME, image_size, (), str(side_data.get("detection_status", "UNKNOWN")), reason="부품 목록 형식이 올바르지 않습니다.")
    ids = [row.get("component_id") for row in raw_components if isinstance(row, dict)]
    if len(ids) != len(raw_components) or len(set(ids)) != len(ids):
        return ComponentOverlayResult("INVALID_DATA", board_id, side, frame_id, VALID_CANONICAL_FRAME, image_size, (), str(side_data.get("detection_status", "UNKNOWN")), rejected_count=len(raw_components), reason="중복 Component ID 또는 잘못된 부품 데이터입니다.")
    valid = []
    rejected = 0
    for row in raw_components:
        try:
            valid.append(_view_model(
                row, project_id=project_id, board_id=board_id, side=side,
                frame_id=frame_id, coordinate_frame=VALID_CANONICAL_FRAME,
                size=image_size, bbox_key="bbox_xyxy", polygon_key="polygon_xy",
            ))
        except (ValueError, TypeError):
            rejected += 1
    detector_status = str(side_data.get("detection_status", "UNKNOWN"))
    status = "READY" if valid else "INVALID_DATA" if rejected else "NO_COMPONENTS"
    return ComponentOverlayResult(status, board_id, side, frame_id, VALID_CANONICAL_FRAME,
                                  image_size, tuple(valid), detector_status, rejected)


def adapt_original_photo_components(
    component_map: dict[str, Any], *, project_id: str, board_id: str, side: str,
    frontend_photo_id: str, backend_photo_id: str, image_size: tuple[int, int],
) -> tuple[ComponentOverlayViewModel, ...]:
    """Use only same-photo EXIF-corrected observations; never project mosaic boxes onto a source image."""
    side_data = component_map.get("sides", {}).get(side, {})
    if not isinstance(side_data, dict):
        return ()
    components = side_data.get("components", [])
    observations = side_data.get("component_observations", [])
    by_id = {item.get("observation_id"): item for item in observations
             if isinstance(item, dict) and item.get("observation_id")}
    result = []
    for component in components:
        if not isinstance(component, dict):
            continue
        refs = component.get("component_observation_ids", [])
        for observation_id in refs:
            observation = by_id.get(observation_id)
            if not observation or observation.get("photo_id") != backend_photo_id:
                continue
            if (observation.get("original_coordinate_frame") != VALID_ORIGINAL_FRAME
                    or observation.get("registration_status") not in {"REFERENCE", "REGISTERED_CANDIDATE"}):
                continue
            raw = {**component, "bbox_xyxy": observation.get("original_bbox_xyxy"),
                   "polygon_xy": None, "detection_confidence": observation.get("confidence"),
                   "ocr_marking": observation.get("ocr_marking"),
                   "component_observation_ids": [observation_id],
                   "source_image_ids": [backend_photo_id]}
            try:
                result.append(_view_model(
                    raw, project_id=project_id, board_id=board_id, side=side,
                    frame_id=frontend_photo_id, coordinate_frame=VALID_ORIGINAL_FRAME,
                    size=image_size, bbox_key="bbox_xyxy", polygon_key="polygon_xy",
                ))
            except (ValueError, TypeError):
                continue
    # A source photo can contain at most one observation for a fused component.
    return tuple({item.component_id: item for item in result}.values())
