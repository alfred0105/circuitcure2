"""Project design-derived reference geometry after image-only inference."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from circuitcare.vision.reconstruction.coordinate_frames import (
    CoordinateFrame, CoordinateTransform, VerificationStatus,
)


GEOMETRY_KEYS = ("outline", "pads", "vias", "segments", "zones")


def summarize_reference_geometry(reference: dict[str, Any]) -> dict[str, Any]:
    """Report available PRISM/KiCad-derived geometry without implying visibility."""
    result: dict[str, Any] = {}
    for key in GEOMETRY_KEYS:
        value = reference.get(key)
        if key == "outline" and isinstance(value, dict):
            count = int(value.get("point_count", 0) or 0)
            available = count > 0 or bool(value.get("polygon") or value.get("points"))
        else:
            count = len(value) if isinstance(value, (list, dict)) else 0
            available = count > 0
        result[key] = {"status": "AVAILABLE" if available else "NOT_AVAILABLE",
                       "count": count,
                       "source": "PRISM evaluation/reference; KiCad-derived, not native-parser output"}
    result["visual_observability"] = "NOT_ASSESSED"
    result["electrical_connectivity"] = "DESIGN_REFERENCE_ONLY_NOT_PHYSICALLY_VERIFIED"
    return result


def project_reference_geometry(reference: dict[str, Any], transform: CoordinateTransform, *,
                               board_side: str, image_size_px: tuple[int, int]) -> dict[str, Any]:
    """Return image-space design references; require independently verified transform."""
    if transform.verification_status != VerificationStatus.VERIFIED:
        raise ValueError("Reference projection for evaluation requires independently verified registration")
    if transform.source_frame != CoordinateFrame.KICAD_DESIGN or transform.target_frame not in {
        CoordinateFrame.IMAGE_RAW, CoordinateFrame.IMAGE_RECTIFIED,
    }:
        raise ValueError("Transform must map KICAD_DESIGN to an image frame")
    if board_side != transform.board_side:
        raise ValueError("Board side does not match the verified transform")
    width, height = image_size_px
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    matrix = np.asarray(transform.matrix, dtype=np.float64)
    projected: dict[str, Any] = {}
    for key in GEOMETRY_KEYS:
        items = reference.get(key)
        if not items:
            projected[key] = {"status": "NOT_AVAILABLE", "items": []}
            continue
        entries = items if isinstance(items, list) else [items]
        mapped = []
        for item in entries:
            coords = _geometry_points(key, item)
            if len(coords) == 0:
                continue
            image_points = cv2.perspectiveTransform(coords.reshape(1, -1, 2), matrix).reshape(-1, 2)
            record = dict(item) if isinstance(item, dict) else {"source": item}
            record["image_points_px"] = image_points.tolist()
            record["inside_image"] = bool(np.all((image_points[:, 0] >= 0) &
                                                  (image_points[:, 0] < width) &
                                                  (image_points[:, 1] >= 0) &
                                                  (image_points[:, 1] < height)))
            record["reference_source"] = "PRISM_KICAD_DERIVED_REFERENCE"
            record["visual_observability"] = "UNKNOWN_OCCLUSION_NOT_MODELED"
            record["electrical_status"] = "DESIGN_REFERENCE_ONLY"
            mapped.append(record)
        projected[key] = {"status": "PROJECTED_NOT_VISIBILITY_GROUNDED", "items": mapped}
    return {
        "coordinate_frame": transform.target_frame.value,
        "board_side": board_side,
        "image_size_px": [width, height],
        "reference_geometry": projected,
        "visibility_mask": "NOT_AVAILABLE",
        "note": "Design copper is not asserted to be optically visible; occlusion must be annotated separately.",
    }


def rasterize_projected_geometry(projected: dict[str, Any], *,
                                 categories: tuple[str, ...] = ("segments", "pads", "vias", "zones", "outline")) -> dict[str, np.ndarray]:
    """Build separate design-reference masks; never combine them with predictions."""
    width, height = projected["image_size_px"]
    output: dict[str, np.ndarray] = {}
    for category in categories:
        mask = np.zeros((height, width), dtype=np.uint8)
        payload = projected["reference_geometry"].get(category, {})
        if payload.get("status") != "PROJECTED_NOT_VISIBILITY_GROUNDED":
            output[category] = mask
            continue
        for item in payload.get("items", []):
            points = np.rint(item["image_points_px"]).astype(np.int32).reshape(-1, 1, 2)
            if not len(points):
                continue
            if category in {"segments", "zones", "outline"} and len(points) >= 2:
                cv2.polylines(mask, [points], category != "segments", 255, 1, cv2.LINE_8)
            else:
                radius = 2
                cv2.circle(mask, tuple(points[0, 0]), radius, 255, -1)
        output[category] = mask
    return output


def _geometry_points(key: str, item: Any) -> np.ndarray:
    if key == "outline" and isinstance(item, dict):
        item = item.get("polygon", item.get("points", []))
    if not isinstance(item, dict):
        return np.asarray(item, dtype=np.float64).reshape(-1, 2)
    if key == "segments":
        coords = [item.get("start"), item.get("end")]
    elif key == "zones" or key == "outline":
        coords = item.get("polygon", item.get("points", []))
    elif key in {"pads", "vias"}:
        coords = [item.get("at", [item.get("x"), item.get("y")])]
    else:
        coords = []
    coords = [point for point in coords if point is not None]
    points = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(points).all():
        raise ValueError(f"Non-finite coordinates in {key} reference")
    return points
