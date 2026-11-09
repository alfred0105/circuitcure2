"""Conservative source/display geometry bookkeeping for the F2 viewer."""

from __future__ import annotations

from typing import Any


def requested_view_transform(
    *, source_size: tuple[int, int], display_size: tuple[int, int], active_photo_id: str,
) -> dict[str, Any]:
    """Return requested display geometry, explicitly unverified against browser CSS."""
    source_width, source_height = (int(value) for value in source_size)
    display_width, display_height = (int(value) for value in display_size)
    if (source_width < 1 or source_height < 1 or display_width < 1
            or display_height < 1 or not active_photo_id):
        raise ValueError("이미지 좌표 정보가 올바르지 않습니다.")
    return {
        "active_photo_id": active_photo_id,
        "source_image_width": source_width,
        "source_image_height": source_height,
        "display_width": display_width,
        "display_height": display_height,
        "scale_x": display_width / source_width,
        "scale_y": display_height / source_height,
        "offset_x": 0.0,
        "offset_y": 0.0,
        "coordinate_frame": "source_image_px",
        "coordinate_status": "UNVERIFIED_RESPONSIVE_CSS",
        "display_size_kind": "requested",
    }
