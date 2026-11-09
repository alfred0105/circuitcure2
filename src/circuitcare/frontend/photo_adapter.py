"""Convert frontend photo metadata to the backend PhotoRecord boundary."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal, Mapping

from circuitcare.frontend.models import ImageRecord
from circuitcare.vision.reconstruction.models import PhotoRecord

VALID_SIDES = {"front", "back", "unknown"}
VALID_CAPTURE_TYPES = {"full_view", "close_up", "oblique", "marking_detail", "unknown"}
BACKEND_CAPTURE_TYPES = {"full_view", "close_up", "oblique", "marking_detail"}
BACKEND_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


@dataclass(frozen=True)
class BackendPhotoInput:
    """Project ownership plus backend photo record; ownership is not in PhotoRecord."""

    project_id: str
    board_id: str
    source_image_id: str
    source_content_sha256: str
    record: PhotoRecord


@dataclass(frozen=True)
class PhotoAdapterResult:
    status: Literal["READY", "UNSUPPORTED"]
    photo: BackendPhotoInput | None
    reason: str | None = None
    legacy_converted: bool = False


@dataclass(frozen=True)
class ComponentMapPhotoInput:
    """Stable frontend-owned request DTO for the optional Phase 21.2 service."""

    board_id: str
    frontend_photo_id: str
    content_sha256: str
    side: Literal["TOP", "BOTTOM"]
    role: Literal["OVERVIEW", "CLOSEUP", "ANGLED"]
    data: bytes
    name: str


@dataclass(frozen=True)
class ComponentMapAdapterResult:
    status: Literal["READY", "UNSUPPORTED"]
    request: ComponentMapPhotoInput | None
    reason: str | None = None


def normalize_photo_metadata(value: ImageRecord | Mapping[str, Any]) -> tuple[ImageRecord, bool]:
    """Normalize current and legacy intake records without guessing unspecified values."""
    if isinstance(value, ImageRecord):
        raw: Mapping[str, Any] = {
            "image_id": value.image_id,
            "side": value.side,
            "capture_type": value.capture_type,
            "filename": value.filename,
            "content_sha256": value.content_sha256,
            "media_type": value.media_type,
            "image_dimensions": value.image_dimensions,
            "image_format": value.image_format,
        }
        legacy_side_detail = value.side == "detail" and value.capture_type == "unknown"
    else:
        raw = value
        side_value = raw.get("side", raw.get("role", "unknown"))
        legacy_side_detail = side_value == "detail" and "capture_type" not in raw

    side = raw.get("side", raw.get("role", "unknown"))
    capture_type = raw.get("capture_type", "unknown")
    if legacy_side_detail:
        side, capture_type = "unknown", "close_up"
    if side not in VALID_SIDES:
        raise ValueError("사진 면은 front, back 또는 unknown이어야 합니다.")
    if capture_type not in VALID_CAPTURE_TYPES:
        raise ValueError("지원하지 않는 촬영 유형입니다.")

    dimensions = raw.get("image_dimensions")
    if dimensions is None and raw.get("width") and raw.get("height"):
        dimensions = (int(raw["width"]), int(raw["height"]))
    if dimensions is not None:
        if len(dimensions) != 2 or any(int(part) < 1 for part in dimensions):
            raise ValueError("이미지 크기 정보가 올바르지 않습니다.")
        dimensions = (int(dimensions[0]), int(dimensions[1]))
    digest = raw.get("content_sha256", raw.get("sha256"))
    media_type = raw.get("media_type")
    image_format = raw.get("image_format")
    if not image_format:
        image_format = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}.get(media_type)
    return ImageRecord(
        image_id=str(raw.get("image_id") or ""), side=side, capture_type=capture_type,
        filename=raw.get("filename"), content_sha256=str(digest or "") or None,
        media_type=media_type, image_dimensions=dimensions,
        image_format=image_format,
    ), legacy_side_detail


def adapt_photo_record(
    metadata: ImageRecord | Mapping[str, Any], *, project_id: str, board_id: str,
) -> PhotoAdapterResult:
    """Build a backend PhotoRecord or explicitly report unsupported capture metadata.

    `unknown` capture type is retained by frontend intake but cannot be silently
    passed through `collect_photos`, which defaults absent capture types to full_view.
    The caller may still create a project draft; only backend photo processing is withheld.
    """
    image, legacy_converted = normalize_photo_metadata(metadata)
    if not project_id or not board_id:
        raise ValueError("PhotoRecord 변환에는 project_id와 board_id가 필요합니다.")
    if not image.image_id or not image.content_sha256:
        raise ValueError("PhotoRecord 변환에는 image_id와 원본 SHA-256이 필요합니다.")
    if not re.fullmatch(r"[0-9a-f]{64}", image.content_sha256):
        raise ValueError("원본 콘텐츠 SHA-256 형식이 올바르지 않습니다.")
    if not image.image_dimensions or not image.image_format:
        return PhotoAdapterResult(
            "UNSUPPORTED", None, "이미지 크기 또는 실제 포맷 정보가 없습니다.", legacy_converted,
        )
    if image.image_format.upper() not in BACKEND_IMAGE_FORMATS:
        return PhotoAdapterResult(
            "UNSUPPORTED", None, f"지원하지 않는 이미지 포맷: {image.image_format}", legacy_converted,
        )
    if image.capture_type not in BACKEND_CAPTURE_TYPES:
        return PhotoAdapterResult(
            "UNSUPPORTED", None,
            "촬영 유형 미지정: 사진 초안은 유지되지만 백엔드 사진 처리는 보류됩니다.",
            legacy_converted,
        )
    record = PhotoRecord(
        photo_id=image.image_id,
        file_sha256=image.content_sha256,
        side=image.side,
        capture_type=image.capture_type,
        image_size_px=image.image_dimensions,
        image_format=image.image_format.upper(),
    )
    return PhotoAdapterResult(
        "READY",
        BackendPhotoInput(project_id, board_id, image.image_id, image.content_sha256, record),
        legacy_converted=legacy_converted,
    )


def adapt_component_map_input(
    metadata: ImageRecord | Mapping[str, Any], *, board_id: str, data: bytes,
) -> ComponentMapAdapterResult:
    """Map known F1.1 axes to Phase 21.2 enum values without guessing reference roles."""
    image, _ = normalize_photo_metadata(metadata)
    if not board_id or not image.image_id or not image.content_sha256:
        raise ValueError("사진 매핑 요청에는 board_id, image_id 및 SHA-256이 필요합니다.")
    if not re.fullmatch(r"[0-9a-f]{64}", image.content_sha256):
        raise ValueError("원본 콘텐츠 SHA-256 형식이 올바르지 않습니다.")
    if not isinstance(data, bytes) or not data:
        raise ValueError("사진 원본 바이트가 필요합니다.")
    side = {"front": "TOP", "back": "BOTTOM"}.get(image.side)
    role = {
        "full_view": "OVERVIEW", "close_up": "CLOSEUP", "oblique": "ANGLED",
    }.get(image.capture_type)
    if side is None or role is None:
        return ComponentMapAdapterResult(
            "UNSUPPORTED", None,
            "미지정 촬영면·유형 또는 부품 표기는 자동 매핑되지 않습니다.",
        )
    return ComponentMapAdapterResult("READY", ComponentMapPhotoInput(
        board_id=board_id, frontend_photo_id=image.image_id,
        content_sha256=image.content_sha256, side=side, role=role,
        data=data, name=image.filename or image.image_id,
    ))
