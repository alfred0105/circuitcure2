"""Local, hash-addressed collection helpers reusing the bounded image loader."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from circuitcare.vision.image_loader import load_board_image
from circuitcare.vision.reconstruction.models import PhotoRecord
from circuitcare.vision.reconstruction.quality import inspect_photo_quality

SIDES = ("front", "back", "unknown")
CAPTURE_TYPES = ("full_view", "close_up", "oblique", "marking_detail")


def collect_photos(uploads: Iterable[Any], *, side_by_name: dict[str, str] | None = None,
                   capture_by_name: dict[str, str] | None = None,
                   max_photos: int = 20) -> list[tuple[PhotoRecord, Any, bytes]]:
    """Decode a bounded multi-upload while preserving each original byte stream."""
    side_by_name = side_by_name or {}
    capture_by_name = capture_by_name or {}
    results = []
    seen: dict[str, str] = {}
    for index, upload in enumerate(uploads):
        if len(results) >= max_photos:
            raise ValueError(f"한 작업에는 사진을 최대 {max_photos}장까지 추가할 수 있습니다.")
        raw = upload.getvalue() if hasattr(upload, "getvalue") else bytes(upload)
        loaded = load_board_image(raw)
        name = str(getattr(upload, "name", "photo"))
        role_key = f"{index}:{name}"
        side = side_by_name.get(role_key, side_by_name.get(name, "unknown"))
        capture = capture_by_name.get(role_key, capture_by_name.get(name, "full_view"))
        if side not in SIDES:
            raise ValueError("PCB 면은 front, back 또는 unknown이어야 합니다.")
        if capture not in CAPTURE_TYPES:
            raise ValueError("지원하지 않는 촬영 유형입니다.")
        photo_id = hashlib.sha256(f"{index}:{loaded.sha256}:{side}:{capture}".encode()).hexdigest()[:20]
        record = PhotoRecord(photo_id, loaded.sha256, side, capture, loaded.original_size,
                             loaded.format, quality=inspect_photo_quality(loaded.image),
                             duplicate_of=seen.get(loaded.sha256))
        seen.setdefault(loaded.sha256, photo_id)
        results.append((record, loaded.image.copy(), raw))
    return results
