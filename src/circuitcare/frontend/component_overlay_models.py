"""Frontend-only view models for conservative PCB component overlays."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComponentOverlayViewModel:
    project_id: str
    board_id: str
    board_side: str
    component_id: str
    component_type: str
    display_label: str | None
    canonical_frame_id: str
    coordinate_frame: str
    bbox_xyxy: tuple[float, float, float, float]
    polygon_xy: tuple[tuple[float, float], ...]
    source_backend_photo_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    detection_confidence: float | None
    identification_status: str | None
    identity_match_status: str | None
    ocr_marking: str | None
    possible_part_number: str | None
    evidence: tuple[str, ...]
    overlay_status: str
    fault_candidate_status: str | None = None
    diagnosis_display_status: str | None = None


@dataclass(frozen=True)
class ComponentOverlayResult:
    status: str
    board_id: str
    board_side: str
    canonical_frame_id: str | None
    coordinate_frame: str | None
    image_size_px: tuple[int, int] | None
    components: tuple[ComponentOverlayViewModel, ...]
    detection_status: str
    rejected_count: int = 0
    reason: str | None = None


IDENTITY_STATUS_LABELS = {
    "SUGGESTED_UNCONFIRMED": "미확인 후보",
    "SUGGESTED_CLASS_ONLY": "종류 후보",
    "NEW_COMPONENT": "신규 후보",
    "IDENTITY_MATCHED": "동일 부품 후보",
    "IDENTITY_AMBIGUOUS": "식별 모호",
    "IDENTITY_CONFLICT": "식별 충돌",
    "USER_CONFIRMED": "사용자 확인",
    "user_confirmed": "사용자 확인",
    "suggested": "미확인 후보",
    "unmatched": "미확인",
    "class_candidate": "종류 후보",
}


def display_identity_status(value: str | None) -> str:
    """Translate known internal identity codes while preserving unknown values."""
    return IDENTITY_STATUS_LABELS.get(value or "", value or "미확인")
