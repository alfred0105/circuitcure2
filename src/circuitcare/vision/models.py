"""Small data records shared by the optional PCB vision tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BoardImage:
    image: Any
    sha256: str
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    scale_xy: tuple[float, float]
    format: str


@dataclass(frozen=True)
class ComponentPosition:
    reference: str
    value: str
    footprint: str
    x_mm: float
    y_mm: float
    rotation_deg: float
    side: str
    position_source: str = "kicad_position_csv"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegistrationResult:
    registration_method: str
    status: str
    correspondences: tuple[dict[str, Any], ...]
    homography: tuple[tuple[float, ...], ...]
    reprojection_error_px: float
    validation_points: tuple[dict[str, Any], ...]
    board_side: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComponentDetection:
    detection_id: str
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    source: str = "trained_object_detector"
    model_id: str = ""
    reference: str | None = None
    identification_status: str = "class_candidate"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
