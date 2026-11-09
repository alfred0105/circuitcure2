"""Typed records for PCB photo observations; no electrical net is implied."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class EvidenceStatus(StrEnum):
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    USER_CONFIRMED = "USER_CONFIRMED"
    DESIGN_VERIFIED = "DESIGN_VERIFIED"
    CONTINUITY_VERIFIED = "CONTINUITY_VERIFIED"
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True)
class PhotoRecord:
    photo_id: str
    file_sha256: str
    side: str
    capture_type: str
    image_size_px: tuple[int, int]
    image_format: str
    quality: dict[str, Any] = field(default_factory=dict)
    duplicate_of: str | None = None
    provenance_status: str = "ORIGINAL_BYTES_PRESERVED_IN_SESSION"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoardGeometryObservation:
    photo_id: str
    board_side: str
    boundary_polygon_px: tuple[tuple[float, float], ...] | None
    orientation_degrees: int | None
    frame_id: str
    verification_status: EvidenceStatus
    source_method: str


@dataclass(frozen=True)
class DiagnosticEvidence:
    evidence_id: str
    evidence_kind: str
    candidate_reference: str | None
    summary: str
    source_photo_ids: tuple[str, ...]
    verification_status: EvidenceStatus
    linked_design_id: str | None = None
    measurement_id: str | None = None


@dataclass(frozen=True)
class RegisteredView:
    source_photo_id: str
    reference_photo_id: str
    board_side: str
    transformation: tuple[tuple[float, ...], ...] | None
    registration_status: str
    method: str
    inlier_count: int = 0
    reprojection_error_px: float | None = None
    overlap_ratio: float = 0.0
    evidence_status: EvidenceStatus = EvidenceStatus.UNKNOWN
    candidate_match_count: int | None = None
    inlier_ratio: float | None = None
    p95_reprojection_error_px: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CopperObservation:
    observation_id: str
    photo_id: str
    board_side: str
    kind: str
    mask_size_px: tuple[int, int]
    mask_rle: tuple[int, ...]
    polylines_px: tuple[tuple[tuple[float, float], ...], ...]
    endpoints_px: tuple[tuple[float, float], ...]
    junctions_px: tuple[tuple[float, float], ...]
    width_estimate_px: float | None
    confidence: float
    uncertainty: str
    evidence_status: EvidenceStatus = EvidenceStatus.INFERRED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PadViaCandidate:
    candidate_id: str
    photo_id: str
    board_side: str
    center_px: tuple[float, float]
    radius_px: float
    shape: str
    candidate_kind: str
    confidence: float
    through_plated: bool | None = None
    evidence_status: EvidenceStatus = EvidenceStatus.INFERRED


@dataclass(frozen=True)
class ConnectivityEvidence:
    source_feature_id: str
    target_feature_id: str
    relationship: str
    evidence_type: str
    verification_status: EvidenceStatus
    source_photo_ids: tuple[str, ...]
    same_side_contact: bool


@dataclass(frozen=True)
class ObservedTopologyGraph:
    nodes: tuple[dict[str, Any], ...]
    edges: tuple[ConnectivityEvidence, ...]
    status: str = "OBSERVED_TOPOLOGY_ONLY"
