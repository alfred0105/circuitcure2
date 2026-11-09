"""UI-owned contracts, intentionally independent of backend domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

VerificationStatus = Literal["VERIFIED", "UNVERIFIED", "DEMO_ONLY", "UNAVAILABLE"]
DiagnosticStatus = Literal[
    "UNANALYZED", "INSPECT", "SUSPECTED", "CONFIRMED", "UNVERIFIED", "UNAVAILABLE"
]


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    side: Literal["front", "back", "unknown"] = "unknown"
    capture_type: Literal["full_view", "close_up", "oblique", "marking_detail", "unknown"] = "unknown"
    filename: str | None = None
    content_sha256: str | None = None
    media_type: str | None = None
    image_dimensions: tuple[int, int] | None = None
    image_format: str | None = None


@dataclass(frozen=True)
class ProjectContext:
    project_id: str
    board_id: str | None = None
    image_records: tuple[ImageRecord, ...] = ()
    symptom_category: str | None = None
    symptom_description: str | None = None
    circuit_source: dict[str, Any] | None = None
    input_fingerprint: str | None = None


@dataclass(frozen=True)
class ComponentViewModel:
    component_ref: str
    display_name: str
    location: dict[str, float] | None = None
    mapping_status: VerificationStatus = "UNAVAILABLE"
    diagnostic_status: DiagnosticStatus = "UNANALYZED"
    evidence_ids: tuple[str, ...] = ()
    status_label: str | None = None


@dataclass(frozen=True)
class MeasurementViewModel:
    probe_id: str
    component_ref: str | None
    node_id: str
    value: float | None
    unit: str
    source: str
    timestamp: str | None = None
    analysis_context_id: str | None = None


@dataclass(frozen=True)
class DiagnosisSummaryViewModel:
    overall_status: str = "UNAVAILABLE"
    candidate_components: tuple[ComponentViewModel, ...] = ()
    suggested_measurements: tuple[dict[str, Any], ...] = ()
    secondary_risks: tuple[dict[str, Any], ...] = ()
    verification_status: VerificationStatus = "UNAVAILABLE"
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionState:
    project_id: str
    board_id: str | None = None
    selected_component: str | None = None
    selected_probe: str | None = None
    active_tab: Literal["pcb", "schematic"] = "pcb"
    active_photo_id: str | None = None
    analysis_context_id: str | None = None

