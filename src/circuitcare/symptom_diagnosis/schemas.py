"""Shared input records for component, observation, and diagnosis state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


COMPONENT_LABELS = (
    "battery", "switch", "buzzer", "capacitor", "clock", "connector", "diode",
    "display", "emi_filter", "ferrite_bead", "fuse", "heatsink", "ic", "inductor",
    "jumper", "led", "ldr", "mov", "oscillator", "potentiometer", "relay", "resistor",
    "speaker", "thermistor", "transformer", "transistor", "voltage_regulator",
)
LABEL_ALIASES = {"integrated_circuit": "ic", "IC": "ic", "LED": "led", "MOV": "mov", "LDR": "ldr"}


class EvidenceStatus(StrEnum):
    OBSERVED = "observed"
    MEASURED = "measured"
    INFERRED = "inferred"
    UNKNOWN = "unknown"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True)
class Observation:
    code: str
    id: str = field(default_factory=lambda: str(uuid4()))
    value: Any = None
    status: EvidenceStatus = EvidenceStatus.OBSERVED
    source: str = "user_report"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    component_id: str | None = None
    confidence: float | None = None
    measurement_conditions: dict[str, Any] | None = None
    evidence_quote: str | None = None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["status"] = self.status.value
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Observation":
        try:
            status = EvidenceStatus(row.get("status", "observed"))
        except ValueError:
            status = EvidenceStatus.UNKNOWN
        confidence = row.get("confidence")
        if confidence is not None:
            try:
                confidence = max(0.0, min(1.0, float(confidence)))
            except (TypeError, ValueError):
                confidence = None
        return cls(
            id=str(row.get("id") or row.get("observation_id") or uuid4()),
            code=str(row.get("code", "")), value=row.get("value"), status=status,
            source=str(row.get("source", "unknown")),
            timestamp=str(row.get("timestamp") or datetime.now(timezone.utc).isoformat()),
            component_id=row.get("component_id"), confidence=confidence,
            measurement_conditions=row.get("measurement_conditions"),
            evidence_quote=row.get("evidence_quote"),
        )


@dataclass(frozen=True)
class ComponentRecord:
    component_id: str
    label: str
    bbox: tuple[float, float, float, float] | None = None
    pins: tuple[dict[str, Any], ...] = ()
    marking: str | None = None
    confirmed_part_number: str | None = None
    electrical_role: str = "unknown"
    detection_confidence: float | None = None
    coordinate_frame: str = "image_pixel"
    ic_identity: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_label(self) -> str:
        label = LABEL_ALIASES.get(self.label, self.label.casefold())
        return label if label in COMPONENT_LABELS else "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "normalized_label": self.normalized_label}

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "ComponentRecord":
        raw_box = row.get("bbox", row.get("bbox_xyxy"))
        bbox = tuple(float(value) for value in raw_box) if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4 else None
        label = str(row.get("label", row.get("component_type", "unknown")))
        marking = row.get("marking", row.get("ocr_marking"))
        ic_identity = row.get("ic_identity") or {}
        if not ic_identity and label.casefold() in {"ic", "integrated_circuit"} and marking:
            from circuitcare.symptom_diagnosis.ic_identity import resolve_ic_identity
            ic_identity = resolve_ic_identity(str(marking))
        return cls(component_id=str(row.get("component_id", "unknown")),
                   label=label,
                   bbox=bbox, pins=tuple(row.get("pins", ())),
                   marking=marking,
                   confirmed_part_number=row.get("confirmed_part_number"),
                   electrical_role=str(row.get("electrical_role", "unknown")),
                   detection_confidence=row.get("detection_confidence"),
                   coordinate_frame=str(row.get("coordinate_frame", "image_pixel")),
                   ic_identity=dict(ic_identity))


@dataclass(frozen=True)
class DiagnosticState:
    diagnosis_id: str
    project_id: str
    board_id: str | None
    components: tuple[ComponentRecord, ...]
    observations: tuple[Observation, ...]
    connectivity_status: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {"diagnosis_id": self.diagnosis_id, "project_id": self.project_id,
                "board_id": self.board_id, "components": [row.to_dict() for row in self.components],
                "observations": [row.to_dict() for row in self.observations],
                "connectivity_status": self.connectivity_status}
