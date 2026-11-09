"""Typed representations for a diagnostic workspace snapshot."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

AnalysisStatus = Literal[
    "NOT_STARTED", "READY", "RUNNING", "COMPLETE", "PARTIAL", "FAILED",
    "STALE", "UNSUPPORTED", "NOT_EVALUATED",
]


@dataclass(frozen=True)
class WorkspaceContext:
    workspace_id: str
    circuit_id: str
    circuit_hash: str
    circuit_version: str
    netlist_path: str
    input_hashes: dict[str, str | None]
    active_fault_scenario: dict[str, Any]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisRecord:
    analysis_type: str
    status: AnalysisStatus
    circuit_hash: str
    source: str
    timestamp: str
    scenario_id: str | None = None
    input_hashes: dict[str, str | None] = field(default_factory=dict)
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
