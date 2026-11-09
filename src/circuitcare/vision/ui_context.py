"""Small immutable context contracts for the PCB image workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VisionViewContext:
    context_key: str
    netlist_component_refs: tuple[str, ...]
    risk_assessment: dict[str, Any] | None
    risk_path: str
    failed_component: str
    fault_type: str
    simulation_status: str
    result_integrity: str
    inspection_ranking: dict[str, Any] | None
    selected_component: str | None

    @property
    def risk_overlay_allowed(self) -> bool:
        return (self.simulation_status in {"SUCCESS", "CACHED_SIMULATION"}
                and self.result_integrity == "VERIFIED")

    @property
    def overlay_risk(self) -> dict[str, Any] | None:
        return self.risk_assessment if self.risk_overlay_allowed else None

    @property
    def blocked_reason(self) -> str | None:
        if self.risk_overlay_allowed:
            return None
        if self.simulation_status not in {"SUCCESS", "CACHED_SIMULATION"}:
            return f"현재 결과 상태 {self.simulation_status}에서는 위험 Overlay를 사용할 수 없습니다."
        return f"Simulation Manifest 검증 상태가 {self.result_integrity}이므로 위험 Overlay를 차단했습니다."


@dataclass(frozen=True)
class ConfirmedOverlayContext:
    registration: dict[str, Any]
    loaded_image: Any
    rectified_image: Any
    parsed_positions: dict[str, Any]
    positions: list[Any]
    board_key: str
    csv_hash: str
    board_side: str
    board_identity: str
    detection_record: dict[str, Any]
    rectified_state: dict[str, Any]
    ai_state: dict[str, Any]
    ai_reference_state: dict[str, Any]
    ai_reference_confirmations: dict[str, str]
    matching: dict[str, Any]
    rectified_image_sha256: str
