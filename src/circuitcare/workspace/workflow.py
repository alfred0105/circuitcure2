"""Guided workflow state derived from existing diagnostic artifacts."""

from __future__ import annotations

from typing import Any


def workflow_steps(*, circuit_status: str, graph_ready: bool, fault_status: str,
                   spice_status: str, report_ready: bool) -> list[dict[str, Any]]:
    return [
        {"step": 1, "name": "회로 선택 및 검증", "status": circuit_status,
         "detail": "회로 파일과 입력 해시 확인"},
        {"step": 2, "name": "연결 그래프", "status": "COMPLETE" if graph_ready else "READY",
         "detail": "부품 및 Net 연결에서 점검 경로 후보 구성"},
        {"step": 3, "name": "고장 정보 및 관측", "status": fault_status,
         "detail": "선택한 부품·유형과 관측 출처 검토"},
        {"step": 4, "name": "SPICE 검증", "status": spice_status,
         "detail": "ngspice 결과 또는 지원/실행 상태 확인"},
        {"step": 5, "name": "결과 검토 및 보고", "status": "READY" if report_ready else "PARTIAL",
         "detail": "근거 및 제한을 포함한 보고서 내보내기"},
        {"step": 6, "name": "수리 이력", "status": "NOT_STARTED",
         "detail": "실제 수리를 완료한 경우에만 기록"},
    ]
