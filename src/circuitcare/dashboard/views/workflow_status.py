"""Read-only workspace and workflow status view."""

from __future__ import annotations

from typing import Any

from circuitcare.workspace.workflow import workflow_steps


def render_workflow_status(st: Any, *, workspace: dict, case: dict, status: str) -> None:
    spice_status = {
        "SUCCESS": "COMPLETE", "CACHED_SIMULATION": "COMPLETE",
        "STALE": "STALE", "PROVENANCE_UNVERIFIED": "NOT_EVALUATED",
        "NOT_RUN": "READY", "FAILED": "FAILED",
    }.get(status, "UNSUPPORTED" if case.get("uploaded") else "NOT_EVALUATED")
    steps = workflow_steps(
        circuit_status=workspace.get("validation_status", "FAILED"), graph_ready=True,
        fault_status="COMPLETE", spice_status=spice_status, report_ready=True,
    )
    st.markdown("### 진단 워크플로")
    st.caption(f"활성 Workspace `{workspace['workspace_id']}` · 회로 버전 `{workspace['circuit_version']}` · 검증 {workspace['validation_status']}")
    st.dataframe(steps, hide_index=True, use_container_width=True)
    if workspace.get("invalidated_analyses"):
        st.warning("회로 또는 고장 Context가 바뀌어 이전 분석이 무효화되었습니다: "
                   + ", ".join(workspace["invalidated_analyses"]))
    st.caption("LLM 요청과 ngspice 실행은 각각 별도 사용자 동작으로만 시작됩니다. 수리 이력 저장은 선택 사항입니다.")
