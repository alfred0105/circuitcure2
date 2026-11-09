"""Validated report preview and download controls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from circuitcare.reporting.exporters import (
    export_bundle,
    export_html,
    export_json,
    export_markdown,
)
from circuitcare.reporting.report_validator import validate_report


def render_report_panel(st: Any, *, snapshot: dict, root: Path, workspace_id: str,
                        expected_circuit_hash: str | None) -> None:
    errors = validate_report(snapshot, expected_circuit_hash=expected_circuit_hash)
    st.markdown("### 근거 보고서")
    st.caption("현재 분석 Snapshot을 내보냅니다. 다운로드 과정에서 SPICE, LLM 또는 다른 분석을 재실행하지 않습니다.")
    if errors:
        st.error("보고서 검증 실패: " + ", ".join(errors))
        return
    st.success("보고서 근거 참조 및 현재 회로 Hash 검증 완료 · 실제 물리 손상은 확인하지 않았습니다.")
    try:
        report_json = export_json(snapshot)
        report_markdown = export_markdown(snapshot)
        report_html = export_html(snapshot)
        report_bundle = export_bundle(snapshot, root=root)
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        st.error(f"보고서를 만들지 못했습니다: {exc} · 회로/분석 자료를 확인하고 다시 열어 주세요.")
        return
    st.download_button("진단 보고서 JSON", report_json, "circuitcare_diagnosis.json", "application/json", key=f"report_json_{workspace_id}")
    st.download_button("진단 보고서 Markdown", report_markdown, "circuitcare_diagnosis.md", "text/markdown", key=f"report_md_{workspace_id}")
    st.download_button("진단 보고서 HTML", report_html, "circuitcare_diagnosis.html", "text/html", key=f"report_html_{workspace_id}")
    st.download_button("Evidence Bundle ZIP", report_bundle, "circuitcare_evidence_bundle.zip", "application/zip", key=f"report_bundle_{workspace_id}")
