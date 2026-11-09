"""Streamlit download controls for raw analysis outputs."""

from __future__ import annotations

import hashlib
import json

from circuitcare.dashboard.downloads import (
    _cached_download_bytes,
    _download_file_digest,
)


def render_downloads(st, case, result_dir, comparison, risk, diagnosis, selected_fault, fault_type, fault_value,
                     *, simulation_verified: bool = False, csv_verified: bool = False,
                     csv_sha256: str | None = None):
    with st.expander("결과 내보내기"):
        summary = {"circuit": case["netlist"], "fault": {"component": selected_fault, "fault_type": fault_type, "parameter": fault_value}, "simulation_source": "live" if "ui_runs" in str(result_dir) else ("cached" if comparison else "unsupported_or_not_run"), "comparison": comparison or None, "risk_assessment": risk or None, "bayesian_diagnosis": diagnosis.get("diagnosis") if diagnosis else None, "damage_confirmed": False, "limitations": ["Simulation does not confirm damage.", "Ratings and demo observations may be assumptions.", "UNKNOWN is not safe."]}
        st.download_button("전체 분석 요약 JSON", json.dumps(summary, ensure_ascii=False, indent=2), "circuitcare_summary.json", "application/json")
        if risk:
            st.download_button("정격 위험 보고서 JSON", json.dumps(risk, ensure_ascii=False, indent=2), "risk_assessment.json", "application/json")
        if diagnosis:
            st.download_button("Bayesian 진단 JSON", json.dumps(diagnosis["diagnosis"], ensure_ascii=False, indent=2), "diagnosis.json", "application/json")
        csv_path = result_dir / "comparison.csv"
        if simulation_verified and csv_verified and csv_path.is_file():
            download_context = hashlib.sha256(json.dumps(
                {"circuit": case.get("netlist"), "fault": selected_fault,
                 "fault_type": fault_type, "fault_value": fault_value},
                sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            try:
                digest = _download_file_digest(csv_path)
                if digest != csv_sha256:
                    raise ValueError("CSV does not match the verified simulation manifest")
                try:
                    data = _cached_download_bytes(str(csv_path), digest, download_context)
                except ValueError:
                    # The file changed between hashing and reading; refresh the snapshot once.
                    digest = _download_file_digest(csv_path)
                    if digest != csv_sha256:
                        raise ValueError("CSV changed after simulation manifest verification")
                    data = _cached_download_bytes(str(csv_path), digest, download_context)
                st.download_button("시뮬레이션 CSV", data, "comparison.csv", "text/csv")
            except (OSError, ValueError) as exc:
                st.warning(f"CSV 파일이 바뀌었거나 다운로드 한도를 넘었습니다. 다시 실행해 주세요. ({exc})")
        elif csv_path.is_file():
            st.caption("시뮬레이션 또는 CSV 무결성을 확인할 수 없어 원시 결과 CSV 다운로드를 차단했습니다.")
        report = ["# CircuitCare AI Analysis", "", f"- Circuit: {case['netlist']}", f"- Fault: {selected_fault} / {fault_type} / {fault_value or 'n/a'}", f"- Simulation source: {'live' if 'ui_runs' in str(result_dir) else ('cached' if comparison else 'unsupported or not run')}", "- Ratings are explicit inputs; demo assumptions are not datasheet facts.", "- UNKNOWN does not mean safe; priority scores are not failure probabilities.", "- Physical damage is not confirmed."]
        st.download_button("통합 Markdown 보고서", "\n".join(report), "circuitcare_report.md", "text/markdown")


