"""Normal-versus-fault SPICE plot and stress table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from circuitcare.dashboard.formatters import csv_rows, stress_table


def render_waveform_view(st: Any, *, comparison: dict, result_dir: Path, stress: dict,
                         csv_verified: bool = False, csv_sha256: str | None = None) -> None:
    st.subheader("정상 / 고장 SPICE 전압 비교")
    if not comparison or not comparison.get("simulation_success"):
        st.warning("저장된 비교 결과가 없습니다. 명시적 SPICE 모델/매핑이 준비되지 않으면 이 단계는 지원되지 않습니다.")
        return
    nodes = list(comparison.get("voltage_nodes", {}))
    if not nodes:
        st.warning("비교 가능한 전압 노드가 없습니다.")
    else:
        node = st.selectbox("전압 노드", nodes)
        rows = csv_rows(result_dir / "comparison.csv", expected_sha256=csv_sha256) if csv_verified else []
        if not csv_verified:
            st.warning("파형 CSV 출처를 확인할 수 없어 원시 파형 표시를 차단했습니다.")
        elif not rows:
            st.warning("파형 CSV가 검증 이후 변경되었거나 읽을 수 없어 표시를 차단했습니다.")
        baseline_key, fault_key = f"baseline_V({node})_V", f"fault_V({node})_V"
        if rows and baseline_key in rows[0] and fault_key in rows[0]:
            import pandas as pd
            st.line_chart(pd.DataFrame({"시간 (s)": [float(row["time_s"]) for row in rows],
                                        "정상 (V)": [float(row[baseline_key]) for row in rows],
                                        "고장 (V)": [float(row[fault_key]) for row in rows]}).set_index("시간 (s)"))
        stats = comparison["voltage_nodes"][node]
        cols = st.columns(4)
        cols[0].metric("정상 마지막 전압", f"{stats['baseline']['last']:.4g} V")
        cols[1].metric("고장 마지막 전압", f"{stats['fault']['last']:.4g} V")
        cols[2].metric("전압 변화", f"{stats['delta_fault_minus_baseline']['last']:+.4g} V")
        cols[3].metric("샘플 수", str(comparison.get("time_range_s", {}).get("samples", "—")))
    if stress:
        st.caption("지원되는 부품 전류·전력 변화 (없거나 미지원인 값은 —)")
        st.dataframe(stress_table(stress), use_container_width=True, hide_index=True)
