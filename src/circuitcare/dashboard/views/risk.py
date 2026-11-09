"""Electrical stress and rating comparison view."""

from __future__ import annotations

from typing import Any

from circuitcare.dashboard.formatters import risk_table


def render_risk_view(st: Any, *, risk: dict) -> None:
    st.subheader("2차 스트레스 및 정격 비교")
    if not risk:
        st.info("이 회로에는 위험 평가 결과가 저장되어 있지 않습니다.")
        return
    st.caption("EXCEEDED / NEAR_LIMIT는 설정 정격과 계산 스트레스 비교입니다. WITHIN_LIMIT는 평가한 정격 항목 한정이며 부품 정상 상태나 실제 손상을 뜻하지 않습니다. UNKNOWN은 안전 판정이 아닙니다.")
    st.dataframe(risk_table(risk), use_container_width=True, hide_index=True)
    st.caption("점수는 점검 우선순위 지표(확률 아님). 정격 출처는 evidence note를 펼쳐 확인하세요.")
    with st.expander("위험 보고서 JSON"):
        st.json(risk)
