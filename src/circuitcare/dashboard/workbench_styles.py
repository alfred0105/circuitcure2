"""Compact dark EDA chrome around a bright, legible schematic canvas."""

from __future__ import annotations

import html
from typing import Any


def apply_workbench_theme(st: Any) -> None:
    st.markdown(
        """
        <style>
        :root { --cc-bg:#0e141b; --cc-panel:#161f29; --cc-line:#2b3947; --cc-muted:#9aa9b8; }
        .stApp { background:var(--cc-bg); }
        [data-testid="stHeader"] { background:rgba(14,20,27,.94); }
        [data-testid="stSidebar"] { background:#111923; border-right:1px solid var(--cc-line); }
        /* Keep Streamlit's responsive top safe-area so the workbench header is not
           painted under the native toolbar on different viewport heights. */
        [data-testid="stMainBlockContainer"] { padding-bottom:1rem; max-width:100%; }
        .cc-workbench-header { box-sizing:border-box; display:flex; align-items:center;
          justify-content:space-between; gap:1rem; width:100%; padding:.55rem .9rem;
          border:1px solid var(--cc-line); border-radius:5px 5px 0 0;
          background:#121b25; color:#e6edf3; font-family:sans-serif; }
        .cc-workbench-brand { color:#f1f5f9 !important; font-size:1rem; font-weight:700; }
        .cc-workbench-subtitle { color:var(--cc-muted); font-size:.74rem; margin-top:.1rem; }
        .cc-workbench-project { color:#c8d5e1; font-size:.8rem; text-align:right; }
        .cc-context-strip { box-sizing:border-box; display:flex; flex-wrap:wrap; gap:.4rem;
          width:100%; padding:.42rem .7rem; margin-bottom:.45rem; border:1px solid var(--cc-line);
          border-top:0; border-radius:0 0 5px 5px; background:#111923; color:#d7e0e8; font-size:.74rem; }
        .cc-context-item { display:inline-flex; align-items:center; gap:.3rem; padding:.12rem .4rem;
          border:1px solid #2a3948; border-radius:3px; background:#17222d; }
        .cc-context-label { color:#9aa9b8; font-weight:650; }
        .cc-context-value { color:#e6edf3; font-weight:600; }
        .cc-context-state { border-color:#705a24; background:#2a2417; }
        [data-testid="stRadio"] > label { display:none; }
        [data-testid="stRadio"] div[role="radiogroup"] { gap:.25rem; }
        [data-testid="stRadio"] div[role="radiogroup"] label { border:1px solid transparent;
          border-radius:4px; padding:.35rem .7rem; background:#151e28; }
        [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
          border-color:#3c75a8; background:#182f43; }
        [data-testid="stPopover"] button { min-height:2rem; }
        [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] iframe) { min-width:0; }
        @media (max-width:1400px) {
          [data-testid="stHorizontalBlock"]:has([class*="st-key-schematic_viewer"]) { display:grid!important;
            grid-template-columns:minmax(145px, .8fr) minmax(0, 4.8fr) minmax(195px, 1.15fr)!important; gap:.65rem!important; }
          [data-testid="stHorizontalBlock"]:has([class*="st-key-schematic_viewer"]) > [data-testid="stColumn"] {
            width:auto!important; min-width:0!important; }
          .cc-workbench-project { text-align:left; }
        }
        @media (max-width:850px) {
          .cc-workbench-header { align-items:flex-start; flex-direction:column; }
          [data-testid="stHorizontalBlock"]:has([class*="st-key-schematic_viewer"]) { grid-template-columns:minmax(0,1fr)!important; }
          [data-testid="stHorizontalBlock"]:has([class*="st-key-schematic_viewer"]) > [data-testid="stColumn"] + [data-testid="stColumn"] {
            border-top:1px solid var(--cc-line); padding-top:.8rem; }
        }
        </style>
        """, unsafe_allow_html=True,
    )


def render_workbench_header(st: Any, *, circuit_name: str, fault_ref: str,
                            fault_type: str, fault_value: str | None,
                            simulation_status: str, verification_status: str) -> None:
    def safe(value: object) -> str:
        return html.escape(str(value))
    from circuitcare.dashboard.navigation import CASE_LABELS, korean_status

    st.markdown(
        f"""
        <section class="cc-workbench-header" aria-label="CircuitCare 작업대">
          <div><div class="cc-workbench-brand">CircuitCare AI <span aria-hidden="true">/</span> 회로 진단 작업대</div>
          <div class="cc-workbench-subtitle">로컬 분석 · 근거 기반 · 실행 전 확인</div></div>
          <div class="cc-workbench-project"><b>회로</b> &nbsp;{safe(CASE_LABELS.get(circuit_name, circuit_name))}</div>
        </section>
        <section class="cc-context-strip" aria-label="현재 진단 조건">
          <span class="cc-context-item"><span class="cc-context-label">고장 가정</span><span class="cc-context-value">{safe(fault_ref)} · {safe(korean_status(fault_type))} · {safe(fault_value or '—')}</span></span>
          <span class="cc-context-item cc-context-state"><span class="cc-context-label">시뮬레이션</span><span class="cc-context-value">{safe(korean_status(simulation_status))}</span></span>
          <span class="cc-context-item"><span class="cc-context-label">결과 검증</span><span class="cc-context-value">{safe(korean_status(verification_status))}</span></span>
        </section>
        """, unsafe_allow_html=True,
    )
