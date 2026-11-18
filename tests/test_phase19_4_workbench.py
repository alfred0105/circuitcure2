from __future__ import annotations

import pytest
from pathlib import Path


def test_streamlit_entry_exposes_only_current_pcb_diagnosis_flow():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    script = Path(__file__).resolve().parents[1] / "src/circuitcare/ui.py"
    app = AppTest.from_file(str(script), default_timeout=30).run()
    assert not app.exception
    assert any("PCB 사진으로 고장 원인 찾기" in item.value for item in app.markdown)
    assert any(item.label == "분석 시작" and item.disabled for item in app.button)
    assert len(app.file_uploader) == 1  # PCB photos only
    assert not any("회로 데이터셋" in item.label for item in app.selectbox)
    assert "active_analysis_view" not in app.session_state


def test_current_diagnosis_statuses_render_as_user_facing_korean():
    from circuitcare.dashboard.views.inspection_panel import _diagnosis_status_label

    assert _diagnosis_status_label(None, None, analysis_snapshot={
        "record": {"status": "RUNNING"}, "result": None,
    }) == "진단 중"
    assert _diagnosis_status_label(None, None, analysis_snapshot={
        "record": {"status": "STALE"}, "result": None,
    }) == "입력 변경 · 재분석 필요"
    assert _diagnosis_status_label(None, None, analysis_snapshot={
        "record": {"status": "FAILED"}, "result": None,
    }) == "진단 실패 · 재시도 필요"
    assert _diagnosis_status_label(None, None, analysis_snapshot={
        "record": {"status": "CURRENT"}, "result": {"status": "ABSTAIN_INSUFFICIENT_EVIDENCE"},
    }) == "추론 보류 · 추가 정보 필요"


def test_current_viewer_is_responsive_and_selection_uses_blue():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    styles = (root / "src/circuitcare/dashboard/components/assets/pcb_viewer.css").read_text()
    assert "@media (max-width:700px)" in styles
    assert ".cc-pcb-component.cc-selected { fill:#60a5fa30; stroke:#60a5fa;" in styles
    assert ".cc-pcb-component.cc-fault-supported" in styles


def test_yolo_and_qwen_label_spaces_remain_separate():
    from circuitcare.component_taxonomy import COMPONENT_CLASSES, QWEN_REVIEW_CLASSES

    assert len(COMPONENT_CLASSES) == 18
    assert "other_component" not in COMPONENT_CLASSES
    assert "board_feature" not in COMPONENT_CLASSES
    assert {"other_component", "board_feature"} <= set(QWEN_REVIEW_CLASSES)
