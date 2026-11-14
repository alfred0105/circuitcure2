from __future__ import annotations

from pathlib import Path

import pytest


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


def test_korean_status_labels_preserve_internal_codes():
    from circuitcare.dashboard.navigation import korean_status

    assert korean_status("EXCEEDED") == "설정 정격 초과"
    assert korean_status("UNKNOWN") == "미평가"
    assert korean_status("PROVENANCE_UNVERIFIED") == "출처 확인 필요"
    assert korean_status("UNRECOGNIZED_INTERNAL_STATE") == "상태 확인 필요"


def test_phase21_responsive_layout_and_overlay_symbols_are_present():
    root = Path(__file__).resolve().parents[1]
    styles = (root / "src/circuitcare/dashboard/workbench_styles.py").read_text()
    viewer = (root / "src/circuitcare/schematic/assets/viewer.js").read_text()
    assert "grid-template-columns:minmax(145px" in styles
    assert "first-child { display:none!important; }" not in styles
    assert 'EXCEEDED: "▲"' in viewer
    assert 'NEAR_LIMIT: "◆"' in viewer
    assert 'WITHIN_LIMIT: "○"' in viewer
    assert 'UNKNOWN: "?"' in viewer


def test_phase21_readme_document_links_exist_and_schema_is_valid_json():
    import json
    import re

    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    for relative in re.findall(r"\]\((docs/[^)#]+\.md)\)", readme):
        assert (root / relative).is_file(), relative
    schema = json.loads((root / "evaluation/experiment_record.schema.json").read_text())
    assert "experiment_id" in schema["required"]
