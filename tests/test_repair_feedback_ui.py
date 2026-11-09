from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_repair_feedback_ui_shows_safe_empty_database_state(tmp_path):
    script = tmp_path / "repair_feedback_app.py"
    local_root = tmp_path / "app_data"
    script.write_text(
        "import streamlit as st\n"
        "from pathlib import Path\n"
        "from circuitcare.repair_feedback_ui import render_repair_feedback_tab\n"
        f"render_repair_feedback_tab(st, root=Path({str(local_root)!r}), "
        "circuit_context={'circuit_id':'demo','circuit_family':'demo','circuit_hash':'hash'}, fault_state=None)\n",
        encoding="utf-8",
    )
    app = AppTest.from_file(str(script), default_timeout=20).run()
    assert not app.exception
    assert any("No Verified Repair Cases" in item.value for item in app.info)
    assert (local_root / "data/private/repair_feedback/repair_cases.sqlite3").is_file()
    assert [tab.label for tab in app.tabs] == ["사례 등록", "검증 및 적격성", "수리 이력", "Prior 학습 및 승인", "진단 비교"]
