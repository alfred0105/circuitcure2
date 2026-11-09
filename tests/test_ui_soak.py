from __future__ import annotations

import json
import subprocess
import os
from pathlib import Path

import pytest


def test_direct_vision_view_switch_keeps_app_live():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "src/circuitcare/ui.py"), default_timeout=30).run()
    next(widget for widget in app.button if widget.label == "PCB 사진 인식").click().run()
    assert not app.exception


@pytest.mark.stress
def test_streamlit_twenty_circuit_switches_keep_app_live():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "src/circuitcare/ui.py"), default_timeout=30).run()
    case_selector = next(widget for widget in app.selectbox if widget.key == "case_name")
    choices = case_selector.options
    samples = {"1": _rss_mb()}
    for index in range(20):
        chosen = choices[index % 3]
        next(widget for widget in app.selectbox if widget.key == "case_name").select(chosen).run()
        assert not app.exception
        assert app.session_state["active_analysis_view"] == "회로도 Workbench"
        if index + 1 in {5, 10, 20}:
            samples[str(index + 1)] = _rss_mb()
    print("ui_switch_rss_samples_mb=" + json.dumps(samples, sort_keys=True))
    output_dir = os.environ.get("CIRCUITCARE_DIAGNOSTICS_DIR")
    if output_dir:
        target = Path(output_dir) / "streamlit_switch_soak.json"
        target.write_text(json.dumps({"status": "PASSED", "switch_count": 20,
                                     "rss_samples_mb": samples}, indent=2) + "\n", encoding="utf-8")


def _rss_mb() -> float | None:
    try:
        result = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                                capture_output=True, text=True, timeout=3, check=True)
        return round(int(result.stdout.strip()) / 1024, 2)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
