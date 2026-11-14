"""Single Streamlit entry point for photo-based PCB fault diagnosis."""

from __future__ import annotations

import sys
import logging
import os
from pathlib import Path

# Allow `streamlit run src/circuitcare/ui.py` directly from an uninstalled checkout.
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

if os.environ.get("CIRCUITCARE_VIEWER_DEBUG", "").lower() in {"1", "true", "yes"}:
    logging.getLogger("circuitcare.frontend.component_map_adapter").setLevel(logging.INFO)
    logging.getLogger("circuitcare.dashboard.components.pcb_workspace").setLevel(logging.INFO)


def main() -> None:
    """Route only between PCB photo intake and its active workbench."""
    import streamlit as st

    from circuitcare.frontend.state import INTAKE_PROJECT_KEY
    from circuitcare.frontend.workbench_state import SCREEN_KEY, enter_workbench

    st.set_page_config(page_title="CircuitCare AI", page_icon="🔧", layout="wide")
    project = st.session_state.get(INTAKE_PROJECT_KEY)
    if project is not None:
        if st.session_state.get(SCREEN_KEY) != "workbench":
            # Recover older sessions that retained a project but not its screen marker.
            enter_workbench(st.session_state, project)
        from circuitcare.dashboard.views.unified_workspace import render_workbench_shell

        render_workbench_shell(st, project)
        return

    from circuitcare.dashboard.views.diagnosis_entry import render_diagnosis_entry

    render_diagnosis_entry(st)


if __name__ == "__main__":
    main()
