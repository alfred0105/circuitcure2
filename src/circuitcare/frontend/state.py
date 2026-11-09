"""Session-state helpers for frontend intake drafts; no persistent storage."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from circuitcare.frontend.models import ProjectContext

INTAKE_IMAGES_KEY = "circuitcare_v2_images"
INTAKE_PROJECT_KEY = "circuitcare_v2_project"
INTAKE_UPLOAD_BATCH_KEY = "circuitcare_v2_upload_batch"
_PROJECT_RESULT_KEYS = {
    "circuitcare_v2_analysis_status", "circuitcare_v2_diagnosis_result",
    "circuitcare_v2_measurements", "circuitcare_v2_selected_component",
    "circuitcare_v2_selected_probe", "circuitcare_v2_analysis_context",
}


def clear_intake_draft(state: MutableMapping[str, Any]) -> None:
    state.pop(INTAKE_PROJECT_KEY, None)
    for key in _PROJECT_RESULT_KEYS:
        state.pop(key, None)


def reset_intake_for_new_project(state: MutableMapping[str, Any]) -> None:
    """Forget current project inputs/results and retire its Streamlit uploader keys."""
    # Imported lazily to keep intake-state helpers independent of image/PIL startup.
    from circuitcare.frontend.component_map_adapter import clear_component_map_state
    from circuitcare.frontend.measurement_state import clear_measurement_workspace_state

    clear_component_map_state(state)
    clear_measurement_workspace_state(state)
    next_batch = int(state.get(INTAKE_UPLOAD_BATCH_KEY, 0)) + 1
    for key in tuple(state):
        if key.startswith(("circuitcare_v2_", "cc_v2_")):
            state.pop(key, None)
    state[INTAKE_IMAGES_KEY] = []
    state[INTAKE_UPLOAD_BATCH_KEY] = next_batch


def draft_is_current(state: MutableMapping[str, Any], fingerprint: str) -> bool:
    project = state.get(INTAKE_PROJECT_KEY)
    return isinstance(project, ProjectContext) and project.input_fingerprint == fingerprint

