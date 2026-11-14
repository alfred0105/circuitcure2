"""Session-scoped workbench navigation and viewer state."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from circuitcare.frontend.models import ProjectContext, SelectionState

SCREEN_KEY = "circuitcare_v2_screen"
SELECTION_KEY = "circuitcare_v2_selection"
BOARD_COMPONENT_SELECTION_KEY = "circuitcare_v7_selected_board_component_id"
MEASUREMENT_TARGET_BOARD_COMPONENT_KEY = "circuitcare_v7_measurement_target_board_component_id"
ACTIVE_PHOTO_KEY = "circuitcare_v2_active_photo_id"
VIEW_ZOOM_KEY = "circuitcare_v2_view_zoom"
VIEW_TRANSFORM_KEY = "circuitcare_v2_view_transform"
MODE_KEY = "circuitcare_v2_workbench_mode"
PHOTO_STATUS_KEY = "circuitcare_v2_photo_status"
WORKFLOW_STATES_KEY = "circuitcare_v8_diagnostic_workflows"

WORKFLOW_REVIEW = "REVIEW"
WORKFLOW_QUESTIONS = "QUESTIONS"
WORKFLOW_INSPECTION = "INSPECTION"
WORKFLOW_REASSESSMENT = "REASSESSMENT"
WORKFLOW_COMPLETE = "COMPLETE"
WORKFLOW_STAGES = (WORKFLOW_REVIEW, WORKFLOW_QUESTIONS, WORKFLOW_INSPECTION,
                   WORKFLOW_REASSESSMENT, WORKFLOW_COMPLETE)
_WORKFLOW_TRANSITIONS = {
    WORKFLOW_REVIEW: {WORKFLOW_QUESTIONS},
    WORKFLOW_QUESTIONS: {WORKFLOW_REVIEW, WORKFLOW_INSPECTION},
    WORKFLOW_INSPECTION: {WORKFLOW_REVIEW, WORKFLOW_QUESTIONS, WORKFLOW_REASSESSMENT},
    WORKFLOW_REASSESSMENT: {WORKFLOW_REVIEW, WORKFLOW_QUESTIONS, WORKFLOW_INSPECTION, WORKFLOW_COMPLETE},
    WORKFLOW_COMPLETE: {WORKFLOW_INSPECTION},
}


def workflow_owner(project: ProjectContext) -> str:
    return f"{project.project_id}|{project.board_id or ''}"


def get_workflow_state(state: MutableMapping[str, Any], project: ProjectContext) -> dict[str, Any]:
    """Return persistent four-step diagnostic state scoped to one project and board."""
    store = state.setdefault(WORKFLOW_STATES_KEY, {})
    owner = workflow_owner(project)
    workflow = store.get(owner)
    if not isinstance(workflow, dict):
        workflow = {
            "project_id": project.project_id,
            "board_id": project.board_id or "",
            "stage": WORKFLOW_REVIEW,
            "previous_stage": None,
            "transition_status": "IDLE",
            "stage_history": [],
            "stage_completion": {stage: False for stage in WORKFLOW_STAGES},
            "symptom_scope": None,
            "question_answers": [],
            "question_plan": None,
            "candidate_ids": [],
            "inspection_targets": [],
            "selected_target_id": None,
            "selected_fault_id": None,
            "selected_test_id": None,
            "measurement_ids": [],
            "reassessment_history": [],
            "inspection_history": [],
            "inspection_cycle": None,
            "inspection_observations": [],
            "transition_reason": None,
            "analysis_fingerprint": None,
            "viewer_focus_id": None,
            "pending_left_status": None,
        }
        store[owner] = workflow
    state[WORKFLOW_STATES_KEY] = store
    return workflow


def transition_workflow(
    state: MutableMapping[str, Any], project: ProjectContext, next_stage: str, *,
    event: str = "user", left_status: str | None = None, reason: str | None = None,
) -> dict[str, Any]:
    """Apply a valid explicit workflow transition without clearing prior inputs."""
    if next_stage not in WORKFLOW_STAGES:
        raise ValueError(f"알 수 없는 진단 단계입니다: {next_stage}")
    workflow = get_workflow_state(state, project)
    current_stage = workflow["stage"]
    if next_stage == current_stage:
        return workflow
    if next_stage not in _WORKFLOW_TRANSITIONS.get(current_stage, set()):
        raise ValueError(f"허용되지 않은 단계 전환입니다: {current_stage} → {next_stage}")
    workflow["previous_stage"] = current_stage
    completion = dict(workflow.get("stage_completion", {}))
    completion[current_stage] = True
    workflow["stage_completion"] = completion
    workflow["stage"] = next_stage
    workflow["transition_status"] = "READY"
    workflow["transition_reason"] = reason
    workflow["stage_history"] = [
        *workflow.get("stage_history", []),
        {"from": current_stage, "to": next_stage, "event": event},
    ][-100:]
    if left_status:
        workflow["pending_left_status"] = left_status
    store = state.setdefault(WORKFLOW_STATES_KEY, {})
    store[workflow_owner(project)] = workflow
    state[WORKFLOW_STATES_KEY] = store
    return workflow


def enter_workbench(
    state: MutableMapping[str, Any], project: ProjectContext, *, preview: bool = False,
) -> SelectionState:
    """Initialize PCB-first state without carrying a previous project's selection."""
    selection = SelectionState(
        project_id=project.project_id,
        board_id=project.board_id,
        active_tab="pcb",
        active_photo_id=project.image_records[0].image_id if project.image_records else None,
    )
    state[SCREEN_KEY] = "workbench"
    state[MODE_KEY] = "PREVIEW_ONLY" if preview else "DIAGNOSTIC"
    state[PHOTO_STATUS_KEY] = "NO_IMAGES" if not project.image_records else "READY"
    state[SELECTION_KEY] = selection
    state.pop(MEASUREMENT_TARGET_BOARD_COMPONENT_KEY, None)
    state[ACTIVE_PHOTO_KEY] = selection.active_photo_id
    state[VIEW_ZOOM_KEY] = 1.0
    state[VIEW_TRANSFORM_KEY] = {
        "active_photo_id": selection.active_photo_id,
        "coordinate_frame": "source_image_px",
        "coordinate_status": "NO_IMAGE" if not project.image_records else "PENDING_RENDER",
    }
    return selection


def update_selection(
    state: MutableMapping[str, Any], *, project: ProjectContext | None = None, **changes: Any,
) -> SelectionState:
    """Update one project's immutable selection snapshot."""
    current = state.get(SELECTION_KEY)
    project = project or state.get("circuitcare_v2_project")
    project_id = getattr(project, "project_id", None)
    if not project_id:
        raise ValueError("Workbench 프로젝트가 없습니다.")
    if not isinstance(current, SelectionState) or current.project_id != project_id:
        current = enter_workbench(state, project)
    values = {name: getattr(current, name) for name in current.__dataclass_fields__}
    values.update(changes)
    updated = SelectionState(**values)
    state[SELECTION_KEY] = updated
    if "active_photo_id" in changes:
        state[ACTIVE_PHOTO_KEY] = changes["active_photo_id"]
    return updated


def update_component_selection(state: MutableMapping[str, Any], *, project: ProjectContext,
                               selected_component: str | None,
                               board_component_id: str | None = None,
                               active_photo_id: str | None = None) -> SelectionState:
    """Update local and board identity together so a rerun cannot restore stale selection."""
    if selected_component and board_component_id:
        state[BOARD_COMPONENT_SELECTION_KEY] = str(board_component_id)
    else:
        state.pop(BOARD_COMPONENT_SELECTION_KEY, None)
    changes = {"selected_component": selected_component}
    if active_photo_id is not None:
        changes["active_photo_id"] = active_photo_id
    return update_selection(state, project=project, **changes)
