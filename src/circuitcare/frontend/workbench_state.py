"""Session-scoped workbench navigation and viewer state."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from circuitcare.frontend.models import ProjectContext, SelectionState

SCREEN_KEY = "circuitcare_v2_screen"
SELECTION_KEY = "circuitcare_v2_selection"
ACTIVE_PHOTO_KEY = "circuitcare_v2_active_photo_id"
VIEW_ZOOM_KEY = "circuitcare_v2_view_zoom"
VIEW_TRANSFORM_KEY = "circuitcare_v2_view_transform"
MODE_KEY = "circuitcare_v2_workbench_mode"
PHOTO_STATUS_KEY = "circuitcare_v2_photo_status"


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
