"""PCB component overlay selection and image preparation."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import base64
import hashlib
import io
import math
from typing import Any

from PIL import Image

from circuitcare.dashboard.components.pcb_viewer import render_pcb_viewer
from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel, display_identity_status
from circuitcare.frontend.models import ProjectContext
from circuitcare.frontend.workbench_state import SELECTION_KEY, update_selection
from circuitcare.vision.image_loader import load_board_image

SELECTED_MAP_KEY = "circuitcare_v4_selection_map_key"


def coordinate_status_for_viewer(*, api_available: bool, client_ready: bool,
                                 frame_valid: bool = True) -> str:
    """Only trust browser-confirmed shared-SVG rendering with a valid source frame."""
    if not frame_valid:
        return "INVALID_COORDINATE_FRAME"
    if not api_available:
        return "VIEWER_UNAVAILABLE"
    if client_ready:
        return "VERIFIED_SHARED_SVG_VIEWBOX"
    return "UNVERIFIED_RESPONSIVE_CSS"


def validate_component_selection_event(
    event: Any, *, project_id: str, board_id: str | None, board_side: str,
    frame_id: str, allowed_ids: set[str], last_sequence: int,
) -> tuple[bool, str | None]:
    if not isinstance(event, dict):
        return False, "선택 이벤트 형식 오류"
    expected = (project_id, board_id, board_side, frame_id)
    observed = (event.get("project_id"), event.get("board_id"), event.get("board_side"), event.get("canonical_frame_id"))
    sequence = event.get("event_seq")
    event_id = event.get("event_id")
    action = event.get("selection_action")
    component_id = event.get("component_id")
    if observed != expected:
        return False, "오래된 프로젝트 또는 이미지 프레임 이벤트"
    if type(sequence) is not int or sequence <= last_sequence:
        return False, "중복 또는 오래된 선택 이벤트"
    if not isinstance(event_id, str) or not event_id.strip():
        return False, "선택 이벤트 ID가 없습니다."
    if action == "select" and component_id in allowed_ids:
        return True, None
    if action == "clear" and component_id is None:
        return True, None
    return False, "현재 Component Map에 없는 부품 선택"


@lru_cache(maxsize=8)
def _prepare_source_image(raw: bytes) -> tuple[str, tuple[int, int]]:
    loaded = load_board_image(raw, max_edge=2400)
    output = io.BytesIO()
    loaded.image.save(output, format="PNG", optimize=False)
    return base64.b64encode(output.getvalue()).decode("ascii"), loaded.processed_size


@lru_cache(maxsize=4)
def _prepare_canonical_image(raw_png: bytes) -> tuple[str, tuple[int, int]]:
    with Image.open(io.BytesIO(raw_png)) as source:
        image = source.convert("RGB")
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False)
    return base64.b64encode(output.getvalue()).decode("ascii"), image.size


def scale_overlay_to_display(
    models: tuple[ComponentOverlayViewModel, ...], *, source_size: tuple[int, int],
    display_size: tuple[int, int],
) -> tuple[ComponentOverlayViewModel, ...]:
    """Apply known EXIF-corrected source → decoded display scaling to every vector point."""
    sx = display_size[0] / source_size[0]
    sy = display_size[1] / source_size[1]
    return tuple(replace(
        item,
        bbox_xyxy=(item.bbox_xyxy[0] * sx, item.bbox_xyxy[1] * sy,
                   item.bbox_xyxy[2] * sx, item.bbox_xyxy[3] * sy),
        polygon_xy=tuple((x * sx, y * sy) for x, y in item.polygon_xy),
    ) for item in models)


def render_component_viewer(
    st: Any, *, image: Any, source_image_bytes: bytes, canonical_png_bytes: bytes | None,
    image_size_px: tuple[int, int], components: tuple[ComponentOverlayViewModel, ...],
    project: ProjectContext, board_side: str, frame_id: str, selected_component_id: str | None,
    map_cache_key: str, image_mode: str,
    on_box_draw: Any | None = None,
) -> tuple[bool, str, str]:
    """Render the viewer and return API availability, key, and coordinate trust state."""
    if image_mode == "통합 이미지" and canonical_png_bytes is not None:
        image_b64, display_size = _prepare_canonical_image(canonical_png_bytes)
    else:
        image_b64, display_size = _prepare_source_image(source_image_bytes)
    if display_size != image_size_px:
        return False, "", "INVALID_COORDINATE_FRAME"
    digest = hashlib.sha256(f"{project.project_id}|{project.board_id}|{map_cache_key}|{board_side}|{frame_id}|{image_mode}".encode()).hexdigest()[:16]
    viewer_key = f"cc_v4_pcb_viewer_{digest}"
    selection = st.session_state.get(SELECTION_KEY)
    selected_id = getattr(selection, "selected_component", None)
    if selected_id and selected_id not in {item.component_id for item in components}:
        selection = update_selection(st.session_state, project=project, selected_component=None)
        selected_id = selection.selected_component

    mount_status_key = f"{viewer_key}_mount_status"
    expected_asset_key = f"{map_cache_key}|{project.project_id}|{project.board_id or ''}|{board_side}|{frame_id}"

    def handle_selection_change() -> None:
        result = st.session_state.get(viewer_key)
        viewer_status = (result.get("viewer_status") if isinstance(result, dict)
                         else getattr(result, "viewer_status", None))
        if isinstance(viewer_status, dict):
            if (viewer_status.get("asset_key") == expected_asset_key
                    and viewer_status.get("status") == "READY"
                    and viewer_status.get("image_size_px") == list(image_size_px)
                    and isinstance(viewer_status.get("event_id"), str)
                    and bool(viewer_status.get("event_id"))):
                st.session_state[mount_status_key] = "READY"
        event = (result.get("selection") if isinstance(result, dict)
                 else getattr(result, "selection", None))
        if not isinstance(event, dict):
            return
        last_id_key = f"{viewer_key}_last_event"
        last_seq_key = f"{viewer_key}_last_seq"
        if event.get("event_id") == st.session_state.get(last_id_key):
            return
        seq = event.get("event_seq")
        if (event.get("selection_action") == "draw"
                and (event.get("project_id"), event.get("board_id"), event.get("board_side"),
                     event.get("canonical_frame_id"))
                == (project.project_id, project.board_id or "", board_side, frame_id)
                and type(seq) is int and seq > int(st.session_state.get(last_seq_key, 0))
                and isinstance(event.get("event_id"), str) and event.get("event_id")
                and on_box_draw is not None):
            bbox = event.get("bbox_xyxy")
            try:
                values = [float(value) for value in bbox]
                if len(values) != 4 or not all(math.isfinite(value) for value in values):
                    return
                width, height = image_size_px
                x1, y1, x2, y2 = values
                x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
                y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    return
            except (TypeError, ValueError):
                return
            st.session_state[last_id_key] = event["event_id"]
            st.session_state[last_seq_key] = seq
            on_box_draw([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))])
            return
        component_id = event.get("component_id")
        allowed_ids = {item.component_id for item in components}
        accepted, _reason = validate_component_selection_event(
            event, project_id=project.project_id, board_id=project.board_id or "",
            board_side=board_side, frame_id=frame_id, allowed_ids=allowed_ids,
            last_sequence=int(st.session_state.get(last_seq_key, 0)),
        )
        if not accepted:
            st.session_state[f"{viewer_key}_event_rejected"] = True
            return
        st.session_state[last_id_key] = event.get("event_id")
        st.session_state[last_seq_key] = seq
        st.session_state[SELECTED_MAP_KEY] = map_cache_key
        update_selection(st.session_state, project=project, selected_component=component_id)

    payload = [{
        "component_id": item.component_id,
        "component_type": item.component_type,
        "display_label": item.display_label,
        "bbox_xyxy": list(item.bbox_xyxy),
        "polygon_xy": [list(point) for point in item.polygon_xy],
        "detection_confidence": item.detection_confidence,
        "overlay_status": item.overlay_status,
        "identification_status": item.identification_status,
        "display_status": display_identity_status(item.identification_status),
        "identity_match_status": item.identity_match_status,
        "ocr_marking": item.ocr_marking,
        "possible_part_number": item.possible_part_number,
        "fault_candidate_status": item.fault_candidate_status,
        "diagnosis_display_status": item.diagnosis_display_status,
    } for item in components]
    result = render_pcb_viewer(
        st, image_png_base64=image_b64, image_size_px=image_size_px, components=payload,
        project_id=project.project_id, board_id=project.board_id or "", board_side=board_side,
        frame_id=frame_id, selected_component_id=selected_id, asset_key=map_cache_key,
        event_sequence_seed=int(st.session_state.get(f"{viewer_key}_last_seq", 0)),
        key=viewer_key, on_selection_change=handle_selection_change,
        on_viewer_status_change=handle_selection_change,
    )
    if result is None:
        return False, viewer_key, coordinate_status_for_viewer(api_available=False, client_ready=False)
    mounted = st.session_state.get(mount_status_key) == "READY"
    return True, viewer_key, coordinate_status_for_viewer(api_available=True, client_ready=mounted)
