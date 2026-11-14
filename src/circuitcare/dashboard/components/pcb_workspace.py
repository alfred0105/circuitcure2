"""PCB component overlay selection and image preparation."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import base64
import hashlib
import io
import math
import logging
import os
import secrets
import time
from typing import Any

from PIL import Image

from circuitcare.dashboard.components.pcb_viewer import render_pcb_viewer
from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel, display_identity_status
from circuitcare.frontend.models import ProjectContext
from circuitcare.frontend.workbench_state import (
    SELECTION_KEY, update_component_selection, update_selection,
)
from circuitcare.vision.image_loader import load_board_image

OVERLAPPING_SELECTION_KEY = "circuitcare_v4_overlapping_box_ids"
SELECTION_TRACE_KEY = "circuitcare_v4_selection_trace"
LOGGER = logging.getLogger(__name__)


def _record_selection_trace(state: Any, stage: str) -> None:
    trace = state.get(SELECTION_TRACE_KEY, [])
    if not isinstance(trace, list):
        trace = []
    trace.append(stage)
    state[SELECTION_TRACE_KEY] = trace[-12:]


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
    event_epoch: str | None = None,
) -> tuple[bool, str | None]:
    if not isinstance(event, dict):
        return False, "선택 이벤트 형식 오류"
    expected = (project_id, board_id, board_side, frame_id)
    observed = (event.get("project_id"), event.get("board_id"), event.get("board_side"), event.get("canonical_frame_id"))
    sequence = event.get("event_seq")
    event_id = event.get("event_id")
    action = event.get("selection_action")
    component_id = event.get("component_id")
    overlap_ids = event.get("overlapping_component_ids", [component_id] if component_id else [])
    if observed != expected:
        return False, "오래된 프로젝트 또는 이미지 프레임 이벤트"
    if event_epoch is not None and event.get("event_epoch") != event_epoch:
        return False, "이전 Viewer 인스턴스의 선택 이벤트"
    if type(sequence) is not int or sequence <= last_sequence:
        return False, "중복 또는 오래된 선택 이벤트"
    if not isinstance(event_id, str) or not event_id.strip():
        return False, "선택 이벤트 ID가 없습니다."
    if (action in {"select", "bulk-select"} and isinstance(component_id, str)
            and component_id in allowed_ids
            and isinstance(overlap_ids, list)
            and component_id in overlap_ids
            and all(isinstance(item, str) for item in overlap_ids)
            and (action != "bulk-select" or bool(overlap_ids))):
        return True, None
    if action == "clear" and component_id is None:
        return True, None
    return False, "현재 Component Map에 없는 부품 선택"


@lru_cache(maxsize=8)
def _prepare_source_image(raw: bytes) -> tuple[str, tuple[int, int], str]:
    loaded = load_board_image(raw, max_edge=2400)
    output = io.BytesIO()
    loaded.image.save(output, format="PNG", optimize=False)
    png = output.getvalue()
    return base64.b64encode(png).decode("ascii"), loaded.processed_size, hashlib.sha256(png).hexdigest()


@lru_cache(maxsize=4)
def _prepare_canonical_image(raw_png: bytes) -> tuple[str, tuple[int, int], str]:
    with Image.open(io.BytesIO(raw_png)) as source:
        image = source.convert("RGB")
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False)
    png = output.getvalue()
    return base64.b64encode(png).decode("ascii"), image.size, hashlib.sha256(png).hexdigest()


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
    photo_id: str | None = None,
    analysis_loading: bool = False,
    on_box_draw: Any | None = None,
) -> tuple[bool, str, str]:
    """Render the viewer and return API availability, key, and coordinate trust state."""
    if image_mode == "통합 이미지" and canonical_png_bytes is not None:
        image_b64, display_size, image_fingerprint = _prepare_canonical_image(canonical_png_bytes)
    else:
        image_b64, display_size, image_fingerprint = _prepare_source_image(source_image_bytes)
    if display_size != image_size_px:
        return False, "", "INVALID_COORDINATE_FRAME"
    # Component identity follows the visible photo frame, not its overlay/data
    # version. Detection and Qwen updates must patch regions in-place.
    digest = hashlib.sha256(
        f"{project.project_id}|{project.board_id or ''}|{board_side}|{frame_id}".encode()
    ).hexdigest()[:16]
    viewer_key = f"cc_v4_pcb_viewer_{digest}"
    bulk_selection_key = f"cc_pcb_bulk_selection_{viewer_key}"
    expected_asset_key = f"{image_fingerprint}|{project.project_id}|{project.board_id or ''}|{board_side}|{frame_id}"
    overlay_version = hashlib.sha256(repr([
        (item.component_id, item.bbox_xyxy, item.display_label, item.component_type,
         item.overlay_status, item.overlay_source, item.ocr_marking,
         item.fault_candidate_status, item.functional_block_roles) for item in components
    ]).encode()).hexdigest()[:16]
    epoch_key = f"{viewer_key}_{image_fingerprint[:16]}_event_epoch"
    event_epoch = st.session_state.setdefault(epoch_key, secrets.token_urlsafe(18))
    active_frame_key = "circuitcare_v4_active_viewer_frame"
    frame_scope = f"{project.project_id}|{project.board_id or ''}|{board_side}|{frame_id}|{image_mode}|{image_fingerprint}"
    prior_frame_scope = st.session_state.get(active_frame_key)
    if prior_frame_scope != frame_scope:
        previous_viewer_key = st.session_state.get("circuitcare_v4_active_viewer_key")
        if previous_viewer_key:
            st.session_state[f"{OVERLAPPING_SELECTION_KEY}_{previous_viewer_key}"] = []
        st.session_state[f"{OVERLAPPING_SELECTION_KEY}_{viewer_key}"] = []
        st.session_state[bulk_selection_key] = []
        update_selection(st.session_state, project=project, selected_component=None)
        st.session_state[active_frame_key] = frame_scope
        st.session_state["circuitcare_v4_active_viewer_key"] = viewer_key
    selection = st.session_state.get(SELECTION_KEY)
    selected_id = getattr(selection, "selected_component", None)
    if selected_id and selected_id not in {item.component_id for item in components}:
        selection = update_selection(st.session_state, project=project, selected_component=None)
        selected_id = selection.selected_component

    mount_status_key = f"{viewer_key}_{image_fingerprint[:16]}_mount_status"
    last_id_key = f"{viewer_key}_last_event"
    rejected_id_key = f"{viewer_key}_rejected_event"

    def handle_selection_change(result_override: Any | None = None) -> bool:
        # Callback-time session state is useful, but read the concrete result
        # from this render as well: CCv2's presented session_state can lag the
        # returned ComponentResult during a fast consecutive click.
        result = result_override if result_override is not None else st.session_state.get(viewer_key)
        viewer_status = (result.get("viewer_status") if isinstance(result, dict)
                         else getattr(result, "viewer_status", None))
        if isinstance(viewer_status, dict):
            if (viewer_status.get("asset_key") == expected_asset_key
                    and viewer_status.get("status") == "READY"
                    and viewer_status.get("image_size_px") == list(image_size_px)
                    and isinstance(viewer_status.get("event_id"), str)
                    and bool(viewer_status.get("event_id"))):
                ready_event_key = f"{mount_status_key}_last_event"
                if st.session_state.get(ready_event_key) != viewer_status["event_id"]:
                    st.session_state[ready_event_key] = viewer_status["event_id"]
                    st.session_state["cc_debug_next_full_script_rerun_cause"] = "viewer_ready_component_state"
                    events = st.session_state.setdefault("cc_performance_events", [])
                    events.append({"stage": "viewer_image_ready", "status": "complete",
                                   "asset_key_hash": hashlib.sha256(expected_asset_key.encode()).hexdigest()[:12]})
                    del events[:-50]
                    LOGGER.info("pcb_viewer event=VIEWER_READY viewer_key=%s project_id=%s photo_id=%s asset_hash=%s frame_id=%s ts_ns=%s",
                                viewer_key, project.project_id, frame_id,
                                hashlib.sha256(expected_asset_key.encode()).hexdigest()[:12],
                                frame_id, time.time_ns())
                st.session_state[mount_status_key] = "READY"
        event = None
        if isinstance(result, dict):
            event = result.get("selection_event") or result.get("selection")
        else:
            event = getattr(result, "selection_event", None) or getattr(result, "selection", None)
        if not isinstance(event, dict):
            LOGGER.debug("pcb_selection stage=python_receive viewer=%s result_type=%s selection=missing",
                         viewer_key, type(result).__name__)
            return False
        last_seq_key = f"{viewer_key}_last_seq"
        if event.get("event_id") == st.session_state.get(last_id_key):
            LOGGER.debug("pcb_selection stage=python_receive viewer=%s event_id=%s result=duplicate",
                         viewer_key, event.get("event_id"))
            return False
        seq = event.get("event_seq")
        LOGGER.debug("pcb_selection stage=python_receive viewer=%s event_id=%s seq=%r action=%s component_id=%s overlaps=%s owner=%s",
                     viewer_key, event.get("event_id"), seq, event.get("selection_action"),
                     event.get("component_id"), event.get("overlapping_component_ids"),
                     (event.get("project_id"), event.get("board_id"), event.get("board_side"),
                      event.get("canonical_frame_id")))
        _record_selection_trace(
            st.session_state,
            f"python_receive · {event.get('event_id')} · {event.get('selection_action')} · {event.get('component_id')}",
        )
        if (event.get("selection_action") == "draw"
                and (event.get("project_id"), event.get("board_id"), event.get("board_side"),
                     event.get("canonical_frame_id"))
                == (project.project_id, project.board_id or "", board_side, frame_id)
                and event.get("event_epoch") == event_epoch
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
            LOGGER.info("pcb_selection stage=python_commit viewer=%s event_id=%s action=draw result=accepted bbox=%s",
                        viewer_key, event["event_id"], [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))])
            return True
        if event.get("selection_action") == "draw":
            LOGGER.warning("pcb_selection stage=python_validate viewer=%s event_id=%s action=draw result=rejected owner_match=%s epoch_match=%s seq=%r last_seq=%s callback_available=%s",
                           viewer_key, event.get("event_id"),
                           (event.get("project_id"), event.get("board_id"), event.get("board_side"),
                            event.get("canonical_frame_id"))
                           == (project.project_id, project.board_id or "", board_side, frame_id),
                           event.get("event_epoch") == event_epoch, seq,
                           int(st.session_state.get(last_seq_key, 0)), on_box_draw is not None)
            return False
        component_id = event.get("component_id")
        allowed_ids = {item.component_id for item in components}
        accepted, reason = validate_component_selection_event(
            event, project_id=project.project_id, board_id=project.board_id or "",
            board_side=board_side, frame_id=frame_id, allowed_ids=allowed_ids,
            last_sequence=int(st.session_state.get(last_seq_key, 0)),
            event_epoch=event_epoch,
        )
        if not accepted:
            st.session_state[f"{viewer_key}_event_rejected"] = True
            st.session_state[rejected_id_key] = event.get("event_id")
            _record_selection_trace(st.session_state, f"python_validate · 거부 · {reason}")
            LOGGER.warning("pcb_selection stage=python_validate viewer=%s event_id=%s accepted=false reason=%s selected=%s allowed_count=%s",
                           viewer_key, event.get("event_id"), reason, component_id, len(allowed_ids))
            return False
        st.session_state.pop(rejected_id_key, None)
        st.session_state[last_id_key] = event.get("event_id")
        st.session_state[last_seq_key] = seq
        raw_overlap_ids = event.get("overlapping_component_ids", [component_id])
        # A delayed browser event can contain stale *other* overlap candidates.
        # Keep the validated selection and only expose candidates in this render.
        overlap_ids = [item for item in raw_overlap_ids if item in allowed_ids]
        if component_id not in overlap_ids:
            overlap_ids.insert(0, component_id)
        if event.get("selection_action") == "bulk-select":
            st.session_state[bulk_selection_key] = list(dict.fromkeys(overlap_ids))
        else:
            st.session_state[bulk_selection_key] = []
        overlap_key = f"{OVERLAPPING_SELECTION_KEY}_{viewer_key}"
        if component_id is None:
            st.session_state[overlap_key] = []
        else:
            st.session_state[overlap_key] = list(dict.fromkeys(overlap_ids))
        selected_model = next((item for item in components if item.component_id == component_id), None)
        board_component_id = getattr(selected_model, "board_component_id", None)
        # Commit local and board identity together before rerun. Otherwise the
        # previous board selection can restore its old photo-local ID.
        update_component_selection(st.session_state, project=project,
                                   selected_component=component_id,
                                   board_component_id=board_component_id)
        LOGGER.info("pcb_selection stage=python_commit viewer=%s event_id=%s action=select result=accepted selected=%s overlaps=%s",
                    viewer_key, event.get("event_id"), component_id, overlap_ids)
        _record_selection_trace(
            st.session_state,
            f"selection_state · {component_id or '선택 해제'} · 겹침 {len(overlap_ids)}개",
        )
        return True

    payload = [{
        "component_id": item.component_id,
        "board_component_id": item.board_component_id,
        "component_type": item.component_type,
        "overlay_source": item.overlay_source,
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
        "fault_hypothesis": item.fault_hypothesis,
        "fault_role_status": item.fault_role_status,
        "fault_history": list(item.fault_history),
        "functional_block_roles": list(item.functional_block_roles),
    } for item in components]
    ack_event_id = st.session_state.get(last_id_key)
    render_count_key = "cc_debug_viewer_render_request_count"
    render_count = int(st.session_state.get(render_count_key, 0)) + 1
    st.session_state[render_count_key] = render_count
    debug_telemetry = str(os.environ.get("CIRCUITCARE_VIEWER_DEBUG", "")).lower() in {"1", "true", "yes"}
    telemetry_key = f"cc_viewer_payload_profile_{viewer_key}"
    previous_profile = st.session_state.get(telemetry_key, {})
    image_unchanged = previous_profile.get("image_fingerprint") == image_fingerprint
    overlay_unchanged = previous_profile.get("overlay_version") == overlay_version
    payload_chars = len(image_b64)
    st.session_state[telemetry_key] = {
        "image_fingerprint": image_fingerprint, "overlay_version": overlay_version,
        "render_count": int(previous_profile.get("render_count", 0)) + 1,
        "base64_chars": payload_chars,
    }
    events = st.session_state.setdefault("cc_performance_events", [])
    events.append({"stage": "pcb_viewer_payload", "image_bytes_base64_chars": payload_chars,
                   "overlay_component_count": len(payload), "image_unchanged": image_unchanged,
                   "overlay_unchanged": overlay_unchanged,
                   "image_href_expected_change": not image_unchanged})
    del events[:-50]
    LOGGER.info("pcb_viewer event=VIEWER_RENDER_REQUESTED viewer_key=%s frame_id=%s render=%s image_hash=%s overlay_version=%s",
                viewer_key, frame_id, render_count, image_fingerprint[:12], overlay_version)
    rejected_event_id = st.session_state.get(rejected_id_key)
    result = render_pcb_viewer(
        st, image_png_base64=image_b64, image_size_px=image_size_px, components=payload,
        project_id=project.project_id, board_id=project.board_id or "", board_side=board_side,
        frame_id=frame_id, photo_id=photo_id or frame_id,
        viewer_ready_ack=st.session_state.get(mount_status_key) == "READY",
        analysis_loading=analysis_loading,
        selected_component_id=selected_id,
        selected_component_ids=list(st.session_state.get(bulk_selection_key, [])),
        asset_key=image_fingerprint,
        overlay_version=overlay_version, debug_telemetry=debug_telemetry,
        selection_ack_event_id=ack_event_id, selection_rejected_event_id=rejected_event_id,
        event_sequence_seed=int(st.session_state.get(f"{viewer_key}_last_seq", 0)),
        event_epoch=event_epoch,
        key=viewer_key, on_selection_change=None,
        on_selection_event_change=handle_selection_change,
        on_viewer_status_change=handle_selection_change,
    )
    result_processed = handle_selection_change(result)
    if result_processed:
        # Components v2 setStateValue()/setTriggerValue() already schedule a
        # Streamlit rerun. Apply the returned value as a late-callback fallback,
        # but do not force another full rerun here; doing so flashes the page
        # and remounts the Viewer after every click.
        LOGGER.info("pcb_selection stage=python_render_apply viewer=%s rerun_source=component_state",
                    viewer_key)
    current = st.session_state.get(SELECTION_KEY)
    LOGGER.debug("pcb_selection stage=render_result viewer=%s result_type=%s selected_state=%s",
                 viewer_key, type(result).__name__, getattr(current, "selected_component", None))
    if result is None:
        return False, viewer_key, coordinate_status_for_viewer(api_available=False, client_ready=False)
    mounted = st.session_state.get(mount_status_key) == "READY"
    return True, viewer_key, coordinate_status_for_viewer(api_available=True, client_ready=mounted)
