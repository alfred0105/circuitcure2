"""PCB-first F2 workbench shell with real intake images and an evidence-neutral panel."""

from __future__ import annotations

import hashlib
import base64
import io
import json
import time
from functools import lru_cache
from dataclasses import replace
from pathlib import Path
import tempfile
from typing import Any

from circuitcare.frontend.models import ProjectContext
from circuitcare.symptom_diagnosis.diagnosis_state import (
    component_inventory_version,
)
from circuitcare.frontend.component_overlay_adapter import (
    adapt_component_map_components, adapt_original_photo_components,
)
from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel
from circuitcare.frontend.board_component_registry import (
    REGISTRY_KEY, build_board_component_registry, confirm_component_match,
    diagnosis_components, measurement_component_aliases, measurement_view_model_for_board_component,
    remove_component_match, source_key_for_component,
)
from circuitcare.component_taxonomy import review_taxonomy
from circuitcare.frontend.measurement_state import get_measurements
from circuitcare.analysis.fingerprints import build_measurement_fingerprint
from circuitcare.analysis.models import AnalysisScope, AnalysisStatus
from circuitcare.frontend.component_map_adapter import (
    COMPONENT_MAP_STATE_KEY, LOGGER, REGISTRATION_LABELS,
    clear_component_map_state,
    component_map_cache_key, component_map_entry_is_current, decode_canonical_image,
    map_backend_photo_ids, optional_backend_provider, photo_input_fingerprint,
    validate_component_map,
)
from circuitcare.frontend.photo_adapter import adapt_component_map_input
from circuitcare.dashboard.components.pcb_workspace import (
    OVERLAPPING_SELECTION_KEY, _record_selection_trace,
    render_component_viewer, scale_overlay_to_display,
)
from circuitcare.dashboard.views.inspection_panel import (
    COMPONENT_EDIT_KEY, COMPONENT_INVENTORY_KEY, MAP_VERSION_KEY,
    board_suspects, render_inspection_panel, render_retraining_workspace,
    _diagnosis_status_label,
)
from circuitcare.dashboard.views.symptom_diagnosis_panel import (
    candidate_rows_for_strategy, candidate_focus_widget_key, current_analysis_snapshot,
    render_symptom_diagnosis_panel, _simple_candidate_rows,
)
from circuitcare.dashboard.views.measurement_panel import render_measurement_panel
from circuitcare.symptom_diagnosis import load_knowledge
from circuitcare.dashboard.views.diagnosis_panel import render_diagnosis_panel
from circuitcare.dashboard.views.report_panel import render_report_panel
from circuitcare.reporting.report_builder import build_pcb_diagnostic_snapshot
from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel
from circuitcare.vision.component_recognition_panel import (
    _observation_fingerprint,
    _review_fingerprint, _review_user_version, ensure_component_detection,
    invalidate_component_review_state, record_box_exclusion,
    merge_ai_review_rows, render_component_recognition_panel,
    review_aux_keys, review_keys, review_scope as component_review_scope,
)
from circuitcare.vision.box_reconciliation import board_visible_review_rows, same_component_box_pair
from circuitcare.frontend.state import (
    INTAKE_IMAGES_KEY, INTAKE_PROJECT_KEY, clear_intake_draft,
)
from circuitcare.frontend.workbench_state import (
    ACTIVE_PHOTO_KEY, BOARD_COMPONENT_SELECTION_KEY, MEASUREMENT_TARGET_BOARD_COMPONENT_KEY,
    MODE_KEY, SCREEN_KEY, SELECTION_KEY, VIEW_ZOOM_KEY,
    VIEW_TRANSFORM_KEY, WORKFLOW_REVIEW, WORKFLOW_QUESTIONS, WORKFLOW_INSPECTION,
    WORKFLOW_REASSESSMENT, WORKFLOW_COMPLETE, get_workflow_state, transition_workflow,
    enter_workbench, update_component_selection, update_selection,
)
from circuitcare.frontend.viewer_geometry import requested_view_transform
from circuitcare.schematic.exporter import export_schematic
from circuitcare.schematic.geometry import read_svg_view_box
from circuitcare.schematic.viewer import render_interactive_viewer
from circuitcare.vision.image_loader import ImageLoadError, load_board_image
from circuitcare.vision.ai_review_jobs import get_ai_review_job

PENDING_PHOTO_FOCUS_KEY = "cc_pending_board_component_photo_focus"


def _commit_review_job(st: Any, *, scope: str) -> bool:
    state = st.session_state
    job_id = state.get(f"cc_ai_review_job_{scope}")
    owner = state.get(f"cc_ai_review_job_owner_{scope}")
    if not job_id or not owner:
        return False
    job = get_ai_review_job(owner=owner, job_id=job_id)
    if not job or job.get("status") in {"RUNNING", "PENDING"}:
        return False
    if state.get(f"cc_ai_review_job_committed_{scope}") == job_id:
        return False
    fingerprint = state.get(f"cc_ai_review_job_fingerprint_{scope}")
    if job.get("fingerprint") != fingerprint:
        state[f"cc_ai_review_job_status_{scope}"] = "STALE"
        state[f"cc_ai_review_job_committed_{scope}"] = job_id
        LOGGER.info("AI_JOB_COMMITTED scope_hash=%s status=STALE reason=job_fingerprint_mismatch",
                    hashlib.sha256(scope.encode()).hexdigest()[:12])
        return True
    records_key = review_keys(scope)[1]
    current_rows = state.get(records_key, [])
    worker_user_version = job.get("user_version")
    current_user_version = _review_user_version(current_rows)
    current_review_fingerprint = _review_fingerprint(current_rows, None)
    submitted_review_fingerprint = state.get(f"cc_ai_review_job_review_fingerprint_{scope}")
    if (_observation_fingerprint(current_rows) != fingerprint
            or job.get("review_fingerprint") != submitted_review_fingerprint
            or (current_review_fingerprint != submitted_review_fingerprint
                and worker_user_version == current_user_version)):
        state[f"cc_ai_review_job_status_{scope}"] = "STALE"
        state[f"cc_ai_review_job_committed_{scope}"] = job_id
        LOGGER.info("AI_JOB_COMMITTED scope_hash=%s status=STALE reason=fingerprint_changed",
                    hashlib.sha256(scope.encode()).hexdigest()[:12])
        return True
    worker_status = str(job.get("status", "FAILED")).upper()
    if worker_status in {"FAILED", "UNAVAILABLE"}:
        state[f"cc_ai_review_job_status_{scope}"] = worker_status
        state[f"cc_ai_review_job_error_{scope}"] = job.get("error") or (
            (job.get("progress") or {}).get("message") if isinstance(job.get("progress"), dict) else None
        )
        state[f"cc_ai_review_job_committed_{scope}"] = job_id
        state[f"cc_ai_review_job_result_ms_{scope}"] = job.get("duration_ms")
        LOGGER.info("AI_JOB_COMMITTED scope_hash=%s status=%s reason=worker_failed attempt=%s",
                    hashlib.sha256(scope.encode()).hexdigest()[:12], worker_status, job.get("attempt"))
        return True
    result_state = job.get("state", {})
    selected_ids = job.get("selected_component_ids")
    worker_review = (result_state.get(review_aux_keys(scope)[1])
                     if isinstance(result_state, dict) else None)
    if (not selected_ids or not isinstance(worker_review, dict)
            or worker_review.get("review_mode") != "selected_class_only"
            or sorted(map(str, selected_ids)) != sorted(map(str, worker_review.get("selected_component_ids", [])))):
        state[f"cc_ai_review_job_status_{scope}"] = "STALE"
        state[f"cc_ai_review_job_committed_{scope}"] = job_id
        LOGGER.info("AI_JOB_COMMITTED scope_hash=%s status=STALE reason=not_selected_class_review",
                    hashlib.sha256(scope.encode()).hexdigest()[:12])
        return True
    worker_rows = result_state.get(records_key, [])
    merged_rows = merge_ai_review_rows(current_rows, worker_rows if isinstance(worker_rows, list) else [])
    if worker_user_version != current_user_version:
        LOGGER.info("AI_REVIEW_USER_VERSION_CHANGED scope_hash=%s action=merge_ai_fields_only",
                    hashlib.sha256(scope.encode()).hexdigest()[:12])
    state[records_key] = merged_rows
    allowed = (*review_aux_keys(scope), f"cc_component_marking_ocr_{scope}",
               f"cc_component_qwen_batches_{scope}", f"cc_component_qwen_pair_queue_{scope}",
               f"cc_ai_review_run_{scope}")
    for key in allowed:
        if key in result_state:
            state[key] = result_state[key]
    # Worker row data is merged by component ID and only AI-owned fields are accepted.
    review_result = state.get(review_aux_keys(scope)[1])
    committed_status = worker_status
    if isinstance(review_result, dict) and review_result.get("verification_status") == "AI_REVIEW_PARTIAL":
        committed_status = "AI_REVIEW_PARTIAL"
    state[f"cc_ai_review_job_status_{scope}"] = committed_status
    state[f"cc_ai_review_job_result_ms_{scope}"] = job.get("duration_ms")
    state[f"cc_ai_review_job_messages_{scope}"] = job.get("messages", [])
    state[f"cc_ai_review_job_committed_{scope}"] = job_id
    metrics = state.setdefault("cc_performance_events", [])
    metrics.append({"stage": "ocr_qwen_review", "duration_ms": job.get("duration_ms"),
                    "status": job.get("status"), "image_bytes": None,
                    "scope_hash": hashlib.sha256(scope.encode()).hexdigest()[:12]})
    del metrics[:-50]
    LOGGER.info("AI_JOB_COMMITTED scope_hash=%s status=%s duration_ms=%s",
                hashlib.sha256(scope.encode()).hexdigest()[:12], committed_status, job.get("duration_ms"))
    return True


def _render_ai_review_progress(st: Any, project_id: str, scope: str | None) -> None:
    if not scope:
        return
    from circuitcare.dashboard.components.review_progress import render_review_progress
    state = st.session_state
    job_id = state.get(f"cc_ai_review_job_{scope}")
    owner = state.get(f"cc_ai_review_job_owner_{scope}")
    if not job_id or not owner:
        return
    job = get_ai_review_job(owner=owner, job_id=job_id)
    if job is None:
        st.warning("AI 작업 상태를 찾을 수 없습니다. 검수 기록은 보존됩니다.")
        return
    def completed():
        # Browser events only request a commit. Identity, terminal status and
        # submission versions are still validated by the existing Python path.
        if (state.get(f"cc_ai_review_job_{scope}") == job_id
                and state.get(f"cc_ai_review_job_owner_{scope}") == owner
                and _commit_review_job(st, scope=scope)):
            state["cc_debug_next_fragment_rerun_cause"] = "ai_review_terminal_commit"
            LOGGER.info("AI_JOB_COMPLETED project=%s job_id=%s transport=sse", project_id, job_id)
    render_review_progress(st, owner=owner, job_id=job_id, job=job, on_completed=completed)


@lru_cache(maxsize=4)
def _cached_photo_pixels(raw: bytes) -> tuple[Any, tuple[int, int], tuple[int, int], bytes]:
    """Decode/normalize a photo once; callers receive a safe copy of the PIL image."""
    loaded = load_board_image(raw)
    output = io.BytesIO()
    loaded.image.save(output, format="PNG")
    return loaded.image.copy(), loaded.original_size, loaded.processed_size, output.getvalue()


def _record_perf(state: Any, stage: str, started: float, *, status: str = "complete") -> None:
    events = state.setdefault("cc_performance_events", [])
    events.append({"stage": stage, "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                   "status": status})
    del events[:-50]


def _style(st: Any) -> None:
    st.markdown("""<style>
    /* Preserve a real top safe-area below Streamlit's toolbar; otherwise the
       top navigation and brand are partially hidden in Safari. */
    header[data-testid="stHeader"], [data-testid="stToolbar"] { display:none !important; }
    [data-testid="stAppViewContainer"] > .main { margin-top:0 !important; padding-top:0 !important; }
    [data-testid="stMainBlockContainer"] { max-width:none !important; padding:0 2rem .9rem !important; overflow:visible; }
    .st-key-cc_workbench_top_navigation { position:relative; top:-.35rem; z-index:2; overflow:visible; margin:0 0 .2rem; padding:0; }
    .st-key-cc_workbench_brand_and_steps { gap:.5rem; margin-bottom:-.35rem; }
    .st-key-cc_workbench_stepper { margin-top:.25rem; margin-bottom:0; }
    .st-key-cc_workbench_stepper [data-testid="stMarkdownContainer"] p { margin:.1rem 0; line-height:1.25; }
    .st-key-cc_workbench_body h4 { margin:.2rem 0 .25rem; }
    .cc-wb-header { display:flex; align-items:center; gap:.6rem; color:#e9eef4; font-weight:700; font-size:1.1rem; margin:0; line-height:1.2; }
    .cc-wb-subtle { color:#9aa9b8; font-size:.82rem; }
    .cc-analysis-hero { background:linear-gradient(135deg,#172536,#111923); border:1px solid #34475c; border-radius:12px; padding:1rem 1.15rem; margin:.25rem 0 1rem; }
    .cc-analysis-step { color:#7fb3ef; font-size:.78rem; font-weight:700; letter-spacing:.04em; text-transform:uppercase; }
    .cc-analysis-title { color:#edf3f9; font-size:1.15rem; font-weight:700; margin:.2rem 0 .45rem; }
    .cc-analysis-copy { color:#aab8c7; font-size:.9rem; }
    .cc-analysis-section { border-top:1px solid #334252; margin-top:1.1rem; padding-top:.75rem; }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.cc-inspection-marker) { position:sticky; top:1.25rem; align-self:start; }
    [class*="st-key-cc_viewer_canvas"] { overflow:auto; max-height:calc(100vh - 13rem); background:#101820; border:1px solid #334252; border-radius:8px; padding:.5rem; }
    div[role="radiogroup"] label { border:1px solid #334252; border-radius:6px; padding:.35rem .8rem; }
    @media (max-width:900px) { [data-testid="stMainBlockContainer"] { padding:0 .8rem .7rem !important; } }
    @media (max-width:900px) {
      .st-key-cc_workbench_body [data-testid="stHorizontalBlock"] { flex-wrap:wrap; }
      .st-key-cc_workbench_body [data-testid="column"] { flex:1 1 100%; width:100%; min-width:100%; }
      [data-testid="stVerticalBlockBorderWrapper"]:has(.cc-inspection-marker) { position:static; }
    }
    @media (max-width:700px) { [class*="st-key-cc_viewer_canvas"] { max-height:60vh; } }
    </style>""", unsafe_allow_html=True)


def _bind_review_models_to_map(
    map_models: tuple[ComponentOverlayViewModel, ...],
    review_models: tuple[ComponentOverlayViewModel, ...],
    review_rows: list[dict[str, Any]],
) -> tuple[tuple[ComponentOverlayViewModel, ...], list[dict[str, Any]]]:
    """Use review rows as canonical boxes; only attach map evidence on certain IDs."""
    review_by_id = {str(row.get("component_id")): row for row in review_rows if row.get("component_id")}
    auto_by_id = {item.component_id: item for item in review_models}
    consumed_map_ids: set[str] = set()
    ambiguous_map_ids: set[str] = set()
    handled_review_ids: set[str] = set()
    rendered: list[ComponentOverlayViewModel] = []

    # Stable aliases are authoritative. Geometry is only a fallback and must be
    # unique in both directions; choosing the highest-confidence nearby map box
    # silently attached evidence to the wrong physical component.
    map_by_id = {item.component_id: item for item in map_models}
    matches_by_row: dict[str, list[str]] = {}
    rows_by_map: dict[str, list[str]] = {}
    for row_id, row in review_by_id.items():
        box = [row.get(key) for key in ("x1", "y1", "x2", "y2")]
        if not all(isinstance(value, (int, float)) for value in box):
            continue
        aliases = row.get("associated_component_ids", [])
        if not isinstance(aliases, (list, tuple, set)):
            aliases = [aliases]
        aliases = [str(item) for item in aliases if item]
        exact = [item for item in aliases if item in map_by_id]
        if exact:
            matches_by_row[row_id] = exact[:1]
            rows_by_map.setdefault(exact[0], []).append(row_id)
            continue
        matches = [item.component_id for item in map_models
                   if same_component_box_pair(box, item.bbox_xyxy)]
        matches_by_row[row_id] = matches
        for map_id in matches:
            rows_by_map.setdefault(map_id, []).append(row_id)

    certain_map_by_row: dict[str, str] = {}
    for row_id, map_ids in matches_by_row.items():
        # Require one candidate and a reciprocal one-row match, even for stored
        # aliases, so duplicated or conflicting review rows remain unbound.
        if len(map_ids) == 1 and len(rows_by_map.get(map_ids[0], [])) == 1:
            certain_map_by_row[row_id] = map_ids[0]
        elif map_ids:
            ambiguous_map_ids.update(map_ids)

    for row_id, row in review_by_id.items():
        box = [row.get(key) for key in ("x1", "y1", "x2", "y2")]
        if not all(isinstance(value, (int, float)) for value in box):
            continue
        handled_review_ids.add(row_id)
        map_id = certain_map_by_row.get(row_id)
        map_match = map_by_id.get(map_id) if map_id else None
        if map_match is not None:
            consumed_map_ids.add(map_match.component_id)
            aliases = row.get("associated_component_ids", [])
            if not isinstance(aliases, (list, tuple, set)):
                aliases = [aliases]
            if map_match.component_id not in aliases:
                row["associated_component_ids"] = [*aliases, map_match.component_id]
            row.pop("map_match_status", None)
        elif matches_by_row.get(row_id):
            row["map_match_status"] = "AMBIGUOUS_GEOMETRY"
        if row.get("상태") == "제외":
            continue
        auto = auto_by_id.get(row_id)
        if auto is None:
            continue
        if map_match is not None:
            auto = replace(auto,
                source_backend_photo_ids=map_match.source_backend_photo_ids,
                observation_ids=map_match.observation_ids,
                identity_match_status=map_match.identity_match_status,
                fault_history=tuple((*auto.fault_history, *map_match.fault_history)),
                evidence=tuple(dict.fromkeys((*auto.evidence, *map_match.evidence))))
        rendered.append(auto)
    rendered.extend(item for item in review_models if item.component_id not in handled_review_ids)
    rendered.extend(item for item in map_models
                    if item.component_id not in consumed_map_ids and item.component_id not in ambiguous_map_ids)
    return tuple(rendered), review_rows


def _attach_verified_history(
    components: tuple[ComponentOverlayViewModel, ...],
    history_by_id: dict[str, list[dict[str, Any]]],
) -> tuple[ComponentOverlayViewModel, ...]:
    """Attach retained verification records without promoting them to live status."""
    return tuple(
        replace(model, fault_history=tuple(history_by_id[model.component_id]))
        if model.component_id in history_by_id else model
        for model in components
    )


def _prepare_photo_component_snapshot(
    st: Any, project: ProjectContext, photos: list[dict[str, Any]], *,
    photo_id: str | None = None, activate: bool = True,
    workflow_status_slot: Any = None, workflow_status: Any = None,
) -> dict[str, Any]:
    """Prepare one photo's review snapshot without depending on Viewer rendering."""
    records = {item.image_id: item for item in project.image_records}
    valid = [item for item in photos if item.get("image_id") in records
             and isinstance(item.get("data"), bytes)]
    if not valid:
        return {"photo_id": None, "components": (), "review_scope": None, "error": None}
    ids = [item["image_id"] for item in valid]
    photo_id = photo_id or st.session_state.get("circuitcare_v2_photo_choice")
    if photo_id not in ids:
        photo_id = st.session_state.get(ACTIVE_PHOTO_KEY)
    if photo_id not in ids:
        photo_id = ids[0]
    if activate:
        st.session_state["circuitcare_v2_photo_choice"] = photo_id
    if activate and st.session_state.get(ACTIVE_PHOTO_KEY) != photo_id:
        update_selection(st.session_state, project=project, active_photo_id=photo_id)
    image_data = next(item["data"] for item in valid if item["image_id"] == photo_id)
    try:
        _cached_image, _source_size, processed_size, detector_image_bytes = _cached_photo_pixels(image_data)
    except (ImageLoadError, ValueError):
        return {"photo_id": photo_id, "components": (), "review_scope": None,
                "error": "PCB 이미지 로드 실패"}
    review_scope = component_review_scope(detector_image_bytes, project.project_id, photo_id)
    result_key, records_key, error_key = review_keys(review_scope)
    detection_needs_run = result_key not in st.session_state and error_key not in st.session_state
    progress = workflow_status
    if detection_needs_run and progress is None and workflow_status_slot is not None:
        with workflow_status_slot.container():
            progress = st.status("PCB 소자 분석 중", expanded=False)
    elif detection_needs_run and progress is None:
        # Non-workbench callers retain a status surface, but the interactive
        # workbench always supplies its dedicated workflow slot.
        progress = st.status("PCB 소자 분석 중", expanded=False)
    if detection_needs_run:
        LOGGER.info("PCB_ANALYSIS_STAGE event=YOLO_STARTED project=%s photo_id=%s mode=synchronous reason=locator_load_and_prediction_run_on_script_thread ts_ns=%s",
                    project.project_id, photo_id, time.time_ns())
    auto_result, auto_error = ensure_component_detection(
        st, image_bytes=image_data, project_id=project.project_id, photo_id=photo_id,
        review_image_bytes=detector_image_bytes, review_image_size=processed_size,
        progress=progress,
    )
    if detection_needs_run and progress is not None:
        if auto_error or auto_result is None:
            progress.update(label="YOLO 탐지 실패 · 검수 기록 보존", state="error", expanded=False)
            LOGGER.info("PCB_ANALYSIS_STAGE event=YOLO_FINISHED project=%s photo_id=%s status=error ts_ns=%s",
                        project.project_id, photo_id, time.time_ns())
        else:
            progress.update(label=f"YOLO 탐지 완료 · {len(auto_result.get('detections', []))}개",
                            state="complete", expanded=False)
            LOGGER.info("PCB_ANALYSIS_STAGE event=YOLO_FINISHED project=%s photo_id=%s status=complete detections=%s ts_ns=%s",
                        project.project_id, photo_id, len(auto_result.get("detections", [])), time.time_ns())
    rows = list(st.session_state.get(records_key, []))
    review_state = "unavailable"
    if (auto_result is not None and not auto_error
            and auto_result.get("image_width") and auto_result.get("image_height")):
        # Qwen classification is explicitly user-triggered for the selected box.
        # Only collect/commit an already submitted job here (for photo switching).
        job_id = st.session_state.get(f"cc_ai_review_job_{review_scope}")
        owner = st.session_state.get(f"cc_ai_review_job_owner_{review_scope}")
        job = get_ai_review_job(owner=owner, job_id=job_id) if owner and job_id else None
        job_state = str(job.get("status")) if isinstance(job, dict) else "NOT_REQUESTED"
        if job_state in {"COMPLETE", "AI_REVIEW_PARTIAL", "FAILED", "UNAVAILABLE", "STALE"}:
            # A job may finish while the user views another photo. Commit its
            # result only when this photo is selected again and its input still matches.
            _commit_review_job(st, scope=review_scope)
            rows = list(st.session_state.get(records_key, []))
        elif job_state in {"PENDING", "RUNNING"}:
            st.session_state[f"cc_ai_review_job_status_{review_scope}"] = job_state
        if job_state == "NOT_REQUESTED":
            review_state = "not_requested"
        else:
            review_state = ("complete" if job_state == "COMPLETE" else
                            "partial" if job_state == "AI_REVIEW_PARTIAL" else
                            "failed" if job_state in {"FAILED", "UNAVAILABLE"} else
                            "pending" if job_state in {"PENDING", "RUNNING"} else
                            "stale" if job_state == "STALE" else "unavailable")
    _ocr_key, qwen_key = (f"cc_component_ocr_{review_scope}", f"cc_component_qwen_review_{review_scope}")
    qwen_result = st.session_state.get(qwen_key, {})
    qwen_types = {str(item.get("detection_id")): item for item in qwen_result.get("class_reviews", [])
                  if isinstance(item, dict)} if isinstance(qwen_result, dict) else {}
    inventory_version = component_inventory_version(
        project_id=project.project_id, board_id=project.board_id, photo_id=photo_id,
        image_sha256=hashlib.sha256(image_data).hexdigest(), components=[
            {"component_id": row.get("component_id"), "component_type": row.get("소자 종류", "unknown"),
             "user_component_type_id": row.get("user_component_type_id"),
             "bbox_xyxy": [row.get("x1", 0), row.get("y1", 0), row.get("x2", 0), row.get("y2", 0)]}
            for row in rows if isinstance(row, dict)
        ],
    )
    if activate:
        st.session_state[MAP_VERSION_KEY] = inventory_version
        st.session_state["circuitcare_v4_component_inventory_scope"] = review_scope
    visible_rows = board_visible_review_rows(rows)
    review_hash = hashlib.sha256(detector_image_bytes).hexdigest()[:16]
    side = {"front": "TOP", "back": "BOTTOM"}.get(records[photo_id].side, records[photo_id].side)
    review_models = tuple(ComponentOverlayViewModel(
        project_id=project.project_id, board_id=project.board_id or "", board_side=side,
        component_id=str(row.get("component_id") or f"auto-{review_hash}-{index + 1}"),
        component_type=str(review_taxonomy(row)["effective_class"]),
        display_label=(f"{row['user_component_type_name']} · {review_taxonomy(row)['effective_class']}"
                       if row.get("user_component_type_name") else
                       f"AI 제안 · {review_taxonomy(row)['effective_class']}" if row.get("source") == "qwen_suggestion"
                       else f"수동 · {review_taxonomy(row)['effective_class']}" if row.get("source") == "manual"
                       else f"{index + 1} · {review_taxonomy(row)['effective_class']}"),
        canonical_frame_id=photo_id, coordinate_frame="SOURCE_PHOTO_PIXEL",
        bbox_xyxy=(float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
        polygon_xy=(), source_backend_photo_ids=(photo_id,), observation_ids=(),
        detection_confidence=float(row.get("신뢰도", 0.0)),
        identification_status="USER_CONFIRMED" if row.get("상태") == "확인" else "SUGGESTED_CLASS_ONLY",
        identity_match_status=None, ocr_marking=row.get("body_marking"),
        possible_part_number=row.get("confirmed_part_number") or row.get("possible_part_number"),
        evidence=("Qwen 미확인 박스 제안" if row.get("source") == "qwen_suggestion"
                  else "사용자 추가 박스" if row.get("source") == "manual"
                  else "YOLO component proposal; awaiting user review",),
        overlay_status="EXCLUDED" if row.get("상태") == "제외" else "CANDIDATE",
        diagnosis_display_status="DIAGNOSIS_NOT_RUN",
        location_confidence=float(row.get("location_confidence", row.get("신뢰도", 0.0))),
        type_confidence=row.get("type_confidence"), match_score=row.get("match_score"),
        component_type_status=review_taxonomy(row)["component_type_status"],
        presence_status="CONFIRMED" if row.get("상태") == "확인" else "UNCONFIRMED",
        ocr_status=row.get("ocr_association_status"),
        ocr_candidates=tuple(row.get("ocr_candidates", [])),
        ocr_class_hints=tuple(row.get("ocr_class_hints", [])),
        reference_designators=tuple(row.get("reference_designators", [])),
        overlay_source=row.get("source", "yolo"),
    ) for index, row in enumerate(visible_rows))
    edits = st.session_state.get(COMPONENT_EDIT_KEY, {})
    if edits:
        review_models = tuple(replace(model,
            ocr_marking=edits.get(model.component_id, {}).get("ocr_marking", model.ocr_marking) or None,
            component_info=edits.get(model.component_id, {}).get("component_info", model.component_info) or None)
            for model in review_models)
    owner = f"{project.project_id}|{project.board_id or ''}|{review_scope or ''}"
    inventories = st.session_state.get(COMPONENT_INVENTORY_KEY, {})
    inventory = inventories.get(owner, {}) if isinstance(inventories, dict) else {}
    review_models = tuple(replace(model,
        confirmed_functional_roles=tuple(inventory.get(model.component_id, {}).get("confirmed_functional_roles", [])),
        part_identity_status=inventory.get(model.component_id, {}).get("part_identity_status", "UNCONFIRMED"),
    ) for model in review_models)
    scope = AnalysisScope(project.project_id, project.board_id or "", str(photo_id))
    image_fingerprint = hashlib.sha256(image_data).hexdigest()
    review_payload = [{"id": item.component_id, "type": item.component_type,
                       "label": item.display_label, "bbox": item.bbox_xyxy,
                       "marking": item.ocr_marking, "info": item.component_info,
                       "presence": item.presence_status, "type_status": item.component_type_status,
                       "identity": item.part_identity_status,
                       "roles": item.confirmed_functional_roles} for item in review_models]
    review_version = hashlib.sha256(json.dumps(review_payload, sort_keys=True, default=str).encode()).hexdigest()
    by_scope = dict(st.session_state.get("circuitcare_v6_photo_component_snapshots", {}))
    previous_snapshot = by_scope.get(scope.key, {})
    cached_final = (previous_snapshot.get("components")
                    if previous_snapshot.get("image_fingerprint") == image_fingerprint
                    and previous_snapshot.get("review_version") == review_version else None)
    models = tuple(cached_final) if cached_final is not None else review_models
    snapshot = {"project_id": project.project_id, "board_id": project.board_id or "",
                "photo_id": photo_id, "review_scope": review_scope,
                "image_fingerprint": image_fingerprint,
                "image_size_px": list(processed_size),
                "review_version": review_version, "review_components": review_models,
                "registry_components": tuple({
                    "component_id": str(row.get("component_id") or ""),
                    **review_taxonomy(row),
                    "component_type": review_taxonomy(row)["effective_class"],
                    "bbox_xyxy": [row.get("x1", 0), row.get("y1", 0), row.get("x2", 0), row.get("y2", 0)],
                    "coordinate_frame": "SOURCE_PHOTO_PIXEL",
                    "identification_status": ("EXCLUDED" if row.get("상태") == "제외" else
                                              "USER_CONFIRMED" if row.get("상태") == "확인" else "UNCONFIRMED"),
                    "presence_status": ("EXCLUDED" if row.get("상태") == "제외" else
                                        "CONFIRMED" if row.get("상태") == "확인" else "UNCONFIRMED"),
                    "component_type_status": review_taxonomy(row)["component_type_status"],
                    "user_component_type_id": row.get("user_component_type_id"),
                    "user_component_type_name": row.get("user_component_type_name"),
                    "user_component_parent_class": row.get("user_component_parent_class"),
                    "ocr_marking": row.get("body_marking"),
                    "body_marking": row.get("body_marking"),
                    "ocr_candidates": row.get("ocr_candidates", []),
                    "marking_candidates": row.get("marking_candidates", []),
                    "ocr_source": row.get("ocr_source", row.get("marking_ocr_source")),
                    "ocr_score": row.get("ocr_score", row.get("marking_ocr_confidence")),
                    "possible_part_number": row.get("possible_part_number"),
                    "part_number_status": row.get("part_number_status", "UNVERIFIED_PART_ID"
                        if row.get("possible_part_number") else "UNREAD"),
                    "confirmed_part_number": row.get("confirmed_part_number"),
                    "datasheet_source": row.get("datasheet_source"),
                    "part_identity_status": row.get("part_identity_status", "UNCONFIRMED"),
                    "ic_identity_candidate": row.get("ic_identity_candidate"),
                    "ocr_class_hints": row.get("ocr_class_hints", []),
                    "ocr_inferred_value": row.get("ocr_inferred_value"),
                    "ocr_inference_method": row.get("ocr_inference_method"),
                    "qwen_component_type_candidate": (row.get("qwen_component_type_candidate")
                        or qwen_types.get(str(row.get("component_id")), {}).get("proposed_base_class",
                            qwen_types.get(str(row.get("component_id")), {}).get("proposed_class"))),
                    "qwen_component_subtype_candidate": (row.get("qwen_component_subtype_candidate")
                        or qwen_types.get(str(row.get("component_id")), {}).get("proposed_subtype")),
                    "qwen_base_class": row.get("qwen_base_class") or row.get("qwen_component_type_candidate")
                        or qwen_types.get(str(row.get("component_id")), {}).get("proposed_base_class"),
                    "qwen_subtype": row.get("qwen_subtype") or row.get("qwen_component_subtype_candidate")
                        or qwen_types.get(str(row.get("component_id")), {}).get("proposed_subtype"),
                    "confirmed_base_class": row.get("confirmed_base_class"),
                    "confirmed_subtype": row.get("confirmed_subtype"),
                    "effective_class": row.get("effective_class") or review_taxonomy(row)["effective_class"],
                    "detector_class": row.get("detector_class") or row.get("소자 종류", "unknown"),
                    "qwen_component_type_alternatives": (row.get("qwen_component_type_alternatives")
                        or qwen_types.get(str(row.get("component_id")), {}).get("alternative_classes", [])),
                    "qwen_component_type_evidence": (row.get("qwen_component_type_evidence")
                        or qwen_types.get(str(row.get("component_id")), {}).get("visual_evidence")
                        or qwen_types.get(str(row.get("component_id")), {}).get("evidence")),
                    "qwen_component_type_confidence": row.get("qwen_component_type_confidence")
                        or qwen_types.get(str(row.get("component_id")), {}).get("confidence"),
                    "qwen_component_type_decision": row.get("qwen_component_type_decision")
                        or qwen_types.get(str(row.get("component_id")), {}).get("decision"),
                    "qwen_component_type_status": (row.get("qwen_component_type_status") or ("AI_PROPOSED_UNVERIFIED"
                        if str(row.get("component_id")) in qwen_types else "NOT_REVIEWED")),
                    "qwen_review_status": row.get("qwen_review_status") or qwen_types.get(
                        str(row.get("component_id")), {}).get("review_status"),
                    "qwen_visual_evidence": row.get("qwen_visual_evidence") or qwen_types.get(
                        str(row.get("component_id")), {}).get("visual_evidence"),
                    "qwen_ocr_evidence": row.get("qwen_ocr_evidence") or qwen_types.get(
                        str(row.get("component_id")), {}).get("ocr_evidence"),
                    "qwen_conflicting_evidence": row.get("qwen_conflicting_evidence") or qwen_types.get(
                        str(row.get("component_id")), {}).get("conflicting_evidence", []),
                    "qwen_marking_candidates": row.get("qwen_marking_candidates") or qwen_types.get(
                        str(row.get("component_id")), {}).get("marking_candidates", []),
                    "qwen_part_number_candidates": row.get("qwen_part_number_candidates") or qwen_types.get(
                        str(row.get("component_id")), {}).get("part_number_candidates", []),
                    "qwen_functional_role_candidates": row.get("qwen_functional_role_candidates") or qwen_types.get(
                        str(row.get("component_id")), {}).get("functional_role_candidates", []),
                    "confirmed_functional_roles": inventory.get(str(row.get("component_id")), {}).get(
                        "confirmed_functional_roles", []),
                    "evidence": ("사용자 제외 기록",) if row.get("상태") == "제외" else
                                ("사용자 검수 확인",) if row.get("상태") == "확인" else
                                ("사진별 탐지 후보",),
                    "overlay_source": row.get("source", "yolo"),
                } for row in rows if row.get("component_id")),
                "components": models, "component_ids": tuple(item.component_id for item in models),
                "detection_status": ("BLOCKED_MODEL_UNAVAILABLE" if auto_error else
                                     "INFERENCE_COMPLETE" if auto_result is not None else "UNKNOWN"),
                "ai_review_status": review_state,
                "error": auto_error, "version": inventory_version,
                "analysis_scope_key": scope.key}
    by_scope[scope.key] = snapshot
    st.session_state["circuitcare_v6_photo_component_snapshots"] = by_scope
    if activate:
        st.session_state["circuitcare_v4_current_components"] = models
        st.session_state["circuitcare_v6_current_component_scope"] = scope.key
        st.session_state["circuitcare_v4_detection_status"] = snapshot["detection_status"]
        st.session_state["circuitcare_v4_component_inventory_scope"] = review_scope
    return snapshot


def _prepare_board_registry_snapshot(
    st: Any, project: ProjectContext, photos: list[dict[str, Any]],
    *, workflow_status_slot: Any = None, workflow_status: Any = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Refresh cached photo snapshots, then build the independent board registry."""
    records = {item.image_id: item for item in project.image_records}
    selection_before = st.session_state.get(SELECTION_KEY)
    previous_photo_id = getattr(selection_before, "active_photo_id", None)
    previous_local_id = getattr(selection_before, "selected_component", None)
    by_photo: dict[str, dict[str, Any]] = {}
    selected_id = st.session_state.get("circuitcare_v2_photo_choice")
    if selected_id not in {str(photo.get("image_id") or "") for photo in photos}:
        selected_id = st.session_state.get(ACTIVE_PHOTO_KEY)
    if selected_id not in {str(photo.get("image_id") or "") for photo in photos}:
        selected_id = next((str(photo.get("image_id")) for photo in photos
                            if photo.get("image_id") in records and isinstance(photo.get("data"), bytes)), None)
    cached_snapshots = st.session_state.get("circuitcare_v6_photo_component_snapshots", {})
    for photo in photos:
        photo_id = str(photo.get("image_id") or "")
        if photo_id not in records or not isinstance(photo.get("data"), bytes):
            continue
        if photo_id == selected_id:
            by_photo[photo_id] = _prepare_photo_component_snapshot(
                st, project, photos, photo_id=photo_id, activate=False,
                workflow_status_slot=workflow_status_slot,
                workflow_status=workflow_status,
            )
            continue
        raw_fingerprint = hashlib.sha256(photo["data"]).hexdigest()
        cached = next((snapshot for snapshot in cached_snapshots.values()
                       if isinstance(snapshot, dict) and snapshot.get("photo_id") == photo_id
                       and snapshot.get("image_fingerprint") == raw_fingerprint), None)
        if cached is not None:
            by_photo[photo_id] = cached
        else:
            # Keep unopened photos out of the first-view inference path. They are
            # analyzed when selected, while previously cached snapshots remain available.
            by_photo[photo_id] = {"project_id": project.project_id,
                "board_id": project.board_id or "", "photo_id": photo_id,
                "image_fingerprint": raw_fingerprint, "review_version": "UNPROCESSED",
                "review_scope": None, "registry_components": (), "components": (),
                "component_ids": (), "detection_status": "PENDING_ON_SELECTION",
                "ai_review_status": "PENDING", "version": None,
                "analysis_scope_key": AnalysisScope(project.project_id,
                    project.board_id or "", photo_id).key}
    by_photo = {photo_id: {**snapshot,
                           "side": records[photo_id].side,
                           "capture_type": records[photo_id].capture_type}
                for photo_id, snapshot in by_photo.items()}
    selected_snapshot = by_photo.get(selected_id)
    if selected_snapshot is not None:
        st.session_state["circuitcare_v2_photo_choice"] = selected_id
        if st.session_state.get(ACTIVE_PHOTO_KEY) != selected_id:
            update_selection(st.session_state, project=project, active_photo_id=selected_id)
        st.session_state[MAP_VERSION_KEY] = selected_snapshot.get("version")
        st.session_state["circuitcare_v4_component_inventory_scope"] = selected_snapshot.get("review_scope")
        st.session_state["circuitcare_v4_current_components"] = tuple(selected_snapshot.get("components", ()))
        st.session_state["circuitcare_v6_current_component_scope"] = selected_snapshot.get("analysis_scope_key")
        st.session_state["circuitcare_v4_detection_status"] = selected_snapshot.get("detection_status", "UNKNOWN")
    registry = build_board_component_registry(
        st.session_state, project_id=project.project_id, board_id=project.board_id,
        photo_snapshots=by_photo,
    )
    LOGGER.info("REGISTRY_REBUILT project=%s photo_count=%s component_count=%s version=%s",
                project.project_id, len(by_photo), len(registry.get("components", [])),
                str(registry.get("version", ""))[-16:])
    for photo_id, snapshot in by_photo.items():
        models = tuple(replace(item, board_component_id=registry["source_to_board_id"].get(
            source_key_for_component(photo_id, item.component_id))) for item in snapshot.get("components", ()))
        by_photo[photo_id] = {**snapshot, "components": models,
                              "component_ids": tuple(item.component_id for item in models)}
    registry_scopes = dict(st.session_state.get("circuitcare_v6_photo_component_snapshots", {}))
    for snapshot in by_photo.values():
        registry_scopes[snapshot["analysis_scope_key"]] = snapshot
    st.session_state["circuitcare_v6_photo_component_snapshots"] = registry_scopes
    selected_snapshot = by_photo.get(selected_id)
    if selected_snapshot is not None:
        st.session_state["circuitcare_v4_current_components"] = tuple(selected_snapshot.get("components", ()))
    st.session_state["circuitcare_v7_photo_snapshots_by_id"] = by_photo
    st.session_state["circuitcare_v7_active_board_registry"] = registry
    selected_board_id = st.session_state.get(BOARD_COMPONENT_SELECTION_KEY)
    if not selected_board_id and previous_photo_id and previous_local_id:
        selected_board_id = registry["source_to_board_id"].get(
            source_key_for_component(previous_photo_id, previous_local_id))
    if selected_board_id:
        target_source = next((source for row in registry.get("components", [])
                             if row.get("board_component_id") == selected_board_id
                             for source in row.get("source_components", [])
                             if source.get("photo_id") == selected_id
                             and source.get("review_status") != "EXCLUDED"), None)
        if target_source:
            local_id = target_source.get("component_id")
            if (getattr(selection_before, "active_photo_id", None) != selected_id
                    or getattr(selection_before, "selected_component", None) != local_id):
                update_selection(st.session_state, project=project, active_photo_id=selected_id,
                                 selected_component=local_id)
        else:
            # An excluded, retired, or non-visible registry source must not
            # overwrite a newer local viewer click with a null/stale selection.
            st.session_state.pop(BOARD_COMPONENT_SELECTION_KEY, None)
    return registry, selected_snapshot


def _render_pcb(st: Any, project: ProjectContext, photos: list[dict[str, Any]]) -> None:
    records = {item.image_id: item for item in project.image_records}
    valid = [item for item in photos if item.get("image_id") in records and isinstance(item.get("data"), bytes)]
    if not valid:
        st.info("PCB 사진 없음")
        if st.button("사진 추가", key="cc_v2_preview_add_photo", use_container_width=True):
            st.session_state["circuitcare_v2_edit_project_id"] = project.project_id
            st.session_state["circuitcare_v2_edit_board_id"] = project.board_id
            clear_intake_draft(st.session_state)
            st.session_state[SCREEN_KEY] = "entry"
            st.session_state.pop(SELECTION_KEY, None)
            st.session_state.pop("circuitcare_v2_photo_choice", None)
            st.session_state.pop(VIEW_ZOOM_KEY, None)
            st.rerun()
        return
    ids = [item["image_id"] for item in valid]
    selected_id = st.session_state.get("circuitcare_v2_photo_choice")
    if selected_id not in ids:
        selected_id = st.session_state.get(ACTIVE_PHOTO_KEY)
    if selected_id not in ids:
        selected_id = ids[0]
    st.session_state["circuitcare_v2_photo_choice"] = selected_id
    prepared_snapshot = st.session_state.get("circuitcare_v7_photo_snapshots_by_id", {}).get(selected_id)
    active_side = records[selected_id].side

    # A single board map covers every supported photo on both physical sides.
    ready_inputs = []
    if project.board_id:
        for item in valid:
            adapted = adapt_component_map_input(records[item["image_id"]], board_id=project.board_id, data=item["data"])
            if adapted.request is not None:
                ready_inputs.append(adapted.request)
    photo_fingerprint = photo_input_fingerprint(list(project.image_records))
    cache_key = component_map_cache_key(
        project_id=project.project_id, board_id=project.board_id or "", inputs=ready_inputs,
        photo_manifest=list(project.image_records),
    )
    provider = optional_backend_provider()
    provider_options = getattr(provider, "cache_options", {}) if provider else {}
    if provider_options:
        cache_key = component_map_cache_key(
            project_id=project.project_id, board_id=project.board_id or "", inputs=ready_inputs,
            photo_manifest=list(project.image_records), options=provider_options,
        )
    entry = st.session_state.get(COMPONENT_MAP_STATE_KEY)
    if not component_map_entry_is_current(
        entry, project_id=project.project_id, board_id=project.board_id,
        photo_fingerprint=photo_fingerprint, cache_key=cache_key,
    ):
        clear_component_map_state(st.session_state, clear_viewer_selection=False)
        entry = None
    if provider is not None and ready_inputs:
        if st.button("사진 정합 실행", key="circuitcare_v3_build_map"):
            try:
                raw_map = provider.build_component_map(
                    board_id=project.board_id,
                    photos=[{"data": p.data, "side": p.side, "role": p.role, "name": p.name}
                            for p in ready_inputs],
                    options=provider_options,
                )
                validated_map = validate_component_map(raw_map, board_id=project.board_id)
                photo_id_map = map_backend_photo_ids(
                    project_id=project.project_id, board_id=project.board_id,
                    inputs=ready_inputs, result=validated_map,
                )
                # Publish map and ID mapping as one owner-checked cache entry.
                entry = {
                    "project_id": project.project_id,
                    "board_id": project.board_id,
                    "input_fingerprint": photo_fingerprint,
                    "component_map_cache_key": cache_key,
                    "component_map": validated_map,
                    "photo_id_map": photo_id_map,
                    "failed_artifacts": [],
                }
                st.session_state[COMPONENT_MAP_STATE_KEY] = entry
                st.session_state.pop("circuitcare_v3_processing_error", None)
            except Exception as exc:
                LOGGER.exception("Component Map build failed")
                st.session_state["circuitcare_v3_processing_error"] = str(exc)
                st.warning("정합 불가 · 원본 사진 보기")
        entry = st.session_state.get(COMPONENT_MAP_STATE_KEY)
        if component_map_entry_is_current(
            entry, project_id=project.project_id, board_id=project.board_id,
            photo_fingerprint=photo_fingerprint, cache_key=cache_key,
        ):
            st.caption("정합 결과 후보 · 정확도 미검증 · 사진면별 결과")

    entry = st.session_state.get(COMPONENT_MAP_STATE_KEY)
    if not component_map_entry_is_current(
        entry, project_id=project.project_id, board_id=project.board_id,
        photo_fingerprint=photo_fingerprint, cache_key=cache_key,
    ):
        entry = None
    component_map = entry["component_map"] if entry else None
    st.session_state[MAP_VERSION_KEY] = cache_key if component_map else None
    side_enum = {"front": "TOP", "back": "BOTTOM"}.get(active_side)
    side_result = component_map.get("sides", {}).get(side_enum, {}) if component_map and side_enum else {}
    # Keep the viewer on the selected source photo; no separate view-mode control.
    canonical = None
    st.session_state.pop("circuitcare_v3_view_mode", None)
    mode = "원본 사진"
    if not ids:
        st.info("선택한 면의 원본 사진 없음")
        return
    # Keep photo identity in session state for analysis/navigation, but do not
    # spend vertical space on a one-photo selector and file metadata above the viewer.
    selected = selected_id
    image_data = next(item["data"] for item in valid if item["image_id"] == selected)
    canonical_png_bytes = None
    try:
        if mode == "통합 이미지" and canonical:
            # Decoded above before the widget is built so failures can safely switch its state.
            image = decoded_canonical.image
            source_size = (decoded_canonical.width, decoded_canonical.height)
            display_size = source_size
            active_image_id = decoded_canonical.frame_id
            canonical_png_bytes = base64.b64decode(canonical["image_png_base64"], validate=True)
            st.caption(f"MOSAIC_PIXEL · 기준 사진 {decoded_canonical.reference_backend_photo_id or '미지정'} · 정합 후보 · 미검증")
        else:
            cached_image, source_size, display_size, _cached_png = _cached_photo_pixels(image_data)
            image = cached_image.copy()
            active_image_id = selected
    except (ImageLoadError, ValueError):
        st.warning("이미지 표시 실패 · 원본 사진 보기")
        return
    auto_detection_result = None
    auto_detection_error = None
    detector_image_bytes = None
    if mode == "원본 사진":
        _cached_image, _source_size, _processed_size, detector_image_bytes = _cached_photo_pixels(image_data)
        review_scope = component_review_scope(detector_image_bytes, project.project_id, selected)
        detection_result_key, _records_key, detection_error_key = review_keys(review_scope)
        auto_detection_result = st.session_state.get(detection_result_key)
        auto_detection_error = st.session_state.get(detection_error_key)
        st.session_state["circuitcare_v4_auto_detection_result"] = auto_detection_result
        st.session_state["circuitcare_v4_auto_detection_error"] = auto_detection_error
        _result_key, records_key, _error_key = review_keys(review_scope)
        review_rows = st.session_state.get(records_key, [])
        active_inventory_version = component_inventory_version(
            project_id=project.project_id, board_id=project.board_id, photo_id=selected,
            image_sha256=hashlib.sha256(image_data).hexdigest(), components=[
                {"component_id": row.get("component_id"), "component_type": row.get("소자 종류", "unknown"),
                 "bbox_xyxy": [row.get("x1", 0), row.get("y1", 0), row.get("x2", 0), row.get("y2", 0)]}
                for row in review_rows if isinstance(row, dict)
            ],
        )
        st.session_state[MAP_VERSION_KEY] = active_inventory_version
        st.session_state["circuitcare_v4_component_inventory_scope"] = review_scope
    side_result = side_result if isinstance(side_result, dict) else {}
    detection_status = str(side_result.get("detection_status", "UNKNOWN"))
    if auto_detection_error:
        detection_status = "BLOCKED_MODEL_UNAVAILABLE"
    elif auto_detection_result is not None:
        detection_status = "INFERENCE_COMPLETE"
    overlay_result = adapt_component_map_components(
        component_map, project_id=project.project_id, board_id=project.board_id or "", side=side_enum or "",
    ) if component_map and side_enum else None
    coordinate_frame_valid = True
    if mode == "통합 이미지" and canonical:
        component_models = overlay_result.components if overlay_result and overlay_result.status == "READY" else ()
        frame_id = decoded_canonical.frame_id
        viewer_size = source_size
        if (overlay_result and overlay_result.status not in {"READY", "NO_COMPONENTS"}) or (
            overlay_result and (overlay_result.frame_id != frame_id or overlay_result.image_size != viewer_size
                                or overlay_result.coordinate_frame != "MOSAIC_PIXEL")
        ):
            coordinate_frame_valid = False
            component_models = ()
    else:
        component_models = ()
        frame_id = selected
        viewer_size = display_size
        if side_enum and entry:
            photo_mapping = next((row for row in entry.get("photo_id_map", [])
                                  if row.project_id == project.project_id and row.board_id == project.board_id
                                  and row.frontend_photo_id == selected and row.mapping_status == "MAPPED"), None)
            if photo_mapping and photo_mapping.backend_photo_id:
                source_models = adapt_original_photo_components(
                    component_map, project_id=project.project_id, board_id=project.board_id or "",
                    side=side_enum, frontend_photo_id=selected,
                    backend_photo_id=photo_mapping.backend_photo_id, image_size=source_size,
                )
                component_models = scale_overlay_to_display(
                    source_models, source_size=source_size, display_size=display_size,
                )
    if overlay_result and overlay_result.status == "INVALID_DATA":
        st.warning("일부 부품 좌표를 검증할 수 없어 표시하지 않았습니다.")
    review_rows: list[dict[str, Any]] = []
    auto_models: tuple[ComponentOverlayViewModel, ...] = ()
    records_key: str | None = None
    if mode == "원본 사진" and detector_image_bytes is not None:
        review_hash = hashlib.sha256(detector_image_bytes).hexdigest()[:16]
        review_scope = component_review_scope(
            detector_image_bytes, project.project_id, selected,
        )
        result_key, records_key, _error_key = review_keys(review_scope)
        st.session_state["cc_active_component_review_scope"] = review_scope
        st.session_state["cc_active_component_review_image"] = detector_image_bytes
        review_rows = st.session_state.get(records_key, [])
        photo_snapshot = prepared_snapshot
        auto_models = tuple(photo_snapshot.get("review_components", ()))
        st.session_state["circuitcare_v4_selected_marking_context"] = None
    else:
        review_scope = None
        active_inventory_version = cache_key if component_map else None
        st.session_state[MAP_VERSION_KEY] = active_inventory_version
        st.session_state["circuitcare_v4_component_inventory_scope"] = active_inventory_version
        st.session_state.pop("cc_active_component_review_scope", None)
        st.session_state.pop("cc_active_component_review_image", None)
        st.session_state["circuitcare_v4_selected_marking_context"] = None
    component_edits = st.session_state.get(COMPONENT_EDIT_KEY, {})
    if component_edits:
        updated_models = []
        for model in component_models:
            edit = component_edits.get(model.component_id, {})
            if edit:
                model = replace(
                    model,
                    ocr_marking=edit.get("ocr_marking", model.ocr_marking) or None,
                    possible_part_number=(
                        edit.get("ocr_marking", model.possible_part_number)
                        if model.component_type.casefold() in {"ic", "integrated_circuit"}
                        else model.possible_part_number
                    ),
                    component_info=edit.get("component_info", model.component_info) or None,
                )
            updated_models.append(model)
        component_models = tuple(updated_models)
    target_id = str(selected or st.session_state.get(ACTIVE_PHOTO_KEY) or "target:default")
    snapshots = st.session_state.get("circuitcare_v6_photo_component_snapshots", {})
    photo_scope = AnalysisScope(project.project_id, project.board_id or "", target_id)
    scoped_snapshot = snapshots.get(photo_scope.key) if isinstance(snapshots, dict) else None
    board_registry = st.session_state.get("circuitcare_v7_active_board_registry")
    board_mode = (isinstance(board_registry, dict)
                  and board_registry.get("project_id") == project.project_id
                  and board_registry.get("board_id") == (project.board_id or ""))
    analysis_scope = (AnalysisScope(project.project_id, project.board_id or "", "board:registry:v1")
                      if board_mode else photo_scope)
    analysis_components = (diagnosis_components(board_registry) if board_mode else
                           tuple(scoped_snapshot.get("components", ())) if isinstance(scoped_snapshot, dict)
                           else tuple(component_models))
    review_scope = (scoped_snapshot.get("review_scope") if isinstance(scoped_snapshot, dict) else review_scope)
    inventory_owner = f"{project.project_id}|{project.board_id or ''}|{review_scope or ''}"
    inventory_rows = st.session_state.get(COMPONENT_INVENTORY_KEY, {})
    analysis_inventory = ({"__board_registry__": {"registry_version": board_registry.get("version")}}
                          if board_mode else
                          inventory_rows.get(inventory_owner, {}) if isinstance(inventory_rows, dict) else {})
    aliases = measurement_component_aliases(board_registry) if board_mode else None
    analysis_snapshot = current_analysis_snapshot(
        st, project, components=analysis_components, component_inventory=analysis_inventory,
        analysis_target_id=analysis_scope.target_id,
        component_id_aliases=aliases,
        board_registry=board_registry if board_mode else None,
        auto_qwen=True,
    )
    analysis_state = analysis_snapshot["record"]
    # Legacy photo analyses and Component Map verdicts do not paint the board.
    # Only the current Analysis State Manager scope may paint live statuses.
    # the live PCB; retained verdicts stay in fault_history.
    component_models = tuple(replace(
        model, fault_candidate_status=None, diagnosis_display_status="DIAGNOSIS_NOT_RUN",
        fault_hypothesis=None, fault_role_status=None, measurement_status=None,
        solder_status=None, fault_score=None, fault_evidence=(), recommended_tests=(),
        functional_block_roles=(),
    ) for model in component_models)
    candidate_state_by_scope = st.session_state.get("circuitcare_v6_diagnosis_candidates_by_scope", {})
    scoped_overlay = (candidate_state_by_scope.get(analysis_scope.key)
                     if isinstance(candidate_state_by_scope, dict) else {})
    current_result = analysis_snapshot.get("result")
    selected_candidate_strategy = st.session_state.get(f"cc_candidate_strategy_{project.project_id}", "baseline")
    overlay_candidates = candidate_rows_for_strategy(current_result, selected_candidate_strategy)         if analysis_state.get("status") == AnalysisStatus.CURRENT.value and isinstance(current_result, dict) else []
    candidate_state: dict[str, dict[str, Any]] = {}
    for candidate in overlay_candidates:
        component_id = str(candidate.get("board_component_id") or candidate.get("component_id") or "")
        if not component_id:
            continue
        existing = candidate_state.get(component_id)
        if existing is None:
            candidate_state[component_id] = dict(candidate)
            candidate_state[component_id]["_fault_hypotheses"] = [str(candidate.get("fault_hypothesis") or candidate.get("fault_id") or "")]
            continue
        fault_id = str(candidate.get("fault_hypothesis") or candidate.get("fault_id") or "")
        if fault_id and fault_id not in existing["_fault_hypotheses"]:
            existing["_fault_hypotheses"].append(fault_id)
        current_status = str(existing.get("visual_status") or existing.get("status") or "SUSPECTED")
        other_status = str(candidate.get("visual_status") or candidate.get("status") or "SUSPECTED")
        if current_status != other_status and {current_status, other_status} & {"FAULT_SUPPORTED", "FAULT_REFUTED", "CONFLICTED"}:
            existing["visual_status"] = "CONFLICTED"
        evidence = list(existing.get("evidence", [])) + list(candidate.get("evidence", []))
        existing["evidence"] = list(dict.fromkeys(str(item) for item in evidence))
        existing["recommended_tests"] = list(dict.fromkeys([
            *existing.get("recommended_tests", []), *candidate.get("recommended_tests", [])
        ]))
    workflow = get_workflow_state(st.session_state, project)
    workflow_stage = workflow.get("stage", WORKFLOW_REVIEW)
    if workflow_stage == WORKFLOW_REVIEW:
        candidate_state.clear()
    elif workflow_stage == WORKFLOW_QUESTIONS:
        for suspect_index, suspect in enumerate(board_suspects(st.session_state, project), start=1):
            board_id = str(suspect.get("board_component_id") or "")
            if board_id and suspect.get("evidence_status") == "UNVERIFIED":
                candidate_state[board_id] = {
                    "visual_status": "SUSPECTED", "candidate_rank": suspect_index,
                    "fault_hypothesis": suspect.get("suspected_fault_mode"),
                    "evidence": ["사용자 의심 소자 · 미검증"], "recommended_tests": [],
                }
    if workflow_stage in {WORKFLOW_INSPECTION, WORKFLOW_REASSESSMENT}:
        for rank, target in enumerate(workflow.get("inspection_targets", []), start=1):
            board_id = str(target.get("board_component_id") or "")
            if not board_id:
                continue
            details = target.get("candidate", {}) if isinstance(target.get("candidate"), dict) else {}
            candidate_state.setdefault(board_id, {
                **details, "visual_status": "SUSPECTED", "candidate_rank": rank,
                "evidence": [target.get("reason", "우선 확인 대상")], "recommended_tests": [],
            })
    for candidate in candidate_state.values():
        hypotheses = [item for item in candidate.pop("_fault_hypotheses", []) if item]
        candidate["fault_hypothesis"] = " · ".join(hypotheses) if hypotheses else candidate.get("fault_hypothesis")
    verified_history = (
        [row for row in scoped_overlay.get("verified_component_history", []) if isinstance(row, dict)]
        if isinstance(scoped_overlay, dict)
        and scoped_overlay.get("project_id") == project.project_id
        and scoped_overlay.get("board_id") == project.board_id else []
    )
    history_by_id: dict[str, list[dict[str, Any]]] = {}
    for history_row in verified_history:
        history_by_id.setdefault(str(history_row.get("component_id") or ""), []).append(history_row)
    if isinstance(candidate_state, dict) and candidate_state:
        status_map = {
            "SUSPECTED": ("SUSPECTED_FAULT", "의심 후보 · 미확정"),
            "FAULT_SUPPORTED": ("FAULT_SUPPORTED", "검증 검사 결과 · 해당 고장 가설 지지"),
            "FAULT_REFUTED": ("FAULT_REFUTED", "검증 검사 결과 · 해당 고장 가설 반박"),
            "INCONCLUSIVE": ("INCONCLUSIVE", "측정 판정 불가"),
            "CONFLICTED": ("CONFLICTED", "측정 결과 충돌 · 판단 보류"),
        }
        component_models = tuple(
            replace(model,
                    fault_candidate_status=status_map.get(candidate_state[str(
                        model.board_component_id if board_mode else model.component_id)].get("visual_status"),
                                                          ("INCONCLUSIVE", "판정 불가"))[0],
                    diagnosis_display_status=status_map.get(candidate_state[str(
                        model.board_component_id if board_mode else model.component_id)].get("visual_status"),
                                                            ("INCONCLUSIVE", "판정 불가"))[1],
                    fault_hypothesis=candidate_state[str(model.board_component_id if board_mode else model.component_id)].get("fault_hypothesis"),
                    fault_role_status=candidate_state[str(model.board_component_id if board_mode else model.component_id)].get("role_status"),
                    measurement_status=candidate_state[str(model.board_component_id if board_mode else model.component_id)].get("measurement_status"),
                    solder_status=candidate_state[str(model.board_component_id if board_mode else model.component_id)].get("solder_status"),
                    fault_score=candidate_state[str(model.board_component_id if board_mode else model.component_id)].get("score"),
                    fault_evidence=tuple(candidate_state[str(model.board_component_id if board_mode else model.component_id)].get("evidence", [])),
                    recommended_tests=tuple(candidate_state[str(model.board_component_id if board_mode else model.component_id)].get("recommended_tests", [])),
                    fault_history=tuple(history_by_id.get(str(model.board_component_id if board_mode else model.component_id), [])))
            if str(model.board_component_id if board_mode else model.component_id) in candidate_state else model
            for model in component_models
        )
    if history_by_id:
        component_models = _attach_verified_history(component_models, history_by_id)
    block_roles_by_component: dict[str, set[str]] = {}
    active_groups = (current_result.get("functional_block_groups", [])
                     if analysis_state.get("status") == AnalysisStatus.CURRENT.value
                     and isinstance(current_result, dict) else [])
    for block in active_groups:
        if not isinstance(block, dict):
            continue
        role = str(block.get("role") or "")
        for component_id in block.get("board_component_ids", block.get("component_ids", [])):
            if component_id:
                block_roles_by_component.setdefault(str(component_id), set()).add(role)
    if block_roles_by_component:
        component_models = tuple(replace(
            model,
            functional_block_roles=tuple(sorted(block_roles_by_component.get(
                str(model.board_component_id or model.component_id), set()))),
        ) for model in component_models)
    if review_scope and records_key:
        previous_rows = [dict(row) for row in review_rows]
        component_models, review_rows = _bind_review_models_to_map(
            component_models, auto_models, review_rows)
        if review_rows != previous_rows:
            st.session_state[records_key] = review_rows
        selection = st.session_state.get(SELECTION_KEY)
        selected_id = getattr(selection, "selected_component", None)
        replacement_id = next((str(row["component_id"]) for row in review_rows
                               if selected_id in row.get("associated_component_ids", [])), None)
        if replacement_id:
            update_selection(st.session_state, project=project, selected_component=replacement_id)
    if board_mode and candidate_state:
        status_map = {
            "SUSPECTED": ("SUSPECTED_FAULT", "의심 후보 · 미확정"),
            "FAULT_SUPPORTED": ("FAULT_SUPPORTED", "검증 검사 결과 · 해당 고장 가설 지지"),
            "FAULT_REFUTED": ("FAULT_REFUTED", "검증 검사 결과 · 해당 고장 가설 반박"),
            "INCONCLUSIVE": ("INCONCLUSIVE", "측정 판정 불가"),
            "CONFLICTED": ("CONFLICTED", "측정 결과 충돌 · 판단 보류"),
        }
        source_to_board = board_registry.get("source_to_board_id", {})
        projected = []
        overlay_match_count = 0
        for model in component_models:
            mapped_board_id = source_to_board.get(
                source_key_for_component(str(selected), model.component_id),
            )
            if mapped_board_id and model.board_component_id != mapped_board_id:
                model = replace(model, board_component_id=str(mapped_board_id))
            board_component_id = str(model.board_component_id or "")
            candidate = candidate_state.get(board_component_id)
            if candidate is None:
                projected.append(model)
                continue
            overlay_match_count += 1
            status, label = status_map.get(candidate.get("visual_status"), ("INCONCLUSIVE", "판정 불가"))
            projected.append(replace(
                model, fault_candidate_status=status, diagnosis_display_status=label,
                fault_hypothesis=candidate.get("fault_hypothesis"),
                fault_role_status=candidate.get("role_status"),
                measurement_status=candidate.get("measurement_status"),
                solder_status=candidate.get("solder_status"), fault_score=candidate.get("score"),
                fault_evidence=tuple(candidate.get("evidence", [])),
                recommended_tests=tuple(candidate.get("recommended_tests", [])),
                fault_history=tuple(history_by_id.get(str(model.board_component_id), [])),
            ))
        component_models = tuple(projected)
        LOGGER.info(
            "PCB_DIAG_OVERLAY project=%s photo_id=%s stage=%s candidate_count=%s "
            "viewer_component_count=%s matched_count=%s statuses=%s",
            project.project_id, selected, workflow_stage, len(candidate_state),
            len(component_models), overlay_match_count,
            sorted({str(row.get("visual_status") or "UNKNOWN") for row in candidate_state.values()}),
        )
    component_models = tuple(component_models)
    target_scope = AnalysisScope(project.project_id, project.board_id or "", str(selected))
    snapshot_map = dict(st.session_state.get("circuitcare_v6_photo_component_snapshots", {}))
    photo_snapshot = snapshot_map.get(target_scope.key)
    if isinstance(photo_snapshot, dict):
        photo_snapshot = {**photo_snapshot, "components": component_models,
                          "component_ids": tuple(item.component_id for item in component_models)}
        snapshot_map[target_scope.key] = photo_snapshot
        st.session_state["circuitcare_v6_photo_component_snapshots"] = snapshot_map
    st.session_state["circuitcare_v4_current_components"] = component_models
    st.session_state["circuitcare_v6_current_component_scope"] = target_scope.key
    st.session_state["circuitcare_v4_detection_status"] = detection_status
    current_selection = st.session_state.get(SELECTION_KEY)
    selected_component_id = getattr(current_selection, "selected_component", None)
    _record_selection_trace(st.session_state,
                            f"unified_workspace · 현재 선택 {selected_component_id or '없음'} · 모델 {len(component_models)}개")
    if selected_component_id and selected_component_id not in {item.component_id for item in component_models}:
        LOGGER.warning("pcb_selection stage=unified_invalidate reason=component_missing selected=%s current_ids=%s",
                       selected_component_id, [item.component_id for item in component_models])
        _record_selection_trace(st.session_state,
                               f"unified_workspace · 현재 박스 목록에 ID 없음 · {selected_component_id}")
        update_selection(st.session_state, project=project, selected_component=None)
        selected_component_id = None
    transform = requested_view_transform(
        source_size=source_size,
        display_size=viewer_size,
        active_photo_id=active_image_id,
    )
    transform.update({
        "project_id": project.project_id,
        "board_id": project.board_id,
        "active_board_side": side_enum or active_side,
        "active_view_mode": mode,
        "source_frame_id": active_image_id,
        "canonical_frame_id": canonical.get("image_id") if canonical else None,
        "source_dimensions": list(source_size),
        "canonical_dimensions": canonical.get("image_size_px") if canonical else None,
        "registration_status": "CANDIDATE_GEOMETRY_ONLY" if mode == "통합 이미지" else "SOURCE_PHOTO",
        "coordinate_status": "UNVERIFIED_RESPONSIVE_CSS",
    })

    def add_manual_box(bbox: list[int]) -> None:
        if mode != "원본 사진" or not detector_image_bytes or not review_scope:
            return
        _result_key, records_key, _error_key = review_keys(review_scope)
        rows = list(st.session_state.get(records_key, []))
        manual_id = f"manual-{review_hash}-{hashlib.sha256(repr(tuple(bbox)).encode()).hexdigest()[:12]}"
        rows.append({"component_id": manual_id, "소자 종류": "unknown", "상태": "검수 필요", "신뢰도": 0.0,
                     "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3], "source": "manual"})
        st.session_state[records_key] = rows
        from circuitcare.vision.component_recognition_panel import invalidate_component_review_state
        invalidate_component_review_state(st.session_state, scope=review_scope,
                                         project_id=project.project_id, board_id=project.board_id)
        update_selection(st.session_state, project=project,
                         selected_component=manual_id)

    interactive, _viewer_key, viewer_coordinate_status = render_component_viewer(
        st, image=image, source_image_bytes=image_data, canonical_png_bytes=canonical_png_bytes,
        image_size_px=viewer_size, components=tuple(component_models), project=project,
        board_side=side_enum or active_side, frame_id=frame_id,
        selected_component_id=selected_component_id,
        photo_id=str(selected),
        analysis_loading=st.session_state.get(f"cc_ai_review_job_status_{review_scope}") in {"PENDING", "RUNNING"},
        # Keep the selection owner key identical to _render_pcb's invalidation
        # key even when this photo has no registered component map yet.
        map_cache_key=cache_key, image_mode=mode,
        on_box_draw=add_manual_box if mode == "원본 사진" and detector_image_bytes is not None else None,
    )
    st.session_state["cc_active_component_overlap_key"] = f"{OVERLAPPING_SELECTION_KEY}_{_viewer_key}"
    current_selection = st.session_state.get(SELECTION_KEY)
    current_local_id = getattr(current_selection, "selected_component", None)
    if board_mode and current_local_id:
        board_component_id = board_registry.get("source_to_board_id", {}).get(
            source_key_for_component(str(selected), current_local_id))
        if board_component_id:
            st.session_state[BOARD_COMPONENT_SELECTION_KEY] = board_component_id
        else:
            st.session_state.pop(BOARD_COMPONENT_SELECTION_KEY, None)
    if candidate_state:
        st.caption("진단 표시 · 노랑 SUSPECTED: 미확정 후보 · 빨강 FAULT_SUPPORTED: 측정이 특정 가설을 지지 · "
                   "초록 FAULT_REFUTED: 특정 가설 반박 · 회색 INCONCLUSIVE: 판정 불가 · "
                   "충돌 CONFLICTED: 증거 상충. 소자 전체의 고장·정상을 확정하지 않습니다.")
    if not interactive:
        with st.container(key="cc_viewer_canvas"):
            st.image(image, width=720)
        if component_models:
            st.warning("인터랙티브 Viewer를 사용할 수 없습니다 · 부품 Overlay를 숨겼습니다.")
        else:
            st.caption("Zoom / Pan Viewer를 사용할 수 없어 정적 이미지로 표시합니다.")
    if not coordinate_frame_valid:
        viewer_coordinate_status = "INVALID_COORDINATE_FRAME"
    transform["coordinate_status"] = viewer_coordinate_status
    st.session_state[VIEW_TRANSFORM_KEY] = transform
    if mode == "원본 사진":
        mapped = entry.get("photo_id_map", []) if entry else []
        photo_mapping = next((row for row in mapped if row.frontend_photo_id == selected and row.project_id == project.project_id), None)
        st.session_state["circuitcare_v4_detection_summary"] = {
            "count": len(auto_detection_result.get("detections", [])) if auto_detection_result else None,
            "error": auto_detection_error,
            "registration": REGISTRATION_LABELS.get(photo_mapping.registration_status, "상태 미확인") if photo_mapping else "정합 결과 없음",
        }


def _workflow_stage_title(stage: str) -> str:
    return {
        WORKFLOW_REVIEW: "1 · 소자 검수",
        WORKFLOW_QUESTIONS: "2 · 고장 증상 질문",
        WORKFLOW_INSPECTION: "3 · 검사 후보 및 측정",
        WORKFLOW_REASSESSMENT: "4 · 측정 결과 재판단",
        WORKFLOW_COMPLETE: "결과 보고",
    }.get(stage, "고장 진단")


def _inspection_targets(
    registry: dict[str, Any] | None, candidates: list[dict[str, Any]],
    suspected_components: list[dict[str, Any]] | None, *, knowledge: dict[str, Any] | None = None,
    inspection_history: list[dict[str, Any]] | None = None,
    measurements: list[Any] | None = None, allow_retest: bool = False,
    symptoms: list[str] | None = None, limit: int | None = 3,
) -> list[dict[str, Any]]:
    """Choose real targets, preferring compatible and not-yet-completed test profiles."""
    rows = registry.get("components", []) if isinstance(registry, dict) else []
    active_rows = [row for row in rows if isinstance(row, dict)
                   and row.get("presence_status") != "EXCLUDED"
                   and row.get("component_type") != "board_feature"
                   and row.get("board_component_id")]
    by_id = {str(row["board_component_id"]): row for row in active_rows}
    ordered: list[tuple[str, dict[str, Any], str, int]] = []
    used: set[str] = set()
    knowledge = knowledge or load_knowledge()
    profiles_by_type = knowledge.get("component_test_profiles", {}).get("profiles", {})
    catalog = knowledge.get("test_catalog", {}).get("tests", {})
    current_symptoms = set(str(value) for value in symptoms or [])
    completed: set[tuple[str, str, str]] = set()
    if not allow_retest:
        for item in inspection_history or []:
            if (not isinstance(item, dict) or item.get("verification_status") not in {
                    "FAULT_SUPPORTED", "FAULT_REFUTED", "CONFLICTED", "VERIFIED", "COMPLETE"}):
                continue
            completed.add((str(item.get("board_component_id") or item.get("target_id") or ""),
                           str(item.get("fault_id") or ""), str(item.get("test_id") or "")))
        for measurement in measurements or []:
            status = str(getattr(measurement, "verification_status", "")).upper()
            test_id = str(getattr(measurement, "test_id", "") or "")
            board_id = str(getattr(measurement, "board_component_id", "") or "")
            if test_id and board_id and status in {"VERIFIED", "VALID", "ACCEPTED", "DIAGNOSTIC"}:
                completed.add((board_id, str(getattr(measurement, "fault_hypothesis", "") or ""), test_id))

    def add(board_id: str, details: dict[str, Any], reason: str) -> None:
        row = by_id.get(str(board_id))
        if (not row or str(board_id) in used
                or not isinstance(row.get("source_components"), list)
                or not row.get("source_components")):
            return
        measurement_component = measurement_view_model_for_board_component(
            registry, board_component_id=str(board_id), preferred_photo_id=None,
        )
        # A row without a validated photo-local box cannot be presented as a
        # physical inspection target. Keep it in the Registry, but do not
        # invent a clickable location or a measurement subject for it.
        if measurement_component is None:
            return
        type_confirmed = row.get("component_type_status") == "USER_CONFIRMED"
        source_rows = row.get("source_components", [])
        source = next((item for item in source_rows if isinstance(item, dict)
                       and item.get("review_status") != "EXCLUDED"), {})
        source_type = str(source.get("confirmed_base_class") or source.get("detector_class")
                          or source.get("component_type") or "unknown")
        component_type = str(row.get("component_type") if type_confirmed
                             else (source_type if source_type != "unknown"
                                   else details.get("component_type") or "unknown"))
        # A detector class is useful for identifying the displayed box, but it
        # is not enough to select an electrical test profile. Only a confirmed
        # Registry type or an already validated diagnosis candidate may do so.
        test_component_type = str(row.get("component_type") if type_confirmed
                                  else details.get("component_type") or "unknown")
        component_profile = profiles_by_type.get(test_component_type.casefold(), {})
        tests = component_profile.get("tests", {}) if isinstance(component_profile, dict) else {}
        hypotheses = details.get("fault_hypotheses", []) if isinstance(details.get("fault_hypotheses"), list) else []
        fault_ids = [str(item.get("fault_id") or item.get("fault_hypothesis") or "")
                     for item in hypotheses if isinstance(item, dict)]
        if not fault_ids:
            fault_ids = [str(details.get("fault_id") or details.get("fault_hypothesis") or "")]
        component_subtype = str(row.get("subtype") or row.get("component_subtype") or "").casefold()
        component_visual_checks = _compatible_visual_checks(
            catalog, target_scope="component", component_type=test_component_type,
            subtype=component_subtype, fault_ids=fault_ids, symptoms=current_symptoms,
        )
        allowed_tests = []
        completed_test_count = 0
        for test_id, profile in tests.items() if isinstance(tests, dict) else []:
            relations = profile.get("fault_relations", {}) if isinstance(profile, dict) else {}
            related_faults = [fault for fault in fault_ids if fault and fault in relations]
            compatible = bool(any(fault_ids) and related_faults)
            unfinished_faults = [fault for fault in related_faults
                                 if (str(board_id), fault, str(test_id)) not in completed]
            was_completed = bool(related_faults) and not unfinished_faults
            if was_completed:
                completed_test_count += 1
            if compatible and (allow_retest or unfinished_faults):
                allowed_tests.append({"test_id": str(test_id), "profile": profile,
                                      "fault_ids": (related_faults if allow_retest else unfinished_faults)})
        ordinal = len(ordered)
        annotated = {**details, "component_type": component_type,
                     "component_type_confirmed": type_confirmed,
                     "available_tests": allowed_tests,
                     "visual_checks": component_visual_checks,
                     "measurement_available": measurement_component is not None,
                     "source_component_id": (measurement_component.component_id if measurement_component
                                             else str(source.get("component_id") or "")),
                     "source_photo_id": (measurement_component.canonical_frame_id if measurement_component
                                         else str(source.get("photo_id") or "")),
                     "source_bbox_xyxy": (list(measurement_component.bbox_xyxy) if measurement_component
                                          else list(source.get("bbox_xyxy") or [])),
                     "inspection_kind": "MEASUREMENT" if allowed_tests else "VISUAL_ONLY",
                     "component_subtype": component_subtype,
                     "completed_test_count": completed_test_count}
        ordered.append((str(board_id), annotated, reason, ordinal))
        used.add(str(board_id))

    for candidate in candidates:
        board_id = str(candidate.get("board_component_id") or "")
        if board_id:
            add(board_id, candidate, "증상·측정·규칙 분석 후보")
    for suspect in suspected_components or []:
        board_id = str(suspect.get("board_component_id") or "")
        if board_id:
            suspected_fault = str(suspect.get("suspected_fault_mode") or "")
            add(board_id, {"component_type": suspect.get("component_type", "unknown"),
                           "fault_hypotheses": ([{"fault_id": suspected_fault}]
                                                if suspected_fault else [])},
                "사용자가 의심 소자로 지정")
    for row in active_rows:
        board_id = str(row["board_component_id"])
        add(board_id, {"component_type": row.get("component_type", "unknown")},
            "추가 확인 대상 · 원인 미확정")
    ordered.sort(key=lambda item: (item[1].get("inspection_kind") != "MEASUREMENT", item[3]))
    return [{"board_component_id": board_id,
             "component_type": str(details.get("component_type") or
                                    by_id[board_id].get("component_type") or "unknown"),
             "component_type_confirmed": bool(details.get("component_type_confirmed")),
             "reference_designators": list(by_id[board_id].get("reference_designators", [])),
             "display_label": next(iter(by_id[board_id].get("reference_designators", [])), None),
             "reason": reason, "candidate": details,
             "inspection_kind": details.get("inspection_kind"),
             "available_tests": details.get("available_tests", []),
             "visual_checks": details.get("visual_checks", []),
             "measurement_available": bool(details.get("measurement_available")),
             "completed_test_count": details.get("completed_test_count", 0),
             "presence_status": by_id[board_id].get("presence_status"),
             "source_components": list(by_id[board_id].get("source_components", []))}
            for board_id, details, reason, _ordinal in (ordered if limit is None else ordered[:max(0, limit)])]


def _measurement_record_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "to_dict"):
        value = row.to_dict()
        if isinstance(value, dict):
            return value
    if hasattr(row, "__dict__"):
        return dict(vars(row))
    return {}


def _measurement_record_id(row: Any) -> str:
    values = _measurement_record_dict(row)
    identifier = values.get("measurement_id") or values.get("id") or getattr(row, "measurement_id", None)
    if identifier is not None:
        return str(identifier)
    return "record:" + hashlib.sha256(json.dumps(
        values, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":")
    ).encode("utf-8")).hexdigest()[:20]


def _measurement_row_fingerprint(row: Any) -> str:
    try:
        return build_measurement_fingerprint([row])
    except (TypeError, ValueError):
        return build_measurement_fingerprint([_measurement_record_dict(row)])


def _compatible_visual_checks(
    catalog: dict[str, Any], *, target_scope: str, component_type: str = "",
    subtype: str = "", fault_ids: list[str] | set[str] | tuple[str, ...] = (),
    symptoms: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return only catalog visual checks whose target and evidence scope match."""
    matched: list[dict[str, Any]] = []
    fault_set = {str(value) for value in fault_ids if value}
    symptom_set = symptoms or set()
    for test_id, raw in catalog.items() if isinstance(catalog, dict) else []:
        if not isinstance(raw, dict) or raw.get("type") != "visual_inspection":
            continue
        configured_scope = str(raw.get("target_scope") or "component")
        scope_matches = configured_scope == target_scope or (
            target_scope == "component" and configured_scope == "connector"
            and component_type.casefold() == "connector"
        )
        if not scope_matches:
            continue
        applicable_types = {str(value).casefold() for value in raw.get("component_types", [])}
        if target_scope != "board" and (not applicable_types or component_type.casefold() not in applicable_types):
            continue
        applicable_subtypes = {str(value).casefold() for value in raw.get("subtypes", [])}
        if applicable_subtypes and subtype not in applicable_subtypes:
            continue
        relevant_faults = {str(value) for value in raw.get("fault_ids", [])}
        relevant_symptoms = {str(value) for value in raw.get("symptom_codes", [])}
        if relevant_faults or relevant_symptoms:
            if not (fault_set & relevant_faults or symptom_set & relevant_symptoms):
                continue
        matched.append({"test_id": str(test_id), **raw})
    return matched


def _visual_observation_fingerprint(
    *, board_id: str, test_id: str, target_scope: str, target_id: str | None,
    result: str, observations: dict[str, Any], safety_checks: dict[str, bool],
) -> str:
    payload = {"board_id": board_id, "test_id": test_id, "target_scope": target_scope,
               "target_id": target_id, "result": result, "observations": observations,
               "safety_checks": safety_checks}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _render_visual_inspection(
    st: Any, project: ProjectContext, workflow: dict[str, Any], *, test: dict[str, Any],
    target_id: str | None, analysis_fingerprint: str | None, preview: bool,
    widget_suffix: str, fault_id: str | None = None,
    related_fault_ids: list[str] | None = None,
) -> None:
    test_id = str(test.get("test_id") or "")
    scope = str(test.get("target_scope") or ("component" if target_id else "board"))
    cycle_id = int((workflow.get("inspection_cycle") or {}).get("cycle_id", 0))
    widget_instance = f"{widget_suffix}_cycle{cycle_id}"
    st.caption(str(test.get("question") or test.get("name") or test_id))
    safety_checks: dict[str, bool] = {}
    if test.get("requires_power_off"):
        safety_checks["power_disconnected"] = bool(st.checkbox(
            "전원 입력을 분리했고 잔류 전압 위험을 확인했습니다",
            key=f"cc_wf_visual_power_off_{widget_instance}_{test_id}",
            help="전원 분리만으로 대용량 커패시터의 잔류 에너지가 안전하다고 판단할 수 없습니다.",
        ))
    if test.get("requires_stored_energy_safe"):
        safety_checks["stored_energy_safe"] = bool(st.checkbox(
            "저장 에너지가 안전하게 제거된 것을 확인했습니다",
            key=f"cc_wf_visual_energy_safe_{widget_instance}_{test_id}",
            help="확인할 수 없으면 기판이나 부품에 접촉하지 말고 검사를 중단하거나 전문가에게 맡기세요.",
        ))
    for condition in test.get("safety_conditions", []):
        st.warning(str(condition))
    result_key = f"cc_wf_visual_result_{widget_instance}_{test_id}"
    visual_result = st.radio(
        "외관 확인 결과", ("이상 발견", "눈에 띄는 이상 없음", "확인 불가"),
        index=None, horizontal=True, key=result_key,
        help="결과는 기본 선택되지 않습니다. 관찰한 결과를 직접 선택하세요.",
    )
    outcome_key = {"이상 발견": "damage_found", "눈에 띄는 이상 없음": "no_visible_damage"}.get(visual_result)
    outcome = test.get("results", {}).get(outcome_key, {}) if outcome_key else {}
    observations = (outcome.get("observations", {}) if isinstance(outcome, dict) else {})
    if visual_result == "확인 불가":
        observations = {}
    safety_ok = (not test.get("requires_power_off") or safety_checks.get("power_disconnected", False)) \
        and (not test.get("requires_stored_energy_safe") or safety_checks.get("stored_energy_safe", False))
    observation_fingerprint = _visual_observation_fingerprint(
        board_id=project.board_id or "", test_id=test_id, target_scope=scope,
        target_id=target_id if scope != "board" else None,
        result=str(visual_result or "UNSELECTED"), observations=observations,
        safety_checks=safety_checks,
    )
    previous = [row for row in workflow.get("inspection_history", [])
                if isinstance(row, dict) and row.get("observation_fingerprint") == observation_fingerprint
                and row.get("verification_status") != "STALE"]
    retest = st.checkbox("같은 외관 검사를 명시적으로 다시 기록", key=f"cc_wf_visual_retest_{widget_instance}_{test_id}")
    if previous and not retest:
        st.info("동일 대상·검사·결과의 기록이 이미 있습니다. 새로운 관측이 있거나 재검사를 선택한 경우에만 다시 기록할 수 있습니다.")
    can_save = bool(visual_result) and safety_ok and not preview and (not previous or retest)
    if st.button("외관 검사 기록", type="primary", disabled=not can_save,
                 key=f"cc_wf_visual_save_{widget_instance}_{test_id}"):
        cycle = workflow.get("inspection_cycle") or {}
        cycle_id = int(cycle.get("cycle_id", 0))
        history_id = hashlib.sha256(
            f"visual|{cycle_id}|{observation_fingerprint}|{len(workflow.get('inspection_history', []))}".encode()
        ).hexdigest()[:24]
        row = {
            "history_id": history_id, "board_component_id": target_id if scope != "board" else None,
            "fault_id": fault_id, "test_id": test_id, "target_scope": scope,
            "related_fault_ids": list(dict.fromkeys(related_fault_ids or ([fault_id] if fault_id else []))),
            "result": visual_result, "safety_checks": safety_checks,
            "observation_codes": sorted(observations),
            "observations": observations,
            "observation_fingerprint": observation_fingerprint,
            "inspection_cycle_id": cycle_id,
            "source_analysis_fingerprint": analysis_fingerprint,
            "analysis_fingerprint": None,
            "verification_status": (
                "PENDING" if visual_result != "확인 불가"
                and (fault_id or related_fault_ids) else "INCONCLUSIVE"
            ),
            "source": "user_visual_inspection",
        }
        diagnosis_key = f"cc_symptom_diagnosis_{project.project_id}"
        diagnosis_state = st.session_state.setdefault(diagnosis_key, {})
        manual = dict(diagnosis_state.get("manual_observations", {}))
        has_new_observation = any(manual.get(code) != value for code, value in observations.items())
        if observations and not has_new_observation:
            row["verification_status"] = "INCONCLUSIVE"
            row["verification_note"] = "기존과 같은 관측값으로 새 판별 정보가 없습니다."
        if has_new_observation:
            for old in workflow.get("inspection_history", []):
                if (isinstance(old, dict) and old.get("source") == "user_visual_inspection"
                        and old.get("board_component_id") == row["board_component_id"]
                        and old.get("test_id") == test_id
                        and old.get("verification_status") in {"FAULT_SUPPORTED", "FAULT_REFUTED", "CONFLICTED", "INCONCLUSIVE"}):
                    old["verification_status"] = "STALE"
        workflow.setdefault("inspection_history", []).append(row)
        workflow["inspection_history"] = workflow["inspection_history"][-100:]
        workflow.setdefault("inspection_observations", []).append(row.copy())
        workflow["inspection_observations"] = workflow["inspection_observations"][-100:]
        if observations and has_new_observation:
            manual.update(observations)
            diagnosis_state["manual_observations"] = manual
            diagnosis_state["run_after_initial_answer"] = True
            st.session_state[diagnosis_key] = diagnosis_state
            _advance_workflow(st, project, WORKFLOW_REASSESSMENT,
                              "외관 확인 결과를 기존 규칙 엔진으로 재평가 중",
                              "visual_observation_saved")
        else:
            st.info("확인 불가 기록은 이력에만 저장했고 손상 근거로 사용하지 않았습니다.")


def _render_workflow_stepper(st: Any, workflow: dict[str, Any]) -> None:
    stages = (WORKFLOW_REVIEW, WORKFLOW_QUESTIONS, WORKFLOW_INSPECTION,
              WORKFLOW_REASSESSMENT, WORKFLOW_COMPLETE)
    with st.container(key="cc_workbench_stepper"):
        cols = st.columns(len(stages), gap="small")
        current = workflow.get("stage", WORKFLOW_REVIEW)
        for col, stage in zip(cols, stages):
            label = _workflow_stage_title(stage)
            if stage == current:
                col.markdown(f"**{label}**")
            elif workflow.get("stage_completion", {}).get(stage):
                col.caption(f"✓ {label}")
            else:
                col.caption(label)


class _LeftProgressSlot:
    """Render work status beside the PCB without clearing or moving the Viewer."""

    def __init__(self, workflow_status_slot: Any, viewer_slot: Any) -> None:
        self.workflow_status_slot = workflow_status_slot
        self.viewer_slot = viewer_slot

    def container(self) -> Any:
        return self.workflow_status_slot.container()


def _advance_workflow(st: Any, project: ProjectContext, stage: str, label: str, event: str, *,
                      reason: str | None = None) -> None:
    workflow = transition_workflow(st.session_state, project, stage, event=event,
                                   left_status=label, reason=reason or event)
    if stage == WORKFLOW_INSPECTION:
        rows = get_measurements(st.session_state, project_id=project.project_id,
                                board_id=project.board_id or "")
        grouped: dict[str, list[Any]] = {}
        grouped_ids: dict[str, dict[str, str]] = {}
        for row in rows:
            pair = "|".join((str(getattr(row, "board_component_id", None) or ""),
                             str(getattr(row, "fault_hypothesis", None) or ""),
                             str(getattr(row, "test_id", None) or "")))
            grouped.setdefault(pair, []).append(row)
            measurement_id = _measurement_record_id(row)
            grouped_ids.setdefault(pair, {})[measurement_id] = _measurement_row_fingerprint(row)
        previous_cycle = workflow.get("inspection_cycle")
        cycle_id = int(previous_cycle.get("cycle_id", 0)) + 1 if isinstance(previous_cycle, dict) else 1
        workflow["inspection_cycle"] = {
            "cycle_id": cycle_id,
            "baseline_by_pair": {pair: build_measurement_fingerprint(values)
                                 for pair, values in grouped.items()},
            "baseline_record_fingerprints": grouped_ids,
            "started_event": event,
        }
    st.rerun()


def _current_board_snapshot(
    st: Any, project: ProjectContext, components: tuple[Any, ...], registry: dict[str, Any],
    aliases: dict[str, str | None], suspected: list[dict[str, Any]],
) -> dict[str, Any]:
    return current_analysis_snapshot(
        st, project, components=components,
        component_inventory={"__board_registry__": {"registry_version": registry.get("version")}},
        analysis_target_id="board:registry:v1", component_id_aliases=aliases,
        board_registry=registry, suspected_components=suspected, auto_qwen=True,
    )


def _inspection_candidate_rows(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return only hypotheses that remain actionable for another test."""
    if not isinstance(result, dict):
        return []
    rows = []
    for candidate in _simple_candidate_rows(result, limit=1000):
        hypotheses = candidate.get("fault_hypotheses", [])
        active = [hypothesis for hypothesis in hypotheses if str(
            hypothesis.get("measurement_verdict") or hypothesis.get("visual_status") or ""
        ).casefold() not in {"refutes", "conflicted", "fault_refuted"}]
        if hypotheses and not active:
            continue
        rows.append({**candidate, "fault_hypotheses": active or hypotheses})
    return rows


def _render_workflow_candidates(
    st: Any, project: ProjectContext, workflow: dict[str, Any], *,
    registry: dict[str, Any], analysis_snapshot: dict[str, Any],
    components: tuple[Any, ...], preview: bool,
) -> None:
    result = analysis_snapshot.get("result")
    current_result = result if analysis_snapshot.get("record", {}).get("status") == AnalysisStatus.CURRENT.value else None
    candidates = _inspection_candidate_rows(current_result)
    suspects = board_suspects(st.session_state, project)
    all_measurements = get_measurements(st.session_state, project_id=project.project_id,
                                        board_id=project.board_id or "")
    retest_key = f"cc_wf_allow_retest_{project.project_id}"
    allow_retest = st.checkbox("완료된 검사도 조건을 바꿔 재검사", key=retest_key,
                               help="검사 조건 변경 또는 명시적인 재검사 시에만 같은 검사를 다시 선택할 수 있습니다.")
    targets = _inspection_targets(
        registry, candidates, suspects, inspection_history=workflow.get("inspection_history", []),
        measurements=all_measurements, allow_retest=allow_retest,
        symptoms=(current_result.get("symptom_codes", []) if isinstance(current_result, dict) else []),
        limit=None,
    )
    from circuitcare.symptom_diagnosis.active_inspection_planner import plan_active_inspections
    knowledge = load_knowledge()
    measurement_status_by_id = {
        str(row.get("measurement_id")): row.get("assessment_status", row.get("verdict"))
        for row in (current_result.get("measurement_evidence", [])
                    if isinstance(current_result, dict) else [])
        if isinstance(row, dict) and row.get("measurement_id")
    }
    inspection_conditions = {}
    for target_row in registry.get("components", []) if isinstance(registry, dict) else []:
        if not isinstance(target_row, dict):
            continue
        board_component_id = str(target_row.get("board_component_id") or "")
        if not board_component_id:
            continue
        details = next((row for row in candidates
                        if str(row.get("board_component_id") or "") == board_component_id), {})
        fault_rows = details.get("fault_hypotheses", []) if isinstance(details, dict) else []
        fault_ids = [str(row.get("fault_id") or "") for row in fault_rows if isinstance(row, dict)]
        if not fault_ids:
            fault_ids = [str(details.get("fault_id") or "general")]
        for fault_id in fault_ids:
            owner = f"{project.project_id}-{project.board_id or ''}-{board_component_id}-{fault_id}"
            selected_test_id = str(st.session_state.get(f"cc_v5_test_{owner}") or "")
            safety_scope = f"{selected_test_id}|{registry.get('version') or ''}"
            scope_matches = st.session_state.get(f"cc_v5_safety_scope_{owner}") == safety_scope
            inspection_conditions[f"{board_component_id}|{fault_id}"] = {
                "power_state": st.session_state.get(f"cc_v5_power_state_{owner}", "UNKNOWN") if scope_matches else "UNKNOWN",
                "stored_energy_status": st.session_state.get(f"cc_v5_stored_energy_{owner}", "UNKNOWN") if scope_matches else "UNKNOWN",
                "component_isolation_status": st.session_state.get(f"cc_v5_isolation_{owner}", "UNKNOWN") if scope_matches else "UNKNOWN",
                "measurement_location_verified": st.session_state.get(f"cc_v5_location_verified_{owner}", False) if scope_matches else False,
                "reference_source_verified": st.session_state.get(f"cc_v5_reference_verified_{owner}", False) if scope_matches else False,
            }
            cycle_id = int((workflow.get("inspection_cycle") or {}).get("cycle_id", 0))
            widget_instance = f"{project.project_id}_{board_component_id}_cycle{cycle_id}"
            for test_id, test in knowledge.get("test_catalog", {}).get("tests", {}).items():
                if not isinstance(test, dict) or test.get("type") != "visual_inspection":
                    continue
                inspection_conditions[f"{board_component_id}|{fault_id}|{test_id}"] = {
                    "power_state": ("POWER_OFF_CONFIRMED" if st.session_state.get(
                        f"cc_wf_visual_power_off_{widget_instance}_{test_id}") else "UNKNOWN"),
                    "stored_energy_status": ("SAFE_CONFIRMED" if st.session_state.get(
                        f"cc_wf_visual_energy_safe_{widget_instance}_{test_id}") else "UNKNOWN"),
                }
    cycle_id = int((workflow.get("inspection_cycle") or {}).get("cycle_id", 0))
    for fault_id in [str(row.get("fault_id") or "") for row in (
            current_result.get("hypotheses", []) if isinstance(current_result, dict) else [])
            if isinstance(row, dict) and row.get("fault_id")]:
        for test_id, test in knowledge.get("test_catalog", {}).get("tests", {}).items():
            if not isinstance(test, dict) or test.get("type") != "visual_inspection" or test.get("target_scope") != "board":
                continue
            widget_instance = f"{project.project_id}_board_cycle{cycle_id}"
            inspection_conditions[f"board|{fault_id}|{test_id}"] = {
                "power_state": ("POWER_OFF_CONFIRMED" if st.session_state.get(
                    f"cc_wf_visual_power_off_{widget_instance}_{test_id}") else "UNKNOWN"),
                "stored_energy_status": ("SAFE_CONFIRMED" if st.session_state.get(
                    f"cc_wf_visual_energy_safe_{widget_instance}_{test_id}") else "UNKNOWN"),
            }
    active_plan = plan_active_inspections(
        hypotheses=current_result.get("hypotheses", []) if isinstance(current_result, dict) else [],
        component_candidates=(current_result.get("all_component_candidates",
                                                current_result.get("component_candidates", []))
                              if isinstance(current_result, dict) else []),
        functional_block_groups=(current_result.get("functional_block_groups", [])
                                 if isinstance(current_result, dict) else []),
        evidence_gaps=current_result.get("evidence_gaps", []) if isinstance(current_result, dict) else [],
        measurements=[
            {**_measurement_record_dict(row),
             "assessment_status": measurement_status_by_id.get(
                 str(_measurement_record_dict(row).get("measurement_id") or ""))}
            for row in all_measurements
        ],
        observations=current_result.get("observations", {}) if isinstance(current_result, dict) else {},
        inspection_history=workflow.get("inspection_history", []), knowledge=knowledge,
        inspection_conditions=inspection_conditions,
        allow_retest=allow_retest,
    )
    workflow["active_inspection_plan"] = active_plan
    ready_plan_rows = [row for row in active_plan.get("candidates", []) if isinstance(row, dict)]
    blocked_plan_rows = [row for row in active_plan.get("blocked", []) if isinstance(row, dict)]
    planner_rows = [*ready_plan_rows, *blocked_plan_rows]
    planned_targets = list(dict.fromkeys(
        str(component_id)
        for item in planner_rows
        for component_id in item.get("target_component_ids", [])
    ))
    if planned_targets:
        planned_rank = {component_id: index for index, component_id in enumerate(planned_targets)}
        selected_target = str(st.session_state.get(f"cc_wf_inspection_target_{project.project_id}")
                              or workflow.get("selected_target_id") or "")
        targets.sort(key=lambda row: (
            str(row.get("board_component_id") or "") != selected_target,
            planned_rank.get(str(row.get("board_component_id") or ""), len(planned_rank) + len(targets)),
        ))
    planned_test_rank = {}
    for plan_rank, plan_item in enumerate(ready_plan_rows):
        for target_id in plan_item.get("target_component_ids", []):
            target_ranks = planned_test_rank.setdefault(str(target_id), {})
            target_ranks.setdefault(str(plan_item.get("test_id")), plan_rank)
    blocked_plan_by_target: dict[str, list[dict[str, Any]]] = {}
    for plan_item in blocked_plan_rows:
        for target_id in plan_item.get("target_component_ids", []):
            blocked_plan_by_target.setdefault(str(target_id), []).append(plan_item)
    for target in targets:
        ranks = planned_test_rank.get(str(target.get("board_component_id") or ""), {})
        target.get("available_tests", []).sort(key=lambda item: (
            ranks.get(str(item.get("test_id") or ""), len(ranks)),
            str(item.get("test_id") or ""),
        ))
    targets = targets[:3]
    workflow["candidate_ids"] = [str(row.get("board_component_id") or row.get("component_id"))
                                  for row in candidates]
    workflow["inspection_targets"] = targets
    workflow["analysis_fingerprint"] = analysis_snapshot.get("input_fingerprint")
    unlocated_count = sum(
        1 for row in registry.get("components", []) if isinstance(row, dict)
        and row.get("presence_status") != "EXCLUDED"
        and row.get("component_type") != "board_feature"
        and row.get("source_components")
        and measurement_view_model_for_board_component(
            registry, board_component_id=str(row.get("board_component_id") or ""),
            preferred_photo_id=st.session_state.get(ACTIVE_PHOTO_KEY),
        ) is None
    )
    if not targets:
        st.warning("사진 좌표를 검증할 수 있는 활성 소자 박스가 없습니다. Registry의 내부 ID만으로는 검사 위치를 만들지 않습니다.")
        if unlocated_count:
            st.info(f"활성 Registry 소자 {unlocated_count}개는 사진·박스 좌표 연결이 확인되지 않아 검사 대상에서 제외했습니다. 1단계에서 실제 박스 위치를 확인하거나 수동 박스를 추가해 주세요.")
        if st.button("소자 검수로 돌아가기", key=f"cc_wf_no_targets_{project.project_id}"):
            _advance_workflow(st, project, WORKFLOW_REVIEW, "소자 검수 화면 준비 중",
                              "no_valid_target",
                              reason="현재 Registry에서 측정·검사 대상으로 해석 가능한 소자가 없습니다.")
        return

    st.markdown("#### 우선 확인할 검사 대상 · 최대 3개")
    measurable = sum(target.get("inspection_kind") == "MEASUREMENT" for target in targets)
    st.caption(f"측정 가능 {measurable}개 · 추가 확인 대상 {len(targets) - measurable}개")
    if unlocated_count:
        st.caption(f"위치 미확인 Registry 소자 {unlocated_count}개는 실제 검사 위치가 없어 후보에서 제외했습니다.")
    for index, target in enumerate(targets, start=1):
        candidate = target.get("candidate", {})
        hypotheses = candidate.get("fault_hypotheses", []) if isinstance(candidate, dict) else []
        hypothesis = hypotheses[0] if hypotheses else {}
        ref = target.get("reference_designators") or []
        label_parts = [f"{index}순위"]
        if ref:
            label_parts.append(str(ref[0]))
        if target.get("component_type_confirmed") and target.get("component_type") not in {None, "", "unknown"}:
            label_parts.append(str(target["component_type"]))
        label = " · ".join(label_parts)
        selected_target_id = st.session_state.get(
            f"cc_wf_inspection_target_{project.project_id}", workflow.get("selected_target_id")
        )
        is_selected = str(selected_target_id or "") == str(target["board_component_id"])
        st.markdown(f"**{label}**" + (" · 측정 대상" if is_selected else ""))
        cause_status = ("측정 근거로 지지" if hypothesis.get("visual_status") == "FAULT_SUPPORTED"
                        else "측정 근거로 반박" if hypothesis.get("visual_status") == "FAULT_REFUTED"
                        else "근거 충돌" if hypothesis.get("visual_status") == "CONFLICTED"
                        else "판단 보류" if hypothesis.get("visual_status") == "INCONCLUSIVE"
                        else "미확정")
        if hypothesis:
            fault_title = hypothesis.get("fault_title") or hypothesis.get("fault_id")
            if target.get("reason") == "사용자가 의심 소자로 지정":
                why_selected = "사용자가 의심 소자로 지정해 확인 대상으로 올렸습니다."
            elif fault_title:
                why_selected = f"{fault_title} 가능성을 확인하기 위해 선정됐습니다. 고장 확정은 아닙니다."
            else:
                why_selected = "증상·측정·규칙 분석과 관련된 확인 대상입니다. 고장 확정은 아닙니다."
            supporting = hypothesis.get("supporting_evidence") or []
            opposing = hypothesis.get("opposing_evidence") or []
            if isinstance(supporting, str):
                supporting = [supporting]
            if isinstance(opposing, str):
                opposing = [opposing]
        elif target.get("reason") == "사용자가 의심 소자로 지정":
            why_selected = "사용자가 의심 소자로 지정해 확인 대상으로 올렸습니다."
            supporting = opposing = []
        elif target.get("reason") == "증상·측정·규칙 분석 후보":
            why_selected = "증상·측정·규칙 분석에서 관련 확인 대상으로 선정됐습니다. 고장 확정은 아닙니다."
            supporting = opposing = []
        else:
            why_selected = "현재 근거만으로 원인이 특정되지 않아 추가 확인 대상으로 유지 중입니다."
            supporting = opposing = []
        st.caption(f"확인 이유 · {why_selected}")
        if supporting:
            st.caption("관련 근거 · " + " · ".join(str(item) for item in supporting[:3]))
        if opposing:
            st.caption("반대 근거 · " + " · ".join(str(item) for item in opposing[:2]))
        st.caption(f"검증 상태 · {cause_status}")
        for blocked_item in blocked_plan_by_target.get(str(target.get("board_component_id") or ""), []):
            prerequisites = [str(value) for value in blocked_item.get("prerequisites", []) if value]
            st.warning(f"검사 대기 · {blocked_item.get('name', blocked_item.get('test_id', '검사'))} · "
                       + (" · ".join(prerequisites) if prerequisites else "필수 검사 조건 미확인"))
        focus_col, measure_col = st.columns([1, 1])
        with focus_col:
            if target.get("measurement_available") and st.button(
                    "PCB에서 위치 보기", key=f"cc_wf_target_focus_{project.project_id}_{target['board_component_id']}"):
                if _focus_board_registry_component(st, project, registry, target["board_component_id"]):
                    workflow["viewer_focus_id"] = target["board_component_id"]
                    st.rerun()
        with measure_col:
            if st.button(
                    "이 소자 측정", key=f"cc_wf_target_measure_{project.project_id}_{target['board_component_id']}",
                    disabled=not target.get("measurement_available"),
                    help=None if target.get("measurement_available") else "검증된 PCB 위치가 없어 측정 입력을 열 수 없습니다."):
                st.session_state[f"cc_wf_inspection_target_{project.project_id}"] = target["board_component_id"]
                st.session_state[MEASUREMENT_TARGET_BOARD_COMPONENT_KEY] = target["board_component_id"]
                workflow["selected_target_id"] = target["board_component_id"]

    target_ids = [row["board_component_id"] for row in targets]
    select_key = f"cc_wf_inspection_target_{project.project_id}"
    selected_id = st.session_state.get(select_key)
    if selected_id not in target_ids:
        selected_id = workflow.get("selected_target_id")
    if selected_id not in target_ids:
        selected_id = None
    if selected_id is None:
        st.info("측정할 소자 카드의 ‘이 소자 측정’을 누르면 해당 소자 전용 측정 입력이 열립니다.")
    else:
        target = next(row for row in targets if row["board_component_id"] == selected_id)
        workflow["selected_target_id"] = selected_id
        st.session_state[MEASUREMENT_TARGET_BOARD_COMPONENT_KEY] = selected_id
        candidate = target.get("candidate", {}) if isinstance(target.get("candidate"), dict) else {}
        hypotheses = [row for row in candidate.get("fault_hypotheses", []) if row.get("fault_id")]
        fault_ids = list(dict.fromkeys(str(row["fault_id"]) for row in hypotheses))
        fault_by_target = workflow.setdefault("selected_fault_by_target", {})
        fault_id = fault_by_target.get(selected_id)
        if fault_id not in fault_ids:
            fault_id = fault_ids[0] if fault_ids else None
        if len(fault_ids) > 1:
            fault_id = st.selectbox("검증할 고장 가설", fault_ids,
                                    key=f"cc_wf_fault_{project.project_id}_{selected_id}",
                                    index=fault_ids.index(fault_id))
        workflow["selected_fault_id"] = fault_id
        fault_by_target[selected_id] = fault_id
        hypothesis = next((row for row in hypotheses if row.get("fault_id") == fault_id), {})
        available_tests = target.get("available_tests", [])
        matching_tests = {str(item["test_id"]): item["profile"] for item in available_tests
                          if isinstance(item, dict) and item.get("test_id") and
                          (not fault_id or fault_id in item.get("fault_ids", []))}
        target_ranks = planned_test_rank.get(selected_id, {})
        ordered_tests = sorted(matching_tests, key=lambda test_id: (
            target_ranks.get(test_id, len(ready_plan_rows)), test_id))
        matching_tests = {test_id: matching_tests[test_id] for test_id in ordered_tests}
        recommended = next((test_id for test_id in matching_tests), None)
        if not recommended and hypothesis.get("recommended_tests"):
            st.caption("추가 검사 안내 · " + " · ".join(str(item) for item in hypothesis["recommended_tests"][:2]))
        if matching_tests:
            test_by_target = workflow.setdefault("selected_test_by_target", {})
            current_test = test_by_target.get(selected_id)
            if current_test not in matching_tests:
                current_test = recommended or matching_tests[0]
            test_id = st.selectbox("검사 방법", list(matching_tests),
                                   key=f"cc_wf_test_{project.project_id}_{selected_id}_{fault_id or 'general'}",
                                   index=list(matching_tests).index(current_test),
                                   format_func=lambda value: matching_tests.get(value, {}).get("title", value))
        else:
            test_id = None
            st.info("새로 실행할 검증 측정이 없습니다. 기존 검사는 반복 추천하지 않습니다. 부품 표기·외관을 확인하거나 조건이 달라졌다면 위의 재검사 선택을 켜세요.")
        workflow["selected_test_id"] = test_id
        workflow.setdefault("selected_test_by_target", {})[selected_id] = test_id
        measurement_component = measurement_view_model_for_board_component(
            registry, board_component_id=selected_id, preferred_photo_id=st.session_state.get(ACTIVE_PHOTO_KEY),
        )
        st.markdown(f"##### 측정 입력 · {target.get('display_label') or target.get('component_type') or '선택 소자'}")
        if measurement_component is None:
            st.warning("소자의 사진 위치를 확인할 수 없어 측정 연결을 중단했습니다.")
        else:
            render_measurement_panel(
                st, project, selected_component=measurement_component, preview=preview,
                component_map_version=registry.get("version"), board_registry=registry,
                compact=True, fault_hypothesis=fault_id, recommended_test_id=test_id,
            )
        visual_checks = _compatible_visual_checks(
            load_knowledge().get("test_catalog", {}).get("tests", {}),
            target_scope="component", component_type=str(target.get("component_type") or ""),
            subtype=str(target.get("component_subtype") or ""),
            fault_ids=[fault_id] if fault_id else [],
            symptoms=(set(current_result.get("symptom_codes", [])) if isinstance(current_result, dict) else set()),
        )
        if visual_checks:
            st.markdown("##### 소자별 육안 검사")
        for visual_test in visual_checks:
            _render_visual_inspection(
                st, project, workflow, test=visual_test, target_id=selected_id,
                analysis_fingerprint=analysis_snapshot.get("input_fingerprint"), preview=preview,
                widget_suffix=f"{project.project_id}_{selected_id}", fault_id=fault_id,
            )
    # Board-wide checks are never attached to an arbitrary component candidate.
    board_symptoms = set(current_result.get("symptom_codes", [])) if isinstance(current_result, dict) else set()
    catalog = load_knowledge().get("test_catalog", {}).get("tests", {})
    possible_board_faults = {
        str(fault_id) for test_row in catalog.values() if isinstance(test_row, dict)
        and test_row.get("type") == "visual_inspection" and test_row.get("target_scope") == "board"
        for fault_id in test_row.get("fault_ids", [])
    } if isinstance(catalog, dict) else set()
    board_fault_ids = [str(row.get("fault_id") or "") for row in (
        current_result.get("hypotheses", []) if isinstance(current_result, dict) else [])
        if isinstance(row, dict) and str(row.get("fault_id") or "") in possible_board_faults]
    board_visual_checks = _compatible_visual_checks(
        catalog,
        target_scope="board", symptoms=board_symptoms,
        fault_ids=board_fault_ids,
    )
    if board_visual_checks:
        with st.expander("기판 전체 외관 검사", expanded=False):
            for visual_test in board_visual_checks:
                _render_visual_inspection(
                    st, project, workflow, test=visual_test, target_id=None,
                    analysis_fingerprint=analysis_snapshot.get("input_fingerprint"), preview=preview,
                    widget_suffix=f"{project.project_id}_board",
                    related_fault_ids=[fault_id for fault_id in board_fault_ids
                                       if fault_id in visual_test.get("fault_ids", [])],
                )
    if selected_id is not None:
        connected = [row for row in all_measurements
                     if str(getattr(row, "board_component_id", None) or "") == selected_id
                     and str(getattr(row, "test_id", None) or "") == str(test_id or "")
                     and (not fault_id or getattr(row, "fault_hypothesis", None) in {None, fault_id})]
        workflow["measurement_ids"] = [_measurement_record_id(row) for row in all_measurements]
        cycle = workflow.get("inspection_cycle")
        if not isinstance(cycle, dict):
            cycle = {"cycle_id": 1, "baseline_by_pair": {}}
            workflow["inspection_cycle"] = cycle
        pair_key = f"{selected_id}|{fault_id or ''}|{test_id or ''}"
        current_measurement_fingerprint = build_measurement_fingerprint(connected)
        baseline_fingerprint = cycle.get("baseline_by_pair", {}).get(pair_key)
        if baseline_fingerprint is None:
            baseline_fingerprint = current_measurement_fingerprint
            cycle.setdefault("baseline_by_pair", {})[pair_key] = baseline_fingerprint
        has_new_measurement = bool(connected) and current_measurement_fingerprint != baseline_fingerprint
        baseline_records = cycle.get("baseline_record_fingerprints", {}).get(pair_key, {})
        changed_measurement_ids = [_measurement_record_id(row) for row in connected
                                   if baseline_records.get(_measurement_record_id(row))
                                   != _measurement_row_fingerprint(row)]
        cycle["active_pair_key"] = pair_key
        cycle["current_pair_fingerprint"] = current_measurement_fingerprint
        cycle["baseline_pair_fingerprint"] = baseline_fingerprint
        cycle["connected_measurement_ids"] = [_measurement_record_id(row) for row in connected]
        cycle["changed_measurement_ids"] = changed_measurement_ids
        if connected and has_new_measurement:
            st.success(f"새 검사 주기의 기록 {len(connected)}개 · 재판단 가능")
        elif connected:
            st.info(f"이전 검사 기록 {len(connected)}개는 보존되어 있습니다. 새 측정이나 유효한 수정 후 재판단할 수 있습니다.")
        if st.button("측정 결과 분석", type="primary", disabled=preview or not has_new_measurement,
                     key=f"cc_wf_reassess_{project.project_id}_{selected_id}"):
            workflow["reassessment_history"] = [
                *workflow.get("reassessment_history", []),
                {"target_id": selected_id, "board_component_id": selected_id,
                 "fault_id": fault_id, "test_id": test_id,
                 "measurement_ids": [_measurement_record_id(row) for row in connected],
                 "changed_measurement_ids": changed_measurement_ids,
                 "measurement_fingerprint": current_measurement_fingerprint,
                 "verification_status": "PENDING",
                 "automatic_run_started": False,
                 "analysis_fingerprint": analysis_snapshot.get("input_fingerprint")},
            ][-100:]
            # This is a one-shot user request. Rebuild the current snapshot on
            # entry to step 4, then consume the flag before running so an error
            # or stale commit cannot send the app into an automatic retry loop.
            workflow["reassessment_run_requested"] = True
            _advance_workflow(st, project, WORKFLOW_REASSESSMENT,
                              "측정 결과 분석 중 · 현재 PCB Registry 유지", "measurement_saved")


def _render_workflow_reassessment(
    st: Any, project: ProjectContext, workflow: dict[str, Any], *,
    registry: dict[str, Any], analysis_snapshot: dict[str, Any], components: tuple[Any, ...],
    aliases: dict[str, str | None], suspected: list[dict[str, Any]], preview: bool,
    left_progress_slot: Any,
) -> dict[str, Any]:
    reassessment_history = workflow.get("reassessment_history", [])
    pending_event = reassessment_history[-1] if reassessment_history else None
    pending_event_unstarted = (
        isinstance(pending_event, dict)
        and pending_event.get("verification_status") == "PENDING"
        and not pending_event.get("automatic_run_started")
    )
    explicit_request = bool(workflow.pop("reassessment_run_requested", False))
    reassessment_run_requested = bool(
        not preview and (explicit_request or pending_event_unstarted)
    )
    if reassessment_run_requested and pending_event_unstarted:
        pending_event["automatic_run_started"] = True
    if reassessment_run_requested:
        LOGGER.info(
            "workflow_reassessment_run_once project=%s fingerprint=%s",
            project.project_id,
            str(getattr(analysis_snapshot.get("fingerprints"), "final_sha256", ""))[:16],
        )
    render_symptom_diagnosis_panel(
        st, project, components=components, preview=preview,
        component_inventory={"__board_registry__": {"registry_version": registry.get("version")}},
        analysis_target_id="board:registry:v1", component_id_aliases=aliases,
        board_registry=registry, suspected_components=suspected, simple_ui=True,
        workflow_stage=WORKFLOW_REASSESSMENT, left_progress_slot=left_progress_slot,
        reassessment_run_requested=reassessment_run_requested,
    )
    latest = _current_board_snapshot(st, project, components, registry, aliases, suspected)
    record = latest.get("record", {})
    if record.get("status") != AnalysisStatus.CURRENT.value or not isinstance(latest.get("result"), dict):
        status = record.get("status", AnalysisStatus.NOT_RUN.value)
        if status == AnalysisStatus.RUNNING.value:
            st.info("현재 측정값을 반영해 재판정 중입니다. 입력과 이력은 보존됩니다.")
        elif status == AnalysisStatus.FAILED.value:
            st.error(f"측정 결과 재판정에 실패했습니다 · {record.get('reason') or '다시 시도해 주세요.'}")
        elif status == AnalysisStatus.STALE.value:
            st.warning(f"측정·증상 입력이 바뀌어 재판정 결과가 필요합니다 · {record.get('reason') or '재판정을 다시 실행해 주세요.'}")
        else:
            st.info("현재 입력으로 재판정 결과를 만들고 있습니다. 이전 결과를 최신 결과로 표시하지 않습니다.")
        return latest
    result = latest["result"]
    fingerprint = latest.get("input_fingerprint")
    if fingerprint and (not workflow.get("reassessment_history")
                        or (workflow["reassessment_history"][-1].get("analysis_fingerprint") != fingerprint
                            and workflow["reassessment_history"][-1].get("verification_status") != "PENDING")):
        if workflow.get("reassessment_history"):
            workflow["reassessment_history"][-1]["analysis_fingerprint"] = fingerprint
        else:
            workflow["reassessment_history"] = [{"analysis_fingerprint": fingerprint,
                                                   "status": "CURRENT"}]
    st.markdown("#### 고장 가설별 재판정")
    candidates = _simple_candidate_rows(result, limit=1000)
    if not candidates:
        st.info("현재 근거로 소자 고장 후보를 특정하지 못했습니다. 임의의 소자 판정을 만들지 않았습니다.")
        missing_symptoms = not result.get("symptom_codes")
        confirmed_component_count = sum(
            getattr(component, "presence_status", None) == "CONFIRMED" for component in components
        )
        needs_component_review = confirmed_component_count == 0
        if missing_symptoms or needs_component_review:
            reason_cols = st.columns(2 if missing_symptoms and needs_component_review else 1)
            if missing_symptoms:
                reason_cols[0].warning("실제 증상이 구조화되지 않았습니다. ‘기타·모르겠음’만으로는 고장 유형을 좁힐 수 없습니다.")
                if reason_cols[0].button("증상 질문으로 돌아가기", key=f"cc_wf_reassess_symptoms_{project.project_id}"):
                    _advance_workflow(st, project, WORKFLOW_QUESTIONS,
                                      "증상 답변 확인 중 · 측정 기록 유지", "reassessment_needs_symptoms")
            if needs_component_review:
                component_col = reason_cols[-1]
                component_col.warning("확인된 소자가 0개입니다. 탐지 박스는 존재하지만 사용자 확인 전이라 특정 소자 고장 후보로 연결하지 않았습니다.")
                if component_col.button("소자 검수로 돌아가기", key=f"cc_wf_reassess_components_{project.project_id}"):
                    _advance_workflow(st, project, WORKFLOW_REVIEW,
                                      "소자 종류·존재 확인 중 · 증상과 측정 기록 유지", "reassessment_needs_component_review")
    for candidate in candidates:
        identity = candidate.get("board_component_id") or candidate.get("component_id")
        for hypothesis in candidate.get("fault_hypotheses", []):
            verdict = str(hypothesis.get("measurement_verdict") or "").casefold()
            status = str(hypothesis.get("visual_status") or {
                "supports": "FAULT_SUPPORTED", "refutes": "FAULT_REFUTED",
                "conflicted": "CONFLICTED", "non_diagnostic": "INCONCLUSIVE",
            }.get(verdict, "SUSPECTED"))
            label = {"FAULT_SUPPORTED": "측정이 이 고장 가설을 지지",
                     "FAULT_REFUTED": "측정이 이 고장 가설을 반박",
                     "INCONCLUSIVE": "판단 보류 · 측정 근거 부족",
                     "CONFLICTED": "판단 보류 · 근거 충돌"}.get(status, "미확정 가설")
            st.markdown(f"**{identity} · {hypothesis.get('fault_title') or hypothesis.get('fault_id', '원인 후보')}**")
            st.caption(label + " · 소자 전체의 고장·정상 확정이 아닙니다.")
            evidence = hypothesis.get("supporting_evidence") or hypothesis.get("evidence") or []
            opposing = hypothesis.get("opposing_evidence") or []
            st.caption("지지 근거 · " + (" · ".join(map(str, evidence)) or "없음"))
            st.caption("반대 근거 · " + (" · ".join(map(str, opposing)) or "없음"))
    measurements = get_measurements(st.session_state, project_id=project.project_id,
                                    board_id=project.board_id or "")
    by_id = {_measurement_record_id(row): row for row in measurements}
    manual_observations = st.session_state.get(
        f"cc_symptom_diagnosis_{project.project_id}", {}
    ).get("manual_observations", {})
    for entry in workflow.get("inspection_history", []):
        if not isinstance(entry, dict) or entry.get("source") != "user_visual_inspection":
            continue
        observations = entry.get("observations", {})
        if any(manual_observations.get(code) != value for code, value in observations.items()):
            if entry.get("verification_status") != "STALE":
                entry["verification_status"] = "STALE"
        elif (entry.get("analysis_fingerprint") and fingerprint
              and entry.get("analysis_fingerprint") != fingerprint
              and entry.get("verification_status") not in {"PENDING", "STALE"}):
            entry["verification_status"] = "STALE"
    for event in workflow.get("reassessment_history", []):
        if not isinstance(event, dict) or not event.get("measurement_ids"):
            continue
        current_rows = [by_id[str(mid)] for mid in event.get("measurement_ids", []) if str(mid) in by_id]
        current_fp = build_measurement_fingerprint(current_rows)
        if len(current_rows) != len(event.get("measurement_ids", [])) or (
                event.get("measurement_fingerprint") and current_fp != event.get("measurement_fingerprint")):
            event["verification_status"] = "STALE"
    for entry in workflow.get("inspection_history", []):
        if (not isinstance(entry, dict) or entry.get("source") != "measurement_reassessment"
                or not entry.get("measurement_ids")):
            continue
        current_rows = [by_id[str(mid)] for mid in entry.get("measurement_ids", []) if str(mid) in by_id]
        current_fp = build_measurement_fingerprint(current_rows)
        if len(current_rows) != len(entry.get("measurement_ids", [])) or (
                entry.get("measurement_fingerprint") and current_fp != entry.get("measurement_fingerprint")):
            entry["verification_status"] = "STALE"
    if workflow.get("reassessment_history"):
        active_cycle = workflow["reassessment_history"][-1]
        changed_ids = set(str(value) for value in active_cycle.get("changed_measurement_ids", []))
        st.markdown("#### 이번 재판단에 사용한 측정")
        for measurement_id in active_cycle.get("measurement_ids", []):
            measured = by_id.get(str(measurement_id))
            if measured is None:
                continue
            qualifier = "새로 입력 또는 수정" if str(measurement_id) in changed_ids else "이전 기록 · 참고 증거"
            st.caption(
                f"{qualifier} · {getattr(measured, 'test_id', None) or '검사 미지정'} · "
                f"{getattr(measured, 'value', '')} {getattr(measured, 'unit', '')} · "
                f"검증 {getattr(measured, 'verification_status', '미확인')} · "
                f"기준 {getattr(measured, 'reference_source', None) or '출처 미확인'}"
            )
    def _hypothesis_status(target_id: str | None, fault_id: str | None) -> str | None:
        if not fault_id:
            return None
        hypothesis = None
        if target_id:
            hypothesis = next((row for candidate in candidates
                               if str(candidate.get("board_component_id") or candidate.get("component_id")) == target_id
                               for row in candidate.get("fault_hypotheses", [])
                               if str(row.get("fault_id") or row.get("fault_hypothesis") or "") == fault_id), None)
        else:
            hypothesis = next((row for row in result.get("hypotheses", [])
                               if isinstance(row, dict)
                               and str(row.get("fault_id") or row.get("fault_hypothesis") or "") == fault_id), None)
        if not isinstance(hypothesis, dict):
            return None
        raw = str(hypothesis.get("measurement_verdict") or hypothesis.get("visual_status") or "").casefold()
        return {"fault_supported": "FAULT_SUPPORTED", "supports": "FAULT_SUPPORTED",
                "supported": "FAULT_SUPPORTED", "fault_refuted": "FAULT_REFUTED", "refutes": "FAULT_REFUTED",
                "refuted": "FAULT_REFUTED",
                "conflicted": "CONFLICTED", "inconclusive": "INCONCLUSIVE",
                "non_diagnostic": "INCONCLUSIVE", "suspected": "INCONCLUSIVE"}.get(raw)

    confirmed_ids = {str(row.get("history_id")) for row in workflow.get("inspection_history", [])
                     if isinstance(row, dict) and row.get("history_id")}
    for visual in workflow.get("inspection_history", []):
        if (not isinstance(visual, dict) or visual.get("source") != "user_visual_inspection"
                or visual.get("verification_status") != "PENDING"):
            continue
        if not fingerprint or fingerprint == visual.get("source_analysis_fingerprint"):
            continue
        current_facts = result.get("observations", {})
        if any(current_facts.get(code) != value
               for code, value in visual.get("observations", {}).items()):
            continue
        target_fault_ids = visual.get("related_fault_ids") or (
            [visual.get("fault_id")] if visual.get("fault_id") else [])
        resolved = [_hypothesis_status(visual.get("board_component_id"), fault_id)
                    for fault_id in target_fault_ids]
        resolved = [value for value in resolved if value]
        status = ("CONFLICTED" if {"FAULT_SUPPORTED", "FAULT_REFUTED"} <= set(resolved)
                  else resolved[0] if resolved else None)
        if status is None:
            continue
        visual["verification_status"] = status
        visual["analysis_fingerprint"] = fingerprint
        if visual.get("history_id") not in confirmed_ids:
            workflow.setdefault("inspection_history", []).append(visual.copy())
            confirmed_ids.add(str(visual.get("history_id")))

    for event in workflow.get("reassessment_history", []):
        if (not isinstance(event, dict) or event.get("verification_status") != "PENDING"
                or not fingerprint or fingerprint == event.get("analysis_fingerprint")):
            continue
        if not event.get("measurement_ids") or len(
                [mid for mid in event["measurement_ids"] if str(mid) in by_id]
        ) != len(event["measurement_ids"]):
            event["verification_status"] = "STALE"
            continue
        evidence_rows = [row for row in result.get("measurement_evidence", [])
                         if isinstance(row, dict)]
        evidenced_ids = {
            str(row.get("measurement_id")) for row in evidence_rows
            if str(row.get("test_id") or "") == str(event.get("test_id") or "")
            and str(row.get("fault_hypothesis") or "") == str(event.get("fault_id") or "")
            and str(row.get("component_id") or "") in {
                str(event.get("board_component_id") or ""), str(event.get("target_id") or ""),
            }
        }
        if not set(str(mid) for mid in event["measurement_ids"]) <= evidenced_ids:
            continue
        status = _hypothesis_status(
            str(event.get("board_component_id") or event.get("target_id") or ""),
            str(event.get("fault_id") or ""),
        )
        if status is None:
            continue
        event["verification_status"] = status
        event["analysis_fingerprint"] = fingerprint
        event["measurement_fingerprint"] = build_measurement_fingerprint(
            [by_id[str(mid)] for mid in event["measurement_ids"]]
        )
        history_id = hashlib.sha256(
            f"measurement|{event.get('board_component_id')}|{event.get('fault_id')}|"
            f"{event.get('test_id')}|{event.get('measurement_fingerprint')}|{fingerprint}".encode()
        ).hexdigest()[:24]
        if history_id not in confirmed_ids:
            workflow.setdefault("inspection_history", []).append({
                "history_id": history_id, "board_component_id": event.get("board_component_id"),
                "fault_id": event.get("fault_id"), "test_id": event.get("test_id"),
                "measurement_ids": list(event.get("measurement_ids", [])),
                "measurement_fingerprint": event.get("measurement_fingerprint"),
                "verification_status": status, "analysis_fingerprint": fingerprint,
                "inspection_cycle_id": (workflow.get("inspection_cycle") or {}).get("cycle_id"),
                "source": "measurement_reassessment",
            })
            confirmed_ids.add(history_id)
    st.caption(f"보존된 측정 기록 · {len(measurements)}개")
    return latest


def _render_workflow_report(
    st: Any, project: ProjectContext, registry: dict[str, Any], analysis_snapshot: dict[str, Any],
) -> None:
    st.markdown("#### 검사 결과 보고서")
    report_snapshot = build_pcb_diagnostic_snapshot(
        root=Path(__file__).resolve().parents[4], project=project,
        registry=registry, analysis_snapshot=analysis_snapshot,
        photos=st.session_state.get(INTAKE_IMAGES_KEY, []),
        selected_board_component_id=st.session_state.get(BOARD_COMPONENT_SELECTION_KEY),
        candidate_strategy=str(st.session_state.get(f"cc_candidate_strategy_{project.project_id}", "baseline")),
    )
    render_report_panel(st, snapshot=report_snapshot, root=Path(__file__).resolve().parents[4],
                        workspace_id=project.project_id, expected_circuit_hash=None)


def render_workbench_shell(st: Any, project: ProjectContext) -> None:
    """Render the workbench in a fragment so its interactions stay local."""
    run_count = int(st.session_state.get("cc_debug_full_script_rerun_count", 0)) + 1
    st.session_state["cc_debug_full_script_rerun_count"] = run_count
    active_photo_id = st.session_state.get("circuitcare_v2_photo_choice") or st.session_state.get(ACTIVE_PHOTO_KEY)
    rerun_cause = st.session_state.pop("cc_debug_next_full_script_rerun_cause", "widget_or_external")
    rerun_causes = st.session_state.setdefault("cc_debug_full_script_rerun_causes", {})
    rerun_causes[rerun_cause] = int(rerun_causes.get(rerun_cause, 0)) + 1
    LOGGER.info("FULL_SCRIPT_RERUN project=%s photo_id=%s count=%s cause=%s cause_count=%s ts_ns=%s",
                project.project_id, active_photo_id or "unknown", run_count, rerun_cause,
                rerun_causes[rerun_cause], time.time_ns())
    # Keep global CSS outside the fragment so a fragment update never removes
    # and reinserts the styles that affect the fixed Viewer layout.
    _style(st)

    fragment = getattr(st, "fragment", None)
    if callable(fragment):
        # This renderer is nested in ``main`` and redefined on every Streamlit
        # full rerun. Without an explicit key Streamlit may retire the previous
        # fragment ID while a component/fragment event is still in flight; the
        # stale event then falls back to a full-script rerun and flashes the
        # Viewer. Keep the fragment identity stable for this project session.
        fragment_key = "cc_workbench_" + hashlib.sha256(project.project_id.encode()).hexdigest()[:16]

        def _render_workbench_fragment() -> None:
            state = st.session_state
            fragment_count = int(state.get("cc_debug_fragment_rerun_count", 0)) + 1
            state["cc_debug_fragment_rerun_count"] = fragment_count
            cause = state.pop("cc_debug_next_fragment_rerun_cause", "widget_or_component")
            last_full_count = state.get("cc_debug_fragment_last_full_count")
            event = "FRAGMENT_RERUN" if last_full_count == run_count else "FRAGMENT_EXECUTION"
            state["cc_debug_fragment_last_full_count"] = run_count
            causes = state.setdefault("cc_debug_fragment_rerun_causes", {})
            causes[cause] = int(causes.get(cause, 0)) + 1
            LOGGER.info("%s project=%s photo_id=%s count=%s cause=%s cause_count=%s ts_ns=%s",
                        event, project.project_id,
                        state.get("circuitcare_v2_photo_choice") or state.get(ACTIVE_PHOTO_KEY) or "unknown",
                        fragment_count, cause, causes[cause], time.time_ns())
            _render_workbench_shell_body(st, project)

        fragment_decorator = fragment(key=fragment_key)
        rendered_fragment = (fragment_decorator(_render_workbench_fragment)
                             if callable(fragment_decorator) else _render_workbench_fragment)
        rendered_fragment()
        return

    # Test doubles and older Streamlit-compatible hosts retain the same renderer.
    _render_workbench_shell_body(st, project)


def _render_workbench_shell_body(st: Any, project: ProjectContext) -> None:
    """Keep the PCB Viewer fixed while rendering one selected task panel."""
    selection = st.session_state.get(SELECTION_KEY)
    if getattr(selection, "project_id", None) != project.project_id:
        enter_workbench(
            st.session_state, project,
            preview=st.session_state.get(MODE_KEY) == "PREVIEW_ONLY",
        )
    workspace_tab_key = f"cc_workbench_top_tab_{project.project_id}"
    with st.container(key="cc_workbench_top_navigation"):
        if callable(getattr(st, "segmented_control", None)):
            workspace_tab = st.segmented_control(
                "작업 탭", ["PCB 검수·진단", "별도 재학습"], default="PCB 검수·진단",
                key=workspace_tab_key, label_visibility="collapsed",
            )
        else:
            workspace_tab = st.radio("작업 탭", ["PCB 검수·진단", "별도 재학습"],
                                     horizontal=True, key=workspace_tab_key,
                                     label_visibility="collapsed")
    if workspace_tab == "별도 재학습":
        render_retraining_workspace(st, project)
        return
    photos = st.session_state.get(INTAKE_IMAGES_KEY, [])
    pending_photo_focus = st.session_state.pop(PENDING_PHOTO_FOCUS_KEY, None)
    valid_photo_ids = {str(item.get("image_id") or "") for item in photos if isinstance(item, dict)}
    if pending_photo_focus in valid_photo_ids:
        # Apply the keyed selectbox value before it is instantiated in _render_pcb.
        # Streamlit rejects changing widget-owned state later in the same run.
        st.session_state["circuitcare_v2_photo_choice"] = pending_photo_focus
    workflow = get_workflow_state(st.session_state, project)
    with st.container(key="cc_workbench_body"):
        left, right = st.columns([1.85, 1], gap="large")
        workflow_status_slot = None
        qwen_progress_slot = None
        viewer_slot = None
        viewer_error_slot = None
        with left:
            # YOLO, transitions, OCR and Qwen share one reserved surface. A
            # disappearing status above a separate Qwen surface shifted Viewer.
            with st.container(height=44, border=False, key="cc_workflow_progress"):
                workflow_status_slot = st.empty()
            qwen_progress_slot = workflow_status_slot
            # Keep the component inside fixed keyed containers. An st.empty()
            # placeholder is replaced on every fragment execution; when it
            # owns a Components v2 iframe this tears down the iframe, fires
            # VIEWER_UNMOUNT, reloads the SVG image, and resets the camera.
            viewer_slot = st.container(key="cc_pcb_viewer_slot")
            viewer_error_slot = st.container(key="cc_pcb_viewer_error_slot")
            pending_label = workflow.pop("pending_left_status", None)
            transition_progress = None
            if pending_label:
                workflow["transition_status"] = "PREPARING"
                with workflow_status_slot.container():
                    transition_progress = st.status(pending_label, expanded=False)
        try:
            prepare_started = time.perf_counter()
            board_registry, photo_snapshot = _prepare_board_registry_snapshot(
                st, project, photos, workflow_status_slot=workflow_status_slot,
                workflow_status=transition_progress,
            )
            _record_perf(st.session_state, "snapshot_registry_prepare", prepare_started)
            snapshot_components = diagnosis_components(board_registry)
            measurement_aliases = measurement_component_aliases(board_registry)
            analysis_target_id = "board:registry:v1"
            st.session_state["circuitcare_v7_analysis_target_id"] = analysis_target_id
            component_inventory = {"__board_registry__": {"registry_version": board_registry.get("version")}}
            suspected_components = board_suspects(st.session_state, project)
            live_analysis_snapshot = current_analysis_snapshot(
                st, project, components=snapshot_components, component_inventory=component_inventory,
                analysis_target_id=analysis_target_id, component_id_aliases=measurement_aliases,
                board_registry=board_registry, suspected_components=suspected_components,
                auto_qwen=True,
            )
            workflow["analysis_fingerprint"] = live_analysis_snapshot.get("input_fingerprint")
            if workflow.get("stage") in {WORKFLOW_INSPECTION, WORKFLOW_REASSESSMENT}:
                live_result = (live_analysis_snapshot.get("result")
                               if live_analysis_snapshot.get("record", {}).get("status") == AnalysisStatus.CURRENT.value
                               else None)
                live_candidates = _inspection_candidate_rows(live_result)
                workflow["inspection_targets"] = _inspection_targets(
                    board_registry, live_candidates, suspected_components,
                    inspection_history=workflow.get("inspection_history", []),
                    measurements=get_measurements(st.session_state, project_id=project.project_id,
                                                  board_id=project.board_id or ""),
                    symptoms=(live_result.get("symptom_codes", []) if isinstance(live_result, dict) else []),
                )
                workflow["candidate_ids"] = [str(row.get("board_component_id") or row.get("component_id"))
                                              for row in live_candidates]
            preview = st.session_state.get(MODE_KEY) == "PREVIEW_ONLY"
            st.session_state[f"cc_candidate_strategy_{project.project_id}"] = (
                "candidate_fusion_v1" if isinstance(live_analysis_snapshot.get("result"), dict)
                and isinstance(live_analysis_snapshot["result"].get("candidate_fusion"), dict) else "baseline"
            )
            workflow["transition_status"] = "PREPARING" if transition_progress is not None else "READY"
            workflow.pop("transition_error", None)
        except Exception as exc:
            workflow["transition_status"] = "FAILED"
            workflow["transition_error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
            if transition_progress is not None:
                transition_progress.update(label="단계 자료 준비 실패 · 기존 검수·측정 기록 보존",
                                           state="error", expanded=True)
            with left:
                try:
                    with viewer_slot:
                        _render_pcb(st, project, photos)
                except Exception:
                    pass
            with right:
                st.error(f"PCB 진단 자료를 준비하지 못했습니다 · {type(exc).__name__}: {str(exc)[:300]}")
            return
        with left:
            active_review_scope = (photo_snapshot or {}).get("review_scope")
            active_review_status = (st.session_state.get(
                f"cc_ai_review_job_status_{active_review_scope}") if active_review_scope else None)
            if (_render_ai_review_progress is not None
                    and active_review_status in {"PENDING", "RUNNING"}):
                with qwen_progress_slot.container():
                    _render_ai_review_progress(st, project.project_id, active_review_scope)
            elif active_review_scope and active_review_status == "QUEUE_FULL":
                with qwen_progress_slot.container():
                    st.warning(st.session_state.get(
                        f"cc_ai_review_job_message_{active_review_scope}",
                        "AI 검수 대기열이 가득 찼습니다. 잠시 후 다시 실행해 주세요."))
            elif active_review_scope and active_review_status in {"COMPLETE", "FAILED", "STALE", "UNAVAILABLE", "AI_REVIEW_PARTIAL"}:
                with qwen_progress_slot.container():
                    if active_review_status == "COMPLETE":
                        st.caption("선택 소자 Qwen 분류 완료 · 제안은 사용자 확인 전까지 미확정")
                    elif active_review_status == "STALE":
                        st.info("입력 박스가 변경되어 오래된 AI 결과를 적용하지 않았습니다. 검수를 다시 실행할 수 있습니다.")
                    elif active_review_status == "AI_REVIEW_PARTIAL":
                        st.warning("AI 검수 일부 실패 · 성공한 제안과 사용자 검수 기록은 보존됩니다.")
                    elif active_review_status in {"FAILED", "UNAVAILABLE"}:
                        st.warning(st.session_state.get(
                            f"cc_ai_review_job_error_{active_review_scope}",
                            "AI 검수가 실패했습니다. ‘AI 소자 검수’를 눌러 다시 시도할 수 있습니다."))
                    else:
                        st.warning("AI 검수가 끝났지만 일부 단계가 실패했습니다. 탐지와 사용자 검수 기록은 보존됩니다.")
            elif photo_snapshot and (photo_snapshot.get("components") or photo_snapshot.get("review_scope")):
                with qwen_progress_slot.container():
                    st.caption("YOLO 탐지 완료 · 사진별 검수 화면입니다. PCB에서 박스를 선택해 Qwen 종류 분류를 실행하세요.")
            try:
                with viewer_slot:
                    _render_pcb(st, project, photos)
            except Exception as exc:
                workflow["transition_status"] = "FAILED"
                workflow["transition_error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
                with viewer_error_slot:
                    failed = st.status("PCB Viewer 준비 실패 · 기존 입력은 유지됩니다", state="error", expanded=True)
                    failed.write(f"{type(exc).__name__}: {str(exc)[:300]}")
            if transition_progress is not None:
                if workflow.get("transition_status") != "FAILED":
                    transition_progress.update(label="단계 자료와 PCB Overlay 준비 완료",
                                               state="complete", expanded=False)
                    workflow["transition_status"] = "READY"
                    workflow_status_slot.empty()
        left_progress_slot = _LeftProgressSlot(workflow_status_slot, viewer_slot)
        with right:
            if workflow.get("transition_status") == "FAILED" and workflow.get("transition_error"):
                st.error(f"단계 준비 실패 · {workflow['transition_error']} · 기존 검수·측정 기록은 보존됩니다.")
            with st.container(key="cc_workbench_brand_and_steps"):
                st.markdown('<div class="cc-wb-header">CircuitCare AI</div>', unsafe_allow_html=True)
                _render_workflow_stepper(st, workflow)
            stage = workflow.get("stage", WORKFLOW_REVIEW)
            if stage == WORKFLOW_REVIEW:
                active = [row for row in board_registry.get("components", [])
                          if isinstance(row, dict) and row.get("presence_status") != "EXCLUDED"
                          and row.get("source_components")]
                needs_type_confirmation = [row for row in active
                    if str(row.get("component_type") or "unknown") != "unknown"
                    and row.get("presence_status") != "CONFIRMED"]
                if needs_type_confirmation:
                    st.warning(
                        f"종류가 지정된 소자 {len(needs_type_confirmation)}개가 아직 사용자 확인 상태가 아닙니다. "
                        "각 소자의 검수 상태를 ‘확인’으로 저장한 뒤 진행해 주세요."
                    )
                _render_component_inspector(st, project, preview, compact=True, board_registry=board_registry)
                if len(active) == 0:
                    st.warning("유효한 소자 관측이 없습니다. 사진을 다시 탐지하거나 직접 박스를 추가해 주세요.")
                if st.button("검수 완료 · 고장 질문으로 진행", type="primary",
                             disabled=preview or len(active) == 0 or bool(needs_type_confirmation),
                             key=f"cc_wf_review_next_{project.project_id}"):
                    for state_key in list(st.session_state.keys()):
                        prefix = f"cc_component_review_rows_{project.project_id}_"
                        if not str(state_key).startswith(prefix):
                            continue
                        scope = str(state_key)[len("cc_component_review_rows_"):]
                        rows = st.session_state.get(state_key, [])
                        changed = False
                        for review_row in rows:
                            if (review_row.get("소자 종류", "unknown") == "unknown"
                                    and review_row.get("상태") != "제외"
                                    and review_row.get("component_type_status") != "USER_CONFIRMED_EXTENSION"):
                                record_box_exclusion(st.session_state, scope=scope, row=review_row,
                                                     reason="unconfirmed_component_not_assigned")
                                changed = True
                        if changed:
                            st.session_state[state_key] = rows
                            invalidate_component_review_state(st.session_state, scope=scope,
                                                              project_id=project.project_id,
                                                              board_id=project.board_id)
                    workflow["stage_completion"][WORKFLOW_REVIEW] = True
                    _advance_workflow(st, project, WORKFLOW_QUESTIONS,
                                      "고장 질문 준비 중 · 검수 결과와 PCB 사진 재사용", "review_complete")
            elif stage == WORKFLOW_QUESTIONS:
                st.markdown("#### 고장 증상 질문")
                st.caption("자연어 증상과 안전한 확인 질문을 기록합니다. 질문은 최대 5개이며 ‘모르겠음’은 근거로 계산하지 않습니다.")
                if st.button("이전 단계", key=f"cc_wf_questions_back_{project.project_id}"):
                    _advance_workflow(st, project, WORKFLOW_REVIEW,
                                      "소자 검수 화면 준비 중 · 기존 검수 기록 재사용", "back")
                render_symptom_diagnosis_panel(
                    st, project, components=snapshot_components, preview=preview,
                    component_inventory=component_inventory, analysis_target_id=analysis_target_id,
                    component_id_aliases=measurement_aliases, board_registry=board_registry,
                    suspected_components=suspected_components, simple_ui=True,
                    workflow_stage=WORKFLOW_QUESTIONS, left_progress_slot=left_progress_slot,
                )
                symptom_state = st.session_state.get(f"cc_symptom_diagnosis_{project.project_id}", {})
                if isinstance(symptom_state, dict):
                    workflow["question_answers"] = list(symptom_state.get("question_answers", {}).values())
                    workflow["question_plan"] = symptom_state.get("question_plan")
                    workflow["symptom_scope"] = live_analysis_snapshot.get("scope")
                question_snapshot = _current_board_snapshot(
                    st, project, snapshot_components, board_registry, measurement_aliases,
                    suspected_components,
                )
                question_scope = question_snapshot.get("scope") or AnalysisScope(
                    project.project_id, project.board_id or "", analysis_target_id or "board:registry:v1")
                has_current = (question_snapshot.get("record", {}).get("status") == AnalysisStatus.CURRENT.value
                               and isinstance(question_snapshot.get("result"), dict))
                symptom_state = st.session_state.get(f"cc_symptom_diagnosis_{project.project_id}", {})
                auto_advance = (symptom_state.get("workflow_auto_advance")
                                if isinstance(symptom_state, dict) else None)
                question_fingerprints = question_snapshot.get("fingerprints")
                question_fingerprint = getattr(question_fingerprints, "final_sha256", None)
                auto_advance_matches = (
                    isinstance(auto_advance, dict)
                    and auto_advance.get("scope_key") == question_scope.key
                    and auto_advance.get("fingerprint") == question_fingerprint
                )
                question_status = question_snapshot.get("record", {}).get("status", "NOT_RUN")
                if auto_advance_matches and has_current:
                    symptom_state.pop("workflow_auto_advance", None)
                    st.session_state[f"cc_symptom_diagnosis_{project.project_id}"] = symptom_state
                    LOGGER.info(
                        "workflow_auto_advance project=%s from=QUESTIONS to=INSPECTION "
                        "scope=%s fingerprint=%s result_status=%s",
                        project.project_id, question_scope.key,
                        str(question_fingerprint or "")[:16],
                        question_snapshot["result"].get("status", "UNKNOWN"),
                    )
                    _advance_workflow(
                        st, project, WORKFLOW_INSPECTION,
                        "고장 검사 후보 준비 중 · 현재 결과와 Registry 확인",
                        "diagnosis_complete",
                    )
                elif (isinstance(auto_advance, dict)
                      and question_status in {AnalysisStatus.FAILED.value, AnalysisStatus.STALE.value}):
                    symptom_state.pop("workflow_auto_advance", None)
                    st.session_state[f"cc_symptom_diagnosis_{project.project_id}"] = symptom_state
                    LOGGER.warning(
                        "workflow_auto_advance_blocked project=%s status=%s scope_match=%s "
                        "fingerprint_match=%s reason=%s",
                        project.project_id, question_status,
                        auto_advance.get("scope_key") == question_scope.key,
                        auto_advance.get("fingerprint") == question_fingerprint,
                        question_snapshot.get("record", {}).get("reason"),
                    )
                submitted_fingerprint = (symptom_state.get("last_analysis_attempt_fingerprint")
                                         if isinstance(symptom_state, dict) else None)
                report_submitted = bool(question_fingerprint and submitted_fingerprint == question_fingerprint)
                LOGGER.info(
                    "workflow_inspection_gate project=%s stage=QUESTIONS diagnosis_status=%s "
                    "has_current=%s submitted_fingerprint_match=%s auto_advance_pending=%s "
                    "scope=%s fingerprint=%s",
                    project.project_id, question_status, has_current, report_submitted,
                    auto_advance_matches, question_scope.key,
                    str(question_fingerprint or "")[:16],
                )
                if st.button("검사 후보 선정", type="primary",
                             disabled=preview or not (has_current or report_submitted),
                             key=f"cc_wf_questions_next_{project.project_id}"):
                    workflow["stage_completion"][WORKFLOW_QUESTIONS] = True
                    _advance_workflow(st, project, WORKFLOW_INSPECTION,
                                      ("고장 검사 후보 준비 중 · 현재 결과와 Registry 확인" if has_current
                                       else "확정 진단 없음 · 안전한 일반 검사 후보 준비 중"),
                                      "questions_complete")
            elif stage == WORKFLOW_INSPECTION:
                st.markdown("#### 고장 검사 후보 및 측정")
                if st.button("이전 단계", key=f"cc_wf_inspection_back_{project.project_id}"):
                    _advance_workflow(st, project, WORKFLOW_QUESTIONS,
                                      "증상 답변 불러오는 중 · 기존 입력 보존", "back")
                _render_workflow_candidates(
                    st, project, workflow, registry=board_registry,
                    analysis_snapshot=_current_board_snapshot(
                        st, project, snapshot_components, board_registry,
                        measurement_aliases, suspected_components,
                    ), components=snapshot_components, preview=preview,
                )
            elif stage == WORKFLOW_REASSESSMENT:
                st.markdown("#### 측정 결과 기반 재판단")
                latest = _render_workflow_reassessment(
                    st, project, workflow, registry=board_registry,
                    analysis_snapshot=live_analysis_snapshot, components=snapshot_components,
                    aliases=measurement_aliases, suspected=suspected_components,
                    preview=preview, left_progress_slot=left_progress_slot,
                )
                current = (latest.get("record", {}).get("status") == AnalysisStatus.CURRENT.value
                           and isinstance(latest.get("result"), dict))
                action_cols = st.columns(2)
                if action_cols[0].button("추가 검사 진행", type="primary", disabled=preview or not current,
                                         key=f"cc_wf_more_tests_{project.project_id}"):
                    workflow["stage_completion"][WORKFLOW_REASSESSMENT] = True
                    _advance_workflow(st, project, WORKFLOW_INSPECTION,
                                      "다음 검사 준비 중 · 기존 측정 결과와 판정 재사용", "additional_test")
                if action_cols[1].button("검사 종료 및 결과 보고서", disabled=preview or not current,
                                         key=f"cc_wf_complete_{project.project_id}"):
                    workflow["stage_completion"][WORKFLOW_REASSESSMENT] = True
                    _advance_workflow(st, project, WORKFLOW_COMPLETE,
                                      "결과 보고서 준비 중 · 현재 판정과 측정 이력 확인", "complete")
            elif stage == WORKFLOW_COMPLETE:
                st.markdown("#### 진단 결과 및 보고서")
                if st.button("추가 검사로 돌아가기", key=f"cc_wf_complete_back_{project.project_id}"):
                    _advance_workflow(st, project, WORKFLOW_INSPECTION,
                                      "검사 후보 불러오는 중 · 저장된 측정 이력 유지", "additional_test")
                _render_workflow_report(st, project, board_registry, live_analysis_snapshot)


def _selected_component(st: Any) -> ComponentOverlayViewModel | None:
    current_components = tuple(st.session_state.get("circuitcare_v4_current_components", ()))
    selection = st.session_state.get(SELECTION_KEY)
    selected_id = getattr(selection, "selected_component", None)
    return next((item for item in current_components if item.component_id == selected_id), None)


def _focus_board_registry_component(
    st: Any, project: ProjectContext, registry: dict[str, Any] | None, board_component_id: str,
) -> bool:
    if not isinstance(registry, dict):
        return False
    component = measurement_view_model_for_board_component(
        registry, board_component_id=board_component_id,
        preferred_photo_id=st.session_state.get(ACTIVE_PHOTO_KEY),
    )
    if component is None:
        return False
    photo_id = component.canonical_frame_id
    st.session_state[PENDING_PHOTO_FOCUS_KEY] = photo_id
    update_component_selection(
        st.session_state, project=project, board_component_id=board_component_id,
        active_photo_id=photo_id, selected_component=component.component_id,
    )
    return True


def _handle_measurement_target_change(
    state: Any, project: ProjectContext, registry: dict[str, Any] | None, widget_key: str,
) -> None:
    """Change the measurement target only; Viewer selection is independent."""
    board_component_id = state.get(widget_key)
    rows = registry.get("components", []) if isinstance(registry, dict) else []
    row = next((item for item in rows
                if item.get("board_component_id") == board_component_id), None)
    if not row or row.get("presence_status") == "EXCLUDED":
        return
    resolved = measurement_view_model_for_board_component(
        registry, board_component_id=str(board_component_id),
        preferred_photo_id=state.get(ACTIVE_PHOTO_KEY),
    )
    if resolved is None:
        return
    state[MEASUREMENT_TARGET_BOARD_COMPONENT_KEY] = str(board_component_id)


def _render_component_inspector(st: Any, project: ProjectContext, preview: bool, *, compact: bool = False,
                                board_registry: dict[str, Any] | None = None) -> None:
    current_components = tuple(st.session_state.get("circuitcare_v4_current_components", ()))
    active_registry = board_registry or st.session_state.get("circuitcare_v7_active_board_registry")
    render_inspection_panel(
        st, project, preview=preview, components=current_components,
        selected_component=_selected_component(st),
        detection_status=st.session_state.get("circuitcare_v4_detection_status", "UNKNOWN"),
        selected_marking_context=st.session_state.get("circuitcare_v4_selected_marking_context"),
        component_map_version=st.session_state.get(MAP_VERSION_KEY),
        inventory_scope=st.session_state.get("circuitcare_v4_component_inventory_scope"),
        review_image_bytes=st.session_state.get("cc_active_component_review_image"),
        review_scope=st.session_state.get("cc_active_component_review_scope"),
        board_registry=active_registry,
        compact=compact,
        on_focus_component=lambda board_id: _focus_board_registry_component(
            st, project, active_registry, board_id),
    )
    if compact:
        active_review_scope = st.session_state.get("cc_active_component_review_scope")
        if active_review_scope:
            _ocr_key, qwen_key = review_aux_keys(active_review_scope)
            qwen_review = st.session_state.get(qwen_key)
            review_status = ("일부 확인 필요" if not isinstance(qwen_review, dict)
                             or qwen_review.get("verification_status") == "AI_REVIEW_FAILED"
                             else "완료 · 제안은 미확인")
            st.caption(f"소자 AI 분석 · {review_status}")
            if st.button("소자 AI 검토 다시 실행", key=f"cc_ai_review_retry_{active_review_scope}"):
                st.session_state.pop(qwen_key, None)
                st.session_state.pop(f"cc_ai_review_run_{active_review_scope}", None)
                st.rerun()
        return
    review_image = st.session_state.get("cc_active_component_review_image")
    review_scope = st.session_state.get("cc_active_component_review_scope")
    summary = st.session_state.get("circuitcare_v4_detection_summary", {})
    if review_image is not None and review_scope and review_scope.startswith(f"{project.project_id}_"):
        photo_id = st.session_state.get("circuitcare_v2_photo_choice")
        detection_count = summary.get("count")
        suffix = f" · {detection_count}개" if detection_count is not None else ""
        with st.expander(f"소자 탐지 · 박스 검수{suffix}", expanded=False):
            if summary.get("error"):
                st.error(f"자동 탐지 실패 · {summary['error']}")
            elif detection_count == 0:
                st.info("이 사진에서 탐지 박스를 찾지 못했습니다.")
            elif detection_count is None:
                st.caption(f"검출 상태 · {st.session_state.get('circuitcare_v4_detection_status', '미확인')}")
            st.caption(f"정합 상태 · {summary.get('registration', '정합 결과 없음')}")
            render_component_recognition_panel(
                st, image_bytes=review_image, project_id=project.project_id,
                board_id=project.board_id, photo_id=photo_id,
                auto_result=st.session_state.get("circuitcare_v4_auto_detection_result"),
                auto_error=st.session_state.get("circuitcare_v4_auto_detection_error"),
            )
    _render_board_identity_controls(st, project)


def _render_board_identity_controls(st: Any, project: ProjectContext) -> None:
    registry = st.session_state.get("circuitcare_v7_active_board_registry")
    if not isinstance(registry, dict):
        return
    inactive_links = registry.get("inactive_links", [])
    sources = [source for row in registry.get("components", [])
               for source in row.get("source_components", [])]
    can_merge = (len(registry.get("source_to_board_id", {})) >= 2
                 and len({source.get("photo_id") for source in sources}) >= 2)
    if not can_merge and not inactive_links:
        return
    labels = {source["source_key"]: (
        f"{source['photo_id']} · {source['component_id']} · "
        f"{source.get('component_type', 'unknown')} · {source.get('review_status', 'UNCONFIRMED')}"
    ) for source in sources}
    entry_owner = json.dumps([project.project_id, project.board_id or ""], separators=(",", ":"))
    entry = st.session_state.get(REGISTRY_KEY, {}).get(entry_owner, {})
    links = entry.get("confirmed_links", [])
    with st.expander("여러 사진에서 같은 소자 확인", expanded=False):
        st.caption("사진 간 자동 위치 병합은 하지 않습니다. 같은 소자임을 직접 확인한 관측만 연결됩니다.")
        if can_merge:
            left = st.selectbox("사진 관측 A", list(labels), format_func=labels.get,
                                key=f"cc_registry_left_{project.project_id}")
            right = st.selectbox("사진 관측 B", list(labels), format_func=labels.get,
                                 key=f"cc_registry_right_{project.project_id}")
            if st.button("같은 보드 소자로 연결", key=f"cc_registry_merge_{project.project_id}",
                         disabled=left == right or labels[left].split(" · ")[0] == labels[right].split(" · ")[0]):
                try:
                    confirm_component_match(st.session_state, project_id=project.project_id,
                                            board_id=project.board_id, left_source_key=left,
                                            right_source_key=right)
                    st.rerun()
                except ValueError as exc:
                    st.warning(str(exc))
            if links:
                st.caption("사용자 확인으로 연결한 관계")
                for index, link in enumerate(links):
                    if len(link) != 2:
                        continue
                    a, b = link
                    st.caption(f"{labels.get(a, a)} ↔ {labels.get(b, b)}")
                    if st.button("연결 해제", key=f"cc_registry_unlink_{project.project_id}_{index}"):
                        remove_component_match(st.session_state, project_id=project.project_id,
                                               board_id=project.board_id, left_source_key=a,
                                               right_source_key=b)
                        st.rerun()
        for item in inactive_links:
            left_key, right_key = item.get("source_keys", ["", ""])
            reason = str(item.get("reason", "RECONFIRMATION_REQUIRED"))
            reason_labels = {
                "PHOTO_OR_COMPONENT_MISSING": "사진 또는 소자 없음",
                "RECONFIRMATION_REQUIRED": "확인 이력 없음",
                "PHOTO_REPLACED_RECONFIRMATION_REQUIRED": "사진 교체됨",
                "PHOTO_REDETECTED_RECONFIRMATION_REQUIRED": "소자 재탐지·검수 변경됨",
                "PHOTO_COMPONENT_MAP_CHANGED_RECONFIRMATION_REQUIRED": "사진별 소자 목록 변경됨",
                "SAME_PHOTO_COMPONENT_COLLISION_RECONFIRMATION_REQUIRED": "같은 사진의 서로 다른 소자 포함",
            }
            st.caption(f"비활성 연결 · {labels.get(left_key, left_key)} ↔ "
                       f"{labels.get(right_key, right_key)} · "
                       f"{reason_labels.get(reason, reason)} · 다시 선택해 확인하세요.")


def _render_fault_analysis_workspace(
    st: Any, project: ProjectContext, preview: bool, *,
    mode: str = "증상",
    analysis_snapshot: dict[str, Any] | None = None,
    photo_snapshot: dict[str, Any] | None = None,
    board_registry: dict[str, Any] | None = None,
    board_components: tuple[Any, ...] | None = None,
    component_id_aliases: dict[str, str | None] | None = None,
    analysis_target_id: str | None = None,
    suspected_components: list[dict[str, Any]] | None = None,
) -> None:
    if photo_snapshot is None:
        target_id = str(st.session_state.get(ACTIVE_PHOTO_KEY) or "target:default")
        scope = AnalysisScope(project.project_id, project.board_id or "", target_id)
        snapshots = st.session_state.get("circuitcare_v6_photo_component_snapshots", {})
        photo_snapshot = snapshots.get(scope.key) if isinstance(snapshots, dict) else None
        if not isinstance(photo_snapshot, dict):
            photo_snapshot = {"components": tuple(st.session_state.get("circuitcare_v4_current_components", ())),
                              "review_scope": st.session_state.get("circuitcare_v4_component_inventory_scope")}
    if board_components is None and isinstance(board_registry, dict):
        board_components = diagnosis_components(board_registry)
    if board_components is not None:
        components = tuple(board_components)
    else:
        components = tuple(photo_snapshot.get("components", ()))
    if analysis_snapshot is None:
        inventory = ({"__board_registry__": {"registry_version": board_registry.get("version")}}
                     if isinstance(board_registry, dict) else {})
        analysis_snapshot = current_analysis_snapshot(
            st, project, components=components,
            component_inventory=inventory,
            analysis_target_id=analysis_target_id, component_id_aliases=component_id_aliases,
            board_registry=board_registry, suspected_components=suspected_components,
            auto_qwen=True,
        )
    selection = st.session_state.get(SELECTION_KEY)
    selected = _selected_component(st)
    review_scope = st.session_state.get("circuitcare_v4_component_inventory_scope")
    owner = f"{project.project_id}|{project.board_id or ''}|{review_scope or st.session_state.get(MAP_VERSION_KEY) or ''}"
    inventories = st.session_state.get(COMPONENT_INVENTORY_KEY, {})
    inventory = inventories.get(owner, {}) if isinstance(inventories, dict) else {}
    result = analysis_snapshot.get("result")
    labels = {"증상": "AI 진단", "원인": "AI 진단", "측정": "측정 확인", "통합": "AI 진단"}
    if mode == "통합":
        render_symptom_diagnosis_panel(
            st, project, components=components, preview=preview,
            component_inventory=({"__board_registry__": {"registry_version": board_registry.get("version")}}
                                 if isinstance(board_registry, dict) else inventory),
            analysis_target_id=analysis_target_id, component_id_aliases=component_id_aliases,
            board_registry=board_registry, suspected_components=suspected_components,
            simple_ui=True,
        )
        current = current_analysis_snapshot(
            st, project, components=components,
            component_inventory=({"__board_registry__": {"registry_version": board_registry.get("version")}}
                                 if isinstance(board_registry, dict) else inventory),
            analysis_target_id=analysis_target_id, component_id_aliases=component_id_aliases,
            board_registry=board_registry, suspected_components=suspected_components,
            auto_qwen=True,
        )
        current_result = current.get("result") if current.get("record", {}).get("status") == AnalysisStatus.CURRENT.value else None
        if not isinstance(current_result, dict):
            return
        strategy = "candidate_fusion_v1" if isinstance(current_result.get("candidate_fusion"), dict) else "baseline"
        candidates = _simple_candidate_rows(current_result)
        candidate_id = None
        selected_candidate = None
        fault_id = None
        recommended_test_id = None
        if candidates:
            candidate_ids = [str(row.get("board_component_id") or row.get("component_id")) for row in candidates]
            candidate_labels = {
                str(row.get("board_component_id") or row.get("component_id")): (
                    f"{row.get('candidate_rank', index + 1)}순위 · {row.get('component_type', '소자')} · "
                    f"{row.get('fault_hypotheses', [{}])[0].get('fault_title') or '원인 가설'} · "
                    f"{row.get('board_component_id') or row.get('component_id')}")
                for index, row in enumerate(candidates)
            }
            candidate_widget_key = f"cc_measure_candidate_{project.project_id}"
            if st.session_state.get(candidate_widget_key) not in candidate_ids:
                st.session_state[candidate_widget_key] = candidate_ids[0]
            selected_candidate_id = st.selectbox(
                "검사할 후보", candidate_ids, key=candidate_widget_key,
                format_func=lambda value: candidate_labels.get(value, value),
            )
            candidate_id = selected_candidate_id if selected_candidate_id in candidate_ids else candidate_ids[0]
            st.session_state[MEASUREMENT_TARGET_BOARD_COMPONENT_KEY] = candidate_id
            selected_candidate = next(row for row in candidates
                                      if str(row.get("board_component_id") or row.get("component_id")) == candidate_id)
            hypotheses = [row for row in selected_candidate.get("fault_hypotheses", []) if row.get("fault_id")]
            if hypotheses:
                hypothesis_ids = list(dict.fromkeys(str(row["fault_id"]) for row in hypotheses))
                fault_id = hypothesis_ids[0]
                if len(hypothesis_ids) > 1:
                    hypothesis_widget_key = f"cc_measure_fault_{project.project_id}_{candidate_id}"
                    if st.session_state.get(hypothesis_widget_key) not in hypothesis_ids:
                        st.session_state[hypothesis_widget_key] = hypothesis_ids[0]
                    fault_id = st.selectbox(
                        "검증할 고장 가설", hypothesis_ids,
                        format_func=lambda value: next((row.get("fault_title") or value for row in hypotheses
                                                        if row.get("fault_id") == value), value),
                        key=hypothesis_widget_key,
                    )
                hypothesis = next((row for row in hypotheses if row.get("fault_id") == fault_id), hypotheses[0])
            else:
                hypothesis = {}
            if isinstance(board_registry, dict):
                registry_component = next((row for row in board_registry.get("components", [])
                                           if row.get("board_component_id") == candidate_id), {})
                component_type = str(registry_component.get("component_type") or selected_candidate.get("component_type") or "unknown")
            else:
                component_type = str(selected_candidate.get("component_type") or "unknown")
            tests = load_knowledge().get("component_test_profiles", {}).get("profiles", {}).get(
                component_type.casefold(), {}).get("tests", {})
            recommended_test_id = next((test_id for test_id, profile in tests.items()
                                        if fault_id in profile.get("fault_relations", {})), None)
            if not recommended_test_id and hypothesis.get("recommended_tests"):
                st.caption("권장 확인 · " + " · ".join(str(item) for item in hypothesis["recommended_tests"][:2]))
        measurement_component = None
        if candidate_id and isinstance(board_registry, dict):
            measurement_component = measurement_view_model_for_board_component(
                board_registry, board_component_id=candidate_id,
                preferred_photo_id=st.session_state.get(ACTIVE_PHOTO_KEY),
            )
        if (measurement_component is None and not isinstance(board_registry, dict)
                and candidate_id):
            measurement_component = next((item for item in components
                if isinstance(item, ComponentOverlayViewModel)
                and (item.board_component_id == candidate_id or item.component_id == candidate_id)), None)
        # Top candidate selection changes the measurement target only. The Viewer
        # remains on the user's clicked component until they explicitly request
        # a location jump.
        if candidate_id and measurement_component is not None and st.button(
            "PCB에서 위치 보기", key=f"cc_measure_candidate_focus_{project.project_id}_{candidate_id}",
        ):
            if isinstance(board_registry, dict):
                _focus_board_registry_component(st, project, board_registry, candidate_id)
            else:
                update_selection(st.session_state, project=project,
                                 active_photo_id=measurement_component.canonical_frame_id,
                                 selected_component=measurement_component.component_id)
                st.session_state[ACTIVE_PHOTO_KEY] = measurement_component.canonical_frame_id
            st.rerun()
        if candidate_id and measurement_component is None:
            st.warning("소자 위치 확인 필요 · 선택한 후보를 현재 사진의 유효한 Registry 관측에 연결할 수 없어 측정을 중단했습니다.")
        if candidates or get_measurements(st.session_state, project_id=project.project_id,
                                          board_id=project.board_id or ""):
            st.markdown("#### 다음 확인 · 측정")
            render_measurement_panel(
                st, project, selected_component=measurement_component, preview=preview,
                component_map_version=(board_registry.get("version") if isinstance(board_registry, dict)
                                       else st.session_state.get(MAP_VERSION_KEY)),
                board_registry=board_registry, compact=True,
                fault_hypothesis=fault_id, recommended_test_id=recommended_test_id,
            )
        if isinstance(board_registry, dict):
            with st.popover("진단 보고서 다운로드"):
                report_snapshot = build_pcb_diagnostic_snapshot(
                    root=Path(__file__).resolve().parents[4], project=project,
                    registry=board_registry, analysis_snapshot=current,
                    photos=st.session_state.get(INTAKE_IMAGES_KEY, []),
                    selected_board_component_id=st.session_state.get(BOARD_COMPONENT_SELECTION_KEY),
                    candidate_strategy=strategy,
                )
                render_report_panel(
                    st, snapshot=report_snapshot, root=Path(__file__).resolve().parents[4],
                    workspace_id=project.project_id, expected_circuit_hash=None,
                )
        return
    with st.container(border=True):
        st.markdown(f'<div class="cc-analysis-step">{labels.get(mode, "분석")}</div>', unsafe_allow_html=True)
        st.markdown('<div class="cc-analysis-copy">규칙 기반 결과를 기준으로 표시합니다. 후보는 측정으로 검증되기 전까지 미확정 가설입니다.</div>', unsafe_allow_html=True)
    record = (analysis_snapshot or {}).get("record", {})
    state_status = str(record.get("status", "NOT_RUN"))
    state_labels = {"NOT_RUN": "분석 전", "RUNNING": "분석 중", "CURRENT": "현재 입력 기준 최신",
                    "STALE": "입력 변경 · 재분석 필요", "FAILED": "분석 실패"}
    st.caption(f"분석 상태 · {state_labels.get(state_status, state_status)}")
    if state_status == "STALE" and record.get("reason"):
        st.warning(f"기존 결과는 현재 입력과 달라 사용하지 않습니다 · {record['reason']}")
    if mode == "증상":
        render_symptom_diagnosis_panel(
            st, project, components=components, preview=preview,
            component_inventory=({"__board_registry__": {"registry_version": board_registry.get("version")}}
                                 if isinstance(board_registry, dict) else inventory),
            analysis_target_id=analysis_target_id, component_id_aliases=component_id_aliases,
            board_registry=board_registry,
            suspected_components=suspected_components,
        )
    elif mode == "원인":
        _render_cause_dashboard(st, project, result, analysis_snapshot=analysis_snapshot,
                                board_registry=board_registry)
    elif mode == "측정":
        st.markdown("#### 측정 기록과 가설 검증")
        if isinstance(board_registry, dict):
            active_targets = [row for row in board_registry.get("components", [])
                              if row.get("presence_status") != "EXCLUDED"
                              and measurement_view_model_for_board_component(
                                  board_registry,
                                  board_component_id=str(row.get("board_component_id")),
                                  preferred_photo_id=st.session_state.get(ACTIVE_PHOTO_KEY),
                              ) is not None]
            component_ids = [str(row.get("board_component_id")) for row in active_targets]
            option_labels = {str(row.get("board_component_id")):
                             f"{row.get('component_type', 'unknown')} · {row.get('board_component_id')}"
                             for row in active_targets}
            viewer_board_id = st.session_state.get(BOARD_COMPONENT_SELECTION_KEY)
            selected_id = st.session_state.get(MEASUREMENT_TARGET_BOARD_COMPONENT_KEY)
            if selected_id not in component_ids:
                selected_id = viewer_board_id if viewer_board_id in component_ids else (component_ids[0] if component_ids else None)
                if selected_id:
                    st.session_state[MEASUREMENT_TARGET_BOARD_COMPONENT_KEY] = selected_id
            target_key = f"cc_measurement_target_{project.project_id}"
            if component_ids and st.session_state.get(target_key) not in component_ids:
                st.session_state[target_key] = selected_id
            if component_ids:
                st.selectbox(
                    "측정 대상 소자", component_ids, format_func=option_labels.get,
                    key=target_key, on_change=_handle_measurement_target_change,
                    kwargs={"state": st.session_state, "project": project,
                            "registry": board_registry, "widget_key": target_key},
                )
                selected_id = st.session_state.get(target_key)
                st.session_state[MEASUREMENT_TARGET_BOARD_COMPONENT_KEY] = selected_id
                measurement_component = measurement_view_model_for_board_component(
                    board_registry, board_component_id=str(selected_id),
                    preferred_photo_id=st.session_state.get(ACTIVE_PHOTO_KEY),
                )
                if measurement_component and st.button(
                    "PCB에서 위치 보기", key=f"cc_measurement_target_focus_{project.project_id}_{selected_id}",
                ):
                    _focus_board_registry_component(st, project, board_registry, str(selected_id))
                    st.rerun()
            else:
                measurement_component = None
        else:
            component_ids = [item.board_component_id or item.component_id for item in components
                             if item.presence_status != "EXCLUDED"]
            selected_id = st.session_state.get(MEASUREMENT_TARGET_BOARD_COMPONENT_KEY)
            if selected_id not in component_ids:
                selected_id = component_ids[0] if component_ids else None
                if selected_id:
                    st.session_state[MEASUREMENT_TARGET_BOARD_COMPONENT_KEY] = selected_id
            measurement_component = next((item for item in components
                                          if (item.board_component_id or item.component_id) == selected_id), None)
        render_measurement_panel(
            st, project, selected_component=measurement_component, preview=preview,
            component_map_version=(board_registry.get("version") if isinstance(board_registry, dict)
                                   else st.session_state.get(MAP_VERSION_KEY)),
            board_registry=board_registry,
        )
        measurements = get_measurements(
            st.session_state, project_id=project.project_id, board_id=project.board_id or "",
        )
        with st.expander("회로 모델 기반 진단 · 고급", expanded=False):
            render_diagnosis_panel(
                st, project, measurements=measurements,
                analysis_context_id=getattr(selection, "analysis_context_id", None),
                component_map_version=st.session_state.get(MAP_VERSION_KEY),
            )

        st.caption("PCB Viewer 선택과 측정 대상은 별도로 유지됩니다. ‘PCB에서 위치 보기’로 선택한 측정을 강조할 수 있습니다.")


def _render_cause_dashboard(
    st: Any, project: ProjectContext, result: Any, *,
    analysis_snapshot: dict[str, Any] | None = None,
    board_registry: dict[str, Any] | None = None,
) -> None:
    """Compact, evidence-aware view of the existing symptom diagnosis result."""
    if not isinstance(result, dict):
        reason = (analysis_snapshot or {}).get("record", {}).get("reason")
        if reason:
            st.warning(f"현재 입력에 대한 분석 결과가 없습니다 · {reason} · 증상 입력 탭에서 다시 분석하세요.")
        else:
            st.info("증상 입력 탭에서 증상을 기록하고 원인 분석을 실행하세요.")
        return
    strategy_key = f"cc_candidate_strategy_{project.project_id}"
    strategy_options = ["baseline", "evidence_v1"]
    if isinstance(result.get("candidate_fusion"), dict):
        strategy_options.append("candidate_fusion_v1")
    strategy_labels = {"baseline": "Baseline 규칙 순위", "evidence_v1": "evidence_v1 실험 순위",
                       "candidate_fusion_v1": "Qwen 통합 실험 순위"}
    current_strategy = st.session_state.get(strategy_key, "baseline")
    if current_strategy not in strategy_options:
        current_strategy = "baseline"
    st.selectbox("후보 표시 전략", strategy_options, key=strategy_key,
                 index=strategy_options.index(current_strategy), format_func=strategy_labels.get,
                 help="증상 분석 탭과 같은 전략을 사용합니다.")
    selected_strategy = st.session_state.get(strategy_key, "baseline")
    candidates = candidate_rows_for_strategy(result, selected_strategy)
    groups = result.get("hypothesis_groups", {})
    next_tests = result.get("next_tests", [])
    next_action = next((row for row in next_tests if row.get("type") != "question"), None)
    if next_action is None:
        next_action = next(iter(next_tests), None)
    with st.container(border=True):
        st.markdown(f"**소자 검사 후보 · {strategy_labels.get(selected_strategy, selected_strategy)} · Top 3**")
        if candidates:
            status_labels = {"SUSPECTED": "검사 전 의심", "FAULT_SUPPORTED": "특정 가설 지지",
                             "FAULT_REFUTED": "특정 가설 반박", "CONFLICTED": "근거 충돌",
                             "INCONCLUSIVE": "판정 불가"}
            for index, candidate in enumerate(candidates[:3], start=1):
                component_id = str(candidate.get("board_component_id") or candidate.get("component_id") or "")
                fault_id = str(candidate.get("fault_hypothesis") or candidate.get("fault_id") or "")
                status = str(candidate.get("visual_status") or candidate.get("status") or "SUSPECTED")
                st.markdown(f"**{index}. `{component_id or '소자 미특정'}` · {candidate.get('component_type', '종류 미확정')}**")
                source_labels = candidate.get("candidate_source", [])
                source_text = ("규칙 기반" if "RULE_BASELINE" in source_labels else
                               "Qwen 미검증" if "QWEN_UNVERIFIED" in source_labels else
                               "사용자 의심 단서" if "USER_SUSPICION" in source_labels else
                               str(candidate.get("verification_status", "미확정")))
                verification = (" · 소자 확인 필요" if candidate.get("verification_status") == "NEEDS_COMPONENT_REVIEW" else "")
                st.caption(f"{candidate.get('fault_title', fault_id or '원인 후보')} · {status_labels.get(status, '미확정')} · {source_text}{verification} · 순위는 검사 우선순위이지 고장 확률이 아닙니다.")
                supporting = candidate.get("supporting_evidence") or candidate.get("evidence", [])
                if supporting:
                    st.caption("지지 · " + " · ".join(supporting[:3]))
                opposing = candidate.get("opposing_evidence", [])
                if opposing:
                    st.caption("반대 · " + " · ".join(opposing[:3]))
                if component_id and board_registry and any(
                    item.get("board_component_id") == component_id
                    for item in board_registry.get("components", [])
                ):
                    if st.button("PCB에서 위치 보기", key=candidate_focus_widget_key("analysis_dashboard", project.project_id, project.board_id,
                                                                    component_id, fault_id),
                                 type="primary" if index == 1 else "secondary", use_container_width=True):
                        _focus_board_registry_component(st, project, board_registry, component_id)
                        st.rerun()
                elif component_id and board_registry is None and any(
                    item.component_id == component_id
                    for item in st.session_state.get("circuitcare_v4_current_components", ())
                ):
                    if st.button("PCB에서 위치 보기", key=candidate_focus_widget_key("analysis_dashboard", project.project_id, project.board_id,
                                                                    component_id, fault_id),
                                 type="primary" if index == 1 else "secondary", use_container_width=True):
                        update_selection(st.session_state, project=project, selected_component=component_id)
                        st.rerun()
        else:
            st.info("확인된 소자 후보가 없습니다. 증상·검수 결과를 확인하고 기판·납땜 또는 외부 시스템 원인을 살펴보세요.")
        if selected_strategy == "candidate_fusion_v1":
            additional = result.get("candidate_fusion", {}).get("additional_qwen_candidates", [])
            if additional:
                with st.expander(f"추가 확인 후보 · Qwen 미검증 {len(additional)}개", expanded=False):
                    for candidate in additional:
                        cid = str(candidate.get("board_component_id") or candidate.get("component_id") or "")
                        st.caption(f"{cid} · {candidate.get('fault_title', candidate.get('fault_id'))} · 미검증")
                        if candidate.get("qwen_reason"):
                            st.caption(candidate["qwen_reason"])
                        if cid and st.button("PCB 위치 보기", key=candidate_focus_widget_key("analysis_dashboard_qwen_extra", project.project_id,
                                                                project.board_id, cid,
                                                                str(candidate.get("fault_hypothesis") or candidate.get("fault_id") or ""))):
                            _focus_board_registry_component(st, project, board_registry, cid)
                            st.rerun()
        if next_action:
            st.info("다음 검사 · " + str(next_action.get("label") or next_action.get("name") or next_action.get("question") or next_action.get("test_id") or "추가 확인"))

    block_groups = result.get("functional_block_groups", [])
    if block_groups:
        st.markdown("#### 기능 블록 후보 · 미확정")
        st.caption("색상은 부품 정보·각인·실크스크린·배치에 따른 역할 그룹입니다. 전기적 연결을 뜻하지 않습니다.")
        for block in block_groups[:8]:
            members = block.get("board_component_ids", block.get("component_ids", []))
            with st.expander(f"{block.get('title', block.get('role', '기능 블록'))} · {len(members)}개 소자 · 추정", expanded=False):
                if block.get("supporting_evidence"):
                    st.caption("근거 · " + " · ".join(block["supporting_evidence"][:4]))
                st.caption("소자 · " + (" · ".join(str(value) for value in members[:16]) or "없음"))
                st.caption("연결 상태 · 확인되지 않음")
                for component_id in members[:8]:
                    if st.button(f"{component_id} 위치 보기", key=candidate_focus_widget_key(
                            "functional_block", project.project_id, project.board_id,
                            str(component_id), str(block.get("block_id")))):
                        _focus_board_registry_component(st, project, board_registry, str(component_id))
                        st.rerun()

    ambiguous_groups = result.get("component_candidate_groups", [])
    if ambiguous_groups:
        with st.expander(f"구별되지 않은 소자 후보 그룹 · {len(ambiguous_groups)}개", expanded=False):
            for group in ambiguous_groups[:8]:
                st.caption(f"{group.get('fault_id')} · " + " · ".join(group.get("component_ids", [])))
                st.caption(group.get("reason", "동일한 근거 수준의 후보들입니다."))

    category_labels = (("component", "소자 내부·기능 블록"), ("board", "기판·납땜"), ("system", "외부 시스템"))
    for group_key, label in category_labels:
        rows = groups.get(group_key, []) if isinstance(groups, dict) else []
        if not rows:
            continue
        st.markdown(f"#### {label}")
        for index, row in enumerate(rows[:5], start=1):
            with st.expander(f"{index}. {row.get('title', '원인 후보')} · 미확정", expanded=index == 1):
                st.caption("지지 근거 · " + (" · ".join(row.get("supporting_evidence", [])) or "직접 근거 없음"))
                st.caption("반대 근거 · " + (" · ".join(row.get("opposing_evidence", [])) or "확인된 반대 근거 없음"))
                if row.get("related_component_ids"):
                    st.caption("관련 소자 · " + " · ".join(row["related_component_ids"]))
                st.caption("검증 상태 · 미확정 가설")

    if result.get("llm_assist") or result.get("llm_component_explanations") or result.get("llm_unverified_suggestions") or result.get("matched_rules") or result.get("functional_blocks") or result.get("measurement_evidence") or result.get("evidence_gaps"):
        with st.expander("상세 근거 · LLM 설명 · 적용 규칙 · 측정", expanded=False):
            assist = result.get("llm_assist")
            if assist:
                st.markdown("**LLM 설명 보조**")
                st.write(assist.get("summary", ""))
                st.caption(f"모델 · {assist.get('model', '확인 불가')} · 후보 확정 없음")
            elif result.get("llm_error"):
                st.warning(f"LLM 설명을 사용할 수 없습니다 · {result['llm_error']} · 규칙 기반 결과는 유지됩니다.")
            if result.get("llm_component_explanations"):
                explanations = result["llm_component_explanations"].get("explanations", {})
                for component_id, explanation in explanations.items():
                    st.caption(f"{component_id} · {explanation}")
            if result.get("matched_rules"):
                st.caption("적용 규칙 · " + " · ".join(row.get("name", "규칙") for row in result["matched_rules"]))
            if result.get("functional_blocks"):
                for block in result["functional_blocks"]:
                    st.caption(f"기능 역할 추정 · {block.get('component_id')} · {', '.join(block.get('candidate_blocks', [])) or '미분류'} · 미확정")
            if result.get("measurement_evidence"):
                for row in result["measurement_evidence"]:
                    st.caption(f"측정 근거 · {row.get('component_id')} · {row.get('measurement_type')} {row.get('value')} {row.get('unit')} · {row.get('verification_status')}")
            if result.get("evidence_gaps"):
                st.markdown("**가설별 부족한 근거와 확인 방법**")
                for gap in result["evidence_gaps"][:8]:
                    st.caption(f"{gap.get('fault_id')} · {gap.get('status')} · " +
                               (" · ".join(item.get("description", "") for item in gap.get("missing_evidence", [])[:3])
                                or "필수 근거 공백 없음; 고장은 여전히 미확정 가설입니다."))
            for suggestion in result.get("llm_unverified_suggestions", []):
                st.caption(f"미등록 원인 제안 · {suggestion.get('title')} · {suggestion.get('reason')}")
    questions = [row for row in next_tests if row.get("type") == "question"]
    if questions:
        st.markdown("#### 추가 확인 질문")
        st.caption("질문 답변과 검사 추천은 증상 분석 패널에서 입력·반영합니다.")
        for row in questions[:3]:
            st.caption("· " + row.get("question", "추가 정보 확인"))
    if analysis_snapshot is not None and isinstance(board_registry, dict):
        with st.expander("진단 보고서 · JSON / Markdown / HTML / Evidence Bundle", expanded=False):
            snapshot = build_pcb_diagnostic_snapshot(
                root=Path(__file__).resolve().parents[4], project=project,
                registry=board_registry, analysis_snapshot=analysis_snapshot,
                photos=st.session_state.get(INTAKE_IMAGES_KEY, []),
                selected_board_component_id=st.session_state.get(BOARD_COMPONENT_SELECTION_KEY),
                candidate_strategy=st.session_state.get(f"cc_candidate_strategy_{project.project_id}", "baseline"),
            )
            render_report_panel(st, snapshot=snapshot,
                                root=Path(__file__).resolve().parents[4],
                                workspace_id=project.project_id,
                                expected_circuit_hash=None)
def _render_top_cause(st: Any, project: ProjectContext, result: Any) -> None:
    selected_strategy = st.session_state.get(f"cc_candidate_strategy_{project.project_id}", "baseline")
    candidates = candidate_rows_for_strategy(result, selected_strategy) if isinstance(result, dict) else []
    next_tests = result.get("next_tests", []) if isinstance(result, dict) else []
    top = candidates[0] if candidates else None
    next_test = next((item for item in next_tests if item.get("type") != "question"), None)
    if next_test is None:
        next_test = next(iter(next_tests), None)
    with st.container(border=True):
        st.markdown("**우선 확인할 내용**")
        if not top:
            st.info("증상을 입력하고 원인 후보 분석을 실행하면 우선 후보와 다음 검사 항목이 표시됩니다.")
            return
        st.markdown(f"**{top.get('component_id', '소자 미특정')} · {top.get('component_type', '종류 미확정')}**  ")
        st.write(top.get("fault_title", "원인 후보"))
        st.caption("미확정 검사 가설 · 상대 점수는 실제 고장 확률을 뜻하지 않습니다.")
        if top.get("evidence"):
            st.caption("근거 · " + " · ".join(top["evidence"][:3]))
        if next_test:
            st.caption("다음 검사 · " + str(next_test.get("label") or next_test.get("name") or next_test.get("question") or next_test.get("test_id") or "추가 확인"))
        if st.button("PCB에서 후보 위치 보기", key=candidate_focus_widget_key("analysis_summary", project.project_id, project.board_id,
                                                        str(top.get("board_component_id") or top.get("component_id") or ""),
                                                        str(top.get("fault_hypothesis") or top.get("fault_id") or "")), use_container_width=True):
            update_selection(st.session_state, project=project, selected_component=top.get("component_id"))
            st.rerun()


def _render_schematic(st: Any, project: ProjectContext, circuit_bytes: bytes | None = None) -> None:
    source = project.circuit_source
    if not source:
        st.info("회로도 없음")
        st.caption("설계 자료 추가")
        return
    if source.get("source_type") == "kicad_schematic":
        st.info("KiCad 회로도 자료가 있습니다.")
        raw = circuit_bytes
        if not raw:
            st.warning("회로도 파일 데이터를 찾을 수 없습니다. 입력을 다시 확인하세요.")
            return
        try:
            pages = _export_schematic_pages(raw, source.get("filename") or "uploaded.kicad_sch")
        except (OSError, ValueError) as exc:
            st.warning(f"회로도 Viewer를 열 수 없습니다 · {exc}")
            return
        if not pages:
            st.warning("KiCad SVG Export 결과가 없습니다.")
            return
        page_names = [page[0] for page in pages]
        page_name = st.selectbox("회로도 페이지", page_names, key="circuitcare_v2_schematic_page") if len(pages) > 1 else page_names[0]
        _, svg, view_box = next(page for page in pages if page[0] == page_name)
        context_key = hashlib.sha256(raw).hexdigest()
        rendered = render_interactive_viewer(
            st, svg=svg, asset_key=f"f2-{context_key}-{page_name}", view_box=view_box,
            geometry_status="UNVERIFIED_GEOMETRY", regions=[], overlay=None,
            selected_ref="", navigation_id=0, context_key=context_key,
            event_sequence_seed=0, key=f"cc-v2-schematic-{context_key[:12]}",
            on_selection_change=lambda *_: None,
        )
        if rendered is None:
            st.html(svg, width="stretch")
        st.caption("KiCad 원본 SVG · PCB 정합 및 진단 미실행")
    elif source.get("source_type") == "kicad_xml_netlist":
        st.info("회로도 레이아웃 없음")
        st.caption("XML Netlist는 연결 데이터이며 원본 회로도 배치를 포함하지 않습니다.")
    else:
        st.info("지원되지 않는 설계 자료")


def _export_schematic_pages(raw: bytes, filename: str) -> list[tuple[str, str, tuple[float, float, float, float]]]:
    """Reuse the bounded existing KiCad SVG exporter inside an ephemeral project root."""
    if not raw or len(raw) > 20_000_000:
        raise ValueError("회로도 파일 크기를 확인해 주세요.")
    project_root = Path(__file__).resolve().parents[4]
    with tempfile.TemporaryDirectory(prefix="cc-f2-schematic-", dir=project_root) as temporary:
        temp_root = Path(temporary)
        source_path = temp_root / Path(filename).name
        if source_path.suffix != ".kicad_sch":
            source_path = source_path.with_suffix(".kicad_sch")
        source_path.write_bytes(raw)
        result = export_schematic(
            source_path, project_root=temp_root, cache_root=temp_root / "svg-cache",
        )
        if result.status != "EXPORTED":
            raise ValueError(result.message)
        pages = []
        for svg_path in result.svg_paths:
            view_box = read_svg_view_box(svg_path)
            if view_box is None:
                continue
            pages.append((svg_path.name, svg_path.read_text(encoding="utf-8"), view_box))
        return pages
