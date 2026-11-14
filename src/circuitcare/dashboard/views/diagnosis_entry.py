"""F1 diagnosis intake screen for local PCB photos and symptom context."""

from __future__ import annotations

import hashlib
import inspect
from typing import Any

from circuitcare.frontend.intake import (
    CAPTURE_TYPES,
    MAX_PHOTOS,
    MAX_SYMPTOM_CHARS,
    PHOTO_SIDES,
    SYMPTOM_CATEGORIES,
    add_image_batch,
    create_project_context,
    normalize_symptom_description,
    validate_circuit_upload,
)
from circuitcare.frontend.photo_adapter import normalize_photo_metadata
from circuitcare.frontend.component_map_adapter import clear_component_map_state
from circuitcare.frontend.state import (
    INTAKE_IMAGES_KEY,
    INTAKE_PROJECT_KEY,
    INTAKE_UPLOAD_BATCH_KEY,
    clear_intake_draft,
    draft_is_current,
)
from circuitcare.frontend.workbench_state import SCREEN_KEY, enter_workbench
from circuitcare.vision.image_loader import load_board_image


def _style(st: Any) -> None:
    st.markdown(
        """<style>
        /* Keep the first line below Streamlit's toolbar/header.  The title was
           rendered beneath that reserved area on Safari, clipping its top. */
        [data-testid="stMainBlockContainer"] { max-width: 940px; padding-top: 2rem; overflow:visible; }
        .cc-entry-brand { text-align:center; font-size:1.05rem; font-weight:700; color:#e8eef5; }
        .cc-entry-title { text-align:center; font-size:1.65rem; font-weight:700; color:#f4f7fa; margin:.5rem 0 1.5rem; }
        .cc-entry-section { margin-top:1.2rem; color:#e4ebf2; font-weight:650; }
        .cc-entry-muted { color:#9aa9b8; font-size:.82rem; }
        [data-testid="stFileUploader"] { border:1px dashed #546474; border-radius:8px; padding:.55rem; }
        @media (max-width:700px) {
          [data-testid="stMainBlockContainer"] { padding:1.5rem .7rem .75rem; }
          .cc-entry-title { font-size:1.35rem; }
        }
        </style>""",
        unsafe_allow_html=True,
    )


def _upload_limit_kwargs(st: Any) -> dict[str, int]:
    return {"max_upload_size": 20} if "max_upload_size" in inspect.signature(st.file_uploader).parameters else {}


def _current_fingerprint(state: Any, category: str, description: str | None) -> str | None:
    images = state.get(INTAKE_IMAGES_KEY, [])
    if not images:
        return None
    try:
        return create_project_context(
            images=images, symptom_category=category, symptom_description=description,
            circuit_source=state.get("circuitcare_v2_circuit_source"),
        ).input_fingerprint
    except ValueError:
        return None


def _normalize_photo_state(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Migrate old session roles while preserving bytes, IDs, and source hashes."""
    normalized_images = []
    for existing in images:
        normalized, _ = normalize_photo_metadata(existing)
        item = dict(existing)
        item.update({
            "image_id": normalized.image_id,
            "side": normalized.side,
            "capture_type": normalized.capture_type,
            "sha256": normalized.content_sha256,
            "image_dimensions": normalized.image_dimensions,
            "image_format": normalized.image_format,
        })
        item.pop("role", None)
        normalized_images.append(item)
    return normalized_images


def render_diagnosis_entry(st: Any) -> None:
    """Render a photo-first intake with symptoms deferred to the AI flow."""
    _style(st)
    state = st.session_state
    images = _normalize_photo_state(state.setdefault(INTAKE_IMAGES_KEY, []))
    state[INTAKE_IMAGES_KEY] = images
    state.setdefault(INTAKE_UPLOAD_BATCH_KEY, 0)
    project = state.get(INTAKE_PROJECT_KEY)
    category = state.get("circuitcare_v2_symptom_category", "unknown")
    description = state.get("circuitcare_v2_symptom_description", "")
    try:
        description = normalize_symptom_description(description)
    except ValueError:
        description = None
    fingerprint = _current_fingerprint(state, category, description)
    if project and draft_is_current(state, fingerprint or ""):
        enter_workbench(state, project)
        st.rerun()
    if project:
        clear_intake_draft(state)

    _, center, _ = st.columns([1, 2.2, 1])
    with center:
        st.markdown('<div class="cc-entry-brand">CircuitCare AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="cc-entry-title">PCB 사진으로 고장 원인 찾기</div>', unsafe_allow_html=True)
        st.caption("사진을 올리면 소자를 찾고, 몇 가지 질문을 통해 확인할 검사를 안내합니다.")
        batch_key = f"cc_v2_photo_upload_{state[INTAKE_UPLOAD_BATCH_KEY]}"
        uploaded = st.file_uploader(
            "PCB 사진 업로드", type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True, key=batch_key,
            help=f"앞면·뒷면·확대 사진을 최대 {MAX_PHOTOS}장까지 추가할 수 있습니다.",
            **_upload_limit_kwargs(st),
        )
        if uploaded:
            new_images, messages = add_image_batch(
                images, ((item.name, item.getvalue()) for item in uploaded),
            )
            state[INTAKE_IMAGES_KEY] = new_images
            state["circuitcare_v2_photo_messages"] = messages
            state[INTAKE_UPLOAD_BATCH_KEY] += 1
            state.pop(INTAKE_PROJECT_KEY, None)
            if new_images != images:
                clear_component_map_state(state)
                # Begin loading Qwen while the user is still on the intake
                # screen. Startup is asynchronous so image preview stays
                # responsive; the first inference acquires the normal runtime
                # lease and cancels its idle timer.
                from circuitcare.ollama_runtime import prewarm_model_server
                prewarm_model_server()
            st.rerun()

        for message in state.pop("circuitcare_v2_photo_messages", []):
            st.warning(message)
        images = state.get(INTAKE_IMAGES_KEY, [])
        if images:
            for start_index in range(0, len(images), 3):
                row = images[start_index:start_index + 3]
                columns = st.columns(len(row))
                for offset, image_record in enumerate(row):
                    index = start_index + offset
                    with columns[offset]:
                        try:
                            preview = load_board_image(image_record["data"], max_edge=720).image
                            st.image(preview, use_container_width=True)
                        except Exception as exc:
                            st.error(f"사진을 읽지 못했습니다 · {exc}")
                        st.caption(image_record["filename"][:40])
                        with st.popover("사진 옵션"):
                            side = st.selectbox(
                                "촬영면", list(PHOTO_SIDES),
                                index=list(PHOTO_SIDES).index(image_record["side"]),
                                format_func=lambda value: PHOTO_SIDES[value],
                                key=f"cc_v2_photo_side_{image_record['image_id']}",
                            )
                            capture = st.selectbox(
                                "사진 종류", list(CAPTURE_TYPES),
                                index=list(CAPTURE_TYPES).index(image_record["capture_type"]),
                                format_func=lambda value: CAPTURE_TYPES[value],
                                key=f"cc_v2_photo_capture_{image_record['image_id']}",
                            )
                            if side != image_record["side"] or capture != image_record["capture_type"]:
                                clear_component_map_state(state)
                                image_record["side"], image_record["capture_type"] = side, capture
                                state.pop(INTAKE_PROJECT_KEY, None)
                                state.pop("circuitcare_v2_analysis_status", None)
                                st.rerun()
                            if st.button("사진 제거", key=f"cc_v2_remove_{image_record['image_id']}"):
                                clear_component_map_state(state)
                                state[INTAKE_IMAGES_KEY] = [item for item in images
                                                            if item["image_id"] != image_record["image_id"]]
                                state.pop(INTAKE_PROJECT_KEY, None)
                                state.pop("circuitcare_v2_analysis_status", None)
                                st.rerun()
        else:
            st.info("PCB 사진을 한 장 이상 추가하세요. 증상 입력과 분석은 다음 화면에서 진행합니다.")

        can_start = bool(images)
        if st.button("분석 시작", type="primary", use_container_width=True, disabled=not can_start):
            try:
                # Retain symptom values from an existing project when editing a
                # previously saved intake; new projects collect symptoms later.
                old_category = state.get("circuitcare_v2_symptom_category", "unknown")
                old_description = state.get("circuitcare_v2_symptom_description", "")
                try:
                    old_description = normalize_symptom_description(old_description)
                except ValueError:
                    old_description = None
                project = create_project_context(
                    images=state[INTAKE_IMAGES_KEY], symptom_category=old_category,
                    symptom_description=old_description,
                    circuit_source=state.get("circuitcare_v2_circuit_source"),
                    project_id=state.get("circuitcare_v2_edit_project_id"),
                    board_id=state.get("circuitcare_v2_edit_board_id"),
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                state[INTAKE_PROJECT_KEY] = project
                state["circuitcare_v2_analysis_status"] = "NOT_RUN"
                state.pop("circuitcare_v2_edit_project_id", None)
                state.pop("circuitcare_v2_edit_board_id", None)
                enter_workbench(state, project)
                st.rerun()
