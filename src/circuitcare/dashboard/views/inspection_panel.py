"""PCB inspection summary and selected-component details."""

from __future__ import annotations

import io
import hashlib
import json
import logging
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel, display_identity_status
from circuitcare.frontend.models import ProjectContext
from circuitcare.vision.component_recognition_panel import (
    APP_DIR, CLASS_OPTIONS, LAST_EXCLUDED_KEY, STATUS_OPTIONS, ensure_review_row_for_component,
    invalidate_component_review_state, record_box_exclusion, restore_review_row,
    review_keys, undo_last_box_exclusion, update_review_row,
)
from circuitcare.vision.weight_paths import resolve_pcb_weights_root
from circuitcare.component_taxonomy import COMPONENT_CLASSES
from circuitcare.frontend.workbench_state import (
    ACTIVE_PHOTO_KEY, SELECTION_KEY, update_component_selection, update_selection,
)

MAP_VERSION_KEY = "circuitcare_v4_selection_map_key"
COMPONENT_EDIT_KEY = "circuitcare_v4_component_edits"
COMPONENT_INVENTORY_KEY = "circuitcare_v4_component_inventory_by_map"
USER_SUSPECTS_KEY = "circuitcare_v4_user_suspects_by_map"
USER_SUSPECTS_MIGRATION_KEY = "circuitcare_v4_user_suspects_migrated_scopes"
USER_SUSPECTS_DISMISSED_KEY = "circuitcare_v4_user_suspects_dismissed"
ROLE_OPTIONS = ("power", "control", "control_input", "control_timing", "driver", "output",
                "input", "input_output", "protection", "thermal", "buzzer", "relay")
LOGGER = logging.getLogger(__name__)
COMPONENT_LABELS = {
    "resistor": "저항", "capacitor": "커패시터", "inductor": "인덕터",
    "diode": "다이오드", "transistor": "트랜지스터", "ic": "IC",
    "integrated_circuit": "IC", "connector": "커넥터", "switch": "스위치",
    "relay": "릴레이", "fuse": "퓨즈", "battery": "배터리", "buzzer": "부저",
    "led": "LED", "unknown": "미확정",
}


def _persist_review_learning(st: Any, project: ProjectContext, image_bytes: bytes | None,
                             rows: list[dict[str, Any]]) -> None:
    if not image_bytes:
        return
    from circuitcare.vision.review_learning import queue_reviewed_boxes

    queue_reviewed_boxes(image_bytes, project_id=project.project_id,
                         photo_id=str(st.session_state.get(ACTIVE_PHOTO_KEY) or "unknown"), rows=rows)


def _render_component_knowledge_form(st: Any, component: ComponentOverlayViewModel,
                                     component_type: str, *, key: str) -> None:
    with st.expander("다음 분류를 위한 소자 정보 추가", expanded=False):
        st.caption("사용자가 확인한 각인·부품번호 정보를 로컬에 저장해, 다음에 같은 표시가 읽히면 Qwen 분류 참고자료로 제공합니다.")
        options = [value for value in CLASS_OPTIONS if value in COMPONENT_CLASSES]
        default_type = component_type if component_type in options else options[0]
        kind = st.selectbox("확인한 소자 종류", options, index=options.index(default_type), key=f"{key}_type")
        marking = st.text_input("소자 각인", value=str(component.ocr_marking or ""), key=f"{key}_marking")
        part_number = st.text_input("부품번호", value=str(getattr(component, "component_info", None) or ""), key=f"{key}_part")
        subtype = st.text_input("세부 종류 (선택)", key=f"{key}_subtype")
        role = st.text_input("기능 역할 (선택)", key=f"{key}_role")
        notes = st.text_input("추가 정보 (선택)", key=f"{key}_notes")
        if st.button("확인한 정보를 다음 분류에 저장", key=f"{key}_save", use_container_width=True):
            try:
                from circuitcare.vision.review_learning import save_component_knowledge

                save_component_knowledge(component_type=kind, marking=marking, part_number=part_number,
                                         subtype=subtype, functional_role=role, notes=notes)
                st.success("사용자 확인 정보가 로컬에 저장됐습니다. 동일 각인이 보이면 이후 Qwen 검수에 참고로 전달됩니다.")
            except ValueError as exc:
                st.warning(str(exc))


def render_retraining_workspace(st: Any, project: ProjectContext) -> None:
    """Separate, opt-in workspace; reviewed samples are collected silently in the review flow."""
    from circuitcare.vision.review_learning import (
        MIN_UNIQUE_IMAGES, flush_review_samples, refresh_training_jobs, start_retraining, training_summary,
        save_user_component_type, user_component_types,
    )

    save_errors = flush_review_samples()
    refresh_training_jobs()
    summary = training_summary()
    st.markdown("### 검수 데이터로 별도 재학습")
    st.caption("사용자가 확정한 박스만 로컬에 자동 축적됩니다. 동일한 픽셀 이미지의 재업로드는 표본 수를 늘리지 않습니다."
               " 재학습 결과는 별도 후보 가중치로 저장되며 현재 탐지 모델을 자동 교체하지 않습니다.")
    st.metric("서로 다른 검수 사진", f"{summary['unique_images']} / {MIN_UNIQUE_IMAGES}")
    st.caption(f"저장된 사용자 확정 박스 · {summary['box_count']}개")
    st.markdown("#### 사용자 소자 종류")
    st.caption("기본 YOLO 18종은 그대로 유지합니다. 새 종류는 별도 레이블로 모으고, 진단에는 선택한 상위 종류를 사용합니다. 추가 레이블은 다음 후보 재학습 데이터에 포함됩니다.")
    with st.form(f"cc_add_user_component_type_{project.project_id}"):
        new_type_name = st.text_input("새 소자 종류 이름", placeholder="예: 전압 감시기")
        new_type_parent = st.selectbox("진단에 사용할 상위 종류", COMPONENT_CLASSES)
        new_type_description = st.text_input("구분 설명 (선택)", placeholder="외관이나 각인으로 구별하는 기준")
        add_type = st.form_submit_button("소자 종류 추가")
    if add_type:
        try:
            saved_type = save_user_component_type(display_name=new_type_name,
                                                  parent_class=new_type_parent,
                                                  description=new_type_description)
            st.success(f"{saved_type['display_name']} 종류를 추가했습니다. 다음 박스 검수에서 선택할 수 있습니다.")
        except ValueError as exc:
            st.warning(str(exc))
    custom_types = user_component_types()
    if custom_types:
        st.dataframe([{"소자 종류": row["display_name"], "진단 상위 종류": row["parent_class"],
                        "학습 레이블": row["type_id"], "설명": row["description"]}
                       for row in custom_types], hide_index=True, use_container_width=True)
    for error in save_errors:
        st.warning(f"백그라운드 표본 저장 실패 · {error}")
    weight_root = resolve_pcb_weights_root(APP_DIR)
    checkpoint = weight_root / "external" / "pcb-yolov8s-detection" / "best.pt"
    same_training = summary.get("already_trained")
    active_training = summary.get("active_job")
    if not checkpoint.is_file():
        st.warning(f"재학습용 초기 가중치가 로컬에 없습니다 · {checkpoint}")
    can_start = (summary["unique_images"] >= MIN_UNIQUE_IMAGES and checkpoint.is_file()
                 and same_training is None and active_training is None)
    if st.button("별도 재학습 시작", key=f"cc_review_retraining_start_{project.project_id}",
                 type="primary", disabled=not can_start):
        try:
            result = start_retraining(checkpoint=checkpoint, epochs=50)
            if result.get("status") == "RUNNING":
                st.success(f"재학습을 별도 작업으로 시작했습니다 · 작업 {result['job_id']}")
            elif result.get("status") == "ALREADY_TRAINED":
                st.info("현재와 같은 검수 표본 묶음으로 이미 학습했습니다. 새 고유 이미지나 수정된 검수 박스가 모인 뒤 다시 시작하세요.")
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"재학습을 시작하지 못했습니다 · {exc}")
    if same_training:
        st.info("현재 표본 묶음은 이미 학습했습니다. 같은 사진을 다시 저장해도 재학습하지 않습니다.")
    if active_training:
        st.info(f"재학습 진행 중 · 작업 {active_training['job_id']}")
        if st.button("재학습 상태 새로고침", key=f"cc_review_retraining_refresh_{project.project_id}"):
            st.rerun()
    last_job = summary.get("last_job")
    if last_job:
        st.caption(f"최근 작업 · {last_job['status']} · {last_job['job_id']}")
        if last_job.get("status") == "SUCCEEDED":
            metadata = Path(last_job["output_dir"]) / "candidate.json"
            if metadata.is_file():
                candidate = json.loads(metadata.read_text()).get("candidate_weights")
                st.code(str(candidate), language=None)
        elif last_job.get("status") == "FAILED":
            st.error(f"학습 실패 · {last_job.get('error') or '로그 확인 필요'}")
            log_path = Path(last_job["log_path"])
            if log_path.is_file():
                with st.expander("학습 오류 자세히 보기"):
                    st.code(log_path.read_text(errors="replace")[-6000:], language=None)


def board_suspects(state: Any, project: ProjectContext) -> list[dict[str, Any]]:
    """Read/migrate user suspicions under project + board + canonical board ID."""
    owner = f"{project.project_id}|{project.board_id or ''}"
    store = state.setdefault(USER_SUSPECTS_KEY, {})
    rows = store.setdefault(owner, {})
    migrated = state.setdefault(USER_SUSPECTS_MIGRATION_KEY, {})
    dismissed_store = state.setdefault(USER_SUSPECTS_DISMISSED_KEY, {})
    dismissed = set(dismissed_store.get(owner, set()))
    if not migrated.get(owner):
        for old_owner, old_rows in list(store.items()):
            if old_owner == owner or not str(old_owner).startswith(f"{owner}|") or not isinstance(old_rows, dict):
                continue
            for old_id, old_row in old_rows.items():
                if isinstance(old_row, dict):
                    canonical_id = str(old_row.get("board_component_id") or old_id)
                    if canonical_id not in dismissed:
                        rows.setdefault(canonical_id, old_row)
        migrated[owner] = True
        state[USER_SUSPECTS_MIGRATION_KEY] = migrated
    store[owner] = rows
    state[USER_SUSPECTS_KEY] = store
    return [row for row in rows.values() if isinstance(row, dict)]


def confirm_qwen_component_type(row: dict[str, Any], *, base_class: str, subtype: str | None = None) -> None:
    """Persist a user-confirmed class without overwriting the detector observation."""
    from circuitcare.component_taxonomy import normalize_qwen_class, normalize_qwen_subtype

    base = normalize_qwen_class(base_class)
    if base in {"unknown", "not_component"}:
        raise ValueError("이 종류는 사용자 확정 소자 종류로 저장할 수 없습니다.")
    subtype_value = normalize_qwen_subtype(base, subtype)
    row["detector_class"] = row.get("detector_class") or row.get("소자 종류", "unknown")
    row["qwen_base_class"] = base
    row["qwen_subtype"] = subtype_value
    row["confirmed_base_class"] = base
    row["confirmed_subtype"] = subtype_value
    row["effective_class"] = base
    row["component_type_status"] = "USER_CONFIRMED_EXTENSION" if base in {"other_component", "board_feature"} else "USER_CONFIRMED"
    row["qwen_component_type_status"] = row["component_type_status"]
    row["qwen_review_status"] = "USER_CONFIRMED"
    row["상태"] = "확인"
    row["presence_status"] = "CONFIRMED"
    row["user_edited"] = True
    if row["component_type_status"] == "USER_CONFIRMED_EXTENSION":
        row["소자 종류"] = "unknown"
        row["diagnosis_class"] = "unknown"
        row["qwen_user_confirmed_extension"] = True
    else:
        row["소자 종류"] = base
        row["diagnosis_class"] = base


def dismiss_board_suspect(state: Any, project: ProjectContext, board_component_id: str) -> None:
    owner = f"{project.project_id}|{project.board_id or ''}"
    state.setdefault(USER_SUSPECTS_KEY, {}).setdefault(owner, {}).pop(str(board_component_id), None)
    dismissed_store = state.setdefault(USER_SUSPECTS_DISMISSED_KEY, {})
    dismissed_store[owner] = set(dismissed_store.get(owner, set())) | {str(board_component_id)}
    state[USER_SUSPECTS_DISMISSED_KEY] = dismissed_store


def add_board_suspect(state: Any, project: ProjectContext, board_component_id: str,
                      suspect: dict[str, Any]) -> None:
    owner = f"{project.project_id}|{project.board_id or ''}"
    dismissed_store = state.setdefault(USER_SUSPECTS_DISMISSED_KEY, {})
    dismissed_store[owner] = set(dismissed_store.get(owner, set())) - {str(board_component_id)}
    state[USER_SUSPECTS_DISMISSED_KEY] = dismissed_store
    state.setdefault(USER_SUSPECTS_KEY, {}).setdefault(owner, {})[str(board_component_id)] = suspect


def render_inspection_panel(
    st: Any, project: ProjectContext, *, preview: bool = False,
    components: tuple[ComponentOverlayViewModel, ...] = (),
    selected_component: ComponentOverlayViewModel | None = None,
    detection_status: str = "UNKNOWN",
    selected_marking_context: dict[str, Any] | None = None,
    component_map_version: str | None = None,
    inventory_scope: str | None = None,
    review_image_bytes: bytes | None = None,
    review_scope: str | None = None,
    board_registry: dict[str, Any] | None = None,
    compact: bool = False,
    on_focus_component: Any = None,
) -> None:
    from circuitcare.vision.review_learning import user_component_types

    custom_type_rows = user_component_types()
    custom_types = {str(row["type_id"]): row for row in custom_type_rows}
    type_options = [*CLASS_OPTIONS, *custom_types]
    type_labels = {type_id: row["display_name"] for type_id, row in custom_types.items()}
    type_format = lambda value: type_labels.get(value, "종류 미확정" if value == "unknown" else value)
    owner = f"{project.project_id}|{project.board_id or ''}|{inventory_scope or component_map_version or ''}"
    suspect_owner = f"{project.project_id}|{project.board_id or ''}"
    inventories = st.session_state.setdefault(COMPONENT_INVENTORY_KEY, {})
    inventory = inventories.setdefault(owner, {})
    board_suspects(st.session_state, project)
    all_suspects = st.session_state.setdefault(USER_SUSPECTS_KEY, {})
    suspects = all_suspects.setdefault(suspect_owner, {})
    with st.container(border=True):
        st.markdown('<span class="cc-inspection-marker"></span>', unsafe_allow_html=True)
        visible_components = [item for item in components if item.presence_status != "EXCLUDED"]
        confirmed_components = sum(item.presence_status == "CONFIRMED" for item in visible_components)
        st.caption(
            f"탐지·검수 대상 {len(visible_components)}개 · 확인 {confirmed_components}개 · "
            f"미확인 {len(visible_components) - confirmed_components}개"
        )
        if suspects:
            with st.popover(f"내가 의심한 소자 · {len(suspects)}개", use_container_width=True):
                registry_rows = board_registry.get("components", []) if isinstance(board_registry, dict) else []
                for board_id, suspect in list(suspects.items()):
                    registry_row = next((row for row in registry_rows
                                         if str(row.get("board_component_id")) == str(board_id)), None)
                    excluded = bool(registry_row and registry_row.get("presence_status") == "EXCLUDED")
                    kind = suspect.get("component_type", registry_row.get("component_type", "unknown")
                                       if registry_row else "unknown")
                    label = suspect.get("reference_designator") or suspect.get("component_id") or board_id
                    kind_label = COMPONENT_LABELS.get(str(kind).casefold(), str(kind))
                    st.markdown(f"**{label}** · {kind_label}" + (" · 제외됨" if excluded else ""))
                    if excluded:
                        st.caption("박스 제외로 현재 후보에서 빠져 있습니다. 복원 후 다시 확인할 수 있습니다.")
                    focus_col, clear_col = st.columns(2)
                    if focus_col.button("PCB 위치 보기", key=f"cc_suspect_focus_{project.project_id}_{board_id}",
                                         disabled=excluded or on_focus_component is None):
                        if on_focus_component(str(board_id)):
                            st.rerun()
                    if clear_col.button("의심 해제", key=f"cc_suspect_clear_{project.project_id}_{board_id}"):
                        dismiss_board_suspect(st.session_state, project, str(board_id))
                        st.rerun()
        if preview:
            st.markdown("**부품** · 인식 결과 없음")
        elif detection_status in {"DETECTOR_NOT_CONFIGURED", "BLOCKED_MODEL_UNAVAILABLE"}:
            st.markdown("**부품 검출** · 검출 모델 미설정")
        elif detection_status not in {"INFERENCE_COMPLETE", "PARTIAL_MAP_READY"}:
            st.markdown("**부품 검출** · 인식 결과 없음")
        elif not components:
            st.markdown("**인식된 부품** · 없음")
        else:
            st.markdown(f"**인식된 부품** · {len(components)}개")
        active_scope = st.session_state.get("cc_active_component_review_scope") or review_scope
        active_records_key = review_keys(active_scope)[1] if active_scope else None
        records_key = review_keys(review_scope)[1] if review_scope else None
        active_rows = list(st.session_state.get(active_records_key, [])) if active_records_key else []
        if active_scope:
            undo_key = f"{LAST_EXCLUDED_KEY}_{active_scope}"
            last_excluded = st.session_state.get(undo_key)
            undo_col, restore_col = st.columns(2)
            undo_id = None
            if isinstance(last_excluded, dict):
                undo_ids = last_excluded.get("component_ids") or []
                undo_id = last_excluded.get("component_id") or (undo_ids[0] if undo_ids else None)
            if undo_col.button("직전 제외 취소", key=f"cc_component_undo_exclusion_{active_scope}",
                               disabled=not bool(undo_id), use_container_width=True,
                               help="가장 최근에 제외한 박스를 복원합니다."):
                restored_id = undo_last_box_exclusion(st.session_state, scope=active_scope, rows=active_rows)
                if restored_id:
                    st.session_state[active_records_key] = active_rows
                    invalidate_component_review_state(st.session_state, scope=active_scope,
                        project_id=project.project_id, board_id=project.board_id)
                    update_selection(st.session_state, project=project, selected_component=restored_id)
                    st.rerun()
            excluded_rows = [row for row in active_rows if row.get("상태") == "제외"]
            if excluded_rows:
                with restore_col.popover(f"제외 복원 · {len(excluded_rows)}",
                                         help="이전에 제외한 박스를 선택해 다시 표시합니다."):
                    for index, row in enumerate(excluded_rows):
                        st.caption(f"{row.get('component_id')} · {row.get('소자 종류', 'unknown')}")
                        if st.button("복원", key=f"cc_restore_box_{active_scope}_{row.get('component_id')}_{index}",
                                     help="이 박스를 다시 활성 상태로 돌리고 Viewer와 진단 목록에 반영합니다."):
                            restore_review_row(row)
                            st.session_state[active_records_key] = active_rows
                            if isinstance(last_excluded, dict) and last_excluded.get("component_id") == row.get("component_id"):
                                st.session_state.pop(undo_key, None)
                            invalidate_component_review_state(st.session_state, scope=active_scope,
                                project_id=project.project_id, board_id=project.board_id)
                            update_selection(st.session_state, project=project,
                                             selected_component=str(row.get("component_id")))
                            st.rerun()
            active_viewer_key = st.session_state.get("circuitcare_v4_active_viewer_key")
            bulk_key = f"cc_pcb_bulk_selection_{active_viewer_key}" if active_viewer_key else None
            selected_ids = st.session_state.get(bulk_key, []) if bulk_key else []
            selected_ids = list(dict.fromkeys(str(value) for value in selected_ids))
            row_by_id = {str(row.get("component_id")): row for row in active_rows if row.get("component_id")}
            target_ids = [component_id for component_id in selected_ids
                          if component_id in row_by_id and row_by_id[component_id].get("상태") != "제외"]
            if selected_ids:
                st.markdown(f"**PCB에서 드래그 선택 · {len(target_ids)}개**")
                st.caption(" · ".join(target_ids[:8]) + (f" 외 {len(target_ids) - 8}개" if len(target_ids) > 8 else ""))
                component_by_id = {str(item.component_id): item for item in components}
                bulk_type_key = f"cc_bulk_review_type_{active_scope}"
                bulk_status_key = f"cc_bulk_review_status_{active_scope}"
                bulk_type = st.selectbox("선택 박스 종류 일괄 변경",
                                         ["변경 안 함", *type_options], key=bulk_type_key,
                                         format_func=lambda value: "변경 안 함" if value == "변경 안 함" else type_format(value))
                bulk_status = st.selectbox("선택 박스 상태 일괄 변경",
                                           ["변경 안 함", "검수 필요", "확인"], key=bulk_status_key)
                apply_col, exclude_col, clear_col = st.columns(3)
                if apply_col.button("일괄 수정 적용", key=f"cc_bulk_apply_{active_scope}",
                                    use_container_width=True,
                                    disabled=(not target_ids or
                                              (bulk_type == "변경 안 함" and bulk_status == "변경 안 함"))):
                    try:
                        for component_id in target_ids:
                            row = row_by_id[component_id]
                            model = component_by_id.get(component_id)
                            if model is not None:
                                row = ensure_review_row_for_component(active_rows, model)
                            update_review_row(
                                row,
                                component_type=(row.get("user_component_type_id") or row.get("소자 종류", "unknown")
                                                if bulk_type == "변경 안 함" else bulk_type),
                                status=(row.get("상태", "검수 필요")
                                        if bulk_status == "변경 안 함" else bulk_status),
                                bbox_xyxy=[row[key] for key in ("x1", "y1", "x2", "y2")],
                                custom_types=custom_types,
                            )
                            if bulk_type != "변경 안 함" or bulk_status == "확인":
                                row["training_label_user_confirmed"] = True
                            if row.get("소자 종류") == "unknown":
                                record_box_exclusion(st.session_state, scope=active_scope, row=row,
                                                     reason="unconfirmed_component_not_assigned")
                        st.session_state[active_records_key] = active_rows
                        _persist_review_learning(st, project, review_image_bytes, active_rows)
                        invalidate_component_review_state(
                            st.session_state, scope=active_scope,
                            project_id=project.project_id, board_id=project.board_id,
                        )
                        st.rerun()
                    except (KeyError, ValueError, TypeError) as exc:
                        st.error(f"일괄 수정에 실패했습니다 · {exc}")
                if exclude_col.button("선택 박스 제외", key=f"cc_bulk_exclude_{active_scope}",
                                      use_container_width=True, disabled=not target_ids):
                    before_rows = {component_id: deepcopy(row_by_id[component_id]) for component_id in target_ids}
                    for component_id in target_ids:
                        record_box_exclusion(st.session_state, scope=active_scope, row=row_by_id[component_id])
                    st.session_state[f"{LAST_EXCLUDED_KEY}_{active_scope}"] = {
                        "component_ids": list(target_ids), "before_rows": before_rows,
                    }
                    st.session_state[active_records_key] = active_rows
                    st.session_state[bulk_key] = []
                    invalidate_component_review_state(
                        st.session_state, scope=active_scope,
                        project_id=project.project_id, board_id=project.board_id,
                    )
                    if selected_component and selected_component.component_id in target_ids:
                        update_component_selection(st.session_state, project=project, selected_component=None)
                    st.rerun()
                if clear_col.button("영역 선택 해제", key=f"cc_bulk_clear_{active_scope}", use_container_width=True):
                    st.session_state[bulk_key] = []
                    st.rerun()
        st.markdown("### 선택 부품")
        if compact:
            if selected_component is None:
                st.caption("PCB에서 박스를 누르면 소자 정보가 여기에 표시됩니다. 소자를 선택하지 않아도 진단할 수 있습니다.")
            else:
                component_id = selected_component.component_id
                board_component_id = str(selected_component.board_component_id or component_id)
                compact_rows_key = review_keys(review_scope)[1] if review_scope else None
                compact_review_row = next((row for row in st.session_state.get(compact_rows_key, [])
                                           if row.get("component_id") == component_id), {}) if compact_rows_key else {}
                shown_type = compact_review_row.get("user_component_type_name") or selected_component.component_type
                st.markdown(f"**{selected_component.reference_designators[0] if selected_component.reference_designators else component_id}** · {shown_type}")
                if selected_component.ocr_marking:
                    st.caption(f"각인 후보 · {selected_component.ocr_marking}")
                if review_scope and review_image_bytes is not None:
                    from circuitcare.vision.ocr_backend import infer_passive_value
                    marking_cache = st.session_state.setdefault("cc_component_marking_ocr", {})
                    marking_key = f"{review_scope}:{component_id}:{selected_component.bbox_xyxy}"
                    marking_result = marking_cache.get(marking_key)
                    if st.button("선택 소자 각인 판독", key=f"cc_marking_ocr_run_{marking_key}"):
                        try:
                            from PIL import Image
                            from circuitcare.vision.ocr_backend import run_component_marking_ocr
                            with Image.open(io.BytesIO(review_image_bytes)) as source_image:
                                marking_result = run_component_marking_ocr(
                                    source_image.convert("RGB"), selected_component.bbox_xyxy,
                                    selected_component.component_type,
                                )
                            marking_cache[marking_key] = marking_result
                            st.session_state["cc_component_marking_ocr"] = marking_cache
                            rows = list(st.session_state.get(records_key, [])) if records_key else []
                            source_row = next((row for row in rows if row.get("component_id") == component_id), None)
                            if source_row is not None:
                                source_row["body_marking"] = marking_result.get("marking")
                                source_row["body_marking_candidates"] = marking_result.get("candidates", [])
                                source_row["ocr_inferred_value"] = marking_result.get("inferred_value")
                                source_row["ocr_inference_method"] = marking_result.get("inference_method")
                                source_row["body_marking_verification_status"] = "unverified"
                                st.session_state[records_key] = rows
                                invalidate_component_review_state(st.session_state, scope=review_scope,
                                    project_id=project.project_id, board_id=project.board_id)
                        except Exception as exc:
                            st.warning(f"각인 판독을 완료하지 못했습니다 · {exc}")
                    if isinstance(marking_result, dict):
                        candidates = marking_result.get("candidates", [])
                        marking = marking_result.get("marking")
                        if marking:
                            st.caption(f"본체 각인 후보 · {marking} · 미확인")
                        inferred = infer_passive_value(selected_component.component_type, marking or "")
                        if inferred and selected_component.component_type.casefold() == "resistor":
                            st.caption(f"OCR 공칭값 후보 · {inferred['value']} · {inferred['method']} · 미확인 (실측값 아님)")
                        if selected_component.component_type.casefold() in {"ic", "integrated_circuit"}:
                            refs = list(selected_component.reference_designators or ())
                            if refs:
                                st.caption("실크스크린 ID 후보 · " + ", ".join(refs[:3]))
                            if marking:
                                st.caption("부품번호 후보 · 데이터시트 확인 전까지 미검증")
                        if not marking and not candidates:
                            st.caption("읽을 수 있는 각인이 없습니다.")
                if isinstance(board_registry, dict):
                    registry_component = next((row for row in board_registry.get("components", [])
                                               if str(row.get("board_component_id")) == str(selected_component.board_component_id or component_id)), None)
                    if registry_component:
                        qwen_candidates = [source for source in registry_component.get("source_components", [])
                                           if source.get("qwen_component_type_candidate")]
                        selected_source = next((source for source in qwen_candidates
                                                if source.get("component_id") == component_id), None)
                        for source in qwen_candidates[:2]:
                            candidate_type = source.get("qwen_component_type_candidate")
                            candidate_subtype = source.get("qwen_component_subtype_candidate")
                            subtype_label = f" / {candidate_subtype}" if candidate_subtype else ""
                            review_status = source.get("qwen_review_status") or "AI_SUGGESTED"
                            st.caption(f"Qwen 종류 제안 · {candidate_type}{subtype_label} · {review_status} · 미확인")
                            if source.get("qwen_component_type_evidence"):
                                st.caption(str(source["qwen_component_type_evidence"]))
                            alternatives = source.get("qwen_component_type_alternatives", [])
                            if alternatives:
                                st.caption("다른 후보 · " + ", ".join(alternatives))
                            conflicts = source.get("qwen_conflicting_evidence", [])
                            if conflicts:
                                st.warning("Qwen/OCR 근거 충돌 · " + " · ".join(str(value) for value in conflicts))
                        if (selected_source and selected_source.get("qwen_component_type_status") == "AI_PROPOSED_UNVERIFIED"
                                and review_scope and records_key):
                            candidate_type = str(selected_source.get("qwen_component_type_candidate") or "")
                            if candidate_type in CLASS_OPTIONS and candidate_type != "not_component":
                                if st.button("Qwen 종류 사용자 확인", key=f"cc_compact_qwen_confirm_{review_scope}_{component_id}"):
                                    rows = list(st.session_state.get(records_key, []))
                                    target = next((row for row in rows if row.get("component_id") == component_id), None)
                                    if target is not None:
                                        confirm_qwen_component_type(target, base_class=candidate_type,
                                            subtype=selected_source.get("qwen_component_subtype_candidate"))
                                        if candidate_type in COMPONENT_CLASSES:
                                            target["training_label_user_confirmed"] = True
                                        st.session_state[records_key] = rows
                                        _persist_review_learning(st, project, review_image_bytes, rows)
                                        invalidate_component_review_state(st.session_state, scope=review_scope,
                                            project_id=project.project_id, board_id=project.board_id)
                                        st.rerun()
                            elif candidate_type in {"other_component", "board_feature"}:
                                if st.button("확장 종류 제안 사용자 확인", key=f"cc_compact_qwen_confirm_extension_{review_scope}_{component_id}"):
                                    rows = list(st.session_state.get(records_key, []))
                                    target = next((row for row in rows if row.get("component_id") == component_id), None)
                                    if target is not None:
                                        confirm_qwen_component_type(target, base_class=candidate_type,
                                            subtype=selected_source.get("qwen_component_subtype_candidate"))
                                        st.session_state[records_key] = rows
                                        _persist_review_learning(st, project, review_image_bytes, rows)
                                        invalidate_component_review_state(st.session_state, scope=review_scope,
                                            project_id=project.project_id, board_id=project.board_id)
                                        st.rerun()
                        if (selected_source and selected_source.get("qwen_component_type_candidate") == "not_component"
                                and selected_source.get("qwen_component_type_decision") == "suggest"
                                and review_scope and records_key):
                            if st.button("Qwen 오탐 제안 · 선택 박스 제외", key=f"cc_compact_qwen_exclude_{review_scope}_{component_id}"):
                                rows = list(st.session_state.get(records_key, []))
                                target = next((row for row in rows if row.get("component_id") == component_id), None)
                                if target is not None:
                                    record_box_exclusion(st.session_state, scope=review_scope, row=target,
                                        reason="qwen_not_component", review={
                                            "decision": selected_source.get("qwen_component_type_decision"),
                                            "proposed_class": "not_component",
                                            "confidence": selected_source.get("qwen_component_type_confidence"),
                                            "evidence": selected_source.get("qwen_component_type_evidence"),
                                        })
                                    st.session_state[records_key] = rows
                                    invalidate_component_review_state(st.session_state, scope=review_scope,
                                        project_id=project.project_id, board_id=project.board_id)
                                    st.rerun()
                edits = st.session_state.setdefault(COMPONENT_EDIT_KEY, {})
                current_type = str(edits.get(component_id, {}).get("component_type") or selected_component.component_type)
                if review_scope and review_image_bytes is not None:
                    edit_key = f"cc_compact_type_{review_scope}_{component_id}"
                    current_review = next((row for row in st.session_state.get(records_key, [])
                                           if row.get("component_id") == component_id), {})
                    current_editor_type = str(current_review.get("user_component_type_id") or current_type)
                    if st.session_state.get(edit_key) not in type_options:
                        st.session_state[edit_key] = current_editor_type if current_editor_type in type_options else "unknown"
                    new_type = st.selectbox("소자 종류", type_options, key=edit_key, format_func=type_format)
                    if new_type == "unknown":
                        st.warning("종류를 지정하지 않고 검수 저장하면 이 박스는 제외됩니다. 제외 메뉴에서 복원할 수 있습니다.")
                    if new_type != current_editor_type and st.button("종류 변경 저장", key=f"cc_compact_type_save_{review_scope}_{component_id}"):
                        rows = list(st.session_state.get(records_key, []))
                        row = ensure_review_row_for_component(rows, selected_component)
                        update_review_row(row, component_type=new_type, status=row.get("상태", "검수 필요"),
                                          bbox_xyxy=[row[key] for key in ("x1", "y1", "x2", "y2")],
                                          custom_types=custom_types)
                        if new_type in COMPONENT_CLASSES or new_type in custom_types:
                            row["training_label_user_confirmed"] = True
                        if new_type == "unknown":
                            record_box_exclusion(st.session_state, scope=review_scope, row=row,
                                                 reason="unconfirmed_component_not_assigned")
                        st.session_state[records_key] = rows
                        _persist_review_learning(st, project, review_image_bytes, rows)
                        invalidate_component_review_state(st.session_state, scope=review_scope,
                            project_id=project.project_id, board_id=project.board_id)
                        st.rerun()
                suspect_owner = f"{project.project_id}|{project.board_id or ''}"
                suspects = st.session_state.setdefault(USER_SUSPECTS_KEY, {}).setdefault(suspect_owner, {})
                if st.button("의심 해제" if board_component_id in suspects else "의심 소자로 지정",
                             key=f"cc_compact_suspect_toggle_{project.project_id}_{board_component_id}"):
                    if board_component_id in suspects:
                        dismiss_board_suspect(st.session_state, project, board_component_id)
                    else:
                        source_photo = (selected_component.source_backend_photo_ids[0]
                                       if len(selected_component.source_backend_photo_ids) == 1 else None)
                        photo_record = next((row for row in project.image_records if row.image_id == source_photo), None)
                        add_board_suspect(st.session_state, project, board_component_id, {
                            "component_id": component_id, "board_component_id": board_component_id,
                            "component_type": selected_component.component_type,
                            "reference_designator": (selected_component.reference_designators[0]
                                                      if selected_component.reference_designators else component_id),
                            "suspected_fault_mode": None, "selection_source": "USER",
                            "evidence_status": "UNVERIFIED", "photo_id": source_photo,
                            "photo_sha256": getattr(photo_record, "content_sha256", None),
                            "bbox_xyxy": list(selected_component.bbox_xyxy),
                        })
                    st.rerun()
                if review_scope and records_key:
                    review_row = next((row for row in active_rows if row.get("component_id") == component_id), None)
                    if review_row and review_row.get("상태") == "제외":
                        st.caption("이 박스는 제외 상태입니다.")
                    elif st.button("선택 박스 제외", key=f"cc_compact_exclude_{review_scope}_{component_id}"):
                        rows = list(st.session_state.get(records_key, []))
                        viewer_key = st.session_state.get("circuitcare_v4_active_viewer_key")
                        bulk_key = f"cc_pcb_bulk_selection_{viewer_key}" if viewer_key else None
                        selected_ids = list(dict.fromkeys(str(value) for value in
                            st.session_state.get(bulk_key, []) if value)) if bulk_key else []
                        exclude_ids = selected_ids if component_id in selected_ids else [component_id]
                        row_by_id = {str(row.get("component_id")): row for row in rows
                                     if row.get("component_id")}
                        exclude_ids = [value for value in exclude_ids
                                       if value in row_by_id and row_by_id[value].get("상태") != "제외"]
                        before_rows = {value: deepcopy(row_by_id[value]) for value in exclude_ids}
                        for value in exclude_ids:
                            record_box_exclusion(st.session_state, scope=review_scope,
                                                 row=row_by_id[value])
                        if exclude_ids:
                            st.session_state[f"{LAST_EXCLUDED_KEY}_{review_scope}"] = {
                                "component_ids": exclude_ids, "before_rows": before_rows,
                            }
                        st.session_state[records_key] = rows
                        if bulk_key:
                            st.session_state[bulk_key] = []
                        invalidate_component_review_state(st.session_state, scope=review_scope,
                            project_id=project.project_id, board_id=project.board_id)
                        update_component_selection(st.session_state, project=project, selected_component=None)
                        st.rerun()
                _render_component_knowledge_form(
                    st, selected_component, current_type,
                    key=f"cc_compact_knowledge_{review_scope}_{component_id}",
                )
            return
        if selected_component is None:
            st.caption("PCB에서 박스를 누르면 이곳에 인식 정보가 표시됩니다.")
        else:
            edits = st.session_state.setdefault(COMPONENT_EDIT_KEY, {})
            edit = edits.get(selected_component.component_id, {})
            current_type = selected_component.component_type
            current_marking = edit.get("ocr_marking", selected_component.ocr_marking or "")
            current_value = edit.get("component_info", selected_component.component_info or "")
            st.caption(f"부품 ID · {selected_component.component_id}")
            inventory_item = inventory.setdefault(selected_component.component_id, {
                "presence_status": "UNCONFIRMED", "component_type_status": "UNCONFIRMED",
                "part_identity_status": "UNCONFIRMED", "confirmed_functional_roles": [],
            })
            review_row = None
            records_key = review_keys(review_scope)[1] if review_scope else None
            if review_image_bytes is not None and records_key:
                review_row = next((row for row in st.session_state.get(records_key, [])
                                   if row.get("component_id") == selected_component.component_id), None)
            current_status = review_row.get("상태", "검수 필요") if review_row else (
                "확인" if selected_component.presence_status == "CONFIRMED" else "검수 필요")
            inventory_item["presence_status"] = (
                "CONFIRMED" if current_status == "확인" else "EXCLUDED" if current_status == "제외" else "UNCONFIRMED")
            inventory_item["component_type_status"] = "USER_CONFIRMED" if current_status == "확인" else "UNCONFIRMED"
            st.caption(f"상태 · {current_status} · 검수 레코드와 동기화됨")
            if review_row is not None and review_row.get("상태") == "제외":
                st.warning("제외된 박스입니다. 복원 전까지 진단과 기본 Viewer에서 제외됩니다.")
            elif review_scope and records_key:
                if st.button("선택 박스 제외", key=f"cc_selected_box_exclude_{selected_component.component_id}",
                             type="primary", use_container_width=True,
                             help="선택한 박스를 기본 Viewer와 진단 대상에서 숨깁니다. 검수 기록은 보존되며 직전 제외 취소 또는 제외 복원으로 되돌릴 수 있습니다."):
                    rows = list(st.session_state.get(records_key, []))
                    viewer_key = st.session_state.get("circuitcare_v4_active_viewer_key")
                    bulk_key = f"cc_pcb_bulk_selection_{viewer_key}" if viewer_key else None
                    selected_ids = list(dict.fromkeys(str(value) for value in
                        st.session_state.get(bulk_key, []) if value)) if bulk_key else []
                    exclude_ids = selected_ids if selected_component.component_id in selected_ids else [selected_component.component_id]
                    row_by_id = {str(row.get("component_id")): row for row in rows if row.get("component_id")}
                    exclude_ids = [value for value in exclude_ids
                                   if value in row_by_id and row_by_id[value].get("상태") != "제외"]
                    before_rows = {value: deepcopy(row_by_id[value]) for value in exclude_ids}
                    for value in exclude_ids:
                        record_box_exclusion(st.session_state, scope=review_scope, row=row_by_id[value])
                    if exclude_ids:
                        st.session_state[f"{LAST_EXCLUDED_KEY}_{review_scope}"] = {
                            "component_ids": exclude_ids, "before_rows": before_rows,
                        }
                    st.session_state[records_key] = rows
                    if bulk_key:
                        st.session_state[bulk_key] = []
                    invalidate_component_review_state(st.session_state, scope=review_scope,
                        project_id=project.project_id, board_id=project.board_id)
                    update_component_selection(st.session_state, project=project,
                                               selected_component=None)
                    st.rerun()
            if review_scope and records_key and review_image_bytes is not None:
                if current_type == "unknown":
                    st.warning("종류가 미확정입니다. 변경 적용 시 종류를 지정하지 않으면 이 박스는 제외되며, 복원 메뉴에서 되돌릴 수 있습니다.")
                box = review_row or {"x1": selected_component.bbox_xyxy[0], "y1": selected_component.bbox_xyxy[1],
                                     "x2": selected_component.bbox_xyxy[2], "y2": selected_component.bbox_xyxy[3]}
                current_editor_type = str((review_row or {}).get("user_component_type_id") or current_type)
                row_signature = (current_editor_type, current_status,
                                 int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"]))
                form_version = hashlib.sha1(repr(row_signature).encode()).hexdigest()[:8]
                form_key = f"cc_review_box_editor_{review_scope}_{selected_component.component_id}_{form_version}"
                with st.form(form_key):
                    st.selectbox("소자 종류", type_options,
                                 index=type_options.index(current_editor_type if current_editor_type in type_options else "unknown"),
                                 key=f"{form_key}_class",
                                 format_func=type_format,
                                 help="검수 결과의 소자 종류를 바꾸면 Viewer와 진단 소자 목록에도 반영됩니다.")
                    editable_statuses = [value for value in STATUS_OPTIONS if value != "제외"]
                    status_value = current_status if current_status in editable_statuses else "검수 필요"
                    st.selectbox("검수 상태", editable_statuses, index=editable_statuses.index(status_value),
                                 key=f"{form_key}_status",
                                 help="확인은 사용자 확인 기록으로 저장됩니다. 제외는 별도 제외 버튼으로 처리합니다.")
                    with st.expander("박스 위치·크기"):
                        cols = st.columns(2)
                        x1 = cols[0].number_input("왼쪽 x", min_value=0, value=int(box["x1"]), key=f"{form_key}_x1",
                                                  help="원본 사진의 픽셀 좌표입니다.")
                        y1 = cols[1].number_input("위쪽 y", min_value=0, value=int(box["y1"]), key=f"{form_key}_y1",
                                                  help="원본 사진의 픽셀 좌표입니다.")
                        x2 = cols[0].number_input("오른쪽 x", min_value=0, value=int(box["x2"]), key=f"{form_key}_x2",
                                                  help="원본 사진의 픽셀 좌표입니다.")
                        y2 = cols[1].number_input("아래쪽 y", min_value=0, value=int(box["y2"]), key=f"{form_key}_y2",
                                                  help="원본 사진의 픽셀 좌표입니다.")
                    apply = st.form_submit_button("변경 적용", use_container_width=True,
                                                   help="종류·상태·좌표를 검수 기록에 저장하고 Viewer와 진단 목록에 반영합니다.")
                if apply:
                    from PIL import Image
                    with Image.open(io.BytesIO(review_image_bytes)) as source_image:
                        width, height = source_image.size
                    coords = [max(0, min(width, int(x1))), max(0, min(height, int(y1))),
                              max(0, min(width, int(x2))), max(0, min(height, int(y2)))]
                    try:
                        rows = list(st.session_state.get(records_key, []))
                        review_row = ensure_review_row_for_component(rows, selected_component)
                        next_status = st.session_state[f"{form_key}_status"]
                        next_type = st.session_state[f"{form_key}_class"]
                        update_review_row(review_row,
                            component_type=next_type,
                            status=next_status, bbox_xyxy=coords, custom_types=custom_types)
                        if next_type != current_type or next_status == "확인":
                            review_row["training_label_user_confirmed"] = True
                        if next_type == "unknown":
                            st.warning("미확정 소자는 검수 저장 후 제외됩니다. 상단 복원 메뉴에서 되돌릴 수 있습니다.")
                            record_box_exclusion(st.session_state, scope=review_scope, row=review_row,
                                                 reason="unconfirmed_component_not_assigned")
                        st.session_state[records_key] = rows
                        _persist_review_learning(st, project, review_image_bytes, rows)
                        inventory_item["presence_status"] = (
                            "CONFIRMED" if review_row["상태"] == "확인" else
                            "EXCLUDED" if review_row["상태"] == "제외" else "UNCONFIRMED")
                        inventory_item["component_type_status"] = (
                            "USER_CONFIRMED" if review_row["상태"] == "확인" else "UNCONFIRMED")
                        st.session_state[COMPONENT_INVENTORY_KEY] = inventories
                        invalidate_component_review_state(st.session_state, scope=review_scope,
                            project_id=project.project_id, board_id=project.board_id)
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
            type_status_options = ("UNCONFIRMED", "USER_CONFIRMED")
            with st.expander("상세 식별·기능 확인", expanded=False):
                identity_key = f"cc_part_identity_status_{hash(owner)}_{selected_component.component_id}"
                if st.session_state.get(identity_key) not in type_status_options:
                    st.session_state[identity_key] = inventory_item.get("part_identity_status", "UNCONFIRMED")
                part_identity_status = st.radio("부품 모델/부품번호 확인", type_status_options,
                    format_func=lambda value: "모델 식별됨" if value == "USER_CONFIRMED" else "모델 미확인",
                    horizontal=True, key=identity_key,
                    on_change=_sync_inventory_widget, args=(st, owner, selected_component.component_id,
                        "part_identity_status", identity_key, project.project_id, project.board_id,
                        str(st.session_state.get(ACTIVE_PHOTO_KEY) or "target:default")))
                inventory_item["part_identity_status"] = part_identity_status
                role_key = f"cc_component_roles_{hash(owner)}_{selected_component.component_id}"
                current_roles = inventory_item.get("confirmed_functional_roles", [])
                if role_key not in st.session_state:
                    st.session_state[role_key] = [role for role in current_roles if role in ROLE_OPTIONS]
                confirmed_roles = st.multiselect(
                    "이 부품에서 확인된 기능 역할", ROLE_OPTIONS, key=role_key,
                    help="직접 확인한 역할만 선택합니다. 선택하지 않은 배치/OCR 추정은 미확정 후보로 유지됩니다.",
                    on_change=_sync_inventory_widget, args=(st, owner, selected_component.component_id,
                        "confirmed_functional_roles", role_key, project.project_id, project.board_id,
                        str(st.session_state.get(ACTIVE_PHOTO_KEY) or "target:default")),
                )
                inventory_item["confirmed_functional_roles"] = list(confirmed_roles)
            board_component_id = str(selected_component.board_component_id or selected_component.component_id)
            active_suspect = board_component_id in suspects
            if st.button("의심 소자 해제" if active_suspect else "의심 소자로 지정",
                         key=f"cc_v4_toggle_suspect_{hash(suspect_owner)}_{board_component_id}"):
                if active_suspect:
                    dismiss_board_suspect(st.session_state, project, board_component_id)
                else:
                    photo_id = (selected_component.source_backend_photo_ids[0]
                                if len(selected_component.source_backend_photo_ids) == 1 else None)
                    photo = next((row for row in project.image_records if row.image_id == photo_id), None)
                    payload = {"component_id": selected_component.component_id,
                               "component_type": selected_component.component_type,
                               "bbox_xyxy": list(selected_component.bbox_xyxy), "photo_id": photo_id,
                               "photo_sha256": getattr(photo, "content_sha256", None),
                               "inventory_scope": inventory_scope or component_map_version}
                    observation_fingerprint = hashlib.sha256(json.dumps(
                        payload, sort_keys=True, ensure_ascii=False, default=str
                    ).encode("utf-8")).hexdigest()
                    registry = board_registry or st.session_state.get("circuitcare_v7_active_board_registry")
                    registry_source = next((source for component in (registry or {}).get("components", [])
                                            if component.get("board_component_id") == board_component_id
                                            for source in component.get("source_components", [])
                                            if source.get("photo_id") == photo_id
                                            and source.get("component_id") == selected_component.component_id), None)
                    add_board_suspect(st.session_state, project, board_component_id, {
                        "component_id": selected_component.component_id,
                        "component_type": selected_component.component_type,
                        "reference_designator": (selected_component.reference_designators[0]
                                                  if selected_component.reference_designators else selected_component.component_id),
                        "suspected_fault_mode": None, "selection_source": "USER",
                        "evidence_status": "UNVERIFIED", "photo_id": photo_id,
                        "observation_fingerprint": observation_fingerprint,
                        "component_map_version": inventory_scope or component_map_version,
                        "selection_version": (registry_source.get("source_observation_fingerprint") if registry_source
                                              else getattr(selected_component, "source_observation_fingerprint", None)),
                    })
                    suspects = st.session_state[USER_SUSPECTS_KEY][suspect_owner]
                all_suspects[suspect_owner] = suspects
                st.session_state[USER_SUSPECTS_KEY] = all_suspects
                st.rerun()
            inventories[owner] = inventory
            st.session_state[COMPONENT_INVENTORY_KEY] = inventories
            selected_component = replace(
                selected_component,
                component_type=current_type,
                presence_status=inventory_item["presence_status"],
                confirmed_functional_roles=tuple(confirmed_roles),
            )
            if review_image_bytes is not None and review_scope:
                if review_row is not None:
                    marking_result_key = f"cc_component_marking_ocr_{review_scope}_{selected_component.component_id}"
                    with st.expander("선택 부품 각인 OCR", expanded=False):
                        st.caption("각인 인식값은 미확인 후보입니다. OCR만으로 부품 모델이나 종류를 확정하지 않습니다.")
                        if st.button("선택 부품에서 각인 읽기", key=f"cc_component_marking_run_{review_scope}_{selected_component.component_id}"):
                            try:
                                from PIL import Image
                                from circuitcare.vision.ocr_backend import run_component_marking_ocr

                                with Image.open(io.BytesIO(review_image_bytes)) as source_image:
                                    result = run_component_marking_ocr(
                                        source_image.convert("RGB"),
                                        [review_row["x1"], review_row["y1"], review_row["x2"], review_row["y2"]],
                                        current_type,
                                    )
                                st.session_state[marking_result_key] = result
                            except Exception as exc:
                                st.warning(f"각인 OCR을 실행하지 못했습니다 · {exc}")
                        marking_result = st.session_state.get(marking_result_key)
                        if isinstance(marking_result, dict):
                            _render_selected_marking(st, {
                                "image_bytes": review_image_bytes,
                                "bbox_xyxy": [review_row["x1"], review_row["y1"], review_row["x2"], review_row["y2"]],
                                "result": marking_result,
                            })
                            choices = [item.get("text", "") for item in marking_result.get("candidates", []) if item.get("text")]
                            if choices:
                                choice_key = f"cc_component_marking_choice_{review_scope}_{selected_component.component_id}"
                                choice = st.selectbox("OCR 원문 후보 선택", choices, key=choice_key)
                                if st.button("선택한 원문을 편집 후보로 사용", key=f"cc_component_marking_use_{review_scope}_{selected_component.component_id}"):
                                    edits = st.session_state.setdefault(COMPONENT_EDIT_KEY, {})
                                    edits[selected_component.component_id] = {
                                        **edits.get(selected_component.component_id, {}),
                                        "ocr_marking": choice,
                                        "component_info": marking_result.get("inferred_value") or current_value,
                                    }
                                    st.session_state[COMPONENT_EDIT_KEY] = edits
                                    invalidate_component_review_state(st.session_state, scope=review_scope,
                                        project_id=project.project_id, board_id=project.board_id)
                                    st.rerun()
            if selected_component.fault_hypothesis:
                st.caption(f"진단 후보 · {selected_component.diagnosis_display_status or '미확정'} · "
                           f"가설 {selected_component.fault_hypothesis}")
                if selected_component.fault_role_status == "INFERRED":
                    st.caption("기능 역할 추정 · 미확정 후보이며 추가 역할 확인이 필요합니다.")
                elif selected_component.fault_role_status == "CONFIRMED":
                    st.caption("기능 역할 확인됨")
                if selected_component.fault_score is not None:
                    st.caption(f"상대 점수 · {selected_component.fault_score:g} · 확률이 아닌 후보 정렬 점수")
                if selected_component.measurement_status:
                    st.caption(f"측정 판정 · {selected_component.measurement_status} · "
                               "이 검사 가설에만 적용")
                if selected_component.fault_evidence:
                    st.caption("후보 근거 · " + " · ".join(selected_component.fault_evidence[:5]))
                if selected_component.recommended_tests:
                    st.caption("권장 확인 · " + " · ".join(selected_component.recommended_tests[:3]))
            for history_item in selected_component.fault_history[-3:]:
                st.caption(
                    f"검사 이력 · {history_item.get('fault_title', history_item.get('fault_hypothesis', '고장 가설'))} · "
                    f"{history_item.get('measurement_verdict', '판정 기록')}"
                )
            marking_key = f"cc_v4_edit_marking_{selected_component.component_id}"
            value_key = f"cc_v4_edit_value_{selected_component.component_id}"
            if marking_key not in st.session_state:
                st.session_state[marking_key] = current_marking
            if value_key not in st.session_state:
                st.session_state[value_key] = current_value
            with st.expander("인식 정보 편집", expanded=False):
                st.text_input("OCR 인식값 · 각인/부품번호", key=marking_key,
                              placeholder="예: 103, LM358")
                st.text_input("소자 값 · OCR 유추값 또는 직접 입력", key=value_key,
                              placeholder="예: 10 kΩ, 100 nF, 3.3 V")
                if st.button("인식 정보 저장", type="primary", key=f"cc_v4_save_component_{selected_component.component_id}"):
                    edits[selected_component.component_id] = {
                        "ocr_marking": st.session_state[marking_key].strip(),
                        "component_info": st.session_state[value_key].strip(),
                    }
                    st.session_state[COMPONENT_EDIT_KEY] = edits
                    if review_scope:
                        invalidate_component_review_state(st.session_state, scope=review_scope,
                            project_id=project.project_id, board_id=project.board_id)
                    st.rerun()
            if current_marking or current_value:
                st.caption(f"자동 판독 상태 · {display_identity_status(selected_component.identification_status)}")
            if selected_component.reference_designators:
                st.caption("주변 실크스크린 Reference 후보 · " + " · ".join(selected_component.reference_designators))
            _render_component_knowledge_form(
                st, selected_component, current_type,
                key=f"cc_component_knowledge_{review_scope or owner}_{selected_component.component_id}",
            )
            if selected_component.ocr_marking:
                st.caption("소자 본체 각인 후보 · " + selected_component.ocr_marking)
            if selected_marking_context and selected_component.component_id.startswith("auto-"):
                _render_selected_marking(st, selected_marking_context)
            if selected_component.identity_match_status:
                st.caption(f"이전 식별 연속성 · {display_identity_status(selected_component.identity_match_status)}")
            st.caption(f"관측 사진 · {len(selected_component.source_backend_photo_ids)}장")
            st.caption("정합 · 후보 · 미검증")
            if selected_component.evidence:
                with st.expander("검출 근거"):
                    st.write(" · ".join(selected_component.evidence))
            if review_row is not None and review_image_bytes is not None and review_scope:
                rows = st.session_state.get(review_keys(review_scope)[1], [])
                excluded_rows = [row for row in rows if row.get("상태") == "제외"]
                if excluded_rows:
                    st.caption(f"제외된 박스 {len(excluded_rows)}개 · 상단의 복원 메뉴에서 되살릴 수 있습니다.")
        # Fault analysis lives in its own full-width workspace. This panel is
        # intentionally limited to PCB component review and editing.


def _diagnosis_status_label(st: Any, project: ProjectContext, *, analysis_snapshot: dict[str, Any] | None = None) -> str:
    if analysis_snapshot is None:
        from circuitcare.dashboard.views.symptom_diagnosis_panel import current_analysis_snapshot

        components = tuple(st.session_state.get("circuitcare_v4_current_components", ()))
        result_scope = st.session_state.get("circuitcare_v4_component_inventory_scope")
        owner = f"{project.project_id}|{project.board_id or ''}|{result_scope or st.session_state.get(MAP_VERSION_KEY) or ''}"
        inventories = st.session_state.get(COMPONENT_INVENTORY_KEY, {})
        inventory = inventories.get(owner, {}) if isinstance(inventories, dict) else {}
        analysis_snapshot = current_analysis_snapshot(
            st, project, components=components, component_inventory=inventory,
        )
    result = analysis_snapshot.get("result")
    if not isinstance(result, dict):
        status = analysis_snapshot.get("record", {}).get("status")
        if status == "RUNNING":
            return "진단 중"
        if status == "STALE":
            return "입력 변경 · 재분석 필요"
        if status == "FAILED":
            return "진단 실패 · 재시도 필요"
        return "진단 미실행"
    status = result.get("status")
    if status == "CANDIDATES_AVAILABLE":
        return f"진단 완료 · 후보 {len(result.get('component_candidates', []))}개"
    if status == "ABSTAIN_INSUFFICIENT_EVIDENCE":
        return "추론 보류 · 추가 정보 필요"
    return "진단 결과 확인 필요"


def _sync_inventory_widget(st: Any, owner: str, component_id: str, field: str, widget_key: str,
                           project_id: str | None = None, board_id: str | None = None,
                           review_scope: str | None = None) -> None:
    inventories = st.session_state.setdefault(COMPONENT_INVENTORY_KEY, {})
    inventory = inventories.setdefault(owner, {})
    item = inventory.setdefault(component_id, {})
    item[field] = st.session_state.get(widget_key)
    st.session_state[COMPONENT_INVENTORY_KEY] = inventories
    if project_id:
        invalidate_component_review_state(st.session_state, scope=review_scope or "target:default",
                                          project_id=project_id, board_id=board_id)


def _render_selected_marking(st: Any, context: dict[str, Any]) -> None:
    """Show the enlarged selected-part crop and the OCR candidates in the inspector."""
    from PIL import Image, ImageEnhance, ImageOps

    raw_image = context.get("image_bytes")
    bbox = context.get("bbox_xyxy")
    result = context.get("result") or {}
    if raw_image and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            image = Image.open(io.BytesIO(raw_image)).convert("RGB")
            x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
            crop = image.crop((max(0, min(x1, x2)), max(0, min(y1, y2)),
                               min(image.width, max(x1, x2)), min(image.height, max(y1, y2))))
            if crop.width and crop.height:
                scale = max(2, min(6, 480 // max(crop.size)))
                enlarged = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
                enlarged = ImageEnhance.Contrast(ImageOps.grayscale(enlarged)).enhance(1.25)
                st.image(enlarged, caption="선택 부품 각인 확대 · OCR 입력", use_container_width=True)
        except (OSError, ValueError, TypeError):
            st.caption("선택 부품 확대 이미지를 만들 수 없습니다.")

    status = result.get("status")
    if status == "OCR_UNAVAILABLE":
        st.warning(f"각인 OCR을 사용할 수 없습니다 · {result.get('error', '엔진 오류')}")
        st.caption("OCR 의존성을 설치한 뒤 다시 선택하면 재시도합니다.")
    elif status == "OCR_NO_TEXT":
        st.info("확대 이미지에서 읽을 수 있는 각인을 찾지 못했습니다. 원본 소자 종류는 유지됩니다.")
    elif status == "OCR_COMPLETE":
        candidates = result.get("candidates", [])
        if len(candidates) > 1:
            st.caption("다른 방향에서 읽힌 후보 · " + " · ".join(item["text"] for item in candidates[1:4]))
        st.caption("OCR 인식값과 계산값은 검수 전 후보이며, 원본 부품 라벨은 바뀌지 않습니다.")
