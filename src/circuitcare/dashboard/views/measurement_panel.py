"""Inline measurement entry and history for the selected visual component."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel
from circuitcare.frontend.measurement_adapter import adapt_measurement_input
from circuitcare.frontend.measurement_models import (
    BASE_UNITS, MEASUREMENT_TYPE_LABELS, MEASUREMENT_UNITS, SOURCE_LABELS,
    MeasurementRecord, parse_finite_number,
)
from circuitcare.frontend.measurement_state import (
    delete_measurement, get_measurements, save_measurement, update_measurement,
)
from circuitcare.frontend.models import ProjectContext

SOURCE_OPTIONS = (
    "실제 측정 · 사용자 입력", "시뮬레이션", "가상 데모", "가져온 측정값", "출처 미확인",
)
SOURCE_CODES = {label: code for code, label in SOURCE_LABELS.items()}
TYPE_OPTIONS = tuple(MEASUREMENT_TYPE_LABELS)


def _sigma_for_input_unit(value: Any, measurement_type: str, unit: str) -> str | None:
    """UI sigma is entered in the measurement type's base unit; adapters expect the selected unit."""
    if value is None or not str(value).strip():
        return None
    sigma_base = parse_finite_number(value, label="불확실도")
    return str(sigma_base / MEASUREMENT_UNITS[measurement_type][unit])


def render_measurement_panel(
    st: Any, project: ProjectContext, *, selected_component: ComponentOverlayViewModel | None,
    preview: bool = False, component_map_version: str | None = None,
) -> None:
    st.markdown("### 측정값")
    if selected_component is None:
        st.caption("PCB에서 부품을 선택하면 측정값을 기록할 수 있습니다.")
        return
    if not project.board_id:
        st.warning("보드 ID가 없어 측정 기록을 연결할 수 없습니다.")
        return
    owner = f"{project.project_id}-{project.board_id}-{selected_component.component_id}"
    records = get_measurements(st.session_state, project_id=project.project_id,
                               board_id=project.board_id, component_id=selected_component.component_id)
    current_records = [row for row in records if (
        component_map_version is not None
        and row.component_map_version == component_map_version
        and row.component_identity_status not in {"IDENTITY_AMBIGUOUS", "IDENTITY_CONFLICT"}
    )]
    stale_records = [row for row in records if row not in current_records]
    st.caption("선택 부품에 측정 기록을 저장합니다. 저장은 진단 실행을 의미하지 않습니다.")
    st.caption("실제 측정 시 회로 전원 상태와 계측기 설정을 확인하세요. 측정 위치는 전기적 안전성 승인을 의미하지 않습니다.")
    if stale_records:
        st.warning(f"이전 Component Map의 측정 기록 {len(stale_records)}개는 현재 부품에 자동 연결하지 않았습니다. 부품 연결을 확인해 주세요.")
    if preview:
        st.info("미리보기에서는 측정 기록을 저장할 수 없습니다.")
        return

    type_key = f"cc_v5_type_{owner}"
    measurement_type = st.selectbox(
        "측정 종류", TYPE_OPTIONS, format_func=MEASUREMENT_TYPE_LABELS.get,
        key=type_key,
    )
    units = tuple(MEASUREMENT_UNITS[measurement_type])
    unit_key = f"cc_v5_unit_{owner}"
    if st.session_state.get(unit_key) not in units:
        st.session_state[unit_key] = units[0]
    with st.form(f"cc_v5_add_measurement_{owner}"):
        value = st.text_input("측정값", key=f"cc_v5_value_{owner}", placeholder="예: 0, -0.25, 3.3")
        unit = st.selectbox("단위", units, key=unit_key)
        uncertainty = st.text_input(
            f"측정 불확실도 σ ({BASE_UNITS[measurement_type]}, 선택)",
            key=f"cc_v5_sigma_{owner}", placeholder="기준 단위로 입력 · 진단에는 양수 필요",
        )
        source_label = st.selectbox("측정 출처", SOURCE_OPTIONS, key=f"cc_v5_source_{owner}")
        submitted = st.form_submit_button("측정값 저장", type="primary")
    if submitted:
        try:
            record = adapt_measurement_input(
                project_id=project.project_id, board_id=project.board_id,
                component_id=selected_component.component_id,
                component_identity_status=selected_component.identification_status,
                measurement_type=measurement_type, value=value, unit=unit,
                uncertainty=_sigma_for_input_unit(uncertainty, measurement_type, unit),
                source=SOURCE_CODES[source_label],
                component_map_version=component_map_version,
            )
            save_measurement(st.session_state, record)
            st.success("측정값을 기록했습니다. 전기적 Node Mapping은 확인되지 않아 진단에는 사용되지 않습니다.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    st.caption(f"현재 Component Map 측정 기록 · {len(current_records)}개")
    if not current_records:
        st.caption("저장된 측정값 없음")
    for record in reversed(current_records):
        _render_record_editor(st, project=project, record=record, owner=owner)
    if stale_records:
        with st.expander(f"이전 Map · 연결 확인 필요 · {len(stale_records)}개"):
            for record in reversed(stale_records):
                st.caption(
                    f"{MEASUREMENT_TYPE_LABELS[record.measurement_type]} · "
                    f"{record.entered_value:g} {record.entered_unit} · "
                    f"{record.timestamp[:16]} · Map {record.component_map_version or '없음'} · "
                    "현재 부품과 연결 미확인"
                )


def _render_record_editor(st: Any, *, project: ProjectContext, record: MeasurementRecord, owner: str) -> None:
    title = (f"{MEASUREMENT_TYPE_LABELS[record.measurement_type]} · "
             f"{record.entered_value:g} {record.entered_unit} · {record.timestamp[:16]}")
    with st.expander(title):
        st.caption(f"출처 · {SOURCE_LABELS[record.source]} · 검증 상태 {record.verification_status}")
        st.caption(f"Component ID · {record.component_id} · 전기적 Mapping 미확인")
        if record.uncertainty is not None:
            st.caption(f"불확실도 σ · {record.uncertainty:g} {record.unit}")
        st.caption(f"Measurement Set · {(record.measurement_set_id or '')[:12]}")
        type_key = f"cc_v5_edit_type_{record.measurement_id}"
        options = tuple(MEASUREMENT_UNITS)
        if st.session_state.get(type_key) not in options:
            st.session_state[type_key] = record.measurement_type
        callback_key = f"cc_v5_edit_type_previous_{record.measurement_id}"
        if callback_key not in st.session_state:
            st.session_state[callback_key] = record.measurement_type

        def reset_changed_kind(*, state: Any, measurement_id: str, key: str, previous_key: str) -> None:
            new_type = state.get(key)
            if new_type != state.get(previous_key):
                state[f"cc_v5_edit_unit_{measurement_id}"] = tuple(MEASUREMENT_UNITS[new_type])[0]
                state[f"cc_v5_edit_sigma_{measurement_id}"] = ""
                state[previous_key] = new_type

        edit_type = st.selectbox("수정 측정 종류", options, format_func=MEASUREMENT_TYPE_LABELS.get,
                                 key=type_key,
                                 on_change=reset_changed_kind,
                                 kwargs={"state": st.session_state, "measurement_id": record.measurement_id,
                                         "key": type_key, "previous_key": callback_key})
        units = tuple(MEASUREMENT_UNITS[edit_type])
        unit_key = f"cc_v5_edit_unit_{record.measurement_id}"
        if st.session_state.get(unit_key) not in units:
            st.session_state[unit_key] = record.entered_unit if record.entered_unit in units else units[0]
        with st.form(f"cc_v5_edit_form_{record.measurement_id}"):
            value = st.text_input("수정 측정값", value=f"{record.entered_value:g}",
                                  key=f"cc_v5_edit_value_{record.measurement_id}")
            unit = st.selectbox("수정 단위", units, key=unit_key)
            sigma_default = ""
            if edit_type == record.measurement_type and record.uncertainty is not None:
                sigma_default = f"{record.uncertainty:g}"
            uncertainty = st.text_input(
                f"수정 불확실도 σ ({BASE_UNITS[edit_type]})", value=sigma_default,
                key=f"cc_v5_edit_sigma_{record.measurement_id}",
            )
            source_label = st.selectbox(
                "수정 출처", SOURCE_OPTIONS, index=SOURCE_OPTIONS.index(SOURCE_LABELS[record.source]),
                key=f"cc_v5_edit_source_{record.measurement_id}",
            )
            edit = st.form_submit_button("변경 저장")
            delete = st.form_submit_button("측정값 삭제")
        if edit:
            try:
                updated = adapt_measurement_input(
                    project_id=record.project_id, board_id=record.board_id,
                    component_id=record.component_id,
                    component_identity_status=record.component_identity_status,
                    measurement_type=edit_type, value=value, unit=unit,
                    uncertainty=_sigma_for_input_unit(uncertainty, edit_type, unit),
                    source=SOURCE_CODES[source_label],
                    component_map_version=record.component_map_version,
                    measurement_id=record.measurement_id, timestamp=record.timestamp,
                )
                updated = replace(updated, changed_at=datetime.now(timezone.utc).isoformat())
                update_measurement(st.session_state, updated)
                st.success("측정 기록을 수정했습니다. 기존 진단 결과는 무효화되었습니다.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if delete:
            delete_measurement(st.session_state, project_id=project.project_id,
                               board_id=project.board_id or "", measurement_id=record.measurement_id)
            st.success("측정 기록을 삭제했습니다. 기존 진단 결과는 무효화되었습니다.")
            st.rerun()
