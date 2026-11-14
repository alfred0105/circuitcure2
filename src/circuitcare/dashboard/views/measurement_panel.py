"""Inline measurement entry and history for the selected visual component."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from circuitcare.frontend.component_overlay_models import ComponentOverlayViewModel
from circuitcare.frontend.board_component_registry import (
    current_source_observation_fingerprint, resolve_measurement_board_component,
    source_component_key,
)
from circuitcare.frontend.measurement_adapter import adapt_measurement_input
from circuitcare.frontend.measurement_models import (
    BASE_UNITS, MEASUREMENT_TYPE_LABELS, MEASUREMENT_UNITS, SOURCE_LABELS,
    MeasurementRecord, parse_finite_number,
)
from circuitcare.frontend.measurement_state import (
    delete_measurement, get_measurements, reconfirm_measurement_identity,
    save_measurement, update_measurement,
)
from circuitcare.frontend.models import ProjectContext
from circuitcare.symptom_diagnosis import load_knowledge
from circuitcare.symptom_diagnosis.measurement_validation import measurement_verification_gaps

SOURCE_OPTIONS = (
    "실제 측정 · 사용자 입력", "시뮬레이션", "가상 데모", "가져온 측정값", "출처 미확인",
)
SOURCE_CODES = {label: code for code, label in SOURCE_LABELS.items()}
TYPE_OPTIONS = tuple(MEASUREMENT_TYPE_LABELS)


def _measurement_verification_gaps(record: MeasurementRecord) -> list[str]:
    """Return UI-facing missing conditions; recording remains allowed without them."""
    profiles = load_knowledge().get("component_test_profiles", {})
    return measurement_verification_gaps(record, profiles=profiles)


def _sigma_for_input_unit(value: Any, measurement_type: str, unit: str) -> str | None:
    """UI sigma is entered in the measurement type's base unit; adapters expect the selected unit."""
    if value is None or not str(value).strip():
        return None
    sigma_base = parse_finite_number(value, label="불확실도")
    return str(sigma_base / MEASUREMENT_UNITS[measurement_type][unit])


def _sync_profile_test_state(
    state: Any, *, test_key: str, type_key: str, unit_key: str, sigma_key: str,
    tests: dict[str, Any],
) -> None:
    """Apply a selected registered profile before the next rerun creates widgets."""
    test_id = state.get(test_key)
    profile = tests.get(test_id, {}) if test_id else {}
    measurement_type = profile.get("measurement_type")
    if measurement_type not in MEASUREMENT_UNITS:
        return
    state[type_key] = measurement_type
    state[unit_key] = tuple(MEASUREMENT_UNITS[measurement_type])[0]
    state[sigma_key] = ""


def _reset_measurement_type_state(
    state: Any, *, type_key: str, unit_key: str, sigma_key: str,
) -> None:
    measurement_type = state.get(type_key)
    if measurement_type in MEASUREMENT_UNITS:
        state[unit_key] = tuple(MEASUREMENT_UNITS[measurement_type])[0]
        state[sigma_key] = ""


def render_measurement_panel(
    st: Any, project: ProjectContext, *, selected_component: ComponentOverlayViewModel | None,
    preview: bool = False, component_map_version: str | None = None,
    board_registry: dict[str, Any] | None = None,
    compact: bool = False,
    fault_hypothesis: str | None = None,
    recommended_test_id: str | None = None,
) -> None:
    if selected_component is None:
        st.info("PCB Viewer에서 소자를 선택하면 해당 소자의 측정값을 기록할 수 있습니다.")
        return
    if not project.board_id:
        st.warning("보드 ID가 없어 측정 기록을 연결할 수 없습니다.")
        return
    board_component_id = selected_component.board_component_id
    if isinstance(board_registry, dict):
        photo_id = selected_component.canonical_frame_id
        local_id = selected_component.component_id
        source_key = source_component_key(photo_id, local_id)
        registry_component = next((row for row in board_registry.get("components", [])
                                   if str(row.get("board_component_id")) == str(board_component_id)), None)
        source = next((row for row in (registry_component or {}).get("source_components", [])
                       if row.get("source_key") == source_key and row.get("review_status") != "EXCLUDED"), None)
        current_fp = current_source_observation_fingerprint(
            board_registry, photo_id=photo_id, component_id=local_id)
        mapped_id = board_registry.get("source_to_board_id", {}).get(source_key)
        photo_manifest = next((row for row in board_registry.get("photos", [])
                               if row.get("photo_id") == photo_id), None)
        source_box = tuple(float(value) for value in source.get("bbox_xyxy", [])) if source else ()
        selected_box = tuple(float(value) for value in selected_component.bbox_xyxy)
        if (not registry_component or registry_component.get("presence_status") == "EXCLUDED"
                or not source or not board_component_id or str(mapped_id) != str(board_component_id)
                or not current_fp or not photo_manifest
                or photo_manifest.get("image_fingerprint") != source.get("image_fingerprint")
                or source_box != selected_box):
            st.warning("소자 위치 확인 필요 · 현재 사진·소자 ID·검수 연결을 Registry에서 검증하지 못해 측정 저장을 중단했습니다.")
            return
    owner = f"{project.project_id}-{project.board_id}-{board_component_id or selected_component.component_id}-{fault_hypothesis or 'no-fault'}"
    all_records = get_measurements(st.session_state, project_id=project.project_id,
                                   board_id=project.board_id)
    current_records: list[MeasurementRecord] = []
    stale_records: list[MeasurementRecord] = []
    for row in all_records:
        resolved_id = row.board_component_id
        resolution = row.identity_resolution_status
        if isinstance(board_registry, dict):
            if resolved_id:
                source_photo = row.source_photo_id
                source_local = row.source_component_id or row.component_id
                if source_photo:
                    actual, resolution = resolve_measurement_board_component(
                        board_registry, photo_id=source_photo, component_id=source_local,
                        photo_fingerprint=row.source_photo_fingerprint,
                        observation_fingerprint=row.source_observation_fingerprint,
                        component_map_version=(row.component_map_version
                                               if not row.source_photo_fingerprint else None))
                    if actual is None:
                        resolved_id = None
                    else:
                        resolved_id = actual
                elif (not source_photo or not source_local or not any(item.get("board_component_id") == resolved_id
                             for item in board_registry.get("components", []))):
                    resolved_id, resolution = None, "UNRESOLVED"
            else:
                resolved_id, resolution = resolve_measurement_board_component(
                    board_registry, photo_id=row.source_photo_id,
                    component_id=row.source_component_id or row.component_id,
                    photo_fingerprint=row.source_photo_fingerprint,
                    observation_fingerprint=row.source_observation_fingerprint,
                    component_map_version=row.component_map_version)
        if (board_component_id and resolved_id == board_component_id
                and str(resolution).startswith(("VERIFIED_", "USER_RECONFIRMED_"))):
            projected = replace(
                row, board_component_id=str(resolved_id),
                identity_resolution_status=resolution,
            )
            current_records.append(projected)
        elif row.component_id == selected_component.component_id and board_registry is None:
            if (component_map_version is not None and row.component_map_version == component_map_version
                    and row.component_identity_status not in {"IDENTITY_AMBIGUOUS", "IDENTITY_CONFLICT"}):
                current_records.append(row)
            else:
                stale_records.append(row)
        elif (not str(resolution).startswith(("VERIFIED_", "USER_RECONFIRMED_"))
              and (row.component_id == selected_component.component_id
                   or row.source_component_id == selected_component.component_id
                   or str(resolution).endswith("RECONFIRMATION_REQUIRED"))):
            stale_records.append(row)
    st.caption(f"측정 대상 · {board_component_id or selected_component.component_id} · {selected_component.component_type}")
    st.caption("측정값을 먼저 기록하세요. 검증 조건이 부족한 기록은 보존되지만 검증 결과로 판정하지 않습니다.")
    if compact:
        suggested = list(selected_component.recommended_tests or ())
        if suggested:
            st.info("확인할 내용 · " + " · ".join(str(item) for item in suggested[:2]))
        st.caption("저항·도통은 전원을 차단한 뒤 측정하세요. 통전 전압 측정은 안전한 측정 조건과 숙련이 확인되지 않으면 진행하지 마세요.")
    if stale_records:
        st.warning(f"이전 Component Map의 측정 기록 {len(stale_records)}개는 현재 부품에 자동 연결하지 않았습니다. 부품 연결을 확인해 주세요.")
    if preview:
        st.info("미리보기에서는 측정 기록을 저장할 수 없습니다.")
        return

    knowledge = load_knowledge()
    fault_modes = knowledge.get("fault_modes", {}).get("fault_modes", {})
    normalized_type = selected_component.component_type.casefold()
    type_profile = knowledge.get("component_test_profiles", {}).get("profiles", {}).get(normalized_type, {})
    test_profiles = type_profile.get("tests", {})
    fault_id_key = f"cc_v5_fault_{owner}"
    requested_fault_id = str(fault_hypothesis or "")
    compatible_tests = {
        test_id: profile for test_id, profile in test_profiles.items()
        if not requested_fault_id or requested_fault_id in profile.get("fault_relations", {})
    }
    default_test_id = (recommended_test_id if recommended_test_id in compatible_tests else
                       next(iter(compatible_tests), ""))
    test_options = [""] + list(compatible_tests)
    test_key = f"cc_v5_test_{owner}"
    test_state = st.session_state.get(test_key)
    if test_state not in test_options:
        st.session_state[test_key] = default_test_id
    selected_test_id = st.session_state.get(test_key, "")
    profile_signature_key = f"cc_v5_profile_applied_{owner}"
    type_key = f"cc_v5_type_{owner}"
    unit_key = f"cc_v5_unit_{owner}"
    sigma_key = f"cc_v5_sigma_{owner}"
    selected_profile = compatible_tests.get(selected_test_id, {})
    if st.session_state.get(profile_signature_key) != selected_test_id:
        expected_type = selected_profile.get("measurement_type")
        if expected_type in MEASUREMENT_UNITS:
            st.session_state[type_key] = expected_type
            st.session_state[unit_key] = tuple(MEASUREMENT_UNITS[expected_type])[0]
            st.session_state[sigma_key] = ""
        elif type_key not in st.session_state:
            st.session_state[type_key] = TYPE_OPTIONS[0]
        st.session_state[profile_signature_key] = selected_test_id
    elif type_key not in st.session_state:
        st.session_state[type_key] = TYPE_OPTIONS[0]
    st.selectbox(
        "부품별 검사 항목", test_options, key=test_key,
        format_func=lambda value: "프로파일 검사 선택 안 함" if not value else value,
        help="등록된 검사만 선택한 고장 가설의 검증에 연결됩니다.",
        on_change=_sync_profile_test_state,
        kwargs={"state": st.session_state, "test_key": test_key, "type_key": type_key,
                "unit_key": unit_key, "sigma_key": sigma_key, "tests": compatible_tests},
    )
    selected_test_id = st.session_state.get(test_key, "")
    selected_profile = compatible_tests.get(selected_test_id, {})
    profile_relations = selected_profile.get("fault_relations", {})
    fault_options = [""] + ([fault_id for fault_id in fault_modes if fault_id in profile_relations]
                              if selected_test_id else
                              ([requested_fault_id] if requested_fault_id in fault_modes else []))
    saved_fault_id = st.session_state.get(fault_id_key)
    default_fault_id = (requested_fault_id if requested_fault_id in fault_options else
                        saved_fault_id if saved_fault_id in fault_options else
                        requested_fault_id if not selected_test_id else "")
    if st.session_state.get(fault_id_key) not in fault_options:
        st.session_state[fault_id_key] = default_fault_id
    if selected_test_id:
        st.selectbox("검증할 고장 가설", fault_options, key=fault_id_key,
                     format_func=lambda code: "선택 안 함" if not code else fault_modes[code]["title"])
        target_keywords = selected_profile.get("target_keywords", [])
        condition_keywords = selected_profile.get("condition_keywords", [])
        if target_keywords:
            st.caption("측정 대상 예시 · " + " · ".join(target_keywords))
        if condition_keywords:
            st.caption("검사 조건 · " + " · ".join(condition_keywords))
    selected_fault_id = str(st.session_state.get(fault_id_key) or "")
    if selected_test_id and selected_fault_id not in profile_relations:
        st.warning("선택한 검사 프로파일은 이 고장 가설과 연결되지 않습니다. 호환되는 검사만 저장할 수 있습니다.")
        selected_test_id = ""
    registry_version = str((board_registry or {}).get("version") or component_map_version or "")
    safety_scope = f"{selected_test_id}|{registry_version}"
    safety_scope_key = f"cc_v5_safety_scope_{owner}"
    if st.session_state.get(safety_scope_key) != safety_scope:
        st.session_state[safety_scope_key] = safety_scope
        st.session_state[f"cc_v5_power_state_{owner}"] = "UNKNOWN"
        st.session_state[f"cc_v5_stored_energy_{owner}"] = "UNKNOWN"
        st.session_state[f"cc_v5_isolation_{owner}"] = "UNKNOWN"
        st.session_state[f"cc_v5_location_verified_{owner}"] = False
        st.session_state[f"cc_v5_reference_verified_{owner}"] = False
    st.caption("검사 전 확인 · 선택한 검사 조건을 저장 전에 확인합니다. 이 확인은 현장 측정의 안전을 보증하지 않습니다.")
    measurement_conditions = {
        "power_state": st.selectbox("전원 상태 확인", ("UNKNOWN", "POWER_OFF_CONFIRMED", "POWER_ON_CONFIRMED"),
            format_func=lambda value: {"UNKNOWN": "미확인", "POWER_OFF_CONFIRMED": "전원 분리 확인", "POWER_ON_CONFIRMED": "전원 연결됨"}[value],
            key=f"cc_v5_power_state_{owner}"),
        "stored_energy_status": st.selectbox("잔류 에너지 확인", ("UNKNOWN", "SAFE_CONFIRMED", "NOT_SAFE_OR_UNKNOWN"),
            format_func=lambda value: {"UNKNOWN": "미확인", "SAFE_CONFIRMED": "안전 여부 확인", "NOT_SAFE_OR_UNKNOWN": "안전 확인 불가/위험"}[value],
            key=f"cc_v5_stored_energy_{owner}"),
        "component_isolation_status": st.selectbox("소자 분리 상태 확인", ("UNKNOWN", "ISOLATED_CONFIRMED", "NOT_ISOLATED"),
            format_func=lambda value: {"UNKNOWN": "미확인", "ISOLATED_CONFIRMED": "측정 대상 분리 확인", "NOT_ISOLATED": "회로에 연결됨"}[value],
            key=f"cc_v5_isolation_{owner}"),
    }
    location_verified = st.checkbox("측정 위치와 지정 단자를 확인함",
                                    key=f"cc_v5_location_verified_{owner}")
    reference_source_verified = st.checkbox("정상 기준과 출처를 확인함",
                                             key=f"cc_v5_reference_verified_{owner}")
    measurement_conditions.update({
        "measurement_location_verified": bool(location_verified),
        "reference_source_verified": bool(reference_source_verified),
    })
    required_conditions = selected_profile.get("required_conditions", {})
    profile_safe_to_offer = bool(
        selected_profile.get("safety_protocol") and required_conditions
        and str(selected_profile.get("risk_level", "unknown")).casefold() in {"low", "none"}
    ) if selected_test_id else True
    safety_confirmed = (profile_safe_to_offer and all(
        measurement_conditions.get(field) == expected for field, expected in required_conditions.items())
        and location_verified and reference_source_verified) if selected_test_id else True
    if selected_test_id and not safety_confirmed:
        st.warning("검사 조건이 모두 확인되기 전에는 이 프로파일의 측정 기록을 저장할 수 없습니다.")
    measurement_type = st.selectbox(
        "측정 종류", TYPE_OPTIONS, format_func=MEASUREMENT_TYPE_LABELS.get,
        key=type_key,
        on_change=_reset_measurement_type_state,
        kwargs={"state": st.session_state, "type_key": type_key,
                "unit_key": unit_key, "sigma_key": sigma_key},
    )
    units = tuple(MEASUREMENT_UNITS[measurement_type])
    unit_key = f"cc_v5_unit_{owner}"
    if st.session_state.get(unit_key) not in units:
        st.session_state[unit_key] = units[0]
    if selected_test_id and selected_profile.get("measurement_type") != measurement_type:
        st.caption("프로파일 검사 종류와 다른 측정 종류를 선택했습니다. 이 값은 기록되지만 해당 프로파일 판정에는 사용되지 않습니다.")
    with st.form(f"cc_v5_add_measurement_{owner}"):
        value = st.text_input("측정값", key=f"cc_v5_value_{owner}", placeholder="예: 0, -0.25, 3.3")
        unit = st.selectbox("단위", units, key=unit_key)
        target = st.text_input("측정 대상/단자", placeholder="예: BZ1 양단 전압, U1 VCC-GND")
        uncertainty = uncertainty_source = ""
        reference_min = reference_max = reference_source = reference_source_id = ""
        reference_status = "UNVERIFIED"
        operating_conditions = reference_conditions = ""
        source_label = SOURCE_OPTIONS[0]
        if compact:
            operating_conditions = st.text_input(
                "측정 조건 (선택)", key=f"cc_v5_condition_{owner}",
                placeholder="예: 전원 차단, 주변 온도",
            )
        st.caption("필수 조건을 충족하지 못한 기록은 검증된 판정 근거가 되지 않습니다.")
        with st.expander("가설 검증 정보 추가" if compact else "검증 정보 · 정상 범위 · 출처", expanded=False):
            uncertainty = st.text_input(
                f"측정 불확실도 σ ({BASE_UNITS[measurement_type]})",
                key=f"cc_v5_sigma_{owner}", placeholder="계측기 정확도 사양에 따른 양수",
            )
            uncertainty_source = st.text_input("불확실도 출처", placeholder="예: 멀티미터 모델·정확도 사양")
            range_col_a, range_col_b = st.columns(2)
            with range_col_a:
                reference_min = st.text_input(f"정상 범위 하한 ({unit})")
            with range_col_b:
                reference_max = st.text_input(f"정상 범위 상한 ({unit})")
            reference_source = st.text_input("정상 범위 출처", placeholder="예: 제조사 데이터시트, 보드 사양")
            reference_source_id = st.text_input("기준 출처 ID", placeholder="예: 데이터시트 문서번호·보드 사양 항목")
            reference_status = st.selectbox("기준 출처", ("UNVERIFIED", "DATASHEET_VERIFIED", "BOARD_SPEC_VERIFIED"),
                                            format_func=lambda value: {
                                                "UNVERIFIED": "미검증",
                                                "DATASHEET_VERIFIED": "데이터시트 확인",
                                                "BOARD_SPEC_VERIFIED": "보드 사양 확인",
                                            }[value])
            if not compact:
                operating_conditions = st.text_input(
                    "실제 측정 조건", key=f"cc_v5_condition_{owner}",
                    placeholder="예: 전원 5 V, 무부하",
                )
            reference_conditions = st.text_input("기준 조건", placeholder="실제 측정 조건과 동일하게")
            if not compact:
                source_label = st.selectbox("측정 출처", SOURCE_OPTIONS, key=f"cc_v5_source_{owner}")
            else:
                st.caption("검사 기준·출처 등 검증 정보가 부족하면 기록은 남지만 판정은 보류됩니다.")
        reference_details_verified = bool(
            reference_status in {"DATASHEET_VERIFIED", "BOARD_SPEC_VERIFIED"}
            and str(reference_min).strip() and str(reference_max).strip()
            and str(reference_source).strip() and str(reference_source_id).strip()
        )
        reference_source_verified = bool(reference_source_verified and reference_details_verified)
        measurement_conditions.update({
            "measurement_location_verified": bool(location_verified),
            "reference_source_verified": reference_source_verified,
        })
        submitted = st.form_submit_button("측정 기록 저장", type="primary",
                                          disabled=bool(selected_test_id and not safety_confirmed))
    if submitted:
        try:
            if selected_test_id and selected_fault_id not in profile_relations:
                raise ValueError("선택한 검사와 고장 가설의 프로파일 관계가 일치하지 않습니다.")
            record = adapt_measurement_input(
                project_id=project.project_id, board_id=project.board_id,
                component_id=selected_component.component_id,
                component_identity_status=selected_component.identification_status,
                measurement_type=measurement_type, value=value, unit=unit,
                uncertainty=_sigma_for_input_unit(uncertainty, measurement_type, unit),
                source=SOURCE_CODES[source_label],
                component_map_version=component_map_version,
                component_type=selected_component.component_type,
                board_component_id=board_component_id,
                source_photo_id=selected_component.canonical_frame_id,
                source_photo_fingerprint=next((item.get("image_fingerprint")
                                                for item in (board_registry or {}).get("photos", [])
                                                if item.get("photo_id") == selected_component.canonical_frame_id), None),
                source_observation_fingerprint=current_source_observation_fingerprint(
                    board_registry or {}, photo_id=selected_component.canonical_frame_id,
                    component_id=selected_component.component_id,
                ),
                source_component_id=selected_component.component_id,
                measurement_location=target,
                identity_resolution_status=("VERIFIED_SOURCE_IDENTITY" if board_component_id and isinstance(board_registry, dict)
                                            else "UNRESOLVED"),
                measurement_target=target, fault_hypothesis=selected_fault_id or requested_fault_id or None,
                test_id=selected_test_id or None,
                reference_min=reference_min, reference_max=reference_max,
                reference_source=reference_source,
                reference_source_id=reference_source_id,
                reference_verification_status=reference_status,
                operating_conditions=operating_conditions,
                reference_conditions=reference_conditions,
                location_verified=location_verified,
                uncertainty_source=uncertainty_source,
                measurement_conditions=measurement_conditions,
            )
            save_measurement(st.session_state, record)
            st.success("측정 기록을 저장했습니다. 저장은 측정 검증 완료를 뜻하지 않습니다.")
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
                if isinstance(board_registry, dict) and st.button(
                    f"현재 선택 소자로 재연결 · {record.measurement_id[:8]}",
                    key=f"cc_reconfirm_measurement_{record.measurement_id}",
                ):
                    photo_id = selected_component.canonical_frame_id
                    photo = next((item for item in board_registry.get("photos", [])
                                  if item.get("photo_id") == photo_id), {})
                    observation_fp = current_source_observation_fingerprint(
                        board_registry, photo_id=photo_id, component_id=selected_component.component_id)
                    try:
                        reconfirm_measurement_identity(
                            st.session_state, measurement_id=record.measurement_id,
                            project_id=project.project_id, board_id=project.board_id,
                            board_component_id=board_component_id or "",
                            source_photo_id=photo_id or "",
                            source_component_id=selected_component.component_id,
                            source_photo_fingerprint=str(photo.get("image_fingerprint") or ""),
                            source_observation_fingerprint=observation_fp or "",
                            component_type=selected_component.component_type,
                            component_map_version=component_map_version,
                        )
                        st.success("사용자 확인 기록을 남기고 측정을 현재 관측에 연결했습니다.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))


def _render_record_editor(st: Any, *, project: ProjectContext, record: MeasurementRecord, owner: str) -> None:
    title = (f"{MEASUREMENT_TYPE_LABELS[record.measurement_type]} · "
             f"{record.entered_value:g} {record.entered_unit} · {record.timestamp[:16]}")
    with st.expander(title):
        check_reasons = _measurement_verification_gaps(record)
        if check_reasons:
            st.warning("기록됨 · 검증 대기 · " + " / ".join(check_reasons))
        else:
            st.success("필수 정보 입력 완료 · 진단 결과와 대조 대기")
        st.caption(f"출처 · {SOURCE_LABELS[record.source]} · 기록 상태 {record.verification_status}")
        st.caption(f"Component ID · {record.component_id} · 전기적 Mapping 미확인")
        if record.measurement_target or record.fault_hypothesis:
            st.caption(f"검사 대상 · {record.measurement_target or '미지정'} · 가설 {record.fault_hypothesis or '미지정'}")
        if record.reference_min is not None and record.reference_max is not None:
            st.caption(f"등록 기준 · {record.reference_min:g}–{record.reference_max:g} {record.unit} · "
                       f"{record.reference_source or '출처 없음'} · 조건 {record.operating_conditions or '미지정'}")
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
        knowledge = load_knowledge()
        fault_modes = knowledge.get("fault_modes", {}).get("fault_modes", {})
        type_profile = knowledge.get("component_test_profiles", {}).get("profiles", {}).get(
            record.component_type.casefold(), {}
        )
        test_profiles = type_profile.get("tests", {})
        test_options = [""] + list(test_profiles)
        selected_test = record.test_id if record.test_id in test_profiles else ""
        with st.form(f"cc_v5_edit_form_{record.measurement_id}"):
            value = st.text_input("수정 측정값", value=f"{record.entered_value:g}",
                                  key=f"cc_v5_edit_value_{record.measurement_id}")
            unit = st.selectbox("수정 단위", units, key=unit_key)
            target = st.text_input("측정 대상/단자", value=record.measurement_target or "",
                                   key=f"cc_v5_edit_target_{record.measurement_id}")
            source_label = st.selectbox(
                "측정 출처", SOURCE_OPTIONS, index=SOURCE_OPTIONS.index(SOURCE_LABELS[record.source]),
                key=f"cc_v5_edit_source_{record.measurement_id}",
            )
            sigma_default = ""
            if edit_type == record.measurement_type and record.uncertainty is not None:
                sigma_default = f"{record.uncertainty:g}"
            with st.expander("고급 검증 조건 수정", expanded=False):
                previous_conditions = record.measurement_conditions or {}
                measurement_conditions = {
                    "power_state": st.selectbox("전원 상태", ("UNKNOWN", "POWER_OFF_CONFIRMED", "POWER_ON_CONFIRMED"),
                        index=("UNKNOWN", "POWER_OFF_CONFIRMED", "POWER_ON_CONFIRMED").index(previous_conditions.get("power_state", "UNKNOWN")),
                        format_func=lambda value: {"UNKNOWN": "미확인", "POWER_OFF_CONFIRMED": "전원 분리 확인", "POWER_ON_CONFIRMED": "전원 연결됨"}[value],
                        key=f"cc_v5_edit_power_state_{record.measurement_id}"),
                    "stored_energy_status": st.selectbox("잔류 에너지", ("UNKNOWN", "SAFE_CONFIRMED", "NOT_SAFE_OR_UNKNOWN"),
                        index=("UNKNOWN", "SAFE_CONFIRMED", "NOT_SAFE_OR_UNKNOWN").index(previous_conditions.get("stored_energy_status", "UNKNOWN")),
                        format_func=lambda value: {"UNKNOWN": "미확인", "SAFE_CONFIRMED": "안전 확인 완료", "NOT_SAFE_OR_UNKNOWN": "안전 확인 불가/위험"}[value],
                        key=f"cc_v5_edit_stored_energy_{record.measurement_id}"),
                    "component_isolation_status": st.selectbox("소자 분리 상태", ("UNKNOWN", "ISOLATED_CONFIRMED", "NOT_ISOLATED"),
                        index=("UNKNOWN", "ISOLATED_CONFIRMED", "NOT_ISOLATED").index(previous_conditions.get("component_isolation_status", "UNKNOWN")),
                        format_func=lambda value: {"UNKNOWN": "미확인", "ISOLATED_CONFIRMED": "측정 대상 분리 확인", "NOT_ISOLATED": "회로에 연결됨"}[value],
                        key=f"cc_v5_edit_isolation_{record.measurement_id}"),
                }
                uncertainty = st.text_input(
                    f"불확실도 σ ({BASE_UNITS[edit_type]})", value=sigma_default,
                    key=f"cc_v5_edit_sigma_{record.measurement_id}",
                )
                uncertainty_source = st.text_input(
                    "불확실도 출처", value=record.uncertainty_source or "",
                    key=f"cc_v5_edit_sigma_source_{record.measurement_id}",
                )
                test_id = st.selectbox(
                    "부품별 검사 항목", test_options,
                    index=test_options.index(selected_test),
                    format_func=lambda item: "프로파일 검사 선택 안 함" if not item else item,
                    key=f"cc_v5_edit_test_{record.measurement_id}",
                )
                relations = test_profiles.get(test_id, {}).get("fault_relations", {})
                fault_options = [""] + [item for item in fault_modes if item in relations]
                fault_default = record.fault_hypothesis if record.fault_hypothesis in fault_options else ""
                fault_hypothesis = st.selectbox(
                    "검사할 고장 가설", fault_options,
                    index=fault_options.index(fault_default),
                    format_func=lambda code: "선택 안 함" if not code else fault_modes[code]["title"],
                    key=f"cc_v5_edit_fault_{record.measurement_id}",
                )
                multiplier = MEASUREMENT_UNITS[edit_type][unit]
                reference_min_default = ""
                reference_max_default = ""
                if edit_type == record.measurement_type:
                    if record.reference_min is not None:
                        reference_min_default = f"{record.reference_min / multiplier:g}"
                    if record.reference_max is not None:
                        reference_max_default = f"{record.reference_max / multiplier:g}"
                range_left, range_right = st.columns(2)
                with range_left:
                    reference_min = st.text_input(
                        f"정상 범위 하한 ({unit})", value=reference_min_default,
                        key=f"cc_v5_edit_ref_min_{record.measurement_id}",
                    )
                with range_right:
                    reference_max = st.text_input(
                        f"정상 범위 상한 ({unit})", value=reference_max_default,
                        key=f"cc_v5_edit_ref_max_{record.measurement_id}",
                    )
                reference_source = st.text_input(
                    "정상 범위 출처", value=record.reference_source or "",
                    key=f"cc_v5_edit_ref_source_{record.measurement_id}",
                )
                reference_source_id = st.text_input(
                    "기준 출처 ID", value=record.reference_source_id or "",
                    key=f"cc_v5_edit_ref_source_id_{record.measurement_id}",
                )
                reference_status_options = ("UNVERIFIED", "DATASHEET_VERIFIED", "BOARD_SPEC_VERIFIED")
                current_reference_status = record.reference_verification_status
                if current_reference_status not in reference_status_options:
                    current_reference_status = "UNVERIFIED"
                reference_status = st.selectbox(
                    "기준 출처 검증", reference_status_options,
                    index=reference_status_options.index(current_reference_status),
                    format_func=lambda item: {"UNVERIFIED": "미검증", "DATASHEET_VERIFIED": "데이터시트 확인", "BOARD_SPEC_VERIFIED": "보드 사양 확인"}[item],
                    key=f"cc_v5_edit_ref_status_{record.measurement_id}",
                )
                operating_conditions = st.text_input(
                    "측정 조건", value=record.operating_conditions or "",
                    key=f"cc_v5_edit_operating_{record.measurement_id}",
                )
                reference_conditions = st.text_input(
                    "기준 조건", value=record.reference_conditions or "",
                    key=f"cc_v5_edit_reference_conditions_{record.measurement_id}",
                )
                location_verified = st.checkbox(
                    "측정 지점이 지정 단자임을 확인함", value=record.location_verified,
                    key=f"cc_v5_edit_location_{record.measurement_id}",
                )
            reference_source_verified = bool(
                reference_status in {"DATASHEET_VERIFIED", "BOARD_SPEC_VERIFIED"}
                and str(reference_min).strip() and str(reference_max).strip()
                and str(reference_source).strip() and str(reference_source_id).strip()
            )
            measurement_conditions.update({
                "measurement_location_verified": bool(location_verified),
                "reference_source_verified": reference_source_verified,
            })
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
                    component_type=record.component_type,
                    measurement_id=record.measurement_id, timestamp=record.timestamp,
                    measurement_target=target,
                    fault_hypothesis=fault_hypothesis or None,
                    test_id=test_id or None,
                    reference_min=reference_min,
                    reference_max=reference_max,
                    reference_source=reference_source,
                    reference_source_id=reference_source_id,
                    reference_verification_status=reference_status,
                    operating_conditions=operating_conditions,
                    reference_conditions=reference_conditions,
                    location_verified=location_verified,
                    uncertainty_source=uncertainty_source,
                    measurement_conditions=measurement_conditions,
                )
                updated = replace(
                    updated, board_component_id=record.board_component_id,
                    source_photo_id=record.source_photo_id,
                    source_photo_fingerprint=record.source_photo_fingerprint,
                    source_observation_fingerprint=record.source_observation_fingerprint,
                    source_component_id=record.source_component_id,
                    measurement_location=target or None,
                    identity_resolution_status=record.identity_resolution_status,
                    component_type=record.component_type,
                )
                updated = replace(updated, changed_at=datetime.now(timezone.utc).isoformat())
                update_measurement(st.session_state, updated)
                st.session_state[f"cc_v5_location_verified_{owner}"] = bool(location_verified)
                st.session_state[f"cc_v5_reference_verified_{owner}"] = reference_source_verified
                st.success("측정 기록을 수정했습니다. 기존 진단 결과는 무효화되었습니다.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if delete:
            delete_measurement(st.session_state, project_id=project.project_id,
                               board_id=project.board_id or "", measurement_id=record.measurement_id)
            st.success("측정 기록을 삭제했습니다. 기존 진단 결과는 무효화되었습니다.")
            st.rerun()
