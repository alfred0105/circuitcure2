"""Shared verification requirements for UI and diagnostic measurement use."""

from __future__ import annotations

from math import isfinite
from typing import Any


VERIFIED_REFERENCE_STATES = {"DATASHEET_VERIFIED", "BOARD_SPEC_VERIFIED"}


def measurement_verification_gaps(
    row: Any, *, profiles: dict[str, Any] | None = None,
    current_context: dict[str, Any] | None = None,
) -> list[str]:
    """Return the single authoritative list of requirements before a reading can
    support/refute a particular hypothesis. Circuit/node mapping is deliberately
    not required: this product does not reconstruct electrical connectivity.
    """
    get = row.get if isinstance(row, dict) else lambda key, default=None: getattr(row, key, default)
    gaps: list[str] = []
    if not str(get("board_component_id") or "").strip():
        gaps.append("보드 소자 연결 미확인")
    identity = str(get("identity_resolution_status") or "")
    component_identity = str(get("component_identity_status") or "")
    if not (identity.startswith(("USER_SELECTED_VERIFIED", "VERIFIED_"))
            or component_identity == "USER_CONFIRMED"):
        gaps.append("보드 소자 동일성 확인 전")
    if not str(get("component_id") or "").strip():
        gaps.append("소자 ID 미지정")
    if not str(get("measurement_target") or get("measurement_location") or "").strip():
        gaps.append("측정 대상/단자 미지정")
    if get("location_verified") is not True:
        gaps.append("측정 지점 확인 전")
    if get("source") != "user_measurement" or get("verification_status") != "USER_REPORTED":
        gaps.append("실측 출처 미확인")
    if not str(get("test_id") or "").strip():
        gaps.append("검사 프로파일 미선택")
    if get("reference_verification_status") not in VERIFIED_REFERENCE_STATES:
        gaps.append("검증된 정상 범위·출처 없음")
    if get("reference_min") is None or get("reference_max") is None or not str(get("reference_source_id") or "").strip() or not str(get("reference_source") or "").strip():
        if "검증된 정상 범위·출처 없음" not in gaps:
            gaps.append("검증된 정상 범위·출처 없음")
    sigma = get("uncertainty")
    if sigma is None or not str(get("uncertainty_source") or "").strip():
        gaps.append("측정 불확실도 또는 출처 미입력")
    else:
        try:
            if not isfinite(float(sigma)) or float(sigma) <= 0:
                gaps.append("측정 불확실도 또는 출처 미입력")
        except (TypeError, ValueError, OverflowError):
            gaps.append("측정 불확실도 또는 출처 미입력")
    conditions = get("operating_conditions") or get("measurement_conditions")
    references = get("reference_conditions")
    if not conditions or not references or conditions != references:
        gaps.append("측정 조건과 기준 조건이 일치하지 않음")
    if not str(get("fault_hypothesis") or "").strip():
        gaps.append("검사할 고장 가설 미지정")

    context = current_context or {}
    if context and any(get(key) != context.get(key) for key in ("project_id", "board_id")):
        gaps.append("프로젝트·보드가 현재 분석과 다릅니다.")
    if context.get("component_map_version") and get("component_map_version") != context.get("component_map_version"):
        gaps.append("소자 Registry 버전이 현재 분석과 다릅니다.")
    expected_status = context.get("component_identity_statuses", {}).get(get("component_id"))
    if expected_status is not None and expected_status != get("component_identity_status"):
        gaps.append("현재 보드의 소자 식별 상태와 측정 기록이 다릅니다.")

    profiles_map = (profiles or {}).get("profiles", {})
    component_type = str(get("component_type") or "unknown").casefold()
    test_id = str(get("test_id") or "")
    test_profile = profiles_map.get(component_type, {}).get("tests", {}).get(test_id, {})
    if test_id:
        if not test_profile:
            gaps.append("이 부품 종류의 검사 프로파일에 등록되지 않은 검사입니다.")
        else:
            if test_profile.get("measurement_type") != get("measurement_type"):
                gaps.append("검사 프로파일의 측정 종류와 기록의 측정 종류가 다릅니다.")
            target = str(get("measurement_target") or get("measurement_location") or "").casefold()
            if not any(str(term).casefold() in target for term in test_profile.get("target_keywords", [])):
                gaps.append("측정 대상이 검사 프로파일에 지정된 단자/위치와 맞지 않습니다.")
            required_conditions = test_profile.get("required_conditions", {})
            actual_conditions = get("measurement_conditions")
            if required_conditions:
                actual_conditions = actual_conditions if isinstance(actual_conditions, dict) else {}
                for field, required in required_conditions.items():
                    if actual_conditions.get(field) != required:
                        labels = {
                            "power_state": "전원 차단 상태 확인 전",
                            "stored_energy_status": "잔류 에너지 안전 확인 전",
                            "component_isolation_status": "측정 대상 분리 상태 확인 전",
                            "measurement_location_verified": "측정 위치·단자 확인 전",
                            "reference_source_verified": "검증된 정상 기준·출처 입력 전",
                        }
                        gaps.append(labels.get(field, f"측정 안전 조건 미충족 · {field}"))
            elif test_profile.get("condition_keywords"):
                # Legacy profiles used free text. Exact token matching avoids
                # treating negated phrases such as '전원 차단 안 함' as safe.
                condition_values = {str(value).strip().casefold() for value in
                                    (conditions if isinstance(conditions, (list, tuple, set)) else [conditions])
                                    if value is not None}
                if not condition_values.intersection(str(term).strip().casefold()
                                                     for term in test_profile.get("condition_keywords", [])):
                    gaps.append("운전 조건이 검사 프로파일에서 요구하는 조건을 포함하지 않습니다.")
            if test_profile.get("requires_reference_verification") and get("reference_verification_status") not in VERIFIED_REFERENCE_STATES:
                gaps.append("검사 프로파일 기준 출처 확인 전")
            fault = str(get("fault_hypothesis") or "")
            if fault not in test_profile.get("fault_relations", {}):
                gaps.append("이 검사 항목은 선택한 고장 가설을 판별하지 않습니다.")
    return list(dict.fromkeys(gaps))


def verified_measurements_for_inference(
    rows: list[dict[str, Any]] | None, *, profiles: dict[str, Any] | None = None,
    current_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Exclude structurally unverified readings before candidate ranking.

    Range-valid contradictions stay in the list so the evaluator can mark them
    CONFLICTED as a group. Incomplete records remain visible as diagnostic
    evidence explanations, but cannot alter rankings.
    """
    return [row for row in rows or [] if isinstance(row, dict)
            and not measurement_verification_gaps(row, profiles=profiles, current_context=current_context)]
