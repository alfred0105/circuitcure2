"""Candidate review controls with the existing human approval gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .evidence_validator import validate_candidate
from .rating_review import save_review
from .ui_helpers import _mpn_in_pdf


def render_candidate_review(st: Any, *, root: Path, selected: dict,
                            document: dict, manufacturer: str, mpn: str,
                            revision: str, component_ref: str, state_id: str,
                            selected_index: int, ds_state: dict,
                            source_status: str | None) -> bool:
    """Render review inputs, persist user decisions, and return condition confirmation."""
    component_match = st.checkbox(
        f"이 PDF의 제조사 모델 {mpn or '(MPN 입력 필요)'}가 실제 {component_ref} 부품이라는 점을 확인했습니다.",
        key=f"ds_match_{state_id}",
    )
    conditions_fit = st.checkbox(
        "데이터시트 적용 조건(온도/연속·펄스/전압 기준)이 현재 회로와 시뮬레이션 조건에 적합함을 확인했습니다.",
        key=f"ds_conditions_{state_id}",
    )
    condition_note = st.text_area(
        "조건 검토 메모 (문서에 명시되지 않은 조건을 적용할 때 근거를 기록)",
        key=f"ds_condition_note_{state_id}",
    )
    source_text = st.text_area(
        "원문 인용 수정 (수정하면 출처 일치 상태가 다시 확인되어야 함)",
        value=selected.get("source_text", ""), key=f"ds_evidence_{state_id}_{selected_index}",
    )
    unit_options = {"max_abs_voltage_v": ["V", "mV", "kV"],
                    "max_current_a": ["A", "mA", "µA"],
                    "max_power_w": ["W", "mW", "µW"]}[selected["rating_key"]]
    scale = {"V": 1.0, "mV": 1e-3, "kV": 1e3, "A": 1.0, "mA": 1e-3,
             "µA": 1e-6, "W": 1.0, "mW": 1e-3, "µW": 1e-6}
    default_unit = selected["normalized_unit"]
    display_value = float(selected["normalized_value"]) / scale[default_unit]
    edit = st.columns(3)
    unit = edit[0].selectbox("검토 단위", unit_options, index=unit_options.index(default_unit),
                             key=f"ds_unit_{state_id}_{selected_index}")
    value = edit[1].number_input("검토 수치", min_value=0.000000001, value=display_value,
                                 format="%.9g", key=f"ds_value_{state_id}_{selected_index}")
    kinds = ["continuous", "peak", "absolute_max", "unknown"]
    current_kind = selected.get("rating_kind")
    rating_kind = edit[2].selectbox(
        "정격 적용 종류", kinds, index=kinds.index(current_kind if current_kind in kinds else "unknown"),
        key=f"ds_kind_{state_id}_{selected_index}",
    )
    actions = st.columns(3)
    if actions[0].button("승인 기록", key=f"ds_approve_{state_id}_{selected_index}"):
        revised = {**selected, "source_text": source_text, "raw_value": f"{value:g} {unit}",
                   "normalized_value": float(value) * scale[unit], "normalized_unit": unit_options[0],
                   "rating_kind": rating_kind, "rating_kind_source": "user_reviewed",
                   "condition_review_note": condition_note.strip(),
                   "source_document_sha256": document["sha256"], "manufacturer": manufacturer.strip(),
                   "mpn": mpn.strip(), "component_reference": component_ref,
                   "document_revision": revision.strip(), "human_approved": True,
                   "review_decision": "approved",
                   "mpn_match_status": "MPN_CONFIRMED" if _mpn_in_pdf(mpn, document) else
                   ("USER_ASSERTED_PART_MATCH" if component_match else "PART_MATCH_UNCERTAIN"),
                   "conditions_compatible_user_confirmed": conditions_fit}
        revised = validate_candidate(revised, document["pages"], manufacturer=manufacturer, mpn=mpn)
        if not revised["manufacturer"] or not revised["mpn"]:
            st.error("제조사와 실제 제조사 MPN을 입력해야 합니다.")
        elif revised["evidence_status"] not in {"SOURCE_MATCHED", "CONDITION_INCOMPLETE"}:
            st.error("원문/수정값 검증 상태가 승인 가능한 값이 아닙니다. 적용을 막았습니다.")
        elif revised["evidence_status"] == "CONDITION_INCOMPLETE" and not revised.get("condition_review_note"):
            st.error("문서에 적용 조건이 충분히 나오지 않습니다. 근거가 있는 조건 검토 메모를 먼저 작성하세요.")
        elif not component_match:
            st.error("KiCad Reference와 실제 MPN의 일치 확인이 필요합니다.")
        else:
            saved = save_review({**revised, "decision": "approved"},
                                root / "data/private/datasheet_ai/review_history.json")
            ds_state.setdefault("reviews", {})[str(selected_index)] = saved
            st.success("사용자 승인 이력을 로컬에 기록했습니다. 아직 적용 전입니다.")
    if actions[1].button("거절 기록", key=f"ds_reject_{state_id}_{selected_index}"):
        saved = save_review({**selected, "component_reference": component_ref,
                             "manufacturer": manufacturer, "mpn": mpn,
                             "source_document_sha256": document["sha256"], "decision": "rejected"},
                            root / "data/private/datasheet_ai/review_history.json")
        ds_state.setdefault("reviews", {})[str(selected_index)] = saved
        st.warning("거절 이력을 기록했습니다.")
    if actions[2].button("문서 미검증 수동 입력으로 보관", key=f"ds_manual_{state_id}_{selected_index}"):
        manual = {**selected, "component_reference": component_ref,
                  "manufacturer": manufacturer.strip(), "mpn": mpn.strip(),
                  "source_document_sha256": document["sha256"],
                  "original_candidate": {key: selected.get(key) for key in
                      ("rating_key", "rating_type", "rating_kind", "normalized_value", "normalized_unit",
                       "source_text", "table_context", "pdf_page_index")},
                  "raw_value": f"{value:g} {unit}", "normalized_value": float(value) * scale[unit],
                  "normalized_unit": unit_options[0], "rating_kind": rating_kind,
                  "decision": "manual_input", "extraction_status": "manual_user_input",
                  "evidence_status": source_status, "human_approved": False,
                  "record_note": "User-entered/edited value is not verified as a datasheet fact and cannot be applied by the datasheet adapter."}
        save_review(manual, root / "data/private/datasheet_ai/review_history.json")
        st.info("비검증 수동 입력 이력을 별도로 저장했습니다. 데이터시트 정격으로 표시하거나 기존 위험도에 적용하지 않았습니다.")
    return conditions_fit
