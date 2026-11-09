"""User-approved rating application and revoke presentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .rating_adapter import (
    RatingApplicationError,
    apply_approved_rating,
    reassess_existing_risk,
)
from .rating_review import applied_rating_session_key, save_application
from .ui_helpers import _comparison_row, _mapping_for_case, _risk_row, _specs_for_case


def render_rating_application(st: Any, *, root: Path, case: dict, risk: dict,
                              stress: dict, component_ref: str, selected_index: int,
                              ds_state: dict, conditions_fit: bool, context_key: str,
                              state_id: str, stress_signature: str | None,
                              current_risk: dict | None) -> None:
    approved = ds_state.get("reviews", {}).get(str(selected_index))
    if approved and approved.get("decision") == "approved":
        if not risk or not stress:
            st.warning("현재 선택한 시나리오에 SPICE 스트레스/위험 결과가 없어 재평가할 수 없습니다.")
        elif st.button("승인 정격 적용 및 위험 재평가", key=f"ds_apply_{state_id}_{selected_index}"):
            try:
                applied_specs = apply_approved_rating(
                    _specs_for_case(root, case), approved, component_ref=component_ref,
                    selected_reference=component_ref,
                    mpn_match_status=approved.get("mpn_match_status", "PART_MATCH_UNCERTAIN"),
                    conditions_compatible=conditions_fit,
                )
                reassess_existing_risk(risk, stress, applied_specs, _mapping_for_case(root, case))
                save_application({"record": approved, "conditions_compatible": conditions_fit,
                                  "component_ref": component_ref, "stress_signature": stress_signature,
                                  "circuit_path": case["netlist"], "scenario_id": risk.get("scenario_id")},
                                 root / "data/private/datasheet_ai/applied_ratings.json")
                st.session_state.pop(applied_rating_session_key(context_key, stress_signature), None)
                st.success("기존 SPICE 결과를 재사용해 Rating Engine을 다시 계산했습니다. SPICE 재실행은 하지 않았습니다.")
                st.rerun()
            except RatingApplicationError as exc:
                st.error(str(exc))
    if not current_risk:
        return
    st.subheader("변경 전 / 현재 활성 평가")
    old = _risk_row(risk, component_ref)
    new = _risk_row(current_risk, component_ref)
    st.dataframe([_comparison_row("기존 정격", old), _comparison_row("승인 데이터시트 정격", new)], hide_index=True)
    st.caption("정격의 출처·조건만 재평가합니다. 파형, SPICE 스트레스, 측정값, PCB 좌표/검출 상자는 변경하지 않습니다.")
    st.write({"이전 등급": old.get("risk_level"), "현재 등급": new.get("risk_level"),
              "이전 점수 / 순위": [old.get("priority_score"), old.get("priority_rank")],
              "현재 점수 / 순위": [new.get("priority_score"), new.get("priority_rank")],
              "점수 의미": "결정론적 점검 우선순위 지수이며 손상 확률이 아닙니다."})
    if current_risk.get("reassessment_source"):
        st.json({"reassessment_source": current_risk["reassessment_source"],
                 "simulation_rerun": current_risk.get("simulation_rerun"),
                 "rating_check": next((row for row in current_risk.get("rating_checks", [])
                                       if row.get("component_ref") == component_ref), None)})
        if st.button("승인 정격 적용 해제", key=f"ds_unapply_{state_id}_{component_ref}"):
            save_application({"event": "revoked", "component_ref": component_ref,
                              "circuit_path": case["netlist"],
                              "scenario_id": current_risk.get("scenario_id"),
                              "stress_signature": stress_signature},
                             root / "data/private/datasheet_ai/applied_ratings.json")
            st.session_state.pop(applied_rating_session_key(context_key, stress_signature), None)
            st.rerun()
