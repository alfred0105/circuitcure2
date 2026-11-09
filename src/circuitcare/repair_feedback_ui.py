"""Local Streamlit workflow for repair cases and approved prior learning."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .fault_ai.hypothesis_ranking import conditional_bayesian_update
from .repair_feedback.case_repository import RepairCaseRepository
from .repair_feedback.diagnosis_feedback import create_diagnosis_snapshot
from .repair_feedback.evidence_review import review_case
from .repair_feedback.models import new_repair_case
from .repair_feedback.outcome_tracker import compare_before_after
from .repair_feedback.prior_estimator import estimate_prior
from .repair_feedback.prior_registry import (activate_prior, active_prior, approve_prior,
    deactivate_prior, propose_prior, rollback_prior)
from .repair_feedback.training_eligibility import assess_training_eligibility


def render_repair_feedback_tab(st: Any, *, root: Path, circuit_context: dict[str, Any],
                               fault_state: dict[str, Any] | None = None) -> None:
    st.subheader("수리 이력 및 AI 학습")
    st.caption("수리 사례는 이 기기에만 저장됩니다. 제안이나 예측은 실제 수리 근거가 아니며, 학습 Prior는 검토·승인 후에만 사용할 수 있습니다.")
    repository = RepairCaseRepository(root / "data/private/repair_feedback/repair_cases.sqlite3")
    cases = repository.list_cases()
    verified = sum(case.get("training_eligibility", {}).get("status") == "ELIGIBLE" for case in cases)
    cards = st.columns(3)
    cards[0].metric("저장된 수리 사례", len(cases))
    cards[1].metric("학습 적격 검증 사례", verified)
    cards[2].metric("실제 수리 기반 Prior", "있음" if repository.list_prior_models() else "없음")
    if not verified:
        st.info("No Verified Repair Cases · 실제 수리 이력이 검증·승인되기 전에는 학습 Prior를 생성하지 않습니다.")
    pages = st.tabs(["사례 등록", "검증 및 적격성", "수리 이력", "Prior 학습 및 승인", "진단 비교"])
    with pages[0]:
        _render_registration(st, repository, root=root, circuit_context=circuit_context, fault_state=fault_state)
    with pages[1]:
        _render_verification(st, repository)
    with pages[2]:
        _render_history(st, repository)
    with pages[3]:
        _render_prior_learning(st, repository)
    with pages[4]:
        _render_diagnosis_comparison(st, repository, fault_state=fault_state,
                                     group_id=circuit_context.get("circuit_family", ""))


def _render_registration(st: Any, repository: RepairCaseRepository, *, root: Path,
                         circuit_context: dict[str, Any], fault_state: dict[str, Any] | None) -> None:
    st.markdown("#### Repair Case 등록")
    st.caption("진단 Snapshot은 등록 시점에 복사해 보존합니다. 이미지·장비 ID·개인 식별정보는 기본 저장하지 않습니다.")
    with st.form("repair_feedback_register_form"):
        circuit_id = st.text_input("회로 식별자", value=circuit_context.get("circuit_id", ""))
        circuit_family = st.text_input("비교 가능한 회로군", value=circuit_context.get("circuit_family", ""),
            help="Prior 학습은 같은 명시적 회로군 안에서만 비교합니다.")
        circuit_hash = st.text_input("회로 버전 Hash", value=circuit_context.get("circuit_hash", ""))
        source = st.selectbox("기록 출처", ["user_reported_repair", "real_repair", "simulated_fault", "synthetic_demo"],
            help="실물 수리를 직접 확인한 뒤에만 real_repair로 표시하세요.")
        symptoms_text = st.text_area("보고된 증상", placeholder="출력 전압이 비정상적으로 높음")
        cols = st.columns(4)
        node = cols[0].text_input("측정 노드", value="VOUT")
        before = cols[1].number_input("수리 전 값", value=0.0, format="%.6g")
        after = cols[2].number_input("수리 후 값", value=0.0, format="%.6g")
        uncertainty = cols[3].number_input("측정 σ", min_value=0.0001, value=0.2, format="%.6g")
        before_recorded = st.checkbox("수리 전 측정값 기록", value=False)
        after_recorded = st.checkbox("수리 후 측정값 기록", value=False)
        unit = st.text_input("단위", value="V")
        measurement_source = st.selectbox("측정 출처", ["user_measurement", "simulated_demo"],
            help="시뮬레이션/가상 측정은 실제 수리 학습에 사용되지 않습니다.")
        conditions_json = st.text_input("측정 조건 JSON (선택)", value="{}")
        after_conditions_json = st.text_input("수리 후 측정 조건 JSON", value="{}")
        limits = st.columns(2)
        expected_minimum = limits[0].number_input("정상 범위 최소 (선택)", value=0.0, format="%.6g")
        expected_maximum = limits[1].number_input("정상 범위 최대 (선택)", value=0.0, format="%.6g")
        cols = st.columns(3)
        refs = circuit_context.get("component_refs", [])
        component_ref = cols[0].selectbox("조치 부품 Reference", ["(없음)", *refs]) if refs else cols[0].text_input("조치 부품 Reference", value="")
        action = cols[1].selectbox("수리 조치", ["component_replaced", "component_removed", "solder_rework", "connector_reconnected", "wiring_modified", "power_check", "additional_measurement", "other"])
        intervention_reason = cols[2].text_input("조치 이유", value="")
        inspection_result = st.text_input("실물 점검 결과 (선택)", value="")
        passed = st.checkbox("수리 후 기능 시험 통과", value=False)
        note = st.text_area("조치 메모", placeholder="사용자가 입력한 정비 기록")
        submitted = st.form_submit_button("로컬 수리 사례 저장")
    if not submitted:
        return
    try:
        conditions = json.loads(conditions_json or "{}")
        if not isinstance(conditions, dict):
            raise ValueError("측정 조건은 JSON 객체여야 합니다.")
        after_conditions = json.loads(after_conditions_json or "{}")
        if not isinstance(after_conditions, dict):
            raise ValueError("수리 후 측정 조건은 JSON 객체여야 합니다.")
        diagnosis = (fault_state or {}).get("bayesian_result") or {}
        snapshot = create_diagnosis_snapshot(circuit_id=circuit_id, circuit_hash=circuit_hash,
            diagnosis=diagnosis, hypothesis_rows=(fault_state or {}).get("hypotheses", []),
            model_version=((fault_state or {}).get("llm_metadata") or {}).get("model", "unknown"),
            prompt_version=((fault_state or {}).get("llm_metadata") or {}).get("prompt_version"),
            model_info=((fault_state or {}).get("llm_metadata") or {}).get("model_info"),
            fault_scenario={"context_key": (fault_state or {}).get("context_key"),
                "hypothesis_definitions": (fault_state or {}).get("hypotheses", [])})
        observations = []
        post_tests = []
        if before_recorded:
            observations.append({"phase": "before", "node": node, "value": float(before), "unit": unit,
                "uncertainty": float(uncertainty), "source": measurement_source, "conditions": conditions})
        if after_recorded:
            after_measurement = {"phase": "after", "node": node, "value": float(after), "unit": unit,
                "uncertainty": float(uncertainty), "source": measurement_source, "conditions": after_conditions}
            post_tests.append({"test_id": "post_repair_function_test", "passed": bool(passed),
                "source": measurement_source, "observations": [after_measurement], "conditions": conditions})
        intervention = [{"component_ref": component_ref.strip() if component_ref != "(없음)" else None, "action": action,
            "reason": intervention_reason, "note": note, "source": "user_reported_action"}]
        case = new_repair_case(circuit_id=circuit_id, circuit_hash=circuit_hash,
            circuit_family=circuit_family, data_source=source,
            component_refs=circuit_context.get("component_refs", []),
            reported_symptoms=[line.strip() for line in symptoms_text.splitlines() if line.strip()],
            observations=observations, diagnosis_snapshot=snapshot, interventions=intervention,
            post_repair_tests=post_tests, hypothesis_set_id=snapshot.get("hypothesis_set_id"),
            expected_ranges=({node: {"minimum": float(expected_minimum), "maximum": float(expected_maximum)}}
                if expected_maximum > expected_minimum else {}),
            inspection_events=([{"component_ref": component_ref if component_ref != "(없음)" else None,
                "finding": inspection_result, "source": "user_reported_inspection"}] if inspection_result.strip() else []))
        repository.create_case(case)
        st.success(f"사례 {case['case_id']}를 로컬 데이터베이스에 저장했습니다. 상태: UNVERIFIED")
        st.json({"case_id": case["case_id"], "verification_status": case["verification_status"],
                 "training_eligibility": case["training_eligibility"], "snapshot_id": snapshot["snapshot_id"]})
    except (ValueError, TypeError, json.JSONDecodeError, KeyError) as exc:
        st.error(f"사례를 저장하지 못했습니다: {exc}")
    except (sqlite3.Error, OSError) as exc:
        # SQLite duplicate IDs and disk errors surface as storage failures without changing existing data.
        st.error(f"로컬 저장 실패: {exc}")


def _render_verification(st: Any, repository: RepairCaseRepository) -> None:
    cases = repository.list_cases()
    if not cases:
        st.info("검토할 수리 사례가 없습니다.")
        return
    case_ids = [case["case_id"] for case in cases]
    selected_id = st.selectbox("검토할 사례", case_ids, key="repair_case_review_select")
    case = repository.get_case(selected_id)
    st.json({"source": case["data_source"], "circuit": case["circuit_id"], "diagnosis_snapshot": case.get("diagnosis_snapshot"),
        "observations": case.get("observations"), "interventions": case.get("interventions"), "post_repair_tests": case.get("post_repair_tests")})
    with st.form(f"repair_review_{selected_id}"):
        reviewer = st.text_input("검토자 이름 또는 이니셜")
        verification = st.selectbox("검증 수준", ["INSUFFICIENT_EVIDENCE", "MEASUREMENT_SUPPORTED", "REPAIR_OUTCOME_VERIFIED", "ROOT_CAUSE_REVIEWED", "CONFLICTING_EVIDENCE"])
        outcome = st.selectbox("수리 결과", ["UNKNOWN", "RESOLVED", "PARTIALLY_RESOLVED", "NOT_RESOLVED", "RECURRENCE"])
        known_refs = case.get("component_refs", [])
        root_ref = st.selectbox("확인된 원인 부품 Reference", ["", *known_refs]) if known_refs else st.text_input("확인된 원인 부품 Reference")
        fault_type = st.text_input("확인된 고장 유형")
        hypothesis_id = st.text_input("일치하는 진단 가설 ID")
        cause_verified = st.checkbox("물리적 검사 근거로 root cause를 확인함")
        isolated = st.checkbox("복수 조치가 있었지만 후속 시험으로 원인을 분리함")
        approve = st.checkbox("이 사례를 Prior 학습 후보로 검토 승인 (적격성 검사 별도)")
        note = st.text_area("검토 근거")
        submit = st.form_submit_button("검토 이벤트 저장")
    if submit:
        if not reviewer.strip():
            st.error("검토자 식별 정보를 입력하세요.")
            return
        root_cause = None
        if root_ref.strip() or fault_type.strip():
            root_cause = {"component_ref": root_ref.strip(), "fault_type": fault_type.strip(),
                "hypothesis_id": hypothesis_id.strip(), "verified": bool(cause_verified),
                "isolated_by_followup_test": bool(isolated)}
        try:
            review_case(repository, selected_id, verification_status=verification, outcome=outcome,
                        reviewer=reviewer.strip(), note=note, root_cause=root_cause, approve_for_training=approve)
            reviewed = repository.get_case(selected_id)
            eligibility = assess_training_eligibility(reviewed, repository.list_cases(),
                supported_hypothesis_ids={row.get("hypothesis_id") for row in (reviewed.get("diagnosis_snapshot") or {}).get("hypotheses", [])})
            reviewed["training_eligibility"] = eligibility
            repository.update_case(reviewed, event_type="TRAINING_ELIGIBILITY_ASSESSED", event_payload=eligibility)
            st.success(f"검토가 저장되었습니다 · 학습 적격성: {eligibility['status']}")
            st.json(eligibility)
        except (KeyError, ValueError) as exc:
            st.error(str(exc))


def _render_history(st: Any, repository: RepairCaseRepository) -> None:
    cases = repository.list_cases()
    if not cases:
        st.info("저장된 수리 이력이 없습니다.")
        return
    statuses = ["전체", "UNVERIFIED", "MEASUREMENT_SUPPORTED", "REPAIR_OUTCOME_VERIFIED", "ROOT_CAUSE_REVIEWED", "INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"]
    selected = st.selectbox("검증 상태 필터", statuses)
    displayed = cases if selected == "전체" else [case for case in cases if case.get("verification_status") == selected]
    st.dataframe([{"case_id": case["case_id"], "circuit": case["circuit_id"], "source": case["data_source"],
        "status": case["case_status"], "verification": case["verification_status"], "outcome": case["outcome"],
        "training": case.get("training_eligibility", {}).get("status")} for case in displayed], use_container_width=True, hide_index=True)
    case_id = st.selectbox("상세 사례", [case["case_id"] for case in displayed], key="repair_case_detail_select")
    detail = repository.get_case(case_id)
    st.json(detail)
    before = [row for row in detail.get("observations", []) if row.get("phase", "before") == "before"]
    after = [measurement for test in detail.get("post_repair_tests", []) for measurement in test.get("observations", [])]
    st.markdown("**수리 전후 측정 비교**")
    st.json(compare_before_after(before, after, expected=detail.get("expected_ranges", {})))
    with st.expander("추가 전용 이벤트 이력"):
        st.json(repository.list_events(case_id))
    cols = st.columns(3)
    full_export = repository.export_json(anonymize=False)
    anon_export = repository.export_json(anonymize=True)
    cols[0].download_button("전체 사례 JSON 내보내기", json.dumps(full_export, ensure_ascii=False, indent=2), "repair_cases_local_export.json", "application/json")
    cols[1].download_button("익명화 평가 JSON 내보내기", json.dumps(anon_export, ensure_ascii=False, indent=2), "repair_cases_anonymized.json", "application/json")
    backup_path = st.text_input("SQLite 백업 저장 경로", value=str(repository.path.with_name("repair_cases_backup.sqlite3")))
    if st.button("로컬 DB 백업 생성"):
        repository.backup(backup_path)
        st.success(f"백업 생성 완료: {backup_path}")
    confirm_delete = st.checkbox("선택 사례를 로컬에서 영구 삭제 확인", key="repair_case_confirm_delete")
    if st.button("선택 사례 삭제", disabled=not confirm_delete):
        repository.delete_case(case_id)
        st.success("사례와 연결된 로컬 증거 이벤트를 삭제했습니다.")
        st.rerun()


def _render_prior_learning(st: Any, repository: RepairCaseRepository) -> None:
    cases = repository.list_cases()
    eligible = [case for case in cases if case.get("training_eligibility", {}).get("status") == "ELIGIBLE"]
    st.markdown("#### Prior 학습 제안")
    if not eligible:
        st.info("적격한 실물 수리 사례가 없어 Prior를 생성하지 않습니다. 기본 Bayesian Prior가 유지됩니다.")
        proposals = []
    else:
        groups = sorted({case.get("circuit_family") for case in eligible if case.get("circuit_family")})
        group = st.selectbox("동일 회로군", groups)
        candidates = [case for case in eligible if case.get("circuit_family") == group and (case.get("diagnosis_snapshot") or {}).get("prior_weights")]
        if not candidates:
            st.warning("이 회로군에서 버전이 지정된 가설 Prior Snapshot을 찾지 못했습니다.")
            proposals = []
        else:
            selected = st.selectbox("가설 집합 기준 Snapshot", [case["case_id"] for case in candidates])
            base_case = repository.get_case(selected)
            snapshot = base_case["diagnosis_snapshot"]
            hypothesis_set_id = snapshot["hypothesis_set_id"]
            minimum_cases = st.number_input("최소 검증 사례 수", min_value=5, value=5, step=1,
                help="운영 기본값 5는 설정값이며 통계적으로 최적이라고 주장하지 않습니다.")
            alpha = st.number_input("Dirichlet smoothing α", min_value=0.1, value=5.0, step=0.5,
                help="기본 Prior 쪽으로 수축하는 명시적 설정값입니다. 최적값으로 검증된 값은 아닙니다.")
            proposal = estimate_prior(cases, group_id=group, hypothesis_set_id=hypothesis_set_id,
                base_prior=snapshot["prior_weights"], alpha=float(alpha), minimum_cases=int(minimum_cases),
                supported_hypothesis_ids=set(snapshot["prior_weights"]))
            st.json(proposal)
            if proposal["status"] == "PROPOSAL_READY" and st.button("Prior 제안 버전 저장"):
                propose_prior(repository, proposal)
                st.success(f"{proposal['version']} 저장됨 · 아직 승인 또는 적용되지 않았습니다.")
        proposals = repository.list_prior_models()
    st.markdown("#### Prior 버전 및 승인")
    proposals = repository.list_prior_models()
    if not proposals:
        st.caption("저장된 Prior 버전 없음 · Active Learned Prior: NONE")
        return
    versions = [item["version"] for item in proposals]
    version = st.selectbox("Prior 버전", versions, key="prior_model_version_select")
    model = repository.get_prior_model(version)
    st.json(model)
    reviewer = st.text_input("Prior 승인자", key="prior_model_reviewer")
    if model.get("status") == "PROPOSED" and st.button("Prior 승인"):
        if not reviewer.strip():
            st.error("승인자 식별 정보를 입력하세요.")
        else:
            approve_prior(repository, version, reviewer=reviewer.strip(), note="Reviewed in local interface")
            st.success("승인 완료. 적용은 별도 버튼에서 수행하세요.")
            st.rerun()
    if model.get("status") == "APPROVED" and st.button("승인된 Prior 활성화"):
        if not reviewer.strip():
            st.error("활성화 요청자 식별 정보를 입력하세요.")
        else:
            activate_prior(repository, version, reviewer=reviewer.strip())
            st.success("Prior가 활성화되었습니다.")
            st.rerun()
    if model.get("status") == "ACTIVE" and st.button("Prior 비활성화"):
        if not reviewer.strip():
            st.error("비활성화 요청자 식별 정보를 입력하세요.")
        else:
            deactivate_prior(repository, version, reviewer=reviewer.strip())
            st.success("Prior를 비활성화했습니다. 이후 진단은 기본 Prior로 돌아갑니다.")
            st.rerun()
    old_versions = [item["version"] for item in proposals if item.get("status") in {"APPROVED", "ACTIVE"}]
    if len(old_versions) > 1:
        target = st.selectbox("Rollback 대상 Prior", old_versions, key="prior_rollback_target")
        if st.button("선택 Prior로 Rollback"):
            if not reviewer.strip():
                st.error("Rollback 요청자 식별 정보를 입력하세요.")
            else:
                rollback_prior(repository, group_id=model["group_id"], target_version=target, reviewer=reviewer.strip())
                st.success(f"{target} 버전을 활성화했습니다.")
                st.rerun()
    with st.expander("Prior 변경 감사 이벤트"):
        st.json(repository.list_prior_events())


def _render_diagnosis_comparison(st: Any, repository: RepairCaseRepository, *,
                                 fault_state: dict[str, Any] | None, group_id: str) -> None:
    st.markdown("#### 동일 증거의 Default / Learned Prior 비교")
    if not fault_state:
        st.info("먼저 AI 고장 가설 탭에서 같은 회로의 가설을 생성하고 SPICE 시뮬레이션을 실행하세요.")
        return
    rows = fault_state.get("hypotheses", [])
    observations = fault_state.get("observations", [])
    versions = repository.list_prior_models(group_id=group_id)
    choices = ["Default Prior"] + [f"{item['version']} · {item['status']}" for item in versions
        if item.get("status") in {"PROPOSED", "APPROVED", "ACTIVE"}]
    selected = st.selectbox("Prior 비교 모드 (미승인 버전은 미리보기만)", choices)
    model = None
    if selected != "Default Prior":
        selected_version = selected.split(" · ", 1)[0]
        model = repository.get_prior_model(selected_version)
        st.caption(f"버전 {model['version']} · 상태 {model['status']} · SPICE 예측과 측정값은 변경하지 않습니다.")
    if not observations:
        st.warning("비교에 필요한 측정 관측이 없습니다.")
        return
    prior_map = model.get("prior_weights") if model else None
    default = conditional_bayesian_update(rows, observations)
    learned = conditional_bayesian_update(rows, observations, prior_weights=prior_map,
                                           prior_model_version=model.get("version") if model else None) if model else None
    st.write("**Default Prior**")
    st.json({"status": default.get("status"), "posterior": default.get("posterior"), "model_mismatch": default.get("model_mismatch")})
    if learned:
        st.write("**Prior 미리보기 / 적용 비교**")
        st.json({"status": learned.get("status"), "prior_model_version": learned.get("prior_model_version"),
                 "posterior": learned.get("posterior"), "model_mismatch": learned.get("model_mismatch")})
        if default.get("posterior") and learned.get("posterior"):
            by_id = {row["hypothesis_id"]: row for row in learned["posterior"]}
            st.dataframe([{"가설": row["hypothesis_id"], "기본 Prior": row["prior_weight"],
                "보정 Prior": by_id.get(row["hypothesis_id"], {}).get("prior_weight"),
                "기본 Posterior": row["posterior"],
                "보정 Posterior": by_id.get(row["hypothesis_id"], {}).get("posterior")}
                for row in default["posterior"]], use_container_width=True, hide_index=True)
            component_mass = {}
            for row in default["posterior"]:
                component_mass.setdefault(row["component_ref"], {"부품": row["component_ref"], "기본 posterior 질량": 0.0, "보정 posterior 질량": 0.0})
                component_mass[row["component_ref"]]["기본 posterior 질량"] += row["posterior"]
            for row in learned["posterior"]:
                component_mass.setdefault(row["component_ref"], {"부품": row["component_ref"], "기본 posterior 질량": 0.0, "보정 posterior 질량": 0.0})
                component_mass[row["component_ref"]]["보정 posterior 질량"] += row["posterior"]
            st.caption("부품별 posterior 질량은 점검 후보 비교용 조건부 합계입니다. 전기적 위험 점수나 손상확률이 아닙니다.")
            st.dataframe(list(component_mass.values()), use_container_width=True, hide_index=True)
    st.caption("정보 이득 추천은 이 화면에서 `NOT_EVALUATED`입니다. 동일한 안전 적격 측정 모델이 연결되지 않았습니다.")
    compatible_cases = [item for item in repository.list_cases(circuit_family=group_id)
        if item.get("diagnosis_snapshot", {}).get("snapshot_id")]
    if compatible_cases:
        case_id = st.selectbox("재분석을 추가 저장할 원래 사례", [item["case_id"] for item in compatible_cases],
            key="repair_reanalysis_case_select")
        if st.button("선택 Prior 결과를 별도 재분석 Snapshot으로 저장"):
            original_case = repository.get_case(case_id)
            selected_result = learned if learned and learned.get("status") == "ACTIVE_CONDITIONAL" else default
            snapshot = create_diagnosis_snapshot(circuit_id=original_case.get("circuit_id", group_id),
                circuit_hash=original_case.get("circuit_hash", ""), diagnosis=selected_result,
                hypothesis_rows=rows, model_version="local-repair-feedback",
                hypothesis_set_id=";".join(sorted(row.get("hypothesis_id", "") for row in selected_result.get("posterior", []))))
            snapshot.update({"snapshot_type": "PRIOR_REANALYSIS",
                "source_snapshot_id": original_case["diagnosis_snapshot"]["snapshot_id"],
                "prior_mode": "APPROVED_LEARNED" if model and model.get("status") == "ACTIVE" else "PREVIEW_OR_DEFAULT",
                "prior_model_version": model.get("version") if model else None})
            repository.add_diagnosis_snapshot(case_id, snapshot)
            st.success("원본 Snapshot은 그대로 두고 재분석 Snapshot을 별도 저장했습니다.")
        with st.expander("선택 사례의 Snapshot 이력"):
            st.json(repository.list_diagnosis_snapshots(case_id))
    st.caption("이 비교는 동일 측정에 대한 조건부 가중치 변화입니다. 독립 정확도 개선이나 실제 부품 고장 확률을 뜻하지 않습니다.")

