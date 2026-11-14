"""Connection-optional, evidence-ranked fault diagnosis UI."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
import json
import logging
import time
import hashlib
from pathlib import Path
from uuid import uuid4

from circuitcare.analysis.models import AnalysisStatus
from circuitcare.frontend.models import ProjectContext
from circuitcare.frontend.measurement_state import get_measurements
from circuitcare.local_llm import LocalLLMError, OllamaClient
from circuitcare.symptom_diagnosis import diagnose, load_knowledge
from circuitcare.symptom_diagnosis.llm_assist import (
    explain_candidates, explain_component_candidates, suggest_unregistered_causes,
    suggest_pcb_fault_candidates, select_follow_up_question,
)
from circuitcare.symptom_diagnosis.pcb_context import build_pcb_context
from circuitcare.symptom_diagnosis.candidate_fusion import fuse_inspection_candidates
from circuitcare.symptom_diagnosis.evidence_strategy import rank_by_verified_evidence
from circuitcare.symptom_diagnosis.evidence_ledger import build_evidence_ledger
from circuitcare.symptom_diagnosis.evidence_gaps import analyze_evidence_gaps
from circuitcare.symptom_diagnosis.active_inspection_planner import plan_active_inspections
from circuitcare.symptom_diagnosis.observation_store import ObservationStore
from circuitcare.symptom_diagnosis.diagnosis_state import build_overlay_state
from circuitcare.symptom_diagnosis.schemas import ComponentRecord, Observation
from circuitcare.symptom_diagnosis.symptom_parser import parse_user_report
from circuitcare.frontend.workbench_state import ACTIVE_PHOTO_KEY, update_selection
from circuitcare.analysis.models import AnalysisScope
from circuitcare.analysis.fingerprints import (
    build_analysis_fingerprints, build_measurement_fingerprint,
    build_symptom_input_fingerprint,
)
from circuitcare.analysis.state_manager import (
    begin_analysis, commit_analysis_result, fail_analysis, get_current_result,
    get_current_state, migrate_legacy_result,
)

LOGGER = logging.getLogger(__name__)


def _project_key(project: ProjectContext) -> str:
    return f"cc_symptom_diagnosis_{project.project_id}"


def candidate_focus_widget_key(screen_scope: str, project_id: str, board_id: str | None,
                               component_id: str, fault_id: str | None) -> str:
    """Stable, scoped key for a physical-component/fault-pair action."""
    identity = "|".join((screen_scope, project_id, board_id or "", component_id, fault_id or ""))
    return f"cc_candidate_focus_{hashlib.sha256(identity.encode()).hexdigest()[:20]}"


def _apply_pending_initial_symptom_widget(session_state: Any, state: dict[str, Any],
                                           widget_key: str) -> None:
    """Apply a queued widget value before the keyed widget is instantiated."""
    pending = state.pop("_pending_initial_symptom_widget_value", None)
    if pending is not None:
        session_state[widget_key] = pending
        state["run_after_initial_answer"] = True


def _diagnosis_fingerprint(
    project: ProjectContext, components: tuple[Any, ...], symptoms: list[str],
    symptom_text: str, observations: dict[str, Any], answers: dict[str, Any],
    measurements: list[Any], silkscreen_labels: list[dict[str, Any]], state: Any,
    confirmed_component_ids: list[str] | None = None,
    inventory_complete: bool = False,
    component_inventory: dict[str, dict[str, Any]] | None = None,
    analysis_target_id: str | None = None,
    suspected_components: list[dict[str, Any]] | None = None,
) -> str:
    target_id = str(analysis_target_id or state.get(ACTIVE_PHOTO_KEY) or "target:default")
    analysis_scope = AnalysisScope(project.project_id, project.board_id or "", target_id)
    photo_snapshots = state.get("circuitcare_v6_photo_component_snapshots", {})
    photo_snapshot = photo_snapshots.get(analysis_scope.key) if isinstance(photo_snapshots, dict) else None
    return build_symptom_input_fingerprint(
        project=project, components=components, symptoms=symptoms, symptom_text=symptom_text,
        observations=observations, answers=answers, silkscreen_labels=silkscreen_labels,
        component_inventory=component_inventory, confirmed_component_ids=confirmed_component_ids,
        inventory_complete=inventory_complete,
        target_id=target_id,
        target_photo_sha256=(photo_snapshot.get("image_fingerprint")
                             if isinstance(photo_snapshot, dict) else None),
        additional_inputs={"user_suspicions": sorted(
            [{"component_id": row.get("board_component_id") or row.get("component_id"),
              "fault_id": row.get("suspected_fault_mode"),
              "observation_fingerprint": row.get("observation_fingerprint"),
              "evidence_status": row.get("evidence_status", "UNVERIFIED")}
             for row in (suspected_components or []) if isinstance(row, dict)],
            key=lambda row: str(row.get("component_id")),
        ), "answered_test_ids": sorted(state.get("answered_test_ids", [])),
        "question_answers": [
            {"test_id": str(test_id), "answer": row.get("answer"),
             "observation_codes": sorted((row.get("observations") or {}).keys())}
            for test_id, row in sorted(state.get("question_answers", {}).items())
            if isinstance(row, dict)
        ]},
    )


def _analysis_scope(project: ProjectContext, state: Any, analysis_target_id: str | None = None) -> AnalysisScope:
    return AnalysisScope(project.project_id, project.board_id or "",
                         str(analysis_target_id or state.get("circuitcare_v7_analysis_target_id")
                             or state.get(ACTIVE_PHOTO_KEY) or "target:default"))


def _registry_valid_suspects(
    rows: list[dict[str, Any]], registry: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep suspicions only while their board ID and source observation remain active."""
    if not isinstance(registry, dict):
        return rows, []
    photos = {str(row.get("photo_id")): row for row in registry.get("photos", []) if isinstance(row, dict)}
    components = {str(row.get("board_component_id")): row for row in registry.get("components", [])
                  if isinstance(row, dict) and row.get("board_component_id")}
    active: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    for suspect in rows:
        board_id = str(suspect.get("board_component_id") or suspect.get("component_id") or "")
        component = components.get(board_id)
        sources = component.get("source_components", []) if component else []
        local_id = str(suspect.get("component_id") or "")
        photo_id = str(suspect.get("photo_id") or "")
        source = next((row for row in sources if isinstance(row, dict)
                       and (not local_id or str(row.get("component_id")) == local_id)
                       and (not photo_id or str(row.get("photo_id")) == photo_id)
                       and str(row.get("review_status", "")).casefold() not in {"excluded", "exclude"}), None)
        manifest = photos.get(photo_id, {})
        image_matches = (not suspect.get("photo_sha256") or not manifest.get("image_fingerprint")
                         or suspect.get("photo_sha256") == manifest.get("image_fingerprint"))
        source_version_matches = (not suspect.get("selection_version") or not source
                                  or suspect.get("selection_version") == source.get("source_observation_fingerprint"))
        if component and source and component.get("presence_status") != "EXCLUDED" and image_matches and source_version_matches:
            active.append(suspect)
        else:
            stale.append(suspect)
    return active, stale


def _auto_qwen_model(state: dict[str, Any], *, refresh_seconds: int = 60) -> str | None:
    """Choose an installed Qwen once per cache window; absence means rules-only fallback."""
    now = time.time()
    models = state.get("ollama_models")
    if now - float(state.get("ollama_checked_at", 0)) > refresh_seconds:
        try:
            models = [str(row.get("name") or row.get("model") or "")
                      for row in OllamaClient(timeout_seconds=2).list_models() if isinstance(row, dict)]
            state["ollama_error"] = None
        except (LocalLLMError, ValueError, OSError) as exc:
            models = []
            state["ollama_error"] = str(exc)[:300]
        state["ollama_models"] = models
        state["ollama_checked_at"] = now
    models = models if isinstance(models, list) else []
    qwen = [name for name in models if isinstance(name, str) and name.casefold().startswith("qwen")]
    preferred = next((name for name in qwen if name.casefold() == "qwen3.5:9b"), None)
    chosen = preferred or next((name for name in qwen if "3.5" in name), None) or (qwen[0] if qwen else None)
    state["auto_qwen_model"] = chosen
    return chosen


def _simple_candidate_rows(result: dict[str, Any], *, limit: int = 3) -> list[dict[str, Any]]:
    """Create physical-component rows, optionally preserving refuted/conflicted hypotheses."""
    fusion = result.get("candidate_fusion")
    if isinstance(fusion, dict):
        all_rows = fusion.get("candidates", fusion.get("top_candidates", []))
        rows = [row for row in all_rows if isinstance(row, dict)]
    else:
        rows = result.get("all_component_candidates", result.get("component_candidates", []))
        rows = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        component_id = str(row.get("board_component_id") or row.get("component_id") or "")
        if not component_id:
            continue
        current = grouped.get(component_id)
        hypothesis = {key: row.get(key) for key in (
            "fault_id", "fault_hypothesis", "fault_title", "visual_status", "measurement_verdict",
            "verification_status", "candidate_source", "supporting_evidence", "opposing_evidence",
            "recommended_tests", "qwen_reason") if key in row}
        if current is None:
            grouped[component_id] = {**row, "component_id": component_id,
                                     "board_component_id": component_id,
                                     "fault_hypotheses": [hypothesis]}
        else:
            current["fault_hypotheses"].append(hypothesis)
    return list(grouped.values())[:max(0, int(limit))]


def _question_plan_fingerprint(
    state: dict[str, Any], *, knowledge: dict[str, Any],
    selected_codes: list[str], symptom_text: str,
    board_registry: dict[str, Any] | None,
    suspected_components: list[dict[str, Any]] | None,
) -> str:
    """Fingerprint only stable inputs that can change which questions are relevant."""
    tests = knowledge.get("test_catalog", {}).get("tests", {})
    catalog = {
        str(test_id): {
            "type": row.get("type"), "question": row.get("question"),
            "risk_level": row.get("risk_level"),
        }
        for test_id, row in tests.items()
        if isinstance(row, dict) and row.get("type") == "question"
    } if isinstance(tests, dict) else {}
    components = board_registry.get("components", []) if isinstance(board_registry, dict) else []
    component_types = sorted({str(row.get("component_type") or "unknown").casefold()
                              for row in components if isinstance(row, dict)
                              and row.get("presence_status") != "EXCLUDED"})
    suspicions = sorted({(
        str(row.get("board_component_id") or row.get("component_id") or ""),
        str(row.get("suspected_fault_mode") or ""),
    ) for row in (suspected_components or []) if isinstance(row, dict)}, key=lambda row: row)
    payload = {
        "policy_version": "question-plan-v2",
        "initial_symptom": state.get("initial_symptom_answer"),
        "symptom_codes": sorted(set(map(str, selected_codes))),
        "symptom_text": " ".join(str(symptom_text or "").casefold().split()),
        "component_types": component_types,
        "user_suspicions": suspicions,
        "catalog_sha256": hashlib.sha256(json.dumps(
            catalog, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _ensure_question_plan(
    state: dict[str, Any], result: dict[str, Any], *, knowledge: dict[str, Any],
    plan_fingerprint: str,
) -> dict[str, Any]:
    """Freeze at most five safe, relevant questions for stable symptom/board inputs."""
    existing = state.get("question_plan")
    if (isinstance(existing, dict) and existing.get("plan_fingerprint") == plan_fingerprint
            and isinstance(existing.get("questions"), list)):
        answered = set(str(value) for value in state.get("answered_test_ids", []))
        skipped = set(str(value) for value in existing.get("skipped_question_ids", []))
        skipped.update(str(value) for value in state.get("skipped_test_ids", []))
        for row in existing["questions"]:
            if isinstance(row, dict):
                question_id = str(row.get("test_id") or "")
                row["status"] = ("SKIPPED" if question_id in skipped else
                                 "ANSWERED" if question_id in answered else "PENDING")
        return existing
    tests = knowledge.get("test_catalog", {}).get("tests", {})
    tests = tests if isinstance(tests, dict) else {}
    answered = set(str(value) for value in state.get("answered_test_ids", []))
    answered.update(str(value) for value in state.get("skipped_test_ids", []))
    ordered_ids: list[str] = []
    reasons: dict[str, str] = {}
    rule_question_ids: list[str] = []
    next_tests = result.get("next_tests", [])
    if isinstance(next_tests, list):
        for row in next_tests:
            question_id = str(row.get("test_id") or "") if isinstance(row, dict) else ""
            if (question_id in tests and tests[question_id].get("type") == "question"
                    and tests[question_id].get("risk_level", "none") == "none"
                    and question_id not in answered):
                rule_question_ids.append(question_id)
                rationale = row.get("reason") or row.get("selection_reason") or row.get("name")
                reasons.setdefault(question_id, str(rationale or "현재 증상 규칙의 다음 확인 질문"))
    proposed = result.get("llm_follow_up_question")
    if isinstance(proposed, dict):
        proposed_id = str(proposed.get("test_id") or "")
        # Qwen may only reprioritize a question already selected by the rules.
        if (proposed_id in rule_question_ids and proposed_id in tests
                and tests[proposed_id].get("type") == "question"):
            ordered_ids.append(proposed_id)
            reasons[proposed_id] = "Qwen 우선순위 제안 · 규칙이 선택한 카탈로그 질문"
    # Rule order carries symptom/hypothesis relevance. Do not pad with generic unrelated questions.
    ordered_ids.extend(rule_question_ids)
    unique_ids = list(dict.fromkeys(ordered_ids))[:5]
    question_rows = [{"test_id": question_id,
                      "question": {"test_id": question_id, **dict(tests[question_id])},
                      "status": "ANSWERED" if question_id in answered else "PENDING",
                      "selection_reason": reasons.get(question_id, "현재 증상 규칙의 다음 확인 질문")}
                     for question_id in unique_ids]
    plan_id = hashlib.sha256(
        f"{plan_fingerprint}|{'|'.join(unique_ids)}".encode("utf-8")
    ).hexdigest()[:20]
    plan = {"plan_id": plan_id, "questions": question_rows,
            "current_question_index": 0, "plan_fingerprint": plan_fingerprint,
            "termination_reason": ("NO_SAFE_CATALOG_QUESTIONS" if not question_rows else None)}
    state["question_plan"] = plan
    return plan


def _simple_candidate_information(result: dict[str, Any]) -> dict[tuple[str, str], tuple[str, str, tuple[str, ...], tuple[str, ...]]]:
    """Stable rule/measurement facts used to decide whether a follow-up helped."""
    information = {}
    for candidate in _simple_candidate_rows(result):
        component_id = str(candidate.get("board_component_id") or candidate.get("component_id") or "")
        for hypothesis in candidate.get("fault_hypotheses", []):
            fault_id = str(hypothesis.get("fault_id") or hypothesis.get("fault_hypothesis") or "")
            if not component_id or not fault_id:
                continue
            information[(component_id, fault_id)] = (
                str(hypothesis.get("visual_status") or "SUSPECTED"),
                str(hypothesis.get("measurement_verdict") or hypothesis.get("verification_status") or ""),
                tuple(sorted(str(value) for value in (hypothesis.get("supporting_evidence") or []))),
                tuple(sorted(str(value) for value in (hypothesis.get("opposing_evidence") or []))),
            )
    return information


def _followup_result_improved(previous: dict, current: dict) -> bool:
    """Count hypothesis separation or evidence/verdict changes, never Qwen rank text."""
    old_keys, new_keys = set(previous), set(current)
    if old_keys - new_keys:
        return True
    for key in old_keys & new_keys:
        old_status, old_verdict, old_support, old_oppose = previous[key]
        new_status, new_verdict, new_support, new_oppose = current[key]
        if (old_status, old_verdict, old_support, old_oppose) != (
            new_status, new_verdict, new_support, new_oppose
        ):
            return True
    return False


def _queue_measurement_change_rerun(
    state: dict[str, Any], *, previous_fingerprint: str | None,
    current_fingerprint: str, prior_analysis_exists: bool,
    current_result: dict[str, Any] | None, central_status: str,
) -> None:
    """Queue exactly one analysis for a changed measurement snapshot."""
    if (previous_fingerprint not in (None, "")
            and previous_fingerprint != current_fingerprint
            and prior_analysis_exists and current_result is None
            and central_status == AnalysisStatus.STALE.value):
        state["run_after_measurement_change"] = current_fingerprint


def _consume_measurement_change_rerun(state: dict[str, Any], current_fingerprint: str) -> bool:
    """Consume the one-shot request only when it still matches current evidence."""
    queued_fingerprint = state.pop("run_after_measurement_change", None)
    return queued_fingerprint == current_fingerprint


def _render_simple_followup_question(
    st: Any, project: ProjectContext, result: dict[str, Any], *, state: dict[str, Any],
    knowledge: dict[str, Any], preview: bool, candidates: list[dict[str, Any]],
) -> None:
    """Render the one active safe question before candidate results for every simple path."""
    next_tests = result.get("next_tests", []) if isinstance(result.get("next_tests"), list) else []
    answered = set(str(item) for item in state.get("answered_test_ids", []))
    plan = state.get("question_plan") if isinstance(state.get("question_plan"), dict) else {}
    planned = plan.get("questions", []) if isinstance(plan.get("questions"), list) else []
    skipped = set(str(value) for value in plan.get("skipped_question_ids", []))
    pending_indices = [index for index, row in enumerate(planned) if isinstance(row, dict)
                       and str(row.get("test_id") or "") not in answered | skipped]
    plan["current_question_index"] = pending_indices[0] if pending_indices else len(planned)
    question = planned[plan["current_question_index"]].get("question") if pending_indices else None
    answered_count = len(answered - {INITIAL_SYMPTOM_QUESTION_ID})
    stopped = bool(state.get("followup_questions_stopped"))
    no_progress = int(state.get("followup_no_improvement_streak", 0))
    if answered_count >= 5 or no_progress >= 2:
        stopped = True
        state["followup_questions_stopped"] = True
        plan["termination_reason"] = "MAX_5_QUESTIONS_ANSWERED" if answered_count >= 5 else "NO_CANDIDATE_SEPARATION"
    if stopped:
        question = None
    if question:
        question_id = str(question.get("test_id") or "")
        if question_id:
            st.markdown("#### 새로운 확인 질문")
            st.caption(f"질문 진행 · {answered_count} / 5")
            if question.get("selection_reason"):
                st.caption(str(question["selection_reason"]))
            st.write(question.get("question", "확인할 내용이 있나요?"))
            answer_key = f"{_project_key(project)}_simple_answer_{question_id}"
            answer = st.radio("답변", ("예", "아니오", "모르겠음"), horizontal=True, key=answer_key)
            if st.button("답변 반영", key=f"{answer_key}_submit", disabled=preview):
                state["followup_pending_comparison"] = {
                    "test_id": question_id,
                    "candidate_information": _simple_candidate_information(result),
                }
                _save_follow_up_answer(state, question, answer)
                state["run_after_initial_answer"] = True
                st.session_state[_project_key(project)] = state
                st.rerun()
            plan_skip_key = f"{answer_key}_skip"
            if st.button("이 질문 건너뛰기", key=plan_skip_key, disabled=preview):
                plan["skipped_question_ids"] = list(dict.fromkeys([
                    *plan.get("skipped_question_ids", []), question_id,
                ]))
                plan["termination_reason"] = "USER_SKIPPED_QUESTION"
                state["skipped_test_ids"] = list(dict.fromkeys([
                    *state.get("skipped_test_ids", []), question_id,
                ]))
                for row in planned:
                    if isinstance(row, dict) and str(row.get("test_id") or "") == question_id:
                        row["status"] = "SKIPPED"
                state["question_plan"] = plan
                st.session_state[_project_key(project)] = state
                st.rerun()
    if not stopped and answered_count < 5 and no_progress < 2:
        if st.button("현재 정보로 검사 진행", key=f"{_project_key(project)}_stop_followup", disabled=preview):
            state["followup_questions_stopped"] = True
            if isinstance(state.get("question_plan"), dict):
                state["question_plan"]["termination_reason"] = "USER_EARLY_STOP"
            st.session_state[_project_key(project)] = state
            st.rerun()
    history = state.get("question_answers", {})
    completed = [(test_id, row) for test_id, row in history.items()
                 if test_id != INITIAL_SYMPTOM_QUESTION_ID and isinstance(row, dict)]
    if completed:
        with st.expander(f"앞서 답한 질문 · {len(completed)}개"):
            tests = knowledge.get("test_catalog", {}).get("tests", {})
            for test_id, row in completed:
                st.caption(f"{row.get('question', {}).get('question', test_id)} · {row.get('answer', '미답변')}")
                question_row = tests.get(test_id, row.get("question", {}))
                edit_key = f"{_project_key(project)}_edit_answer_{test_id}"
                edited = st.radio("답변 수정", ("예", "아니오", "모르겠음", "답변 취소"),
                                  horizontal=True, key=edit_key, index=(
                                      ("예", "아니오", "모르겠음", "답변 취소").index(row.get("answer"))
                                      if row.get("answer") in ("예", "아니오", "모르겠음", "답변 취소") else 0))
                if st.button("수정 저장", key=f"{edit_key}_save"):
                    state["followup_pending_comparison"] = {
                        "test_id": str(test_id),
                        "candidate_information": _simple_candidate_information(result),
                    }
                    _save_follow_up_answer(state, question_row, edited)
                    state.pop("followup_questions_stopped", None)
                    state["run_after_initial_answer"] = True
                    st.session_state[_project_key(project)] = state
                    st.rerun()


def _render_simple_diagnosis_result(
    st: Any, project: ProjectContext, result: dict[str, Any], *, state: dict[str, Any],
    knowledge: dict[str, Any], preview: bool, board_registry: dict[str, Any] | None,
    suspected_components: list[dict[str, Any]] | None = None,
    show_questions: bool = True, show_candidates: bool = True,
) -> None:
    if state.get("ollama_error"):
        st.caption("로컬 AI에 연결할 수 없어 규칙 기반 기본 진단으로 진행합니다.")
    pending = state.pop("followup_pending_comparison", None)
    if isinstance(pending, dict):
        previous_information = pending.get("candidate_information", {})
        current_information = _simple_candidate_information(result)
        improved = _followup_result_improved(previous_information, current_information)
        state["followup_no_improvement_streak"] = (0 if improved else
            int(state.get("followup_no_improvement_streak", 0)) + 1)
    if result.get("status") == "NEEDS_SYMPTOMS":
        st.info("증상 정보가 더 필요합니다. 모르는 항목은 ‘모르겠음’을 선택할 수 있습니다.")
    candidates = _simple_candidate_rows(result)
    if show_questions:
        symptom_text = str(st.session_state.get(f"{_project_key(project)}_description",
                                                 project.symptom_description or ""))
        selected_codes = list(state.get("symptoms", []))
        state["valid_suspected_components"] = [dict(row) for row in (suspected_components or [])
                                                if isinstance(row, dict)]
        plan_fingerprint = _question_plan_fingerprint(
            state, knowledge=knowledge, selected_codes=selected_codes,
            symptom_text=symptom_text, board_registry=board_registry,
            suspected_components=state.get("valid_suspected_components", []),
        )
        _ensure_question_plan(state, result, knowledge=knowledge,
                              plan_fingerprint=plan_fingerprint)
        _render_simple_followup_question(st, project, result, state=state, knowledge=knowledge,
                                         preview=preview, candidates=candidates)
    if not show_candidates:
        return
    if candidates:
        st.markdown("#### AI 추천 검사 후보")
        for index, candidate in enumerate(candidates, start=1):
            component_id = str(candidate.get("board_component_id") or candidate.get("component_id"))
            st.markdown(f"**{index}. {component_id} · {candidate.get('component_type', '종류 미확정')}**")
            for hypothesis in candidate.get("fault_hypotheses", []):
                source = hypothesis.get("candidate_source", [])
                source_label = "Qwen 미검증 제안" if "QWEN_UNVERIFIED" in source else "검사 후보"
                title = hypothesis.get("fault_title") or hypothesis.get("fault_hypothesis") or hypothesis.get("fault_id") or "원인 확인 필요"
                status = hypothesis.get("visual_status") or "SUSPECTED"
                status_label = {"FAULT_SUPPORTED": "특정 가설 지지", "FAULT_REFUTED": "특정 가설 반박",
                                "CONFLICTED": "근거 충돌", "INCONCLUSIVE": "판정 보류"}.get(status, "미확정")
                st.caption(f"{title} · {source_label} · {status_label}")
                reason = hypothesis.get("qwen_reason")
                if reason:
                    st.caption(reason)
                for test in (hypothesis.get("recommended_tests") or [])[:2]:
                    st.caption(f"다음 확인 · {test}")
            with st.expander(f"{index}번 후보 근거와 검사 안내"):
                for hypothesis in candidate.get("fault_hypotheses", []):
                    title = hypothesis.get("fault_title") or hypothesis.get("fault_id") or "고장 가설"
                    st.markdown(f"**{title}**")
                    support = hypothesis.get("supporting_evidence") or hypothesis.get("evidence") or []
                    oppose = hypothesis.get("opposing_evidence") or []
                    st.caption("지지 근거 · " + (" · ".join(str(item) for item in support) or "직접 근거 없음"))
                    st.caption("반대 근거 · " + (" · ".join(str(item) for item in oppose) or "확인된 반대 근거 없음"))
                    tests = hypothesis.get("recommended_tests") or []
                    if tests:
                        st.caption("확인할 검사 · " + " · ".join(str(item) for item in tests))
                    if hypothesis.get("qwen_reason"):
                        st.write(hypothesis["qwen_reason"])
            if st.button("PCB에서 위치 보기", key=candidate_focus_widget_key(
                    "simple_top3", project.project_id, project.board_id, component_id,
                    ",".join(str(row.get("fault_id") or row.get("fault_hypothesis") or "")
                            for row in candidate.get("fault_hypotheses", [])))):
                focused = _focus_board_component(st, project, board_registry, component_id) if board_registry else False
                if not focused:
                    update_selection(st.session_state, project=project, selected_component=component_id)
                st.rerun()
    if any(result.get(key) for key in (
        "llm_assist", "llm_component_explanations", "llm_unverified_suggestions",
        "matched_rules", "functional_blocks", "measurement_evidence", "hypothesis_groups",
        "functional_block_groups", "hierarchical_hypotheses", "evidence_gaps", "evidence_ledger",
    )):
        with st.expander("전체 근거와 추가 진단 정보", expanded=False):
            assist = result.get("llm_assist")
            if isinstance(assist, dict):
                st.markdown("**AI 설명 보조 · 미확정**")
                st.write(assist.get("summary") or assist.get("explanation") or "")
            groups = result.get("hypothesis_groups", {})
            for group_key, label in (("component", "소자 내부·기능 블록"),
                                     ("board", "기판·납땜"), ("system", "외부 시스템")):
                rows = groups.get(group_key, []) if isinstance(groups, dict) else []
                for row in rows:
                    st.markdown(f"**{label} · {row.get('title', '원인 후보')}**")
                    st.caption("지지 · " + (" · ".join(row.get("supporting_evidence", [])) or "근거 없음"))
                    st.caption("반대 · " + (" · ".join(row.get("opposing_evidence", [])) or "근거 없음"))
            for row in result.get("matched_rules", []):
                st.caption("적용 규칙 · " + str(row.get("name", row.get("rule_id", "규칙"))))
            for block in result.get("functional_blocks", []):
                st.caption(f"기능 역할 후보 · {block.get('component_id')} · {', '.join(block.get('candidate_blocks', []))} · 미확정")
            for block in result.get("functional_block_groups", [])[:12]:
                st.caption(f"기능 블록 그룹 · {block.get('role')} · " +
                           " · ".join(block.get("component_ids", [])[:12]) +
                           " · 추정이며 연결 확정 아님")
            for gap in result.get("evidence_gaps", [])[:8]:
                missing = [item.get("description", "") for item in gap.get("missing_evidence", [])[:3]]
                st.caption(f"부족 근거 · {gap.get('fault_id')} · " +
                           (" · ".join(missing) if missing else "검증 전 가설"))
            ledger = result.get("evidence_ledger", [])
            if ledger:
                status_counts: dict[str, int] = {}
                for item in ledger:
                    status = str(item.get("status", "UNKNOWN"))
                    status_counts[status] = status_counts.get(status, 0) + 1
                st.caption("근거 출처 상태 · " + " · ".join(f"{key} {value}" for key, value in sorted(status_counts.items())))
            for row in result.get("measurement_evidence", []):
                st.caption(f"측정 근거 · {row.get('component_id')} · {row.get('measurement_type')} {row.get('value')} {row.get('unit')} · {row.get('verification_status')}")
            for suggestion in result.get("llm_unverified_suggestions", []):
                st.caption(f"추가 원인 제안 · {suggestion.get('title')} · {suggestion.get('reason')} · 미검증")
    elif result.get("status") == "INCONCLUSIVE":
        st.info(result.get("message") or "현재 정보로는 소자를 특정할 수 없습니다. 사진 검수나 안전한 확인 정보가 더 필요합니다.")
    else:
        st.caption(result.get("message", "후보를 만들 수 있는 근거가 아직 부족합니다."))



def current_analysis_snapshot(
    st: Any, project: ProjectContext, *, components: tuple[Any, ...] = (),
    component_inventory: dict[str, dict[str, Any]] | None = None,
    symptom_text: str | None = None, selected_codes: list[str] | None = None,
    observations: dict[str, Any] | None = None, inventory_complete: bool | None = None,
    analysis_target_id: str | None = None,
    component_id_aliases: dict[str, str | None] | None = None,
    board_registry: dict[str, Any] | None = None,
    suspected_components: list[dict[str, Any]] | None = None,
    auto_qwen: bool = False,
) -> dict[str, Any]:
    """Rebuild and validate a photo-scoped or board-scoped analysis version."""
    key = _project_key(project)
    state = st.session_state.get(key, {})
    if not isinstance(state, dict):
        state = {}
    kb = load_knowledge()
    symptom_items = kb["symptom_ontology"]["symptoms"]
    code_to_label = {code: item["label"] for code, item in symptom_items.items()}
    if selected_codes is None:
        labels = st.session_state.get(f"{key}_symptoms", [
            code_to_label[code] for code in state.get("symptoms", []) if code in code_to_label
        ])
        selected_codes = [code for code, label in code_to_label.items() if label in labels]
        if auto_qwen:
            simple_choice = st.session_state.get(f"{key}_simple_initial_symptom")
            if simple_choice in INITIAL_SYMPTOM_OPTIONS:
                simple_code = INITIAL_SYMPTOM_OPTIONS[simple_choice]
                selected_codes = [simple_code] if simple_code else []
    if symptom_text is None:
        symptom_text = st.session_state.get(f"{key}_description", project.symptom_description or "")
    if inventory_complete is None:
        inventory_complete = bool(st.session_state.get(
            f"{key}_inventory_complete", state.get("inventory_complete", False)))
        if auto_qwen:
            inventory_complete = bool(st.session_state.get(
                f"{key}_simple_inventory_complete", inventory_complete))
    if observations is None:
        observations = dict(state.get("manual_observations", state.get("observations", {})))
        if auto_qwen:
            for fact in ("POWER_SOURCE_VERIFIED", "BUZZER_VERIFIED_FAILED", "BUZZER_VERIFIED_WORKING",
                         "BUZZER_DRIVE_ABNORMAL", "BUZZER_DRIVE_VERIFIED_NORMAL", "VISIBLE_SOLDER_DAMAGE"):
                widget_value = st.session_state.get(f"{key}_simple_{fact}")
                widget_value = {"예": True, "아니오": False, "모름": None}.get(widget_value, widget_value)
                if widget_value in {True, False, None}:
                    observations[fact] = widget_value
            if st.session_state.get(f"{key}_simple_power_off", False):
                observations["POWER_OFF_CONFIRMED"] = True
            else:
                observations.pop("POWER_OFF_CONFIRMED", None)
        for fact in ("POWER_SOURCE_VERIFIED", "BUZZER_VERIFIED_FAILED", "BUZZER_VERIFIED_WORKING",
                     "BUZZER_DRIVE_ABNORMAL", "BUZZER_DRIVE_VERIFIED_NORMAL", "VISIBLE_SOLDER_DAMAGE"):
            widget_value = st.session_state.get(f"{key}_{fact}")
            widget_value = {"예": True, "아니오": False, "모름": None}.get(widget_value, widget_value)
            if widget_value in {True, False, None}:
                observations[fact] = widget_value
        if st.session_state.get(f"{key}_power_off", False):
            observations["POWER_OFF_CONFIRMED"] = True
        else:
            observations.pop("POWER_OFF_CONFIRMED", None)
    # `None` means unknown and the rule engine ignores it. Hidden simple-workflow
    # widgets can contribute explicit None values while reassessment uses only
    # persisted observations; normalize both paths so they share one fingerprint.
    if isinstance(observations, dict):
        observations = {code: value for code, value in observations.items() if value is not None}
    inventory = component_inventory or {}
    rows = _component_rows(components, inventory)
    confirmed_ids = sorted(row["component_id"] for row in rows if row.get("presence_status") == "CONFIRMED")
    measurements = get_measurements(
        st.session_state, project_id=project.project_id, board_id=project.board_id or "",
    )
    if isinstance(board_registry, dict):
        from circuitcare.frontend.board_component_registry import project_measurements_to_registry
        projected, measurements = project_measurements_to_registry(measurements, board_registry)
        # Unresolved/retired identities stay in the measurement archive and UI,
        # but are not evidence for the current diagnosis.
    elif component_id_aliases is not None:
        measurements = [replace(
            row, component_id=component_id_aliases.get(row.component_id) or "",
            component_identity_status=(row.component_identity_status
                                       if component_id_aliases.get(row.component_id) else "UNKNOWN"),
        ) for row in measurements]
    silkscreen = st.session_state.get("circuitcare_v4_silkscreen_labels", [])
    if not isinstance(silkscreen, list):
        silkscreen = []
    input_fingerprint = _diagnosis_fingerprint(
        project, components, selected_codes, symptom_text, observations, state.get("answers", {}),
        measurements, silkscreen, st.session_state, confirmed_ids,
        bool(inventory_complete), inventory, analysis_target_id,
        suspected_components=suspected_components,
    )
    selected_model = _auto_qwen_model(state) if auto_qwen else st.session_state.get(f"{key}_model")
    llm_enabled = bool(selected_model) if auto_qwen else bool(st.session_state.get(f"{key}_llm_enabled", False))
    fingerprints = build_analysis_fingerprints(
        input_payload={"input_sha256": input_fingerprint}, measurements=measurements,
        extra_rules_payload={"llm_enabled": llm_enabled,
                             "llm_model": selected_model if llm_enabled else None,
                             "qwen_pcb_strategy": (bool(selected_model) if auto_qwen else
                                                    bool(st.session_state.get(f"{key}_qwen_pcb_strategy", False)))},
    )
    scope = _analysis_scope(project, st.session_state, analysis_target_id)
    record = get_current_state(st.session_state, scope)
    legacy_matches = (
        state.get("input_fingerprint") == input_fingerprint
        and state.get("measurement_fingerprint") == _measurement_fingerprint(project, measurements)
        and state.get("rules_sha256") == fingerprints.rules_sha256
    )
    if state.get("result") and not record.get("results") and not record.get("fingerprints") and legacy_matches:
        migrate_legacy_result(
            st.session_state, scope=scope, fingerprints=fingerprints,
            result=state["result"], legacy_fingerprint=state.get("input_fingerprint"),
        )
    record = get_current_state(st.session_state, scope, fingerprints)
    return {
        "scope": scope, "fingerprints": fingerprints, "record": record,
        "result": get_current_result(st.session_state, scope=scope, fingerprints=fingerprints),
        "input_fingerprint": input_fingerprint,
        "measurement_fingerprint": _measurement_fingerprint(project, measurements),
        "measurements": measurements, "symptoms": selected_codes,
        "symptom_text": symptom_text, "observations": observations,
        "inventory_complete": bool(inventory_complete), "components": components,
        "component_inventory": inventory, "llm_enabled": llm_enabled,
        "selected_model": selected_model,
        "component_id_aliases": component_id_aliases, "board_registry": board_registry,
    }


def get_current_diagnosis_result(
    st: Any, project: ProjectContext, *, components: tuple[Any, ...] = (),
    component_inventory: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    return current_analysis_snapshot(
        st, project, components=components, component_inventory=component_inventory,
    )["result"]


def _measurement_fingerprint(project: ProjectContext, measurements: list[Any]) -> str:
    return build_measurement_fingerprint(measurements)


def _sync_component_candidates(st: Any, project: ProjectContext, result: dict[str, Any]) -> None:
    projects = dict(st.session_state.get("circuitcare_v6_diagnosis_candidates_by_scope", {}))
    diagnosis_state = st.session_state.get(_project_key(project), {})
    scope = _analysis_scope(project, st.session_state)
    projects[scope.key] = build_overlay_state(
        project_id=project.project_id, board_id=project.board_id,
        component_map_version=st.session_state.get("circuitcare_v4_selection_map_key"),
        result={**result, "input_fingerprint": diagnosis_state.get("input_fingerprint")},
        previous=projects.get(scope.key),
    )
    st.session_state["circuitcare_v6_diagnosis_candidates_by_scope"] = projects


def _focus_board_component(st: Any, project: ProjectContext, registry: dict[str, Any] | None,
                           board_component_id: str) -> bool:
    rows = registry.get("components", []) if isinstance(registry, dict) else []
    row = next((item for item in rows if item.get("board_component_id") == board_component_id), None)
    sources = row.get("source_components", []) if row else []
    if not sources:
        return False
    source = sources[0]
    photo_id, local_id = source.get("photo_id"), source.get("component_id")
    st.session_state["circuitcare_v2_photo_choice"] = photo_id
    update_selection(st.session_state, project=project,
                     active_photo_id=photo_id, selected_component=local_id)
    st.session_state[f"cc_analysis_viewer_open_{project.project_id}"] = True
    return True


def _component_rows(components: tuple[Any, ...],
                    component_inventory: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    inventory = component_inventory or {}
    result = []
    for item in components:
        review_owned = getattr(item, "overlay_source", None) in {"yolo", "manual", "qwen_suggestion", "board_registry"}
        presence_status = ("CONFIRMED" if item.identification_status == "USER_CONFIRMED" else "UNCONFIRMED") \
            if review_owned else inventory.get(item.component_id, {}).get("presence_status", "UNCONFIRMED")
        type_status = ("USER_CONFIRMED" if item.identification_status == "USER_CONFIRMED" else "UNCONFIRMED") \
            if review_owned else inventory.get(item.component_id, {}).get(
                "component_type_status", getattr(item, "component_type_status", "UNCONFIRMED"))
        roles = (list(getattr(item, "confirmed_functional_roles", ()) or ())
                 if review_owned else inventory.get(item.component_id, {}).get("confirmed_functional_roles", []))
        result.append({"component_id": item.component_id, "component_type": item.component_type,
             "board_component_id": getattr(item, "board_component_id", None),
             "photo_id": (getattr(item, "source_backend_photo_ids", ())[0]
                          if len(getattr(item, "source_backend_photo_ids", ())) == 1 else
                          getattr(item, "source_photo_ids", ())[0]
                          if len(getattr(item, "source_photo_ids", ())) == 1 else None),
             "source_backend_photo_ids": list(getattr(item, "source_backend_photo_ids", ()) or ()),
             "photo_local_component_id": (getattr(item, "observation_ids", ())[0]
                                          if len(getattr(item, "observation_ids", ())) == 1 else item.component_id),
             "coordinate_frame": getattr(item, "coordinate_frame", "UNKNOWN"),
             "detection_confidence": getattr(item, "detection_confidence", None),
             "identification_status": getattr(item, "identification_status", None),
             "bbox_xyxy": (list(item.bbox_xyxy) if getattr(item, "bbox_xyxy", None) else None),
             "ocr_marking": item.ocr_marking,
             "component_info": getattr(item, "component_info", None),
             "board_component_id": getattr(item, "board_component_id", None),
             "source_photo_ids": list(getattr(item, "source_photo_ids", ()) or ()),
             "identity_status": getattr(item, "identity_status", None),
             "presence_status": presence_status,
             "component_type_status": type_status,
             "part_identity_status": inventory.get(item.component_id, {}).get("part_identity_status", "UNCONFIRMED"),
             "confirmed_functional_roles": roles,
             "candidate_component_types": list(getattr(item, "candidate_component_types", ()) or ()),
             "ai_review_evidence": list(getattr(item, "ai_review_evidence", ()) or ()),
             "ocr_marking_candidates": list(getattr(item, "ocr_marking_candidates", ()) or ()),
             "type_evidence_status": "CONFIRMED" if type_status == "USER_CONFIRMED" else "TENTATIVE",
             "role_status": "CONFIRMED" if roles else "UNKNOWN",
             "role_evidence": ("사용자가 선택한 기능 역할",) if roles else ()})
    return result


def _current_analysis_context(project: ProjectContext, state: Any,
                              component_inventory: dict[str, dict[str, Any]]) -> dict[str, Any]:
    registry = state.get("circuitcare_v7_active_board_registry", {})
    statuses = {
        str(row.get("board_component_id")): "USER_CONFIRMED"
        for row in registry.get("components", [])
        if isinstance(registry, dict) and row.get("presence_status") == "CONFIRMED"
    } if isinstance(registry, dict) else {
        component_id: "USER_CONFIRMED"
        for component_id, row in component_inventory.items()
        if isinstance(row, dict) and row.get("presence_status") == "CONFIRMED"
    }
    return {
        "project_id": project.project_id,
        "board_id": project.board_id or "",
        "component_map_version": (registry.get("version") if isinstance(registry, dict)
                                  else state.get("circuitcare_v4_selection_map_key")),
        "component_identity_statuses": statuses,
    }


def _combined_observations(state: dict[str, Any], answers: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows = list(state.get("parsed_observations", []))
    rows.extend({"code": code, "value": value, "status": "observed", "source": "user_report"}
                for code, value in state.get("manual_observations", {}).items())
    saved_answers = state.get("question_answers", {})
    if isinstance(saved_answers, dict) and saved_answers:
        rows.extend({"code": code, "value": value, "status": "observed", "source": "legacy_follow_up_answer"}
                    for code, value in state.get("legacy_answers", {}).items())
        for test_id, entry in saved_answers.items():
            if not isinstance(entry, dict) or entry.get("answer") in {"답변 취소", "모르겠음", "기타·모르겠음"}:
                continue
            rows.extend({"code": code, "value": value, "status": "observed",
                         "source": "follow_up_answer", "test_id": test_id}
                        for code, value in (entry.get("observations") or {}).items())
    else:
        rows.extend({"code": code, "value": value, "status": "observed", "source": "follow_up_answer"}
                    for code, value in (answers if answers is not None else state.get("answers", {})).items())
    return rows


def _save_follow_up_answer(state: dict[str, Any], question: dict[str, Any], answer: str) -> None:
    """Persist/replace one catalog answer and rebuild compatibility state safely."""
    test_id = str(question.get("test_id") or "")
    if not test_id:
        return
    entries = state.setdefault("question_answers", {})
    if not entries and state.get("answers") and "legacy_answers" not in state:
        state["legacy_answers"] = dict(state.get("answers", {}))
    if answer == "답변 취소":
        entries.pop(test_id, None)
        state["answered_test_ids"] = [value for value in state.get("answered_test_ids", []) if value != test_id]
    elif answer in {"예", "아니오", "모르겠음"}:
        outcomes = question.get("results", {})
        outcome_key = "yes" if answer == "예" else "no"
        outcome = (outcomes.get(outcome_key, outcomes.get(answer == "예", {}))
                   if answer != "모르겠음" and isinstance(outcomes, dict) else {})
        observations = outcome.get("observations", {}) if isinstance(outcome, dict) else {}
        # Unknown is stored as an answer but yields no positive or negative fact.
        entries[test_id] = {"test_id": test_id, "question": question,
                            "answer": answer,
                            "observations": dict(observations) if answer in {"예", "아니오"} else {},
                            "source": "follow_up_answer"}
        state["answered_test_ids"] = list(dict.fromkeys([*state.get("answered_test_ids", []), test_id]))
    state["question_answers"] = entries
    # Legacy readers expect a fact map; conflicts remain represented in the
    # per-question rows consumed by _combined_observations and are not collapsed there.
    flattened: dict[str, Any] = {}
    for entry in entries.values():
        if not isinstance(entry, dict) or entry.get("answer") in {"답변 취소", "모르겠음", "기타·모르겠음"}:
            continue
        for code, value in (entry.get("observations") or {}).items():
            flattened.setdefault(str(code), value)
    state["answers"] = flattened
    plan = state.get("question_plan")
    if isinstance(plan, dict):
        answered_ids = set(str(value) for value in state.get("answered_test_ids", []))
        skipped_ids = set(str(value) for value in plan.get("skipped_question_ids", []))
        for row in plan.get("questions", []):
            if not isinstance(row, dict):
                continue
            question_id = str(row.get("test_id") or "")
            row["status"] = ("SKIPPED" if question_id in skipped_ids else
                             "ANSWERED" if question_id in answered_ids else "PENDING")


INITIAL_SYMPTOM_QUESTION_ID = "INITIAL_SYMPTOM_CAPTURE"
INITIAL_SYMPTOM_OPTIONS = {
    "전원 안 켜짐": "NO_POWER",
    "입력 반응 없음": "SWITCH_NO_RESPONSE",
    "출력 작동 안 함": "OUTPUT_FAILURE",
    "불안정하게 작동": "INTERMITTENT_OPERATION",
    "기타·모르겠음": None,
}


def _save_initial_symptom_answer(state: dict[str, Any], answer: str) -> None:
    """Save the initial symptom choice through the same versioned answer store."""
    if answer not in INITIAL_SYMPTOM_OPTIONS:
        return
    code = INITIAL_SYMPTOM_OPTIONS[answer]
    state.setdefault("question_answers", {})[INITIAL_SYMPTOM_QUESTION_ID] = {
        "test_id": INITIAL_SYMPTOM_QUESTION_ID,
        "question": {"test_id": INITIAL_SYMPTOM_QUESTION_ID, "type": "question",
                     "question": "어떤 문제가 발생하나요?"},
        "answer": answer,
        "observations": {code: True} if code else {},
        "source": "initial_symptom_question",
    }
    state["answered_test_ids"] = list(dict.fromkeys([
        *state.get("answered_test_ids", []), INITIAL_SYMPTOM_QUESTION_ID,
    ]))
    state["initial_symptom_answer"] = answer
    state["symptoms"] = [code] if code else []


def _safe_initial_followup(knowledge: dict[str, Any], answered_test_ids: set[str]) -> dict[str, Any] | None:
    """Pick one unasked catalog question by ontology coverage; never invent a test."""
    tests = knowledge.get("test_catalog", {}).get("tests", {})
    modes = knowledge.get("fault_modes", {}).get("fault_modes", {})
    if not isinstance(tests, dict) or not isinstance(modes, dict):
        return None
    choices: list[tuple[int, str, dict[str, Any]]] = []
    for test_id, test in tests.items():
        if (not isinstance(test, dict) or test.get("type") != "question"
                or str(test_id) in answered_test_ids):
            continue
        observed = {str(code) for outcome in (test.get("results") or {}).values()
                    if isinstance(outcome, dict)
                    for code in (outcome.get("observations") or {})}
        coverage = sum(bool(observed & (set(mode.get("supported_symptoms", {}))
                                       | set(mode.get("supporting_evidence", {}))
                                       | set(mode.get("opposing_evidence", {}))
                                       | set(mode.get("supporting_evidence_when_false", {}))
                                       | set(mode.get("opposing_evidence_when_false", {}))))
                       for mode in modes.values() if isinstance(mode, dict))
        choices.append((-coverage, str(test_id), {"test_id": str(test_id), **test}))
    return min(choices, key=lambda row: (row[0], row[1]))[2] if choices else None


def _persist_diagnosis(project: ProjectContext, state: dict[str, Any], result: dict[str, Any],
                       components: tuple[Any, ...], observations: list[dict[str, Any]]) -> None:
    target_id = str(state.get("analysis_target_id") or "target:default")
    ids_by_target = state.setdefault("diagnosis_ids_by_target", {})
    legacy_id = state.get("diagnosis_id") if state.get("analysis_target_id") in (None, target_id) else None
    diagnosis_id = ids_by_target.setdefault(target_id, legacy_id or str(uuid4()))
    state["diagnosis_id"] = diagnosis_id  # legacy readers still use the most recent ID
    result["diagnosis_id"] = diagnosis_id
    component_records = [ComponentRecord.from_mapping({
        "component_id": item.component_id, "label": item.component_type,
        "bbox_xyxy": (list(item.bbox_xyxy) if getattr(item, "bbox_xyxy", None) else None),
        "detection_confidence": item.detection_confidence,
        "marking": item.ocr_marking,
    }).to_dict() for item in components]
    normalized_observations = [Observation.from_dict(row).to_dict() for row in observations]
    result["component_snapshot"] = component_records
    db_path = Path(__file__).resolve().parents[4] / "data" / "private" / "diagnosis" / "diagnosis.sqlite3"
    ObservationStore(db_path).save(
        diagnosis_id=diagnosis_id, project_id=project.project_id, board_id=project.board_id,
        status=result.get("status", "UNKNOWN"), result=result, observations=normalized_observations,
        input_fingerprint=state.get("input_fingerprint"),
    )


def _apply_qwen_pcb_assist(
    result: dict[str, Any], *, project: ProjectContext, components: list[dict[str, Any]],
    suspected_components: list[dict[str, Any]], selected_model: str, knowledge: dict[str, Any],
    answered_test_ids: list[str], board_registry: dict[str, Any] | None = None,
    measurement_records: list[Any] | None = None, inspection_history: list[dict[str, Any]] | None = None,
) -> None:
    """Attach bounded Qwen proposals without changing rule ranks or verdicts."""
    started = time.perf_counter()
    context = build_pcb_context(components=components, photo_records=project.image_records,
                                diagnosis=result, suspected_components=suspected_components,
                                board_registry=board_registry)
    result["pcb_context_summary"] = {
        "summary": context["summary"],
        "focus_component_ids": [row["component_id"] for row in context["focus_components"]],
        "omitted_component_ids": context["omitted_component_ids"],
        "context_token_estimate": context["context_token_estimate"],
    }
    try:
        result["qwen_fault_proposals"] = suggest_pcb_fault_candidates(
            OllamaClient(auto_start=True, timeout_seconds=120), selected_model,
            pcb_context=context, diagnosis=result, knowledge=knowledge,
        )
    except (LocalLLMError, ValueError, TypeError, OSError) as exc:
        message = str(exc)
        folded = message.casefold()
        category = ("TIMEOUT" if isinstance(exc, TimeoutError) or "timed out" in folded or "timeout" in folded else
                    "TRUNCATED_RESPONSE" if "truncated" in folded or "output token limit" in folded else
                    "INVALID_JSON" if "invalid json" in folded else
                    "INVALID_SCHEMA" if "must be an object" in folded or "schema" in folded else
                    "MODEL_UNAVAILABLE" if "not installed" in folded or "could not reach" in folded else
                    "MODEL_OR_TRANSPORT_ERROR")
        result["qwen_fault_proposals"] = {"proposals": [], "status": "FAILED",
                                          "failure_category": category, "error": message[:400]}
    qwen_proposals = result.get("qwen_fault_proposals", {}).get("proposals", [])
    component_by_id = {}
    for component_row in components:
        if isinstance(component_row, dict):
            for alias in (component_row.get("component_id"), component_row.get("board_component_id")):
                if alias:
                    component_by_id[str(alias)] = component_row
    qwen_proposals = [proposal for proposal in qwen_proposals if isinstance(proposal, dict)
                      and (component := component_by_id.get(str(proposal.get("component_id") or "")))
                      and str(component.get("component_type", component.get("type", "unknown"))).casefold()
                      not in {"unknown", "not_component"}
                      and str(component.get("presence_status") or "").upper() != "EXCLUDED"
                      and str(proposal.get("component_review_status") or "").upper() != "EXCLUDED"]
    if qwen_proposals:
        # Feed validated Qwen role hints into the structural hypothesis view,
        # but keep them explicitly AI_SUGGESTED and separate from rule ranks.
        groups = result.setdefault("functional_block_groups", [])
        role_rows = {}
        for role_row in result.get("functional_blocks", []):
            if isinstance(role_row, dict):
                for alias in (role_row.get("component_id"), role_row.get("board_component_id")):
                    if alias:
                        role_rows[str(alias)] = role_row
        suggested_groups = []
        allowed_roles = {"power", "input", "control", "control_input", "control_timing",
                         "driver", "output", "protection", "thermal", "buzzer", "relay"}
        for proposal in qwen_proposals:
            component_id = str(proposal.get("component_id") or "")
            role = str(proposal.get("functional_block") or "UNKNOWN")
            component = component_by_id.get(component_id)
            if not component or role not in allowed_roles:
                continue
            board_component_id = str(component.get("board_component_id") or component_id)
            role_row = role_rows.get(component_id) or role_rows.get(board_component_id)
            if role_row is not None:
                role_row["ai_suggested_functional_roles"] = list(dict.fromkeys([
                    *role_row.get("ai_suggested_functional_roles", []), role]))
                role_row["candidate_blocks"] = list(dict.fromkeys([
                    *role_row.get("candidate_blocks", []), role]))
                role_row.setdefault("role_evidence", {}).setdefault(role, []).append(
                    "Qwen 기능 역할 제안 · 사용자 확인 전")
            block_id = str(proposal.get("functional_block_id") or "")
            member = {"component_id": board_component_id, "local_component_id": component_id,
                      "board_component_id": board_component_id,
                      "component_type": component.get("component_type", component.get("type", "unknown")),
                      "identity_status": "VERIFIED" if component.get("component_type_status") == "USER_CONFIRMED" else "UNKNOWN",
                      "evidence_status": "UNKNOWN", "role_status": "AI_SUGGESTED",
                      "role_verified": False, "role_evidence": ["Qwen 기능 역할 제안 · 사용자 확인 전"]}
            seed = json.dumps([role, board_component_id, "qwen"], ensure_ascii=False, separators=(",", ":"))
            ai_block_id = "ai-block-" + hashlib.sha256(seed.encode()).hexdigest()[:12]
            suggested_groups.append({"block_id": ai_block_id, "functional_block_id": ai_block_id,
                    "version": hashlib.sha256((seed + "|AI_SUGGESTED").encode()).hexdigest()[:16],
                    "suggested_parent_block_id": block_id or None,
                    "role": role, "title": role.replace("_", " ").title(),
                    "component_ids": [board_component_id], "board_component_ids": [board_component_id],
                    "members": [member], "supporting_evidence": ["Qwen 기능 역할 제안 · 사용자 확인 전"],
                    "opposing_evidence": [], "verification_status": "AI_SUGGESTED",
                    "status": "AI_SUGGESTED", "verified": False, "partition_status": "AI_SUGGESTED",
                    "connectivity_status": "NOT_INFERRED", "unknown_connectivity_component_ids": [board_component_id],
                    "unverified_relations": [],
                    "connectivity_note": "AI 역할 제안은 전기적 연결을 의미하지 않습니다."})
        existing_group_ids = {str(row.get("block_id")) for row in groups if isinstance(row, dict)}
        groups.extend(row for row in suggested_groups if row["block_id"] not in existing_group_ids)
        ledger_rows = [*result.get("evidence_ledger", []), *build_evidence_ledger(
            ai_suggestions=qwen_proposals,
            functional_roles=list({id(row): row for row in role_rows.values()}.values()),
            functional_blocks=groups,
            source_version=str(result.get("input_fingerprint") or "") or None)]
        result["evidence_ledger"] = list({row["evidence_id"]: row for row in ledger_rows}.values())
        result["evidence_gaps"] = analyze_evidence_gaps(
            hypotheses=result.get("hypotheses", []), knowledge=knowledge,
            confirmed_types=set(result.get("component_inventory", {}).get("confirmed_component_types", [])),
            functional_blocks=groups,
            component_candidates=result.get("all_component_candidates",
                                            result.get("component_candidates", [])),
            measurement_evidence=result.get("measurement_evidence", []),
            evidence_ledger=result["evidence_ledger"],
            connectivity_status=result.get("connectivity_status", "none"),
            component_roles=result.get("functional_blocks", []),
        )
        measurement_status = {str(row.get("measurement_id")): row.get("assessment_status", row.get("verdict"))
                              for row in result.get("measurement_evidence", [])
                              if isinstance(row, dict) and row.get("measurement_id")}
        measurement_rows = [row.to_dict() if hasattr(row, "to_dict") else dict(row)
                            for row in measurement_records or [] if isinstance(row, dict) or hasattr(row, "to_dict")]
        for measurement_row in measurement_rows:
            measurement_row["assessment_status"] = measurement_status.get(
                str(measurement_row.get("measurement_id") or ""))
        result["active_inspection_plan"] = plan_active_inspections(
            hypotheses=result.get("hypotheses", []),
            component_candidates=result.get("all_component_candidates",
                                            result.get("component_candidates", [])),
            functional_block_groups=groups, evidence_gaps=result["evidence_gaps"],
            measurements=measurement_rows, observations=result.get("observations", {}),
            inspection_history=inspection_history or [], knowledge=knowledge,
        )
    result["candidate_fusion"] = fuse_inspection_candidates(
        rule_candidates=result.get("all_component_candidates", result.get("component_candidates", [])),
        qwen_proposals=qwen_proposals,
        components=components, suspected_components=suspected_components, top_k=3,
        allowed_fault_ids=set(knowledge.get("fault_modes", {}).get("fault_modes", {})),
    )
    eligible_questions = [row for row in result.get("next_tests", [])
                          if row.get("type") == "question" and row.get("test_id") not in set(answered_test_ids)]
    if len(result.get("hypotheses", [])) > 1 and eligible_questions:
        try:
            result["llm_follow_up_question"] = select_follow_up_question(
                OllamaClient(auto_start=True, timeout_seconds=120), selected_model, pcb_context=context,
                diagnosis=result, answered_test_ids=answered_test_ids,
            )
        except (LocalLLMError, ValueError, TypeError, OSError) as exc:
            result["llm_follow_up_question_error"] = str(exc)
    context_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str)
    qwen_result = result.get("qwen_fault_proposals", {})
    qwen_result = qwen_result if isinstance(qwen_result, dict) else {}
    model_info = qwen_result.get("model_info")
    model_info = model_info if isinstance(model_info, dict) else {}
    result["qwen_run_metadata"] = {
        "model": qwen_result.get("model", selected_model),
        "model_digest": model_info.get("digest"),
        "context_char_count": len(context_json),
        "context_token_estimate": context.get("context_token_estimate"),
        "estimated_request_tokens": int(context.get("context_token_estimate", 0)) + 1200,
        "token_estimate_method": "serialized JSON characters / 3.2 plus fixed prompt/schema reserve; estimate only",
        "omitted_component_count": context.get("omitted_component_count", 0),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "usage": qwen_result.get("usage", {}),
        "failure_category": qwen_result.get("failure_category"),
    }


def _attach_evidence_v1(
    result: dict[str, Any], *, components: list[dict[str, Any]], measurements: list[Any],
    observations: list[dict[str, Any]],
) -> None:
    """Attach an opt-in ranking without changing the baseline result fields."""
    measurement_rows = [row.to_dict() if hasattr(row, "to_dict") else row
                        for row in measurements if isinstance(row, dict) or hasattr(row, "to_dict")]
    try:
        result["evidence_v1"] = rank_by_verified_evidence(
            result, components=components, measurements=measurement_rows,
            observations=observations, top_k=3,
        )
    except (TypeError, ValueError, KeyError):
        result["evidence_v1"] = {"ranking_strategy": "evidence_v1", "component_candidates": [],
                                 "status": "UNAVAILABLE"}


def candidate_rows_for_strategy(result: dict[str, Any], strategy: str) -> list[dict[str, Any]]:
    """Return display rows from exactly one strategy; never merge rankings."""
    if strategy == "candidate_fusion_v1":
        fusion = result.get("candidate_fusion")
        if isinstance(fusion, dict):
            return [row for row in fusion.get("top_candidates", []) if isinstance(row, dict)]
    elif strategy == "evidence_v1":
        evidence = result.get("evidence_v1")
        if isinstance(evidence, dict):
            return [row for row in evidence.get("component_candidates", []) if isinstance(row, dict)]
    rows = result.get("component_candidates", [])
    return [row for row in rows[:3] if isinstance(row, dict)] if isinstance(rows, list) else []


def render_symptom_diagnosis_panel(
    st: Any, project: ProjectContext, *, components: tuple[Any, ...] = (), preview: bool = False,
    component_inventory: dict[str, dict[str, Any]] | None = None,
    analysis_target_id: str | None = None,
    component_id_aliases: dict[str, str | None] | None = None,
    board_registry: dict[str, Any] | None = None,
    suspected_components: list[dict[str, Any]] | None = None,
    simple_ui: bool = False,
    workflow_stage: str | None = None,
    left_progress_slot: Any = None,
    reassessment_run_requested: bool = False,
) -> None:
    """Collect symptoms, run local rules, and render candidates and safe next questions."""
    kb = load_knowledge()
    symptom_items = kb["symptom_ontology"]["symptoms"]
    key = _project_key(project)
    state = st.session_state.setdefault(key, {"manual_observations": {}, "result": None})
    state.setdefault("manual_observations", state.pop("observations", {}))
    if state.pop("measurement_refresh_notice", False):
        st.info("측정 기록이 변경되어 기존 결과를 무효화하고 새 입력으로 원인 후보를 다시 계산했습니다. 측정 근거와 판정 상태를 확인하세요.")
    if state.get("invalidated_reason"):
        st.warning(f"입력이 변경되어 이전 분석 결과를 무효화했습니다 · {state['invalidated_reason']} · 다시 분석해 주세요.")
    reassessment_view = simple_ui and workflow_stage == "REASSESSMENT"
    if simple_ui:
        st.markdown("#### AI 진단")
        initial_key = f"{key}_simple_initial_symptom"
        initial_options = tuple(INITIAL_SYMPTOM_OPTIONS)
        previous_answer = state.get("initial_symptom_answer")
        if st.session_state.get(initial_key) not in initial_options:
            st.session_state[initial_key] = previous_answer if previous_answer in initial_options else initial_options[-1]
        if reassessment_view:
            simple_choice = previous_answer if previous_answer in initial_options else initial_options[-1]
            st.caption(f"기록된 증상 · {simple_choice}")
        else:
            st.caption("증상을 고르거나 짧게 적어 주세요. 소자를 선택하지 않아도 시작할 수 있습니다.")
            simple_choice = st.radio("어떤 문제가 발생하나요?", initial_options, key=initial_key,
                                     horizontal=True)
    else:
        st.markdown("#### 증상 입력 및 원인 분석")
        st.caption("자연어로 증상을 설명하세요. 원인 후보는 검사 전 가설이며 실제 고장 확률로 해석하지 않습니다.")
    description_key = f"{key}_description"
    if description_key not in st.session_state:
        st.session_state[description_key] = project.symptom_description or ""
    if reassessment_view:
        symptom_text = str(st.session_state.get(description_key, project.symptom_description or ""))
        if symptom_text.strip():
            st.caption(f"증상 설명 · {symptom_text}")
    else:
        symptom_text = st.text_area("증상 설명", key=description_key,
                                     placeholder="예: 전원은 켜지지만 LED가 들어오지 않고 버저가 울리지 않음", height=82)
    code_to_label = {code: item["label"] for code, item in symptom_items.items()}
    symptom_widget_key = f"{key}_symptoms"
    _apply_pending_initial_symptom_widget(st.session_state, state, symptom_widget_key)
    if symptom_widget_key not in st.session_state:
        st.session_state[symptom_widget_key] = [
            code_to_label[code] for code in state.get("symptoms", []) if code in code_to_label
        ]
    selected_labels = st.session_state.get(symptom_widget_key, [])
    selected_codes = [code for code, label in code_to_label.items() if label in selected_labels]
    if reassessment_view:
        selected_codes = list(state.get("symptoms", []))
    elif simple_ui:
        selected_code = INITIAL_SYMPTOM_OPTIONS.get(simple_choice)
        selected_codes = [selected_code] if selected_code else []
    component_inventory = component_inventory or {}
    component_rows = _component_rows(components, component_inventory)
    active_component_ids: set[str] = set()
    excluded_component_ids: set[str] = set()
    for component_row in component_rows:
        canonical_id = str(component_row.get("component_id") or "")
        aliases = {canonical_id, str(component_row.get("board_component_id") or ""),
                   str(component_row.get("photo_local_component_id") or "")}
        active_component_ids.update(value for value in aliases if value)
        if component_row.get("presence_status") == "EXCLUDED":
            excluded_component_ids.update(value for value in aliases if value)
    active_suspects = [row for row in (suspected_components or []) if isinstance(row, dict)
                      and ({str(row.get("board_component_id") or ""), str(row.get("component_id") or "")}
                           & active_component_ids)
                      and not ({str(row.get("board_component_id") or ""), str(row.get("component_id") or "")}
                               & excluded_component_ids)
                      and row.get("evidence_status") == "UNVERIFIED"]
    valid_suspects, stale_suspects = _registry_valid_suspects(active_suspects, board_registry)
    if stale_suspects:
        st.warning(f"저장된 사용자 의심 {len(stale_suspects)}개가 사진·Registry 관측과 달라 현재 진단에서 제외했습니다. PCB에서 다시 지정하면 기존 기록은 보존됩니다.")
    confirmed_component_ids = sorted(
        row["component_id"] for row in component_rows if row.get("presence_status") == "CONFIRMED"
    )
    st.caption(f"PCB 검수 목록 · 확인 {len(confirmed_component_ids)} / 표시 {len(component_rows)}개")
    if not simple_ui:
      with st.expander("추가 관측·고급 분석 설정", expanded=False):
        st.multiselect("구조화된 증상", list(code_to_label.values()), key=symptom_widget_key,
                       help="자연어 설명 외에 분류된 증상을 추가합니다.")
        inventory_complete = st.checkbox(
            "PCB 전체 부품 목록을 확인 완료했습니다",
            value=bool(state.get("inventory_complete", False)),
            key=f"{key}_inventory_complete",
            help="누락 여부 분석에만 사용됩니다. 확실하지 않으면 선택하지 마세요.",
        )
        yes_no_unknown = ("모름", "예", "아니오")
        evidence_fields = {
            "POWER_SOURCE_VERIFIED": "외부 전원/배터리 입력이 정상임",
            "BUZZER_VERIFIED_FAILED": "버저 자체가 고장임을 별도 확인함",
            "BUZZER_VERIFIED_WORKING": "버저 자체는 정상임을 별도 확인함",
            "BUZZER_DRIVE_ABNORMAL": "버저 구동 신호 이상을 확인함",
            "BUZZER_DRIVE_VERIFIED_NORMAL": "버저 구동 신호 정상임을 확인함",
            "VISIBLE_SOLDER_DAMAGE": "납땜 손상을 사진/육안으로 확인함",
        }
        observations = dict(state.get("manual_observations", {}))
        for fact, label in evidence_fields.items():
            current = observations.get(fact)
            default = "모름" if current is None else ("예" if current is True else "아니오")
            answer = st.selectbox(label, yes_no_unknown, index=yes_no_unknown.index(default), key=f"{key}_{fact}")
            observations[fact] = True if answer == "예" else False if answer == "아니오" else None
        power_off = st.checkbox("전원을 차단했고 기판이 식은 상태임", key=f"{key}_power_off")
        if power_off:
            observations["POWER_OFF_CONFIRMED"] = True
        else:
            observations.pop("POWER_OFF_CONFIRMED", None)
        model_cache = state.get("ollama_models")
        checked_at = float(state.get("ollama_checked_at", 0))
        if time.time() - checked_at > 30:
            try:
                local_client = OllamaClient(timeout_seconds=2)
                model_cache = [str(item.get("name")) for item in local_client.list_models() if item.get("name")]
                state["ollama_error"] = None
            except (LocalLLMError, ValueError) as exc:
                model_cache = []
                state["ollama_error"] = str(exc)
            state["ollama_models"] = model_cache
            state["ollama_checked_at"] = time.time()
            st.session_state[key] = state
        if model_cache:
            selected_model = st.selectbox("로컬 LLM", model_cache, key=f"{key}_model")
            llm_key = f"{key}_llm_enabled"
            if llm_key not in st.session_state:
                st.session_state[llm_key] = True
            llm_enabled = st.checkbox("LLM 설명 보조 사용", key=llm_key)
            qwen_pcb_strategy = st.checkbox(
                "Qwen PCB 구조 기반 고장 후보 제안 (미검증)",
                key=f"{key}_qwen_pcb_strategy", value=bool(state.get("qwen_pcb_strategy", True)),
                help="규칙 기반 순위와 분리해 보조 제안으로 표시합니다. Qwen 제안만으로 판정을 확정하지 않습니다.",
            )
            state["qwen_pcb_strategy"] = qwen_pcb_strategy
        else:
            qwen_pcb_strategy = False
            st.caption("Ollama를 사용할 수 없어 규칙 기반 분석으로 실행합니다.")
    elif reassessment_view:
        observations = dict(state.get("manual_observations", {}))
        inventory_complete = bool(st.session_state.get(
            f"{key}_simple_inventory_complete", state.get("inventory_complete", False)))
        for fact in ("POWER_SOURCE_VERIFIED", "BUZZER_VERIFIED_FAILED", "BUZZER_VERIFIED_WORKING",
                     "BUZZER_DRIVE_ABNORMAL", "BUZZER_DRIVE_VERIFIED_NORMAL", "VISIBLE_SOLDER_DAMAGE"):
            widget_value = st.session_state.get(f"{key}_simple_{fact}")
            widget_value = {"예": True, "아니오": False, "모름": None}.get(widget_value, widget_value)
            if widget_value in {True, False, None}:
                observations[fact] = widget_value
        if st.session_state.get(f"{key}_simple_power_off", False):
            observations["POWER_OFF_CONFIRMED"] = True
        else:
            observations.pop("POWER_OFF_CONFIRMED", None)
        qwen_pcb_strategy = bool(state.get("qwen_pcb_strategy", True))
    else:
        observations = dict(state.get("manual_observations", {}))
        inventory_complete = bool(state.get("inventory_complete", False))
        qwen_pcb_strategy = True
        with st.expander("추가로 확인한 증상", expanded=False):
            inventory_complete = st.checkbox(
                "PCB 부품 목록을 모두 확인함", value=inventory_complete,
                key=f"{key}_simple_inventory_complete",
            )
            evidence_fields = {
                "POWER_SOURCE_VERIFIED": "외부 전원·배터리 입력이 정상임",
                "BUZZER_VERIFIED_FAILED": "버저 자체 고장을 별도로 확인함",
                "BUZZER_VERIFIED_WORKING": "버저가 정상 작동함을 확인함",
                "BUZZER_DRIVE_ABNORMAL": "버저 구동 신호 이상을 확인함",
                "BUZZER_DRIVE_VERIFIED_NORMAL": "버저 구동 신호가 정상임을 확인함",
                "VISIBLE_SOLDER_DAMAGE": "납땜 손상을 사진이나 육안으로 확인함",
            }
            for fact, label in evidence_fields.items():
                value = observations.get(fact)
                default = "모름" if value is None else "예" if value is True else "아니오"
                answer = st.selectbox(label, ("모름", "예", "아니오"),
                                      index=("모름", "예", "아니오").index(default),
                                      key=f"{key}_simple_{fact}")
                observations[fact] = True if answer == "예" else False if answer == "아니오" else None
            power_off = st.checkbox("전원을 차단했고 기판이 식은 상태임",
                                    value=bool(observations.get("POWER_OFF_CONFIRMED")),
                                    key=f"{key}_simple_power_off")
            if power_off:
                observations["POWER_OFF_CONFIRMED"] = True
            else:
                observations.pop("POWER_OFF_CONFIRMED", None)
    snapshot = current_analysis_snapshot(
        st, project, components=components, component_inventory=component_inventory,
        symptom_text=symptom_text, selected_codes=selected_codes,
        observations=observations, inventory_complete=inventory_complete,
        analysis_target_id=analysis_target_id, component_id_aliases=component_id_aliases,
        board_registry=board_registry, suspected_components=valid_suspects,
        auto_qwen=simple_ui,
    )
    measurements = snapshot["measurements"]
    silkscreen_labels = st.session_state.get("circuitcare_v4_silkscreen_labels", [])
    if not isinstance(silkscreen_labels, list):
        silkscreen_labels = []
    fingerprint = snapshot["input_fingerprint"]
    measurement_fingerprint = snapshot["measurement_fingerprint"]
    llm_enabled = snapshot["llm_enabled"]
    selected_model = snapshot["selected_model"]
    analysis_scope = snapshot["scope"]
    analysis_fingerprints = snapshot["fingerprints"]
    confirmed_component_ids = sorted(
        row["component_id"] for row in _component_rows(components, component_inventory)
        if row.get("presence_status") == "CONFIRMED"
    )
    central_state = get_current_state(st.session_state, analysis_scope)
    legacy_result_matches_inputs = (
        state.get("input_fingerprint") == fingerprint
        and state.get("measurement_fingerprint") == measurement_fingerprint
        and state.get("rules_sha256") == analysis_fingerprints.rules_sha256
    )
    if state.get("result") and not central_state.get("results"):
        migrated = False
        if legacy_result_matches_inputs:
            migrated = migrate_legacy_result(
                st.session_state, scope=analysis_scope, fingerprints=analysis_fingerprints,
                result=state["result"], legacy_fingerprint=state.get("input_fingerprint"),
            )
        if not migrated:
            legacy_history = list(state.get("history", []))
            legacy_history.append({"result": state["result"],
                                   "reason": "중앙 상태 버전 전환 전의 진단 기록"})
            state["history"] = legacy_history[-100:]
            state["result"] = None
    central_state = get_current_state(st.session_state, analysis_scope, analysis_fingerprints)
    current_result = get_current_result(
        st.session_state, scope=analysis_scope, fingerprints=analysis_fingerprints,
    )
    previous_measurement_fingerprint = state.get("measurement_fingerprint")
    prior_analysis_exists = bool(state.get("result")) or state.get("analysis_status") == "CURRENT"
    # Measurement writes already invalidate the central result. Queue one run
    # for this exact fingerprint; subsequent reruns see the same fingerprint
    # and cannot enqueue a duplicate diagnosis.
    _queue_measurement_change_rerun(
        state, previous_fingerprint=previous_measurement_fingerprint,
        current_fingerprint=measurement_fingerprint,
        prior_analysis_exists=prior_analysis_exists, current_result=current_result,
        central_status=central_state.get("status", ""),
    )
    if state.get("input_fingerprint") != fingerprint or state.get("measurement_fingerprint") != measurement_fingerprint:
        if state.get("result") and not current_result:
            state["invalidated_reason"] = central_state.get("reason") or "분석 입력 변경"
        state["input_fingerprint"] = fingerprint
        state["measurement_fingerprint"] = measurement_fingerprint
        state["result"] = current_result
        state["analysis_status"] = central_state.get("status", "NOT_RUN")
        st.session_state[key] = state
    elif current_result is not None:
        state["result"] = current_result
        state["analysis_status"] = "CURRENT"
    else:
        state["result"] = None
        state["analysis_status"] = central_state.get("status", "NOT_RUN")
    run_clicked = False
    if not reassessment_view:
        run_clicked = st.button("AI 진단 시작" if not current_result else "다시 분석",
                                type="primary", disabled=preview, key=f"{key}_run")
    elif (current_result is None and central_state.get("status") in {
            AnalysisStatus.NOT_RUN.value, AnalysisStatus.FAILED.value, AnalysisStatus.STALE.value,
    }):
        run_clicked = st.button("재판정 실행" if state.get("last_analysis_attempt_fingerprint")
                                != analysis_fingerprints.final_sha256 else "재판정 다시 시도",
                                type="primary", disabled=preview,
                                key=f"{key}_reassessment_retry")
    run_after_measurement_change = _consume_measurement_change_rerun(state, measurement_fingerprint)
    reuse_qwen_for_measurement_change = bool(
        run_after_measurement_change
        and state.get("last_qwen_input_fingerprint") == fingerprint
    )
    run_after_initial_answer = bool(state.pop("run_after_initial_answer", False))
    # A stage transition requests one automatic run. Failures and stale commits
    # must wait for an explicit retry click; otherwise every rerun can enqueue
    # another diagnosis for the same input version.
    run_after_reassessment = bool(
        reassessment_view and reassessment_run_requested and current_result is None
        and central_state.get("status") != AnalysisStatus.RUNNING.value and not preview
    )
    run_requested = bool(run_clicked or run_after_initial_answer or run_after_measurement_change
                         or run_after_reassessment)
    if run_requested:
        LOGGER.info(
            "diagnosis_run_requested project=%s stage=%s scope=%s trigger=%s symptoms=%d "
            "symptom_text=%s current_before_run=%s model=%s",
            project.project_id, workflow_stage or "standalone", analysis_scope.key,
            "button" if run_clicked else "measurement_change" if run_after_measurement_change
            else "reassessment_missing_current" if run_after_reassessment
            else "initial_symptom_answer", len(selected_codes), bool(symptom_text.strip()),
            bool(current_result), selected_model or "rules_only",
        )
    if run_requested:
        if left_progress_slot is not None:
            progress_container = left_progress_slot.container()
            analysis_progress = progress_container.status(
                "측정 결과 재판정 중" if reassessment_view else "고장 후보 분석 중", expanded=True)
        else:
            analysis_progress = st.status(
                "측정 결과 재판정 중" if reassessment_view else "고장 후보 분석 중", expanded=True)
        # The simple workflow intentionally does not start Ollama merely by
        # opening the page. Start it only after the user requests diagnosis;
        # otherwise the earlier non-starting model probe leaves Qwen disabled
        # for the entire run even though the client supports on-demand startup.
        if simple_ui and not selected_model:
            try:
                analysis_progress.write("로컬 Qwen 서버 시작 · 진행 중")
                available_models = [
                    str(row.get("name") or row.get("model") or "")
                    for row in OllamaClient(timeout_seconds=30, auto_start=True).list_models()
                    if isinstance(row, dict)
                ]
                qwen_models = [name for name in available_models if name.casefold().startswith("qwen")]
                selected_model = (
                    next((name for name in qwen_models if name.casefold() == "qwen3.5:9b"), None)
                    or next((name for name in qwen_models if "3.5" in name.casefold()), None)
                    or (qwen_models[0] if qwen_models else None)
                )
                state["ollama_models"] = available_models
                state["ollama_checked_at"] = time.time()
                state["ollama_error"] = (
                    None if selected_model else "Ollama에 연결됐지만 사용할 Qwen 모델이 설치되어 있지 않습니다."
                )
                llm_enabled = bool(selected_model)
                qwen_pcb_strategy = bool(selected_model)
                if selected_model:
                    analysis_progress.write(f"로컬 Qwen 연결 · {selected_model}")
                    # The LLM setting is part of the analysis fingerprint. Refresh
                    # it after on-demand startup so the result commits under the
                    # exact engine configuration used for this run.
                    snapshot = current_analysis_snapshot(
                        st, project, components=components, component_inventory=component_inventory,
                        symptom_text=symptom_text, selected_codes=selected_codes,
                        observations=observations, inventory_complete=inventory_complete,
                        analysis_target_id=analysis_target_id, component_id_aliases=component_id_aliases,
                        board_registry=board_registry, suspected_components=valid_suspects,
                        auto_qwen=True,
                    )
                    fingerprint = snapshot["input_fingerprint"]
                    measurement_fingerprint = snapshot["measurement_fingerprint"]
                    analysis_fingerprints = snapshot["fingerprints"]
                    analysis_scope = snapshot["scope"]
                    measurements = snapshot["measurements"]
                    silkscreen_labels = st.session_state.get("circuitcare_v4_silkscreen_labels", [])
                    if not isinstance(silkscreen_labels, list):
                        silkscreen_labels = []
            except (LocalLLMError, ValueError, OSError) as exc:
                state["ollama_models"] = []
                state["ollama_checked_at"] = time.time()
                state["ollama_error"] = str(exc)[:300]
                llm_enabled = False
                selected_model = None
                qwen_pcb_strategy = False
                analysis_progress.write("로컬 Qwen 연결 실패 · 규칙 기반 진단은 계속합니다")
        analysis_progress.write("증상·검수·현재 측정 Fingerprint · 확인 완료")
        if simple_ui:
            _save_initial_symptom_answer(state, simple_choice)
        state["symptoms"] = selected_codes
        state["confirmed_component_ids"] = confirmed_component_ids
        state["inventory_complete"] = inventory_complete
        state["manual_observations"] = {fact: value for fact, value in observations.items() if value is not None}
        # The initial answer is written above on an explicit button click and
        # participates in the input fingerprint. Rebuild the run snapshot only
        # after that write; otherwise commit_analysis_result correctly rejects
        # every completed run as stale, leaving the next workflow step disabled.
        st.session_state[key] = state
        snapshot = current_analysis_snapshot(
            st, project, components=components, component_inventory=component_inventory,
            symptom_text=symptom_text, selected_codes=selected_codes,
            observations=observations, inventory_complete=inventory_complete,
            analysis_target_id=analysis_target_id, component_id_aliases=component_id_aliases,
            board_registry=board_registry, suspected_components=valid_suspects,
            auto_qwen=simple_ui,
        )
        fingerprint = snapshot["input_fingerprint"]
        measurement_fingerprint = snapshot["measurement_fingerprint"]
        analysis_fingerprints = snapshot["fingerprints"]
        analysis_scope = snapshot["scope"]
        measurements = snapshot["measurements"]
        # Remember that the user explicitly submitted this exact report. The
        # workflow may still continue to safe inspection when diagnosis is
        # inconclusive or a model/commit fails; this is never treated as a
        # confirmed fault or a normal-board result.
        state["last_analysis_attempt_fingerprint"] = analysis_fingerprints.final_sha256
        if simple_ui and workflow_stage == "QUESTIONS" and run_clicked:
            # The next workbench run advances only after it sees this exact
            # fingerprint committed as CURRENT. A stale/failed run stays here.
            state["workflow_auto_advance"] = {
                "scope_key": analysis_scope.key,
                "fingerprint": analysis_fingerprints.final_sha256,
            }
        st.session_state[key] = state
        run_id = begin_analysis(
            st.session_state, scope=analysis_scope, fingerprints=analysis_fingerprints,
            result_kind="symptom",
        )
        component_rows = _component_rows(components, component_inventory)
        parsed = {"observations": [], "uncertain_facts": [], "parser_status": "SKIPPED"}
        if (llm_enabled and selected_model and symptom_text.strip()
                and not reuse_qwen_for_measurement_change):
            try:
                analysis_progress.write("증상 구조화 · 진행 중")
                with st.spinner("AI가 증상을 정리하는 중…" if simple_ui else f"{selected_model}이 증상을 구조화하는 중…"):
                    parsed = parse_user_report(OllamaClient(auto_start=True, timeout_seconds=120), selected_model, symptom_text)
                analysis_progress.write("증상 구조화 · 완료")
            except (LocalLLMError, ValueError, TypeError, OSError) as exc:
                parsed = {"observations": [], "uncertain_facts": [], "parser_status": "FAILED", "error": str(exc)}
                LOGGER.warning("diagnosis symptom_structuring_failed project=%s model=%s error=%s",
                               project.project_id, selected_model, str(exc)[:500])
                analysis_progress.write("증상 구조화 · 실패 · 규칙 진단 계속")
        elif reuse_qwen_for_measurement_change:
            cached_parsed = state.get("parsed_observations")
            parsed = ({"observations": cached_parsed, "uncertain_facts": [],
                       "parser_status": "REUSED_SYMPTOM_INPUT"}
                      if isinstance(cached_parsed, list) else
                      {"observations": [], "uncertain_facts": [], "parser_status": "SKIPPED"})
            analysis_progress.write("측정 변경 · 증상 구조화 결과 재사용 · Qwen 호출 생략")
        state["parsed_observations"] = parsed.get("observations", [])
        combined_observations = _combined_observations(state)
        parsed_symptoms = [row["code"] for row in state["parsed_observations"]
                           if row.get("status") in {"observed", "measured"} and row.get("value") is True
                           and row.get("code") in symptom_items]
        try:
            analysis_progress.write("규칙 기반 고장 후보 분석 · 진행 중")
            result = diagnose(
                symptoms=list(dict.fromkeys([*selected_codes, *parsed_symptoms])),
                symptom_text=symptom_text,
                observations=combined_observations, components=component_rows,
                confirmed_component_ids=confirmed_component_ids,
                inventory_complete=inventory_complete,
                measurements=[row.to_dict() for row in measurements],
                current_context=_current_analysis_context(project, st.session_state, component_inventory),
                silkscreen_labels=silkscreen_labels,
                connectivity_status="none",
            )
            analysis_progress.write("규칙 기반 고장 후보 분석 · 완료")
        except Exception as exc:
            fail_analysis(st.session_state, scope=analysis_scope, run_id=run_id, error=str(exc))
            state["result"] = None
            state["analysis_status"] = "FAILED"
            st.session_state[key] = state
            analysis_progress.update(label="고장 분석 실패 · 입력과 측정 기록은 보존됩니다",
                                     state="error", expanded=True)
            st.error(f"규칙 기반 분석을 완료하지 못했습니다: {exc}")
            return
        # If the user reached the end of the safe catalog without identifying
        # an ontology symptom, stop the question loop as explicitly inconclusive.
        answered_followups = set(str(value) for value in state.get("answered_test_ids", [])) - {
            INITIAL_SYMPTOM_QUESTION_ID,
        }
        if (not result.get("symptom_codes") and state.get("initial_symptom_answer") == "기타·모르겠음"
                and answered_followups
                and _safe_initial_followup(kb, set(str(value) for value in state.get("answered_test_ids", []))) is None):
            result["status"] = "INCONCLUSIVE"
            result["message"] = "안전 질문으로도 증상을 특정하지 못했습니다. 추가 정보나 측정 없이 소자 고장을 추정하지 않습니다."
        result["llm_parser"] = parsed
        result["symptom_text"] = symptom_text
        result["silkscreen_labels"] = silkscreen_labels
        _attach_evidence_v1(result, components=component_rows, measurements=measurements,
                            observations=combined_observations)
        if (result.get("status") != "NEEDS_SYMPTOMS" and result.get("symptom_codes")
                and llm_enabled and selected_model and qwen_pcb_strategy
                and not reuse_qwen_for_measurement_change):
            with st.spinner("AI가 PCB 구조를 검토하는 중…" if simple_ui else f"{selected_model}이 PCB 구조를 참고해 미검증 후보를 검토하는 중…"):
                analysis_progress.write("Qwen PCB 보조 검토 · 진행 중")
                _apply_qwen_pcb_assist(
                    result, project=project, components=component_rows,
                    suspected_components=valid_suspects, selected_model=selected_model,
                    knowledge=kb, answered_test_ids=list(state.get("answered_test_ids", [])),
                    board_registry=board_registry,
                    measurement_records=measurements,
                )
            if result.get("qwen_pcb_assist_error"):
                analysis_progress.write("Qwen PCB 보조 검토 · 실패 · 규칙 후보 유지")
            else:
                analysis_progress.write("Qwen PCB 보조 검토 · 완료")
        if (len(result.get("hypotheses", [])) < 3 and llm_enabled and selected_model
                and not reuse_qwen_for_measurement_change):
            try:
                with st.spinner("AI가 추가 가능성을 확인하는 중…" if simple_ui else f"{selected_model}이 미등록 원인 가능성을 별도 검토하는 중…"):
                    result["llm_unverified_suggestions"] = suggest_unregistered_causes(
                        OllamaClient(auto_start=True, timeout_seconds=120), selected_model,
                        symptom_text=symptom_text, diagnosis=result,
                    )
            except (LocalLLMError, ValueError, TypeError, OSError) as exc:
                result["llm_suggestion_error"] = str(exc)
        if (llm_enabled and selected_model and result.get("hypotheses")
                and not reuse_qwen_for_measurement_change):
            try:
                with st.spinner("AI가 근거를 정리하는 중…" if simple_ui else f"{selected_model}이 근거를 정리하는 중…"):
                    result["llm_assist"] = explain_candidates(
                        OllamaClient(auto_start=True, timeout_seconds=120), selected_model,
                        symptom_text=symptom_text, diagnosis=result,
                    )
            except (LocalLLMError, ValueError, TypeError, OSError) as exc:
                result["llm_error"] = str(exc)
        if (llm_enabled and selected_model and result.get("component_candidates")
                and not reuse_qwen_for_measurement_change):
            try:
                with st.spinner("AI가 소자별 검사 이유를 정리하는 중…" if simple_ui else f"{selected_model}이 소자별 검사 이유를 짧게 정리하는 중…"):
                    result["llm_component_explanations"] = explain_component_candidates(
                        OllamaClient(auto_start=True, timeout_seconds=120), selected_model,
                        candidates=result["component_candidates"],
                    )
            except (LocalLLMError, ValueError, TypeError, OSError) as exc:
                result["llm_component_error"] = str(exc)
        if llm_enabled and selected_model:
            if reuse_qwen_for_measurement_change:
                LOGGER.info("diagnosis_qwen_skipped project=%s reason=measurement_only input_fingerprint=%s",
                            project.project_id, str(fingerprint)[:16])
            else:
                state["last_qwen_input_fingerprint"] = fingerprint
        analysis_progress.write("검사 추천 준비 · 완료")
        live_snapshot = current_analysis_snapshot(
            st, project, components=components, component_inventory=component_inventory,
            symptom_text=symptom_text, selected_codes=selected_codes,
            observations=observations, inventory_complete=inventory_complete,
            analysis_target_id=analysis_target_id, component_id_aliases=component_id_aliases,
            board_registry=board_registry, suspected_components=valid_suspects,
            auto_qwen=simple_ui,
        )
        committed = commit_analysis_result(
            st.session_state, scope=analysis_scope, run_id=run_id,
            fingerprints=analysis_fingerprints, current_fingerprints=live_snapshot["fingerprints"],
            current_scope=live_snapshot["scope"],
            result=result, result_kind="symptom",
        )
        if not committed:
            LOGGER.warning(
                "diagnosis result_commit_rejected project=%s scope=%s run_id=%s "
                "started_fingerprint=%s live_fingerprint=%s",
                project.project_id, analysis_scope.key, run_id,
                analysis_fingerprints.final_sha256, live_snapshot["fingerprints"].final_sha256,
            )
            state["result"] = None
            state["analysis_status"] = "STALE"
            state["invalidated_reason"] = "분석 중 입력이 바뀌어 이전 버전 결과를 저장하지 않았습니다."
            st.session_state[key] = state
            analysis_progress.update(label="분석 입력이 바뀌어 결과를 저장하지 않았습니다",
                                     state="error", expanded=True)
            st.warning(state["invalidated_reason"])
            return
        LOGGER.info(
            "diagnosis_result_committed project=%s stage=%s scope=%s run_id=%s "
            "fingerprint=%s result_status=%s hypotheses=%d component_candidates=%d next_tests=%d",
            project.project_id, workflow_stage or "standalone", analysis_scope.key, run_id,
            analysis_fingerprints.final_sha256[:16], result.get("status", "UNKNOWN"),
            len(result.get("hypotheses", [])), len(result.get("component_candidates", [])),
            len(result.get("next_tests", [])),
        )
        state["analysis_target_id"] = analysis_scope.target_id
        st.session_state["circuitcare_v7_analysis_target_id"] = analysis_scope.target_id
        _persist_diagnosis(project, state, result, components, combined_observations)
        _sync_component_candidates(st, project, result)
        state["result"] = result
        state["analysis_status"] = "CURRENT"
        state.pop("invalidated_reason", None)
        state["input_fingerprint"] = fingerprint
        state["measurement_fingerprint"] = measurement_fingerprint
        st.session_state[key] = state
        has_llm_error = (
            result.get("llm_parser", {}).get("parser_status") == "FAILED"
            or any(result.get(field) for field in (
                "llm_suggestion_error", "llm_error", "llm_component_error",
                "qwen_pcb_assist_error",
            ))
        )
        analysis_progress.update(
            label=("분석 완료 · 일부 AI 보조 실패, 규칙 결과 유지" if has_llm_error
                   else f"분석 완료 · 원인 후보 {len(_simple_candidate_rows(result))}개"),
            state="complete", expanded=False,
        )
        st.rerun()
    result = current_result
    if not result:
        if not reassessment_view:
            st.info("증상 입력을 마치고 **AI 진단 시작**을 누르세요.")
        return
    if simple_ui:
        _render_simple_diagnosis_result(
            st, project, result, state=state, knowledge=kb, preview=preview,
            board_registry=board_registry,
            suspected_components=valid_suspects,
            show_questions=workflow_stage != "REASSESSMENT",
            show_candidates=workflow_stage != "QUESTIONS",
        )
        return
    if result.get("evidence_conflicts"):
        st.warning("서로 다른 입력이 있어 해당 사실은 확정하지 않았습니다. 증상/측정값을 확인해 주세요.")
        for conflict in result["evidence_conflicts"]:
            st.caption(f"충돌 · {conflict['code']} · {conflict['values']} · {conflict['resolution']}")
    if result["status"] == "NEEDS_SYMPTOMS":
        st.info("PCB 사진과 소자 목록을 준비했습니다. 사진만으로 전기적 고장을 확정하지 않습니다.")
        initial_answer = state.get("initial_symptom_answer")
        if not initial_answer:
            choice_key = f"{key}_initial_symptom"
            options = tuple(INITIAL_SYMPTOM_OPTIONS)
            if choice_key not in st.session_state:
                st.session_state[choice_key] = options[0]
            initial_choice = st.selectbox("어떤 문제가 발생하나요?", options, key=choice_key)
            if st.button("답변으로 분석 시작", key=f"{key}_submit_initial_symptom",
                         type="primary", disabled=preview):
                _save_initial_symptom_answer(state, initial_choice)
                code = INITIAL_SYMPTOM_OPTIONS[initial_choice]
                state["_pending_initial_symptom_widget_value"] = [code_to_label[code]] if code else []
                st.session_state[key] = state
                st.rerun()
        elif initial_answer == "기타·모르겠음":
            st.caption("모르겠다면 기존 안전 질문에서 다음 확인 항목을 안내합니다.")
            fallback = _safe_initial_followup(kb, set(str(value) for value in state.get("answered_test_ids", [])))
            if fallback:
                st.markdown(f"**확인 질문** · {fallback.get('question', '')}")
                answer_key = f"{key}_answer_{fallback['test_id']}"
                answer = st.selectbox("답변", ("미답변", "예", "아니오", "모르겠음"), key=answer_key)
                if st.button("답변 저장", key=f"{key}_fallback_submit",
                             disabled=preview or answer == "미답변"):
                    _save_follow_up_answer(state, fallback, answer)
                    state["run_after_initial_answer"] = True
                    st.session_state[key] = state
                    st.rerun()
    inventory = result.get("component_inventory", {})
    if not inventory.get("confirmed_count"):
        st.info("실제 존재한다고 확인한 소자가 없습니다. 소자 내부 고장 후보는 생성되지 않았습니다.")
    if not inventory.get("complete"):
        st.caption("전체 부품 목록은 아직 미확정입니다. 미검출 소자가 실제로 없는지 판단하지 않으며, 선택한 실재 확인 ID만 부품 고장 후보에 사용합니다.")
    st.markdown("#### 원인 후보")
    st.success(result["message"])
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
                 help="전략별 후보 목록을 따로 표시합니다. 점수는 고장 확률이 아닙니다.")
    selected_strategy = st.session_state.get(strategy_key, "baseline")
    component_candidates = candidate_rows_for_strategy(result, selected_strategy)
    st.caption(f"표시 전략 · {strategy_labels.get(selected_strategy, selected_strategy)} · 후보 순위는 검사 검토 순서입니다.")
    status_labels = {
        "SUSPECTED": "검사 후보 · 아직 미확정",
        "FAULT_SUPPORTED": "검증된 검사 · 해당 고장 가설 지지",
        "FAULT_REFUTED": "검증된 검사 · 해당 고장 가설 반박",
        "INCONCLUSIVE": "측정 판정 불가",
        "CONFLICTED": "측정 결과 충돌 · 판단 보류",
    }
    if component_candidates:
        st.markdown("**우선 확인할 후보 · 최대 3개**")
        explanations = (result.get("llm_component_explanations", {}).get("explanations", {})
                        if isinstance(result.get("llm_component_explanations"), dict) else {})
        for candidate in component_candidates:
            label = status_labels.get(candidate.get("visual_status"), "판정 불가")
            display_rank = candidate.get("candidate_rank", candidate.get("rank", 0))
            display_id = candidate.get("board_component_id") or candidate.get("component_id", "?")
            st.markdown(f"**{display_rank}. {display_id}** · "
                        f"{candidate.get('component_type', '종류 미확정')} · "
                        f"{candidate.get('fault_title', candidate.get('fault_id', '원인 후보'))} · {label}")
            reason = explanations.get(candidate["component_id"])
            if reason:
                st.caption(reason)
            with st.expander("이 소자 후보의 근거·판정", expanded=False):
                supporting = candidate.get("supporting_evidence") or candidate.get("evidence", [])
                opposing = candidate.get("opposing_evidence", [])
                st.caption("뒷받침 · " + (" · ".join(supporting[:5]) if supporting else "직접 근거 없음"))
                st.caption("반대 근거 · " + (" · ".join(opposing[:5]) if opposing else "확인된 반대 근거 없음"))
                if candidate.get("measurement_reason"):
                    st.caption("측정 판정 · " + str(candidate["measurement_reason"]))
                elif candidate.get("recommended_tests"):
                    st.caption("다음 검사 · " + " · ".join(candidate["recommended_tests"][:3]))
                elif not supporting:
                    st.caption("부품 종류와 증상으로 좁힌 미확정 검사 후보입니다.")
            if any(item.component_id == candidate.get("component_id") for item in components):
                if st.button("PCB에서 위치 보기", key=candidate_focus_widget_key("symptom", project.project_id, project.board_id,
                                                        str(candidate.get("board_component_id") or candidate.get("component_id") or ""),
                                                        str(candidate.get("fault_hypothesis") or candidate.get("fault_id") or ""))):
                    focused = (_focus_board_component(st, project, board_registry,
                               str(candidate.get("component_id"))) if board_registry else False)
                    if not focused:
                        update_selection(st.session_state, project=project,
                                         selected_component=candidate.get("component_id"))
                        st.session_state[f"cc_analysis_viewer_open_{project.project_id}"] = True
                    st.rerun()
        if result.get("llm_component_explanations"):
            st.caption(f"후보별 설명 · {result['llm_component_explanations'].get('model', '')} · 후보 추가/확정 없음")
        elif result.get("llm_component_error"):
            st.caption(f"LLM 후보 설명을 만들지 못해 규칙 근거를 표시합니다 · {result['llm_component_error']}")
    else:
        if inventory.get("unconfirmed_detection_ids"):
            st.caption("확인되지 않은 자동 검출은 고장 후보에서 제외했습니다. 실제 존재하는 부품만 위 목록에서 선택해 다시 추론하세요.")
        else:
            st.caption("현재 근거로 개별 부품을 좁힐 수 없습니다. 소자를 임의로 강조하지 않았습니다.")
    if valid_suspects:
        st.caption("사용자가 지정한 의심 단서 · 미검증이며 규칙 근거/측정 판정이 아닙니다.")
        mode_table = kb.get("fault_modes", {}).get("fault_modes", {})
        for suspect in valid_suspects:
            suspect_id = str(suspect.get("board_component_id") or suspect.get("component_id"))
            mode_id = suspect.get("suspected_fault_mode")
            mode_title = mode_table.get(mode_id, {}).get("title", "AI가 판단") if mode_id else "AI가 판단"
            st.caption(f"사용자 의심 · {suspect_id} · {mode_title} · 미검증")
    qwen_proposals = result.get("qwen_fault_proposals", {}).get("proposals", [])
    if qwen_proposals:
        st.markdown("**Qwen PCB 구조 기반 제안 · 미검증 · 규칙 순위와 별도**")
        for index, proposal in enumerate(qwen_proposals[:3], start=1):
            fault_title = kb.get("fault_modes", {}).get("fault_modes", {}).get(
                proposal.get("suspected_fault_id"), {}).get("title", proposal.get("suspected_fault_id"))
            st.markdown(f"{index}. **{proposal['component_id']} · {fault_title}** · 미검증 제안")
            if proposal.get("functional_block") not in (None, "UNKNOWN"):
                st.caption(f"기능부 후보 · {proposal['functional_block']} · 미확인 추정")
            if proposal.get("reason"):
                st.caption(proposal["reason"])
            if proposal.get("supporting_evidence_ids"):
                st.caption("참조한 입력 근거 · " + " · ".join(proposal["supporting_evidence_ids"]))
            if proposal.get("opposing_evidence_ids"):
                st.caption("반대 입력 근거 · " + " · ".join(proposal["opposing_evidence_ids"]))
            if proposal.get("missing_information"):
                st.caption("부족한 정보 · " + " · ".join(proposal["missing_information"]))
            if proposal.get("recommended_next_action"):
                st.caption("다음 작업 · " + proposal["recommended_next_action"])
            if st.button("PCB에서 위치 보기", key=candidate_focus_widget_key("symptom_qwen", project.project_id, project.board_id,
                                                            str(proposal.get("board_component_id") or proposal.get("component_id") or ""),
                                                            str(proposal.get("suspected_fault_id") or proposal.get("fault_id") or ""))):
                focused = (_focus_board_component(st, project, board_registry, proposal["component_id"])
                           if board_registry else False)
                if not focused and any(item.component_id == proposal["component_id"] for item in components):
                    update_selection(st.session_state, project=project,
                                     selected_component=proposal["component_id"])
                    focused = True
                if focused:
                    st.session_state[f"cc_analysis_viewer_open_{project.project_id}"] = True
                st.rerun()
    elif result.get("qwen_fault_proposals", {}).get("status") == "FAILED":
        failure = result.get("qwen_fault_proposals", {})
        st.caption(f"Qwen 구조 제안 실패 · {failure.get('failure_category', 'MODEL_ERROR')} · 기존 규칙 기반 결과는 유지됩니다.")
    if result.get("llm_assist"):
        assist = result["llm_assist"]
        st.markdown("**로컬 LLM 설명 보조**")
        st.write(assist["summary"])
        st.caption(f"모델 · {assist['model']} · 후보 추가/확정 없음")
        for note in assist.get("candidate_notes", []):
            st.caption(f"{note['fault_id']} · {note['explanation']}")
            st.caption("사용 근거 · " + " · ".join(note.get("evidence_used", [])))
    elif result.get("llm_error"):
        st.warning(f"로컬 LLM 설명 보조 실패 · {result['llm_error']} · 규칙 기반 후보는 유지됩니다.")
    parsed = result.get("llm_parser", {})
    if parsed.get("observations"):
        with st.expander("LLM이 구조화한 사용자 관측값", expanded=False):
            for row in parsed["observations"]:
                label = symptom_items.get(row["code"], kb["symptom_ontology"].get("observations", {}).get(row["code"], {})).get("label", row["code"])
                st.caption(f"{label} · {'예' if row['value'] else '아니오'} · 원문: {row.get('evidence_quote', '')}")
            for item in parsed.get("uncertain_facts", []):
                st.caption(f"미분류/불확실 · {item}")
    if parsed.get("error"):
        st.caption(f"LLM 증상 파싱 실패 · {parsed['error']} · 키워드 규칙으로 처리했습니다.")
    if result.get("llm_unverified_suggestions"):
        with st.expander("LLM 미등록 원인 제안 · 순위 미반영", expanded=False):
            for suggestion in result["llm_unverified_suggestions"]:
                st.markdown(f"**{suggestion['title']}** · 미검증 제안")
                st.caption(suggestion["reason"])
                if suggestion["evidence_used"]:
                    st.caption("참조 관측 · " + " · ".join(suggestion["evidence_used"]))
    if result.get("matched_rules"):
        st.caption("적용 규칙 · " + " · ".join(row["name"] for row in result["matched_rules"]))
    groups = result.get("hypothesis_groups", {})
    for group_key, group_label in (("component", "소자 내부·기능 블록 원인"),
                                   ("board", "기판·납땜 원인"),
                                   ("system", "외부·시스템 원인")):
        rows = groups.get(group_key, []) if isinstance(groups, dict) else []
        if not rows:
            continue
        st.markdown(f"**{group_label}**")
        for index, row in enumerate(rows[:5], start=1):
            title = f"{index}. {row['title']} · 미확정 후보"
            with st.expander(title, expanded=(group_key == "component" and index == 1)):
                if row["supporting_evidence"]:
                    st.markdown("**뒷받침한 사실** · " + " · ".join(row["supporting_evidence"]))
                else:
                    st.caption("직접 뒷받침하는 사실은 아직 없습니다.")
                if row["opposing_evidence"]:
                    st.markdown("**반대 사실** · " + " · ".join(row["opposing_evidence"]))
                if row.get("related_component_ids"):
                    st.caption("확인된 관련 부품 · " + " · ".join(row["related_component_ids"]))
                st.caption("상태 · 검증 전 후보. 이 결과만으로 부품 고장을 확정하지 않습니다.")
    if result.get("functional_blocks"):
        with st.expander("부품 기능 역할 추정", expanded=False):
            for block in result["functional_blocks"]:
                st.caption(f"{block['component_id']} · {block['component_type']} → "
                           f"{', '.join(block['candidate_blocks']) or '역할 미분류'} · 배치/OCR 기반 미확정 추정")
                identity = block.get("ic_identity")
                if identity:
                    st.caption(f"IC 각인 {identity.get('marking') or '없음'} · "
                               f"{', '.join(identity.get('role_candidates', [])) or '기능 미분류'} · "
                               f"{identity.get('status')} · 데이터시트 검증 전")
    if result.get("measurement_evidence"):
        with st.expander("진단에 연결된 측정 기록", expanded=False):
            for row in result["measurement_evidence"]:
                st.caption(f"{row['component_id']} · {row['measurement_type']} {row['value']} {row['unit']} · "
                           f"{row['verification_status']}")
            st.caption("빨강/초록은 검사 프로파일과 출처가 검증된 측정이 고장 가설을 직접 지지/반박할 때만 표시됩니다.")
    if result.get("joint_candidates"):
        with st.expander("소자 내부 고장과 분리한 납땜·기판 후보", expanded=False):
            for candidate in result["joint_candidates"]:
                st.caption(f"{candidate['title']} · 미확정 · " + " · ".join(candidate.get("evidence", [])))
    questions = [item for item in result.get("next_tests", []) if item.get("type") == "question"]
    answered_ids = set(state.get("answered_test_ids", []))
    question_history = state.get("question_answers", {})
    if isinstance(question_history, dict) and question_history:
        with st.expander(f"추가 질문 답변 기록 · {len(question_history)}개", expanded=False):
            for history_id, history_row in question_history.items():
                if not isinstance(history_row, dict):
                    continue
                old_question = history_row.get("question", {})
                prompt = old_question.get("question", history_id) if isinstance(old_question, dict) else history_id
                st.caption(f"{prompt} · 답변: {history_row.get('answer', '미답변')}")
                if st.button("이 답변 수정·취소", key=f"{key}_edit_answer_{history_id}"):
                    state["editing_test_id"] = history_id
                    st.session_state[key] = state
                    st.rerun()
    editing_test_id = state.get("editing_test_id")
    selected_question = result.get("llm_follow_up_question")
    allowed_question_ids = {row.get("test_id") for row in questions}
    if editing_test_id and isinstance(question_history, dict):
        history_item = question_history.get(editing_test_id, {})
        selected_question = history_item.get("question") if isinstance(history_item, dict) else None
    valid_edit_question = bool(editing_test_id and isinstance(selected_question, dict)
                               and selected_question.get("type") == "question"
                               and selected_question.get("test_id") == editing_test_id)
    if (not selected_question or (selected_question.get("test_id") in answered_ids
                                  and selected_question.get("test_id") != editing_test_id)
            or (selected_question.get("test_id") not in allowed_question_ids and not valid_edit_question)):
        selected_question = next((row for row in questions if row.get("test_id") not in answered_ids), None)
    followup_limit = 5
    followup_answered_ids = answered_ids - {INITIAL_SYMPTOM_QUESTION_ID}
    followup_limit_reached = len(followup_answered_ids) >= followup_limit and not editing_test_id
    if followup_limit_reached:
        selected_question = None
    if selected_question:
        st.markdown("**추가 질문 답변**" if editing_test_id else "**다음 확인 질문 · 한 번에 하나**")
        if selected_question.get("selection_reason"):
            st.caption(selected_question["selection_reason"])
        question_id = selected_question["test_id"]
        answer_key = f"{key}_answer_{question_id}"
        options = ("미답변", "예", "아니오", "모르겠음", "답변 취소") if editing_test_id else (
            "미답변", "예", "아니오", "모르겠음")
        if answer_key not in st.session_state:
            prior_answer = question_history.get(question_id, {}).get("answer") if isinstance(question_history, dict) else None
            st.session_state[answer_key] = prior_answer if prior_answer in options else "미답변"
        answer = st.selectbox(selected_question["question"], options, key=answer_key)
        if st.button("답변 반영하고 다시 추론", key=f"{key}_apply_answers", disabled=preview or answer == "미답변"):
            _save_follow_up_answer(state, selected_question, answer)
            state.pop("editing_test_id", None)
            answers = state.get("answers", {})
            answer_input_fingerprint = _diagnosis_fingerprint(
                project, components, selected_codes, symptom_text,
                observations, answers, measurements,
                silkscreen_labels, st.session_state, confirmed_component_ids, inventory_complete,
                component_inventory, analysis_target_id, suspected_components=valid_suspects,
            )
            answer_fingerprints = build_analysis_fingerprints(
                input_payload={"input_sha256": answer_input_fingerprint}, measurements=measurements,
                extra_rules_payload={"llm_enabled": llm_enabled,
                                     "llm_model": selected_model if llm_enabled else None,
                                     "qwen_pcb_strategy": qwen_pcb_strategy},
            )
            followup_run_id = begin_analysis(
                st.session_state, scope=analysis_scope, fingerprints=answer_fingerprints,
                result_kind="symptom",
            )
            combined_observations = _combined_observations(state, answers)
            try:
                state["result"] = diagnose(
                    symptoms=state.get("symptoms", []), symptom_text="",
                    observations=combined_observations,
                    components=_component_rows(components, component_inventory),
                    confirmed_component_ids=confirmed_component_ids,
                    inventory_complete=inventory_complete,
                    measurements=[row.to_dict() for row in measurements],
                    current_context=_current_analysis_context(project, st.session_state, component_inventory),
                    silkscreen_labels=silkscreen_labels, connectivity_status="none",
                )
            except Exception as exc:
                fail_analysis(st.session_state, scope=analysis_scope, run_id=followup_run_id, error=str(exc))
                state["result"] = None
                state["analysis_status"] = "FAILED"
                st.session_state[key] = state
                st.error(f"추가 답변 분석을 완료하지 못했습니다: {exc}")
                return
            state["result"]["symptom_text"] = symptom_text
            state["result"]["silkscreen_labels"] = silkscreen_labels
            _attach_evidence_v1(state["result"], components=_component_rows(components, component_inventory),
                                measurements=measurements, observations=combined_observations)
            if state["result"].get("status") != "NEEDS_SYMPTOMS" and llm_enabled and selected_model and qwen_pcb_strategy:
                _apply_qwen_pcb_assist(
                    state["result"], project=project,
                    components=_component_rows(components, component_inventory),
                    suspected_components=valid_suspects, selected_model=selected_model, knowledge=kb,
                    answered_test_ids=list(state.get("answered_test_ids", [])),
                    board_registry=board_registry,
                    measurement_records=measurements,
                )
            if llm_enabled and selected_model and state["result"].get("hypotheses"):
                try:
                    state["result"]["llm_assist"] = explain_candidates(
                        OllamaClient(auto_start=True, timeout_seconds=120), selected_model,
                        symptom_text=symptom_text, diagnosis=state["result"],
                    )
                except (LocalLLMError, ValueError, TypeError, OSError) as exc:
                    state["result"]["llm_error"] = str(exc)
            if llm_enabled and selected_model and state["result"].get("component_candidates"):
                try:
                    state["result"]["llm_component_explanations"] = explain_component_candidates(
                        OllamaClient(auto_start=True, timeout_seconds=120), selected_model,
                        candidates=state["result"]["component_candidates"],
                    )
                except (LocalLLMError, ValueError, TypeError, OSError) as exc:
                    state["result"]["llm_component_error"] = str(exc)
            answer_result = state["result"]
            live_snapshot = current_analysis_snapshot(
                st, project, components=components, component_inventory=component_inventory,
                symptom_text=symptom_text, selected_codes=selected_codes,
                observations=observations, inventory_complete=inventory_complete,
                analysis_target_id=analysis_target_id, component_id_aliases=component_id_aliases,
                board_registry=board_registry, suspected_components=valid_suspects,
            )
            committed = commit_analysis_result(
                st.session_state, scope=analysis_scope, run_id=followup_run_id,
                fingerprints=answer_fingerprints, current_fingerprints=live_snapshot["fingerprints"],
                current_scope=live_snapshot["scope"], result=answer_result, result_kind="symptom",
            )
            if not committed:
                state["result"] = None
                state["analysis_status"] = "STALE"
                state["invalidated_reason"] = "추가 답변 처리 중 분석 입력이 변경되어 결과를 저장하지 않았습니다."
                st.session_state[key] = state
                st.warning(state["invalidated_reason"])
                return
            state["input_fingerprint"] = answer_input_fingerprint
            state["measurement_fingerprint"] = measurement_fingerprint
            state["analysis_status"] = "CURRENT"
            state["analysis_target_id"] = analysis_scope.target_id
            st.session_state["circuitcare_v7_analysis_target_id"] = analysis_scope.target_id
            _persist_diagnosis(project, state, answer_result, components, combined_observations)
            _sync_component_candidates(st, project, answer_result)
            st.session_state[key] = state
            st.rerun()
    elif followup_limit_reached:
        st.info("추가 질문 한도에 도달했습니다. 현재 근거로 판단이 어려우면 권장 검사 또는 측정 검증을 확인하세요.")
    visuals = [item for item in result.get("next_tests", []) if item.get("type") == "visual_inspection"]
    if visuals:
        st.caption("육안 점검은 전원 차단 및 냉각을 확인한 뒤에만 표시됩니다.")
        for item in visuals:
            st.write(f"**{item['name']}** · {item['question']}")
