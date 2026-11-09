"""Connection-optional, evidence-ranked fault diagnosis UI."""

from __future__ import annotations

from typing import Any
import time
from pathlib import Path
from uuid import uuid4

from circuitcare.frontend.models import ProjectContext
from circuitcare.local_llm import LocalLLMError, OllamaClient
from circuitcare.symptom_diagnosis import diagnose, load_knowledge
from circuitcare.symptom_diagnosis.llm_assist import explain_candidates, suggest_unregistered_causes
from circuitcare.symptom_diagnosis.observation_store import ObservationStore
from circuitcare.symptom_diagnosis.schemas import ComponentRecord, Observation
from circuitcare.symptom_diagnosis.symptom_parser import parse_user_report


def _project_key(project: ProjectContext) -> str:
    return f"cc_symptom_diagnosis_{project.project_id}"


def _persist_diagnosis(project: ProjectContext, state: dict[str, Any], result: dict[str, Any],
                       components: tuple[Any, ...], observations: list[dict[str, Any]]) -> None:
    diagnosis_id = state.setdefault("diagnosis_id", str(uuid4()))
    result["diagnosis_id"] = diagnosis_id
    component_records = [ComponentRecord.from_mapping({
        "component_id": item.component_id, "label": item.component_type,
        "bbox_xyxy": list(item.bbox_xyxy), "detection_confidence": item.detection_confidence,
        "marking": item.ocr_marking,
    }).to_dict() for item in components]
    normalized_observations = [Observation.from_dict(row).to_dict() for row in observations]
    result["component_snapshot"] = component_records
    db_path = Path(__file__).resolve().parents[4] / "data" / "private" / "diagnosis" / "diagnosis.sqlite3"
    ObservationStore(db_path).save(
        diagnosis_id=diagnosis_id, project_id=project.project_id, board_id=project.board_id,
        status=result.get("status", "UNKNOWN"), result=result, observations=normalized_observations,
    )


def render_symptom_diagnosis_panel(
    st: Any, project: ProjectContext, *, components: tuple[Any, ...] = (), preview: bool = False,
) -> None:
    """Collect symptoms, run local rules, and render candidates and safe next questions."""
    kb = load_knowledge()
    symptom_items = kb["symptom_ontology"]["symptoms"]
    key = _project_key(project)
    state = st.session_state.setdefault(key, {"manual_observations": {}, "result": None})
    state.setdefault("manual_observations", state.pop("observations", {}))
    st.markdown("### 증상 기반 고장 추론")
    st.caption("부품 연결도가 없어도 증상과 관측 사실로 원인 후보를 정리합니다. 모든 후보는 검증 전 상태입니다.")
    description_key = f"{key}_description"
    if description_key not in st.session_state:
        st.session_state[description_key] = project.symptom_description or ""
    symptom_text = st.text_area("증상 설명", key=description_key,
                                 placeholder="예: 전원은 켜지지만 LED가 들어오지 않고 버저가 울리지 않음", height=82)
    code_to_label = {code: item["label"] for code, item in symptom_items.items()}
    symptom_widget_key = f"{key}_symptoms"
    if symptom_widget_key not in st.session_state:
        st.session_state[symptom_widget_key] = [
            code_to_label[code] for code in state.get("symptoms", []) if code in code_to_label
        ]
    selected_labels = st.multiselect(
        "확인된 증상", list(code_to_label.values()), key=symptom_widget_key,
    )
    selected_codes = [code for code, label in code_to_label.items() if label in selected_labels]
    with st.expander("추가로 확인한 사실", expanded=False):
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
    llm_enabled = False
    selected_model = None
    model_cache = state.get("ollama_models")
    checked_at = float(state.get("ollama_checked_at", 0))
    if time.time() - checked_at > 30:
        try:
            local_client = OllamaClient(timeout_seconds=2)
            model_cache = [str(item.get("name")) for item in local_client.list_models()
                           if item.get("name")]
            state["ollama_error"] = None
        except (LocalLLMError, ValueError) as exc:
            model_cache = []
            state["ollama_error"] = str(exc)
        state["ollama_models"] = model_cache
        state["ollama_checked_at"] = time.time()
        st.session_state[key] = state
    if model_cache:
        selected_model = st.selectbox("로컬 LLM 모델", model_cache, key=f"{key}_model")
        llm_key = f"{key}_llm_enabled"
        if llm_key not in st.session_state:
            st.session_state[llm_key] = True
        llm_enabled = st.checkbox("LLM으로 후보 설명 보조", key=llm_key)
    else:
        st.caption("로컬 LLM 연결을 사용할 수 없어 규칙 기반 진단만 실행합니다. Ollama가 켜지면 이곳에서 모델을 선택할 수 있습니다.")
    if st.button("고장 원인 후보 추론", type="primary", disabled=preview,
                 key=f"{key}_run"):
        state["symptoms"] = selected_codes
        state["manual_observations"] = {fact: value for fact, value in observations.items() if value is not None}
        component_rows = [{"component_id": item.component_id, "component_type": item.component_type,
                           "bbox_xyxy": list(item.bbox_xyxy), "ocr_marking": item.ocr_marking}
                          for item in components]
        parsed = {"observations": [], "uncertain_facts": [], "parser_status": "SKIPPED"}
        if llm_enabled and selected_model and symptom_text.strip():
            try:
                with st.spinner(f"{selected_model}이 증상을 구조화하는 중…"):
                    parsed = parse_user_report(OllamaClient(timeout_seconds=120), selected_model, symptom_text)
            except (LocalLLMError, ValueError, TypeError, OSError) as exc:
                parsed = {"observations": [], "uncertain_facts": [], "parser_status": "FAILED", "error": str(exc)}
        state["parsed_observations"] = parsed.get("observations", [])
        combined_observations = list(state["parsed_observations"])
        combined_observations.extend({"code": fact, "value": value, "status": "observed", "source": "user_report"}
                                     for fact, value in state["manual_observations"].items())
        combined_observations.extend({"code": fact, "value": value, "status": "observed", "source": "follow_up_answer"}
                                     for fact, value in state.get("answers", {}).items())
        parsed_symptoms = [row["code"] for row in state["parsed_observations"]
                           if row.get("status") in {"observed", "measured"} and row.get("value") is True
                           and row.get("code") in symptom_items]
        result = diagnose(
            symptoms=list(dict.fromkeys([*selected_codes, *parsed_symptoms])),
            symptom_text=symptom_text,
            observations=combined_observations, components=component_rows,
            connectivity_status="none",
        )
        result["llm_parser"] = parsed
        if len(result.get("hypotheses", [])) < 3 and llm_enabled and selected_model:
            try:
                with st.spinner(f"{selected_model}이 미등록 원인 가능성을 별도 검토하는 중…"):
                    result["llm_unverified_suggestions"] = suggest_unregistered_causes(
                        OllamaClient(timeout_seconds=120), selected_model,
                        symptom_text=symptom_text, diagnosis=result,
                    )
            except (LocalLLMError, ValueError, TypeError, OSError) as exc:
                result["llm_suggestion_error"] = str(exc)
        if llm_enabled and selected_model and result.get("hypotheses"):
            try:
                with st.spinner(f"{selected_model}이 근거를 정리하는 중…"):
                    result["llm_assist"] = explain_candidates(
                        OllamaClient(timeout_seconds=120), selected_model,
                        symptom_text=symptom_text, diagnosis=result,
                    )
            except (LocalLLMError, ValueError, TypeError, OSError) as exc:
                result["llm_error"] = str(exc)
        _persist_diagnosis(project, state, result, components, combined_observations)
        candidates_by_project = dict(st.session_state.get("circuitcare_v4_diagnosis_candidates_by_project", {}))
        candidate_ids = list(dict.fromkeys(
            component_id for row in result.get("hypotheses", [])[:5]
            for component_id in row.get("component_candidates", []) if component_id
        ))
        if candidate_ids:
            candidates_by_project[project.project_id] = candidate_ids
        else:
            candidates_by_project.pop(project.project_id, None)
        st.session_state["circuitcare_v4_diagnosis_candidates_by_project"] = candidates_by_project
        state["result"] = result
        st.session_state[key] = state
    result = state.get("result")
    if not result:
        st.info("증상을 입력하고 **고장 원인 후보 추론**을 눌러 시작하세요.")
        return
    if result["status"] == "NEEDS_SYMPTOMS":
        st.info(result["message"])
        return
    st.success(result["message"])
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
    for index, row in enumerate(result.get("hypotheses", [])[:5], start=1):
        title = f"{index}. {row['title']} · 미확정 후보"
        with st.expander(title, expanded=index == 1):
            if row["supporting_evidence"]:
                st.markdown("**뒷받침한 사실** · " + " · ".join(row["supporting_evidence"]))
            else:
                st.caption("직접 뒷받침하는 사실은 아직 없습니다.")
            if row["opposing_evidence"]:
                st.markdown("**반대 사실** · " + " · ".join(row["opposing_evidence"]))
            if row["component_candidates"]:
                st.caption("관련 부품 후보 · " + " · ".join(row["component_candidates"]))
            st.caption("상태 · 검증 전 후보. 이 결과만으로 부품 고장을 확정하지 않습니다.")
    if result.get("functional_blocks"):
        with st.expander("부품 기능 역할 추정", expanded=False):
            for block in result["functional_blocks"]:
                st.caption(f"{block['component_id']} · {block['component_type']} → "
                           f"{', '.join(block['candidate_blocks'])} · 부품 종류 기반 추정")
    questions = [item for item in result.get("next_tests", []) if item.get("type") == "question"]
    if questions:
        st.markdown("**다음 확인 질문**")
        answers: dict[str, Any] = {}
        for question in questions[:3]:
            test_id = question["test_id"]
            answer = st.selectbox(question["question"], ("미답변", "예", "아니오"),
                                  key=f"{key}_answer_{test_id}")
            if answer in {"예", "아니오"}:
                outcome = question.get("results", {}).get("yes" if answer == "예" else "no", {})
                answers.update(outcome.get("observations", {}))
        if st.button("답변 반영하고 다시 추론", key=f"{key}_apply_answers", disabled=preview):
            state["answers"] = answers
            combined_observations = list(state.get("parsed_observations", []))
            combined_observations.extend({"code": fact, "value": value, "status": "observed", "source": "user_report"}
                                         for fact, value in state.get("manual_observations", {}).items())
            combined_observations.extend({"code": fact, "value": value, "status": "observed", "source": "follow_up_answer"}
                                         for fact, value in answers.items())
            state["result"] = diagnose(
                symptoms=state.get("symptoms", []), symptom_text="",
                observations=combined_observations,
                components=[{"component_id": item.component_id, "component_type": item.component_type,
                             "bbox_xyxy": list(item.bbox_xyxy), "ocr_marking": item.ocr_marking}
                            for item in components], connectivity_status="none",
            )
            if llm_enabled and selected_model and state["result"].get("hypotheses"):
                try:
                    with st.spinner(f"{selected_model}이 갱신된 근거를 정리하는 중…"):
                        state["result"]["llm_assist"] = explain_candidates(
                            OllamaClient(timeout_seconds=120), selected_model,
                            symptom_text=symptom_text, diagnosis=state["result"],
                        )
                except (LocalLLMError, ValueError, TypeError, OSError) as exc:
                    state["result"]["llm_error"] = str(exc)
            _persist_diagnosis(project, state, state["result"], components, combined_observations)
            candidates_by_project = dict(st.session_state.get("circuitcare_v4_diagnosis_candidates_by_project", {}))
            candidate_ids = list(dict.fromkeys(
                component_id for row in state["result"].get("hypotheses", [])[:5]
                for component_id in row.get("component_candidates", []) if component_id
            ))
            if candidate_ids:
                candidates_by_project[project.project_id] = candidate_ids
            else:
                candidates_by_project.pop(project.project_id, None)
            st.session_state["circuitcare_v4_diagnosis_candidates_by_project"] = candidates_by_project
            st.session_state[key] = state
            st.rerun()
    visuals = [item for item in result.get("next_tests", []) if item.get("type") == "visual_inspection"]
    if visuals:
        st.caption("육안 점검은 전원 차단 및 냉각을 확인한 뒤에만 표시됩니다.")
        for item in visuals:
            st.write(f"**{item['name']}** · {item['question']}")

