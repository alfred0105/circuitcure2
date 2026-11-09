"""Local LLM and deterministic repair advice view."""

from __future__ import annotations

import json
import time
from typing import Any

from circuitcare.conversation import answer_question
from circuitcare.dashboard.context import (
    _advisor_matches,
    _measurement_set_id,
    _simulation_result_id,
)
from circuitcare.evidence_builder import build_evidence
from circuitcare.local_llm import LocalLLMError
from circuitcare.repair_advisor import build_repair_advice


def render_repair_advisor_view(st: Any, *, case: dict, case_name: str, client: Any,
                               models: list[dict], supported: bool, context_key: Any,
                               risk: dict, stress: dict, risk_path: Any, stress_path: Any,
                               result_dir: Any) -> None:
    st.subheader("로컬 LLM 수리 지원")
    if client and models:
        names = [item.get("name") for item in models if item.get("name")]
        selected_model = st.selectbox("설치된 로컬 모델", names,
            index=names.index("gemma4:12b") if "gemma4:12b" in names else 0)
        st.caption("질문을 제출할 때만 로컬 모델을 호출합니다. 추론 중 다른 분석 단계는 다시 실행하지 않습니다.")
    else:
        selected_model = None
        st.warning("Ollama를 사용할 수 없습니다. 결정론적 점검 안내와 분석 결과는 계속 사용할 수 있습니다.")
    with st.form("chat_form"):
        question = st.text_input("회로에 대해 질문", placeholder="C1을 왜 먼저 점검해야 해?")
        ask = st.form_submit_button("질문 제출")
    if ask:
        from circuitcare.services.diagnosis_service import advisor_inputs
        diagnosis, inspection, recommendations = advisor_inputs(case) if supported else ({}, {}, {})
        active_diagnosis = st.session_state.get("diagnosis_result")
        if active_diagnosis and supported:
            diagnosis = active_diagnosis["diagnosis"]
            inspection = active_diagnosis["inspection_ranking"]
            recommendations = active_diagnosis["recommendations"]
        evidence = build_evidence(diagnosis, inspection, recommendations, risk or {}, stress or {},
            source_files={"diagnosis": case.get("prior") or "unavailable",
                          "risk_assessment": str(risk_path), "component_stress": str(stress_path)})
        advice = build_repair_advice(evidence, question)
        evidence["recommendations"] = advice["recommended_checks"]
        started = time.perf_counter()
        if client and selected_model:
            with st.spinner("로컬 Ollama 모델 응답 생성 중…"):
                try:
                    result = answer_question(client, selected_model, question, evidence, advice)
                except (LocalLLMError, ValueError) as exc:
                    result = {"status": "failed", "error": str(exc), "fallback_used": True}
        else:
            result = {"status": "skipped", "fallback_used": True}
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        result["fallback_used"] = result.get("status") != "success"
        if result["fallback_used"]:
            from circuitcare.dashboard.formatters import fallback_text
            result["fallback_explanation"] = fallback_text(evidence)
        observations = st.session_state.get("measurements", []) if supported else []
        st.session_state.last_advisor = {
            "analysis_context_key": context_key,
            "measurement_set_id": _measurement_set_id(observations),
            "simulation_result_id": _simulation_result_id(result_dir),
            "result": result, "evidence": evidence, "advice": advice,
        }
    saved = st.session_state.get("last_advisor")
    observations = st.session_state.get("measurements", []) if supported else []
    if saved and not _advisor_matches(saved, context_key, observations, result_dir):
        st.session_state.pop("last_advisor", None)
        saved = None
    if not saved:
        return
    result = saved["result"]
    model_label = result.get("model", selected_model) if client else "Local LLM unavailable"
    st.caption(f"상태: {result['status']} · 응답 시간: {result['elapsed_seconds']} s · " +
               ("Deterministic Fallback" if result["fallback_used"] else f"Local LLM · {model_label}"))
    if result.get("validation_error"):
        st.caption(f"LLM 응답 검증 사유: {result['validation_error']}")
    elif result.get("error"):
        st.caption(f"로컬 모델 오류: {result['error']}")
    if result.get("response"):
        st.markdown(result["response"]["answer"])
        with st.expander("Facts / Assumptions / Recommendations / Unknowns"):
            st.json(result["response"])
    else:
        st.info(result.get("fallback_explanation", "결정론적 안내를 구성할 수 없습니다."))
    with st.expander("근거 자료 (Evidence IDs)"):
        st.json(saved["evidence"])
    st.download_button("수리 조언 JSON", json.dumps({"repair_advice": saved["advice"],
        "llm_explanation": result, "evidence": saved["evidence"]}, ensure_ascii=False, indent=2),
        "repair_advice.json", "application/json")
