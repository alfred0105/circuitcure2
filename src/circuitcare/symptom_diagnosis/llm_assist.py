"""Evidence-bounded local LLM assistance for symptom diagnosis."""

from __future__ import annotations

import json
from typing import Any

from circuitcare.local_llm import LocalLLMError, OllamaClient
from circuitcare.symptom_diagnosis.pcb_context import normalize_review_status


def suggest_pcb_fault_candidates(
    client: OllamaClient, model: str, *, pcb_context: dict[str, Any],
    diagnosis: dict[str, Any], knowledge: dict[str, Any],
) -> dict[str, Any]:
    """Return schema-bounded Qwen proposals, separate from deterministic ranks."""
    raw_components = pcb_context.get("focus_components", [])
    components = {str(row.get("component_id")): row for row in raw_components
                  if isinstance(row, dict) and isinstance(row.get("component_id"), (str, int))
                  and str(row.get("component_id")).strip()}
    eligible_states = {"USER_CONFIRMED", "DETECTED_UNVERIFIED"}
    eligible_ids = sorted(cid for cid, row in components.items()
                          if normalize_review_status(row) in eligible_states
                          and str(row.get("identity_status") or "").upper() not in {
                              "STALE", "UNRESOLVED", "STALE_OR_UNRESOLVED", "RECONFIRMATION_REQUIRED"})
    raw_faults = knowledge.get("fault_modes", {})
    fault_modes = raw_faults.get("fault_modes", {}) if isinstance(raw_faults, dict) else {}
    if not isinstance(fault_modes, dict):
        fault_modes = {}
    modes = sorted(fault_modes)
    if not modes or not eligible_ids:
        return {"proposals": [], "model": model, "status": "NO_ELIGIBLE_COMPONENTS"}
    block_values = sorted({"power", "input", "control", "control_input", "control_timing",
                           "driver", "output", "protection", "thermal", "buzzer", "relay"})
    block_groups = {str(row.get("block_id")): row for row in pcb_context.get("functional_block_groups", [])
                    if isinstance(row, dict) and row.get("block_id")}
    raw_observations = pcb_context.get("observations", {})
    evidence_values = ({str(key) for key in raw_observations}
                       if isinstance(raw_observations, dict) else set())
    for row in diagnosis.get("matched_rules", []) if isinstance(diagnosis.get("matched_rules"), list) else []:
        if isinstance(row, dict) and row.get("id"):
            evidence_values.add(str(row["id"]))
    for row in pcb_context.get("measurement_evidence", []) if isinstance(pcb_context.get("measurement_evidence"), list) else []:
        if isinstance(row, dict) and row.get("measurement_id"):
            evidence_values.add(f"measurement:{row['measurement_id']}")
    for row in pcb_context.get("evidence_ledger", []) if isinstance(pcb_context.get("evidence_ledger"), list) else []:
        if isinstance(row, dict) and row.get("evidence_id"):
            evidence_values.add(str(row["evidence_id"]))
    evidence_options = sorted(evidence_values) or ["__none__"]
    schema = {
        "type": "object", "additionalProperties": False, "required": ["proposals"],
        "properties": {"proposals": {"type": "array", "maxItems": 3, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["component_id", "suspected_fault_id", "functional_block", "reason",
                         "supporting_evidence_ids", "opposing_evidence_ids", "missing_information",
                         "recommended_next_action", "uncertainty_status"],
            "properties": {
                "component_id": {"type": "string", "enum": eligible_ids},
                "suspected_fault_id": {"type": "string", "enum": modes},
                "functional_block": {"type": "string", "enum": block_values + ["UNKNOWN"]},
                "functional_block_id": {"type": "string", "enum": sorted(block_groups) + ["__none__"]},
                "reason": {"type": "string", "maxLength": 500},
                "supporting_evidence_ids": {"type": "array", "maxItems": 6,
                                             "items": {"type": "string", "enum": evidence_options}},
                "opposing_evidence_ids": {"type": "array", "maxItems": 6,
                                           "items": {"type": "string", "enum": evidence_options}},
                "missing_information": {"type": "array", "maxItems": 5,
                                        "items": {"type": "string", "maxLength": 160}},
                "recommended_next_action": {"type": "string", "maxLength": 240},
                "uncertainty_status": {"type": "string", "enum": ["UNVERIFIED", "INCONCLUSIVE"]},
            },
        }}}
    }
    payload = {"pcb_context": pcb_context, "rule_hypotheses": pcb_context.get("hypotheses", []),
               "allowed_component_ids": eligible_ids, "allowed_fault_ids": modes,
               "allowed_functional_block_ids": sorted(block_groups),
               "evidence_ids": evidence_options}
    response = client.chat_json(model, [
        {"role": "system", "content": (
            "PCB 진단 보조자다. 확인된 소자는 진단 후보로, DETECTED_UNVERIFIED 소자는 검수 필요 검사 후보로만 제안한다. "
            "type_evidence_status와 ocr_evidence_status가 CONFIRMED인 정보만 확인 사실로 사용한다. TENTATIVE 표시는 검수 단서일 뿐이다. "
            "미검수 소자의 종류는 잠정값이므로 확정 사실처럼 말하지 않는다. 입력 JSON의 허용 고장 유형과 증거만 사용한다. "
            "사용자 의심은 미검증 단서다. 존재하지 않는 측정·배선·핀·연결·소자·증거를 만들지 않는다. "
            "기능 블록 ID는 supplied functional_block_groups에 있는 ID만 선택하고, 없으면 __none__으로 둔다. "
            "근거가 부족하면 후보를 생략하거나 INCONCLUSIVE로 제안한다. Qwen 제안만으로 고장을 확정하거나 "
            "FAULT_SUPPORTED 판정을 내리지 않는다. 기능부는 추정 후보일 뿐 연결 사실이 아니다. "
            "JSON 스키마에 맞는 제안 최대 3개를 반환한다."
        )},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
    ], output_schema=schema)
    if not isinstance(response, dict):
        raise LocalLLMError("Qwen candidate response must be an object")
    if response.get("done_reason") in {"length", "max_tokens"}:
        raise LocalLLMError("Qwen candidate JSON response was truncated by the output token limit.")
    response_content = response.get("content")
    if not isinstance(response_content, dict) or not isinstance(response_content.get("proposals"), list):
        return {"proposals": [], "model": response.get("model", model),
                "status": "FAILED", "failure_category": "INVALID_SCHEMA",
                "rejected_proposal_count": 0}
    rows = response_content["proposals"]
    validated = []
    rejected = 0
    for row in rows[:3] if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            rejected += 1
            continue
        raw_cid, raw_fault_id = row.get("component_id"), row.get("suspected_fault_id")
        if not isinstance(raw_cid, (str, int)) or not isinstance(raw_fault_id, str):
            rejected += 1
            continue
        cid, fault_id = str(raw_cid).strip(), raw_fault_id.strip()
        if cid not in eligible_ids or not fault_id:
            rejected += 1
            continue
        mode = fault_modes.get(fault_id)
        component = components.get(cid)
        if not isinstance(component, dict):
            continue
        review_status = normalize_review_status(component)
        if review_status not in eligible_states or not isinstance(mode, dict):
            rejected += 1
            continue
        raw_accepted_types = mode.get("candidate_component_types", [])
        raw_actual_types = component.get("candidate_component_types", [])
        accepted_types = {str(value).casefold() for value in raw_accepted_types
                          if isinstance(value, str)} if isinstance(raw_accepted_types, list) else set()
        actual_types = {str(value).casefold() for value in raw_actual_types
                        if isinstance(value, str) and value.casefold() != "unknown"} \
            if isinstance(raw_actual_types, list) else set()
        actual_type = str(component.get("type") or "unknown").casefold()
        if actual_type != "unknown":
            actual_types.add(actual_type)
        aliases = {"integrated_circuit": "ic", "ic": "integrated_circuit"}
        if not accepted_types or not any(kind in accepted_types or aliases.get(kind) in accepted_types
                                         for kind in actual_types):
            rejected += 1
            continue
        raw_supports = row.get("supporting_evidence_ids", [])
        raw_opposes = row.get("opposing_evidence_ids", [])
        raw_missing = row.get("missing_information", [])
        supports = [value for value in raw_supports if isinstance(value, str) and value in evidence_values] \
            if isinstance(raw_supports, list) else []
        opposes = [value for value in raw_opposes if isinstance(value, str) and value in evidence_values] \
            if isinstance(raw_opposes, list) else []
        missing = [str(value)[:160] for value in raw_missing[:5] if isinstance(value, (str, int, float))] \
            if isinstance(raw_missing, list) else []
        reason = row.get("reason") if isinstance(row.get("reason"), str) else ""
        next_action = row.get("recommended_next_action") if isinstance(row.get("recommended_next_action"), str) else ""
        proposed_block = row.get("functional_block")
        if not isinstance(proposed_block, str):
            proposed_block = "UNKNOWN"
        proposed_block_id = row.get("functional_block_id")
        if proposed_block_id not in block_groups:
            proposed_block_id = None
        validated.append({
            "component_id": cid, "suspected_fault_id": fault_id,
            "board_component_id": cid,
            "component_type": actual_type if actual_type != "unknown" else sorted(actual_types)[0],
            "functional_block": proposed_block if proposed_block in block_values else "UNKNOWN",
            "functional_block_id": proposed_block_id,
            "reason": reason[:500],
            "supporting_evidence_ids": supports, "opposing_evidence_ids": opposes,
            "missing_information": missing,
            "recommended_next_action": next_action[:240],
            "uncertainty_status": row.get("uncertainty_status") if row.get("uncertainty_status") in {"UNVERIFIED", "INCONCLUSIVE"} else "UNVERIFIED",
            "evidence_status": "UNVERIFIED", "rank_status": "NOT_MERGED",
            "component_review_status": review_status,
            "component_type_status": component.get("component_type_status", "UNCONFIRMED"),
            "verification_status": ("NEEDS_COMPONENT_REVIEW" if review_status == "DETECTED_UNVERIFIED"
                                     or component.get("component_type_status") != "USER_CONFIRMED"
                                     else "NEEDS_MEASUREMENT_OR_INSPECTION"),
        })
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in validated:
        unique.setdefault((row["component_id"], row["suspected_fault_id"]), row)
    return {"proposals": list(unique.values()), "model": response.get("model", model),
            "model_info": response.get("model_info", {}),
            "usage": {key: response.get(key) for key in (
                "prompt_eval_count", "eval_count", "total_duration_ns", "load_duration_ns",
                "prompt_eval_duration_ns", "eval_duration_ns")},
            "status": "LOCAL_LLM_PROPOSED_UNVERIFIED" if unique else "NO_VALID_PROPOSALS",
            "failure_category": "INVALID_SCHEMA" if rejected and not unique else None,
            "rejected_proposal_count": rejected}


def select_follow_up_question(client: OllamaClient, model: str, *, pcb_context: dict[str, Any],
                              diagnosis: dict[str, Any], answered_test_ids: list[str]) -> dict[str, Any] | None:
    """Select one unasked, pre-authored diagnostic question; never invent a test."""
    available = [row for row in diagnosis.get("next_tests", [])
                 if isinstance(row, dict) and row.get("type") == "question"
                 and row.get("test_id") and row.get("test_id") not in set(answered_test_ids)]
    if not available:
        return None
    allowed = [str(row["test_id"]) for row in available]
    schema = {"type": "object", "additionalProperties": False, "required": ["test_id", "reason"],
              "properties": {"test_id": {"type": "string", "enum": allowed},
                             "reason": {"type": "string", "maxLength": 220}}}
    response = client.chat_json(model, [
        {"role": "system", "content": (
            "한 개의 다음 질문만 선택한다. 입력의 기존 안전 질문 중 후보를 구분하는 데 가장 유용한 질문을 고른다. "
            "질문·측정값·조건을 새로 만들지 않는다. 통전/고전압 검사 대신 기존 질문만 고른다. JSON만 반환한다."
        )},
        {"role": "user", "content": json.dumps({"pcb_context": pcb_context,
                                                   "candidates": available[:8]}, ensure_ascii=False, default=str)},
    ], output_schema=schema)
    if response.get("done_reason") in {"length", "max_tokens"}:
        raise LocalLLMError("Qwen follow-up JSON response was truncated by the output token limit.")
    data = response.get("content", {})
    choice = next((row for row in available if row.get("test_id") == data.get("test_id")), None)
    return ({**choice, "selection_reason": str(data.get("reason", ""))[:220],
             "question_source": "QWEN_SELECTED_EXISTING_SAFE_QUESTION"} if choice else None)


def explain_candidates(client: OllamaClient, model: str, *, symptom_text: str,
                       diagnosis: dict[str, Any]) -> dict[str, Any]:
    """Ask a local model to explain existing candidates without adding diagnoses."""
    payload = {
        "symptom_text": symptom_text[:2000],
        "hypotheses": [{"fault_id": row["fault_id"], "title": row["title"],
                        "supporting_evidence": row["supporting_evidence"],
                        "opposing_evidence": row["opposing_evidence"]}
                       for row in diagnosis.get("hypotheses", [])[:5]],
        "questions_available": [row.get("question", "") for row in diagnosis.get("next_tests", [])
                                if row.get("type") == "question"][:5],
    }
    schema = {
        "type": "object", "additionalProperties": False,
        "required": ["summary"],
        "properties": {
            "summary": {"type": "string", "maxLength": 900},
        },
    }
    response = client.chat_json(model, [
        {"role": "system", "content": (
            "너는 PCB 고장 진단 설명 보조자다. 제공된 JSON의 사실과 후보만 사용한다. "
            "새 고장 원인이나 후보 ID를 추가하지 말고, 확률·확정 판정을 말하지 않는다. "
            "근거가 부족하면 부족하다고 명시한다. 한국어 JSON 객체로만 답한다. "
            '고장 가설별 근거는 새로 만들지 말고 한 문장으로 전체 결과를 설명한다. '
            '형식은 반드시 {"summary":"한국어 한두 문장"} 하나뿐이다.'
        )},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ], output_schema=schema)
    data = response.get("content", {})
    if not isinstance(data.get("summary"), str):
        raise LocalLLMError("로컬 모델 설명에 summary 문자열이 없습니다.")
    return {"summary": data["summary"][:900], "model": response.get("model", model),
            "status": "LOCAL_LLM_ASSISTED"}


def explain_component_candidates(client: OllamaClient, model: str, *,
                                 candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Explain only the supplied Top-3 component candidates in concise Korean."""
    candidates = candidates[:3]
    allowed_ids = [str(row.get("component_id", "")) for row in candidates]
    payload = [{"component_id": row.get("component_id"), "component_type": row.get("component_type"),
                "fault_hypothesis": row.get("fault_hypothesis"), "fault_title": row.get("fault_title"),
                "evidence": row.get("evidence", [])[:6],
                "supporting_evidence": row.get("supporting_evidence", [])[:6],
                "opposing_evidence": row.get("opposing_evidence", [])[:6],
                "measurement_status": row.get("measurement_status")} for row in candidates]
    schema = {"type": "object", "additionalProperties": False, "required": ["explanations"],
              "properties": {"explanations": {"type": "array", "maxItems": 3, "items": {
                  "type": "object", "additionalProperties": False,
                  "required": ["component_id", "explanation"],
                  "properties": {"component_id": {"type": "string", "enum": allowed_ids},
                                 "explanation": {"type": "string", "maxLength": 220}},
              }}}}
    response = client.chat_json(model, [
        {"role": "system", "content": (
            "PCB 고장 후보를 설명한다. 입력에 있는 후보 ID와 근거만 사용한다. "
            "후보마다 쉬운 한국어 한 문장으로, 어떤 관측이 왜 이 부품을 검사 후보로 올렸는지 말한다. "
            "근거가 약하면 그 점을 그대로 말한다. 확률, 확정 고장, 새 측정값, 새 후보를 만들지 않는다. "
            "정상 측정은 명시된 검사 가설에만 적용한다. JSON만 반환한다."
        )},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ], output_schema=schema)
    rows = response.get("content", {}).get("explanations", [])
    allowed = set(allowed_ids)
    explanations = {str(row["component_id"]): str(row["explanation"])[:220]
                    for row in rows if isinstance(row, dict)
                    and str(row.get("component_id")) in allowed
                    and isinstance(row.get("explanation"), str)}
    return {"explanations": explanations, "model": response.get("model", model),
            "status": "LOCAL_LLM_ASSISTED"}


def suggest_unregistered_causes(client: OllamaClient, model: str, *, symptom_text: str,
                                diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a separate, low-priority LLM-only bucket; never merge into rule ranks."""
    payload = {"symptom_text": symptom_text[:2000],
               "known_hypotheses": [{"id": row["fault_id"], "title": row["title"]}
                                    for row in diagnosis.get("hypotheses", [])[:8]],
               "known_evidence": diagnosis.get("observations", {}),
               "confirmed_component_inventory": diagnosis.get("component_inventory", {})}
    evidence_values = sorted(diagnosis.get("observations", {}).keys()) or ["__none__"]
    schema = {"type": "object", "additionalProperties": False,
              "required": ["suggestions"], "properties": {
                  "suggestions": {"type": "array", "maxItems": 2, "items": {
                      "type": "object", "additionalProperties": False,
                      "required": ["title", "reason", "evidence_used"],
                      "properties": {"title": {"type": "string", "maxLength": 160},
                                     "reason": {"type": "string", "maxLength": 500},
                                     "evidence_used": {"type": "array", "maxItems": 5,
                                                       "items": {"type": "string", "enum": evidence_values}}},
                  }}},
              }
    response = client.chat_json(model, [
        {"role": "system", "content": (
            "PCB 고장 원인 보완 검토자다. 규칙 엔진 후보에 없는 가능성을 최대 2개 제안한다. "
            "제안은 미검증 가설이며 순위를 매기거나 확정하지 않는다. 없는 배선·핀·측정값을 만들지 않는다. "
            "부품 고장은 규칙 엔진이 확인된 소자 목록으로만 평가한다. 확인 목록에 없는 소자 고장을 제안하지 말고, "
            "구체 부품 ID나 위치를 지목하지 않는다. 목록이 미완료면 부품의 존재/부재를 단정하지 않는다. "
            "JSON: {\"suggestions\":[{\"title\":\"...\","
            "\"reason\":\"입력 증상과의 관계\",\"evidence_used\":[\"입력에 존재하는 관측 키\"]}]}"
        )},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ], output_schema=schema)
    data = response.get("content", {})
    observed = set(diagnosis.get("observations", {}))
    suggestions = []
    suggestion_rows = data.get("suggestions", [])
    if not isinstance(suggestion_rows, list):
        suggestion_rows = []
    for row in suggestion_rows[:2]:
        if not isinstance(row, dict) or not isinstance(row.get("title"), str) or not isinstance(row.get("reason"), str):
            continue
        evidence_used = [code for code in row.get("evidence_used", []) if code in observed]
        suggestions.append({"title": row["title"][:160], "reason": row["reason"][:500],
                            "evidence_used": evidence_used, "status": "UNVERIFIED_LLM_SUGGESTION",
                            "ranked_with_rule_candidates": False,
                            "id": f"LLM_UNVERIFIED_{len(suggestions) + 1}"})
    return suggestions
