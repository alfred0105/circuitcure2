"""Evidence-bounded local LLM assistance for symptom diagnosis."""

from __future__ import annotations

import json
from typing import Any

from circuitcare.local_llm import LocalLLMError, OllamaClient


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


def suggest_unregistered_causes(client: OllamaClient, model: str, *, symptom_text: str,
                                diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a separate, low-priority LLM-only bucket; never merge into rule ranks."""
    payload = {"symptom_text": symptom_text[:2000],
               "known_hypotheses": [{"id": row["fault_id"], "title": row["title"]}
                                    for row in diagnosis.get("hypotheses", [])[:8]],
               "known_evidence": diagnosis.get("observations", {})}
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
            "구체 부품 위치를 지목하지 않는다. JSON: {\"suggestions\":[{\"title\":\"...\","
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
