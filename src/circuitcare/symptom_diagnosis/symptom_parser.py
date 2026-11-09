"""Schema-checked local LLM parser for explicitly reported PCB symptoms."""

from __future__ import annotations

import unicodedata
from typing import Any

from circuitcare.local_llm import LocalLLMError, OllamaClient
from circuitcare.symptom_diagnosis.engine import infer_symptom_codes, load_knowledge
from circuitcare.symptom_diagnosis.schemas import EvidenceStatus, Observation


def _contains_quote(source: str, quote: str) -> bool:
    source = unicodedata.normalize("NFC", source).casefold()
    quote = unicodedata.normalize("NFC", quote).casefold().strip()
    return bool(quote) and quote in source


def parse_user_report(client: OllamaClient, model: str, text: str) -> dict[str, Any]:
    """Extract only allowlisted facts with a verbatim quote from user input."""
    if not text.strip():
        return {"observations": [], "uncertain_facts": [], "requires_clarification": True,
                "parser_status": "EMPTY_INPUT"}
    kb = load_knowledge()
    allowed = {**kb["symptom_ontology"]["symptoms"], **kb["symptom_ontology"].get("observations", {})}
    allowed_desc = {code: row["label"] for code, row in allowed.items()}
    schema = {
        "type": "object", "additionalProperties": False,
        "required": ["observations", "uncertain_facts", "requires_clarification"],
        "properties": {
            "observations": {"type": "array", "maxItems": 32, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["code", "value", "evidence_quote"],
                "properties": {"code": {"type": "string", "enum": list(allowed)},
                               "value": {"type": "boolean"},
                               "evidence_quote": {"type": "string", "maxLength": 250}},
            }},
            "uncertain_facts": {"type": "array", "maxItems": 8,
                                 "items": {"type": "string", "maxLength": 300}},
            "requires_clarification": {"type": "boolean"},
        },
    }
    response = client.chat_json(model, [
        {"role": "system", "content": (
            "너는 PCB 증상 문장을 관측값으로 구조화한다. 고장 원인은 추론하지 않는다. "
            "입력에 명시된 사실만 추출하고, 각 항목의 evidence_quote는 원문에서 그대로 복사한다. "
            "가정·조건 표현(예: ~면, ~라면, ~할 경우)은 발생한 증상으로 기록하지 않는다. "
            "이중 부정이나 정정(예: 안 켜지는 것은 아님)은 증상 발생으로 기록하지 않는다. "
            "전원 표시등이 켜졌다는 말은 표시등만 확인된 것이며 보드 전체 전원이 정상이라는 뜻이 아니다. "
            "어떤 소자/기능이 언급되지 않은 것은 false가 아니라 unknown이다. "
            "목록에 없는 코드는 만들지 않고 uncertain_facts에 자연어로 남긴다. "
            "JSON만 반환: {\"observations\":[{\"code\":\"allowlist code\",\"value\":true,"
            "\"evidence_quote\":\"원문 인용\"}],\"uncertain_facts\":[],\"requires_clarification\":false}"
        )},
        {"role": "user", "content": "허용 코드와 설명: " + str(allowed_desc) + "\n원문: " + text[:2000]},
    ], output_schema=schema)
    data = response.get("content", {})
    accepted: list[Observation] = []
    uncertain_rows = data.get("uncertain_facts", [])
    if not isinstance(uncertain_rows, list):
        uncertain_rows = []
    uncertain = [str(item)[:300] for item in uncertain_rows[:8]
                 if isinstance(item, (str, int, float))]
    observation_rows = data.get("observations", [])
    if not isinstance(observation_rows, list):
        observation_rows = []
    for row in observation_rows[:32]:
        if not isinstance(row, dict):
            continue
        code, value = row.get("code"), row.get("value")
        quote = row.get("evidence_quote")
        if code not in allowed or type(value) is not bool:
            uncertain.append(str(row.get("code", "분류할 수 없는 관측"))[:300])
            continue
        verified_quote = quote if isinstance(quote, str) and _contains_quote(text, quote) else None
        if verified_quote is None:
            verified_quote = next((str(keyword) for keyword in allowed[code].get("keywords", [])
                                   if _contains_quote(text, str(keyword))), None)
        if verified_quote is None:
            uncertain.append(f"원문 근거를 확인하지 못함: {allowed[code]['label']}")
            continue
        if code in kb["symptom_ontology"]["symptoms"] and value is True:
            if code not in infer_symptom_codes(verified_quote, kb):
                uncertain.append(f"조건/부정 여부를 확정할 수 없음: {allowed[code]['label']}")
                continue
        accepted.append(Observation(code=code, value=value, status=EvidenceStatus.OBSERVED,
                                    source="user_report", evidence_quote=verified_quote))
    values_by_code: dict[str, set[bool]] = {}
    for item in accepted:
        values_by_code.setdefault(item.code, set()).add(bool(item.value))
    if any(len(values) > 1 for values in values_by_code.values()):
        accepted = [Observation(code=item.code, value=item.value,
                                status=EvidenceStatus.CONTRADICTED if len(values_by_code[item.code]) > 1 else item.status,
                                source=item.source, evidence_quote=item.evidence_quote)
                    for item in accepted]
    return {"observations": [item.to_dict() for item in accepted],
            "uncertain_facts": list(dict.fromkeys(uncertain)),
            "requires_clarification": bool(data.get("requires_clarification", False)) or not accepted,
            "parser_status": "PARSED", "model": response.get("model", model)}
