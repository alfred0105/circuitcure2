"""Schema-checked local LLM parser for explicitly reported PCB symptoms."""

from __future__ import annotations

import unicodedata
import re
from typing import Any

from circuitcare.local_llm import LocalLLMError, OllamaClient
from circuitcare.symptom_diagnosis.engine import infer_symptom_codes, load_knowledge
from circuitcare.symptom_diagnosis.schemas import EvidenceStatus, Observation
from circuitcare.component_taxonomy import COMPONENT_CLASSES, normalize_component_class


def _contains_quote(source: str, quote: str) -> bool:
    source = unicodedata.normalize("NFC", source).casefold()
    quote = unicodedata.normalize("NFC", quote).casefold().strip()
    return bool(quote) and quote in source


def _quote_context(source: str, quote: str) -> str:
    """Return the complete sentence containing a quote so trailing negation is checked."""
    normalized_source = unicodedata.normalize("NFC", source)
    normalized_quote = unicodedata.normalize("NFC", quote).strip()
    start = normalized_source.casefold().find(normalized_quote.casefold())
    if start < 0:
        return ""
    end = start + len(normalized_quote)
    left_marks = [normalized_source.rfind(mark, 0, start) for mark in (".", "!", "?", "\n", ";")]
    right_marks = [position for mark in (".", "!", "?", "\n", ";")
                   if (position := normalized_source.find(mark, end)) >= 0]
    return normalized_source[max(left_marks, default=-1) + 1:min(right_marks, default=len(normalized_source))]


def _explicitly_denies_symptom(context: str, code: str, knowledge: dict[str, Any]) -> bool:
    """Accept a symptom=false only when its complete sentence explicitly denies it."""
    folded = context.casefold()
    keywords = knowledge["symptom_ontology"]["symptoms"].get(code, {}).get("keywords", [])
    has_keyword = any(str(keyword).casefold() in folded for keyword in keywords)
    return has_keyword and bool(re.search(
        r"(?:것은|건|게|것이|건은)?\s*(?:아님|아니(?:다|고|라|야)|아닙니다|않(?:다|고|음)|없(?:다|음))",
        folded,
    ))


def _quoted_number(value: int, quote: str, *, ordinal: bool = False) -> bool:
    if value <= 0:
        return True
    forms = {
        1: ("첫 번째", "첫번째", "1번째", "한 번째", "하나"),
        2: ("두 번째", "두번째", "2번째", "둘째", "두 개", "2개"),
        3: ("세 번째", "세번째", "3번째", "셋째", "세 개", "3개"),
        4: ("네 번째", "네번째", "4번째", "넷째", "네 개", "4개"),
        5: ("다섯 번째", "다섯번째", "5번째", "다섯 개", "5개"),
    }
    choices = forms.get(value, (f"{value}번째", f"{value}개"))
    if ordinal:
        choices = tuple(item for item in choices if "번째" in item or "째" in item) or (f"{value}번째",)
    return any(item.casefold() in quote.casefold() for item in choices)


def _validated_target(row: Any, source: str, known_refs: set[str]) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(row, dict):
        return None, "소자 대상 표현 형식이 올바르지 않음"
    reference_id = str(row.get("reference_id") or "").strip()[:40]
    quote = row.get("evidence_quote")
    component_type = normalize_component_class(row.get("component_type"), allow_not_component=False)
    if (not reference_id or reference_id in known_refs or component_type not in COMPONENT_CLASSES
            or not isinstance(quote, str) or not _contains_quote(source, quote)):
        return None, "원문에서 확인할 수 없는 소자 대상 표현"
    group_size = row.get("group_size") if type(row.get("group_size")) is int else 0
    ordinal = row.get("ordinal") if type(row.get("ordinal")) is int else 0
    if group_size < 0 or group_size > 100 or ordinal < 0 or ordinal > 100:
        return None, "소자 개수 또는 순번 범위가 올바르지 않음"
    if group_size and not _quoted_number(group_size, quote):
        group_size = 0
    if ordinal and not _quoted_number(ordinal, quote, ordinal=True):
        ordinal = 0
    ordering = str(row.get("ordering") or "UNSPECIFIED").upper()
    ordering_terms = {"LEFT_TO_RIGHT": ("왼쪽에서", "좌측부터", "왼쪽부터", "좌에서 우"),
                      "TOP_TO_BOTTOM": ("위에서", "위쪽부터", "상단부터", "위에서 아래")}
    if ordering not in {"UNSPECIFIED", *ordering_terms} or (
        ordering != "UNSPECIFIED" and not any(term in quote for term in ordering_terms.get(ordering, ()))
    ):
        ordering = "UNSPECIFIED"
    region = str(row.get("region") or "UNSPECIFIED").upper()
    region_terms = {"UPPER": ("위쪽", "상단", "위에", "위의"), "LOWER": ("아래쪽", "하단", "아래에"),
                    "LEFT": ("왼쪽", "좌측"), "RIGHT": ("오른쪽", "우측"), "CENTER": ("중앙", "가운데")}
    if region not in {"UNSPECIFIED", *region_terms} or (
        region != "UNSPECIFIED" and not any(term in quote for term in region_terms.get(region, ()))
    ):
        region = "UNSPECIFIED"
    color = str(row.get("color") or "").strip()[:40]
    if color and color.casefold() not in quote.casefold():
        color = ""
    reference_designator = str(row.get("reference_designator") or "").strip().upper()[:24]
    if reference_designator and reference_designator.casefold() not in quote.casefold():
        reference_designator = ""
    return ({"reference_id": reference_id, "component_type": component_type,
             "color": color, "region": region, "group_size": group_size or None,
             "ordinal": ordinal or None, "ordering": ordering,
             "reference_designator": reference_designator, "evidence_quote": quote.strip(),
             "resolution_status": "UNRESOLVED"}, None)


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
            "observations": {"type": "array", "maxItems": 4, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["code", "value", "evidence_quote"],
                "properties": {"code": {"type": "string", "enum": list(allowed)},
                               "value": {"type": "boolean"},
                               "evidence_quote": {"type": "string", "maxLength": 160},
                               "target_reference": {"type": "string", "maxLength": 40}},
            }},
            "uncertain_facts": {"type": "array", "maxItems": 4,
                                 "items": {"type": "string", "maxLength": 160}},
            "requires_clarification": {"type": "boolean"},
            "targets": {"type": "array", "maxItems": 4, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["reference_id", "component_type", "color", "region", "group_size",
                             "ordinal", "ordering", "reference_designator", "evidence_quote"],
                "properties": {
                    "reference_id": {"type": "string", "maxLength": 40},
                    "component_type": {"type": "string", "enum": list(COMPONENT_CLASSES)},
                    "color": {"type": "string", "maxLength": 40},
                    "region": {"type": "string", "enum": ["UPPER", "LOWER", "LEFT", "RIGHT", "CENTER", "UNSPECIFIED"]},
                    "group_size": {"type": "integer", "minimum": 0, "maximum": 100},
                    "ordinal": {"type": "integer", "minimum": 0, "maximum": 100},
                    "ordering": {"type": "string", "enum": ["UNSPECIFIED", "LEFT_TO_RIGHT", "TOP_TO_BOTTOM"]},
                    "reference_designator": {"type": "string", "maxLength": 24},
                    "evidence_quote": {"type": "string", "maxLength": 160},
                },
            }},
            "peer_observations": {"type": "array", "maxItems": 4, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["code", "value", "target_reference", "evidence_quote"],
                "properties": {
                    "code": {"type": "string", "enum": list(kb["symptom_ontology"].get("observations", {}))},
                    "value": {"type": "boolean"},
                    "target_reference": {"type": "string", "maxLength": 40},
                    "evidence_quote": {"type": "string", "maxLength": 160},
                },
            }},
        },
    }
    response = client.chat_json(model, [
        {"role": "system", "content": (
            "너는 PCB 증상 문장을 관측값으로 구조화한다. 고장 원인은 추론하지 않는다. "
            "명시적 근거가 있는 관측만 최대 4개 반환하고, 증상 추측·누락된 사실의 부정 추론은 하지 않는다. "
            "불명확하면 observations와 uncertain_facts를 비워 둔다. "
            "입력에 명시된 사실만 추출하고, 각 항목의 evidence_quote는 원문에서 그대로 복사한다. "
            "가정·조건 표현(예: ~면, ~라면, ~할 경우)은 발생한 증상으로 기록하지 않는다. "
            "이중 부정이나 정정(예: 안 켜지는 것은 아님)은 증상 발생으로 기록하지 않는다. "
            "전원 표시등이 켜졌다는 말은 표시등만 확인된 것이며 보드 전체 전원이 정상이라는 뜻이 아니다. "
            "어떤 소자/기능이 언급되지 않은 것은 false가 아니라 unknown이다. "
            "목록에 없는 코드는 만들지 않고 uncertain_facts에 자연어로 남긴다. "
            "사용자가 언급한 소자 대상은 targets에 종류와 원문 인용을 기록한다. 색상·영역·개수·순번·방향·실크스크린은 "
            "원문에 명시된 경우에만 채운다. '세 번째'만 있고 왼쪽/위쪽 기준이 없으면 ordering은 UNSPECIFIED다. "
            "각 증상에 target_reference를 연결하되, 실제 PCB ID는 만들거나 추측하지 않는다. '나머지는 정상'은 "
            "peer_observations로 기록하고 대상 그룹 참조만 연결한다. "
            "JSON만 반환: {\"observations\":[{\"code\":\"allowlist code\",\"value\":true,"
            "\"evidence_quote\":\"원문 인용\",\"target_reference\":\"target-1 또는 빈 문자열\"}],"
            "\"targets\":[{\"reference_id\":\"target-1\",\"component_type\":\"led\",\"color\":\"빨간\","
            "\"region\":\"UPPER\",\"group_size\":4,\"ordinal\":3,\"ordering\":\"UNSPECIFIED\","
            "\"reference_designator\":\"\",\"evidence_quote\":\"원문 인용\"}],"
            "\"peer_observations\":[],\"uncertain_facts\":[],\"requires_clarification\":false}"
        )},
        {"role": "user", "content": "허용 코드와 설명: " + str(allowed_desc) + "\n원문: " + text[:2000]},
    ], output_schema=schema, output_budget=480)
    data = response.get("content", {})
    accepted: list[Observation] = []
    uncertain_rows = data.get("uncertain_facts", [])
    if not isinstance(uncertain_rows, list):
        uncertain_rows = []
    uncertain = [str(item)[:300] for item in uncertain_rows[:8]
                 if isinstance(item, (str, int, float))]
    targets: list[dict[str, Any]] = []
    target_rows = data.get("targets", [])
    for row in target_rows[:4] if isinstance(target_rows, list) else []:
        target, error = _validated_target(row, text, {str(item["reference_id"]) for item in targets})
        if target:
            targets.append(target)
        elif error:
            uncertain.append(error)
    target_ids = {str(item["reference_id"]) for item in targets}
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
        target_reference = str(row.get("target_reference") or "").strip()
        if target_reference and target_reference not in target_ids:
            uncertain.append(f"관측 대상 참조를 확인하지 못함: {target_reference[:40]}")
            target_reference = ""
        if code in kb["symptom_ontology"]["symptoms"]:
            context = _quote_context(text, verified_quote)
            if value is True and code not in infer_symptom_codes(context, kb):
                uncertain.append(f"원문 전체 문장에서 증상 긍정이 확인되지 않음: {allowed[code]['label']}")
                continue
            if value is False and not _explicitly_denies_symptom(context, code, kb):
                uncertain.append(f"원문 전체 문장에서 증상 부정이 확인되지 않음: {allowed[code]['label']}")
                continue
        accepted.append(Observation(code=code, value=value, status=EvidenceStatus.OBSERVED,
                                    source="user_report", evidence_quote=verified_quote,
                                    target_reference=target_reference or None))
    values_by_code: dict[str, set[bool]] = {}
    for item in accepted:
        values_by_code.setdefault(item.code, set()).add(bool(item.value))
    if any(len(values) > 1 for values in values_by_code.values()):
        accepted = [Observation(code=item.code, value=item.value,
                                status=EvidenceStatus.CONTRADICTED if len(values_by_code[item.code]) > 1 else item.status,
                                source=item.source, evidence_quote=item.evidence_quote,
                                target_reference=item.target_reference)
                    for item in accepted]
    peer_observations = []
    peer_rows = data.get("peer_observations", [])
    for row in peer_rows[:4] if isinstance(peer_rows, list) else []:
        if not isinstance(row, dict):
            continue
        code = str(row.get("code") or "")
        quote = row.get("evidence_quote")
        target_reference = str(row.get("target_reference") or "")
        if (code not in kb["symptom_ontology"].get("observations", {})
                or type(row.get("value")) is not bool or target_reference not in target_ids
                or not isinstance(quote, str) or not _contains_quote(text, quote)):
            uncertain.append("나머지 소자 정상 관측의 대상 또는 원문 근거를 확인하지 못함")
            continue
        peer_observations.append({"code": code, "value": row["value"], "status": "observed",
                                  "source": "user_report_peer", "target_reference": target_reference,
                                  "evidence_quote": quote})
    return {"observations": [item.to_dict() for item in accepted],
            "targets": targets, "peer_observations": peer_observations,
            "uncertain_facts": list(dict.fromkeys(uncertain)),
            "requires_clarification": bool(data.get("requires_clarification", False)) or not accepted,
            "parser_status": "PARSED", "model": response.get("model", model)}
