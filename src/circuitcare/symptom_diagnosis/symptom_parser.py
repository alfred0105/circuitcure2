"""Schema-checked local LLM parser for explicitly reported PCB symptoms."""

from __future__ import annotations

import unicodedata
import re
from typing import Any

from circuitcare.local_llm import OllamaClient
from circuitcare.symptom_diagnosis.engine import infer_symptom_codes, load_knowledge
from circuitcare.symptom_diagnosis.schemas import EvidenceStatus, Observation
from circuitcare.component_taxonomy import COMPONENT_CLASSES, normalize_component_class


def _contains_quote(source: str, quote: str) -> bool:
    source = unicodedata.normalize("NFC", source).casefold()
    quote = unicodedata.normalize("NFC", quote).casefold().strip()
    return bool(quote) and quote in source


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


def _rule_target_fallback(text: str, existing_targets: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Observation]]:
    """Recover only literal component group, count, and ordinal facts omitted by the LLM."""
    component_terms = {
        "integrated_circuit": r"(?:집적\s*회로|통합\s*회로|IC)",
        "switch": r"(?:스위치|버튼|switch)", "buzzer": r"(?:버저|부저|buzzer)",
        "capacitor": r"(?:커패시터|콘덴서|capacitor)", "clock": r"(?:클록|크리스털|오실레이터|crystal|oscillator)",
        "connector": r"(?:커넥터|단자|connector)", "diode": r"(?:다이오드|diode)",
        "display": r"(?:디스플레이|화면|display)", "fuse": r"(?:퓨즈|fuse)",
        "heatsink": r"(?:방열판|heatsink)", "inductor": r"(?:인덕터|inductor)",
        "led": r"(?:LED|발광\s*다이오드|표시등|램프)",
        "potentiometer": r"(?:가변\s*저항|포텐셔미터|potentiometer)",
        "relay": r"(?:릴레이|relay)", "resistor": r"(?:저항|resistor)",
        "transformer": r"(?:변압기|transformer)", "transistor": r"(?:트랜지스터|transistor)",
    }
    korean_numbers = {"한": 1, "하나": 1, "첫": 1, "첫째": 1, "두": 2, "둘": 2, "둘째": 2,
                      "세": 3, "셋": 3, "셋째": 3, "네": 4, "넷": 4, "넷째": 4,
                      "다섯": 5, "다섯째": 5}
    targets: list[dict[str, Any]] = []
    observations: list[Observation] = []
    known_refs = {str(row.get("reference_id") or "") for row in existing_targets}
    for clause in (part.strip() for part in re.split(r"[.!?;\n]+", text) if part.strip()):
        matched = None
        component_type = ""
        for kind, term in component_terms.items():
            pattern = re.compile(
                rf"(?P<component>{term})\s*(?P<count>\d+|하나|한|두|둘|세|셋|네|넷|다섯)\s*개\s*중\s*"
                rf"(?P<direction>왼쪽에서|왼쪽부터|좌측부터|위에서|위쪽부터|상단부터)?\s*"
                rf"(?P<ordinal>\d+|첫|첫째|두|둘째|세|셋째|네|넷째|다섯|다섯째)\s*(?:번째|째)", re.IGNORECASE,
            )
            matched = pattern.search(clause)
            if matched:
                component_type = kind
                break
        if not matched:
            continue
        count_raw, ordinal_raw = matched.group("count"), matched.group("ordinal")
        group_size = int(count_raw) if count_raw.isdigit() else korean_numbers.get(count_raw)
        ordinal = int(ordinal_raw) if ordinal_raw.isdigit() else korean_numbers.get(ordinal_raw)
        if not group_size or not ordinal or ordinal > group_size:
            continue
        existing_match = next((row for row in existing_targets
                               if row.get("component_type") == component_type
                               and row.get("group_size") == group_size and row.get("ordinal") == ordinal
                               and isinstance(row.get("evidence_quote"), str)
                               and (row["evidence_quote"] in clause or clause in row["evidence_quote"])), None)
        reference_id = str(existing_match.get("reference_id")) if existing_match else (
            f"rule-target-{len(existing_targets) + len(targets) + 1}")
        if not existing_match and reference_id in known_refs:
            continue
        region = next((value for terms, value in (
            (("위쪽", "상단", "위에", "위의"), "UPPER"), (("아래쪽", "하단", "아래에"), "LOWER"),
            (("왼쪽", "좌측"), "LEFT"), (("오른쪽", "우측"), "RIGHT"), (("중앙", "가운데"), "CENTER"),
        ) if any(term in clause for term in terms)), "UNSPECIFIED")
        color = next((value for terms, value in (
            (("빨간", "빨강", "적색", "red"), "빨간"), (("파란", "파랑", "청색", "blue"), "파란"),
            (("초록", "녹색", "green"), "초록"), (("노란", "황색", "yellow"), "노란"),
        ) if any(term in clause.casefold() for term in terms)), "")
        direction = matched.group("direction") or ""
        ordering = "LEFT_TO_RIGHT" if direction in {"왼쪽에서", "왼쪽부터", "좌측부터"} else (
            "TOP_TO_BOTTOM" if direction else "UNSPECIFIED")
        raw_target = {"reference_id": reference_id, "component_type": component_type,
                      "color": color, "region": region, "group_size": group_size,
                      "ordinal": ordinal, "ordering": ordering, "reference_designator": "",
                      "evidence_quote": clause[:160]}
        if not existing_match:
            target, _error = _validated_target(raw_target, text, known_refs)
            if not target:
                continue
            target["source"] = "rule_fallback_from_explicit_text"
            targets.append(target)
            known_refs.add(reference_id)
        observations.extend(Observation(
            code=code, value=True, status=EvidenceStatus.OBSERVED,
            source="user_report_rule_fallback", evidence_quote=clause[:160],
            target_reference=reference_id,
        ) for code in infer_symptom_codes(clause))
    return targets, observations


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
            "너는 PCB 증상 문장을 한국어 의미로 해석해 관측값으로 구조화한다. 단어가 등록 키워드와 글자 그대로 일치하는지로 판단하지 않는다. "
            "예: '세 번째 LED만 켜지지 않습니다'는 LED_NOT_LIT=true이고, '나머지 LED는 정상입니다'는 해당 peer observation이다. "
            "고장 원인은 추론하지 않는다. "
            "명시적 근거가 있는 관측만 최대 4개 반환하고, 증상 추측·누락된 사실의 부정 추론은 하지 않는다. "
            "불명확하면 observations와 uncertain_facts를 비워 둔다. "
            "입력에 명시된 사실만 추출하고, 각 항목의 evidence_quote는 원문에서 그대로 복사한다. "
            "원인 추측과 관측 증상을 분리한다: '커패시터가 나간 것 같다'는 원인 추측이므로 관측 코드로 만들지 않지만, "
            "같은 문장에 '제어 기능이 가끔 멈춘다'가 있으면 그 관측은 별도로 기록한다. "
            "구어체·축약·띄어쓰기 누락도 의미가 분명하면 허용 코드로 정규화한다: '보드 안켜짐'은 NO_POWER, "
            "'특정 기능 먹통/영 안 됨'은 PARTIAL_FUNCTION_FAILURE, 'LED 안들어옴'은 LED_NOT_LIT, "
            "'가끔 됐다 안됐다'는 INTERMITTENT_OPERATION, '연결된 장치가 반응하지 않음'은 OUTPUT_FAILURE다. "
            "'전원은 들어오는데 LED 하나만 안 켜짐'은 NO_POWER가 아니라 LED_NOT_LIT이다. "
            "예: '전원은 들어오는데 제어 쪽 몇 개가 먹통'은 PARTIAL_FUNCTION_FAILURE만 기록한다. "
            "이 문장으로 NO_POWER=false를 만들거나 전원 규격이 정상이라고 단정하지 않는다. "
            "'기판은 도는 것 같은데'는 작동 중이라는 표현이지 INTERMITTENT_OPERATION이 아니다. "
            "간헐 동작은 '가끔 멈춤/됐다 안 됨/켜졌다 꺼짐/움직일 때 변함'처럼 반복 또는 변화가 명시된 경우에만 기록한다. "
            "'연결된 장치가 반응하지 않는다'는 OUTPUT_FAILURE이며, 장치 고장이라고 사용자가 추측해도 원인 확정이 아니다. "
            "'전압이 떨어지는 것 같기도 하고 아닐지도'처럼 서로 반대 가능성을 함께 말하면 관측으로 확정하지 말고 clarification을 요청한다. "
            "'커넥터가 끊겼다더라'처럼 직접 관찰하지 않은 전언이나 '커패시터 고장' 같은 추측은 확인된 관측으로 승격하지 않는다. "
            "'기판이 돈다/돌아간다'는 구어체로 기판이 동작한다는 뜻일 수 있으므로 물리 회전으로 해석하지 않는다. "
            "가정·조건 표현(예: ~면, ~라면, ~할 경우)은 발생한 증상으로 기록하지 않는다. "
            "이중 부정이나 정정(예: 안 켜지는 것은 아님)은 증상 발생으로 기록하지 않는다. 긍정/부정은 문장 전체의 의미로 판단한다. "
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
    fallback_targets, fallback_observations = _rule_target_fallback(text, targets)
    targets.extend(fallback_targets[:max(0, 4 - len(targets))])
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
        folded_quote = str(quote or "").casefold()
        if code == "NO_POWER" and value is False and not re.search(
                r"(?:안\s*켜|무전원|전원\s*반응이\s*없).{0,16}(?:아니|아님|않)", text.casefold()):
            uncertain.append("전체 무전원이 아니라고 확인할 명시적 정정은 없습니다.")
            continue
        if code == "INTERMITTENT_OPERATION" and value is True and not any(term in folded_quote for term in (
                "간헐", "가끔", "때때로", "왔다 갔다", "왔다갔다", "됐다 안", "되다 안", "켜졌다 꺼",
                "불안정", "움직이면", "흔들면", "건드리면", "만지면", "깜빡")):
            uncertain.append("반복·변화가 명시되지 않아 간헐 동작으로 분류하지 않았습니다.")
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
        # Qwen has already interpreted the sentence. Rechecking its chosen code
        # against keyword rules discarded valid semantic parses such as
        # “세 번째 LED만 켜지지 않습니다”. Keep strict source-quote, enum,
        # target-reference, and contradiction validation instead.
        accepted.append(Observation(code=code, value=value, status=EvidenceStatus.OBSERVED,
                                    source="user_report", evidence_quote=verified_quote,
                                    target_reference=target_reference or None))
    accepted_codes = {(item.code, item.target_reference) for item in accepted}
    for observation in fallback_observations:
        # Qwen can recognize a symptom but attach a missing reference (or attach
        # the reference for “the other LEDs”) while the literal fallback finds
        # the positional target. Merge those two pieces of evidence so the
        # accepted symptom is bound to the rule-resolved target instead of
        # persisting as a duplicate system-level observation.
        matching_index = next((index for index, item in enumerate(accepted)
                               if item.code == observation.code and item.evidence_quote
                               and observation.evidence_quote
                               and (item.evidence_quote in observation.evidence_quote
                                    or observation.evidence_quote in item.evidence_quote)), None)
        if matching_index is not None:
            item = accepted[matching_index]
            target_by_reference = {str(target.get("reference_id")): target for target in targets}
            matched_target = target_by_reference.get(str(item.target_reference or ""))
            evidence_matches_target = bool(
                matched_target and isinstance(matched_target.get("evidence_quote"), str)
                and (item.evidence_quote in matched_target["evidence_quote"]
                     or matched_target["evidence_quote"] in item.evidence_quote)
            )
            if not item.target_reference or not evidence_matches_target:
                accepted[matching_index] = Observation(
                    code=item.code, value=item.value, status=item.status, source=item.source,
                    evidence_quote=item.evidence_quote, target_reference=observation.target_reference,
                )
                accepted_codes.discard((item.code, item.target_reference))
                accepted_codes.add((item.code, observation.target_reference))
                continue
        if (observation.code, observation.target_reference) not in accepted_codes:
            accepted.append(observation)
            accepted_codes.add((observation.code, observation.target_reference))
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
