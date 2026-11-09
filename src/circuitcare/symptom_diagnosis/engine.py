"""Deterministic, connection-optional symptom and evidence diagnosis engine."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from circuitcare.symptom_diagnosis.schemas import EvidenceStatus, Observation

KNOWLEDGE_DIR = Path(__file__).with_name("knowledge")
FAILURE_FUNCTION = {
    "NO_POWER": "power", "SUPPLY_VOLTAGE_DROOP": "power",
    "LED_NOT_LIT": "led", "BUZZER_NO_SOUND": "buzzer", "BUZZER_ABNORMAL_SOUND": "buzzer",
    "SWITCH_NO_RESPONSE": "control_input", "DISPLAY_BLANK": "display",
    "RELAY_NO_ACTUATION": "relay", "RELAY_ACTUATES_LOAD_UNRESPONSIVE": "relay_load",
    "TIMING_UNSTABLE": "control_timing", "OVERHEATING": "thermal",
}


@lru_cache(maxsize=1)
def load_knowledge() -> dict[str, Any]:
    """Load bundled knowledge through PyYAML's safe loader."""
    names = ("symptom_ontology", "component_taxonomy", "fault_modes", "symptom_rules", "test_catalog")
    return {name: yaml.safe_load((KNOWLEDGE_DIR / f"{name}.yaml").read_text(encoding="utf-8")) or {}
            for name in names}


def infer_symptom_codes(text: str, knowledge: dict[str, Any] | None = None) -> list[str]:
    """Fallback parser for reviewed symptom keywords when the local LLM is unavailable."""
    ontology = (knowledge or load_knowledge())["symptom_ontology"]["symptoms"]
    folded = (text or "").casefold()
    return [code for code, item in ontology.items()
            if any(str(word).casefold() in folded for word in item.get("keywords", []))]


def _condition_matches(condition: dict[str, Any], facts: dict[str, Any]) -> bool:
    """Interpret a small safe YAML rule language; never evaluate source text."""
    if "all" in condition:
        return all(_condition_matches(child, facts) for child in condition["all"])
    if "any" in condition:
        return any(_condition_matches(child, facts) for child in condition["any"])
    key, op, expected = condition.get("fact"), condition.get("operator"), condition.get("value")
    actual = facts.get(key)
    try:
        if op == "eq": return actual == expected
        if op == "neq": return actual != expected
        if op == "exists": return (key in facts) == bool(expected)
        if op == "gt": return actual is not None and float(actual) > float(expected)
        if op == "lt": return actual is not None and float(actual) < float(expected)
    except (TypeError, ValueError):
        return False
    return False


def _normalized_type(value: Any) -> str:
    aliases = {"integrated_circuit": "ic", "IC": "ic", "LED": "led", "unknown_component": "unknown"}
    text = str(value or "unknown").strip()
    return aliases.get(text, text.casefold())


def _normalize_observations(rows: dict[str, Any] | list[dict[str, Any]] | None) -> tuple[dict[str, Any], dict[str, Observation]]:
    facts: dict[str, Any] = {}
    provenance: dict[str, Observation] = {}
    if isinstance(rows, dict):
        items = [Observation(code=str(code), value=value, status=EvidenceStatus.OBSERVED,
                             source="user_report") for code, value in rows.items() if value is not None]
    else:
        items = [Observation.from_dict(row) for row in (rows or []) if isinstance(row, dict)]
    for observation in items:
        # Unknown, inferred, or contradicted items do not become hard rule facts.
        if observation.status not in {EvidenceStatus.OBSERVED, EvidenceStatus.MEASURED}:
            continue
        if observation.code:
            facts[observation.code] = observation.value
            provenance[observation.code] = observation
    return facts, provenance


def _derive_function_facts(facts: dict[str, Any]) -> None:
    failed_functions = {function for code, function in FAILURE_FUNCTION.items() if facts.get(code) is True}
    if facts.get("MULTIPLE_FUNCTIONS_FAILED") is True or len(failed_functions) >= 2:
        facts["MULTIPLE_FUNCTIONS_FAILED"] = True
    positive_functions = {"power" for code in ("POWER_INDICATOR_WORKING", "POWER_SOURCE_VERIFIED") if facts.get(code) is True}
    positive_functions.update(function for code, function in {
        "LED_WORKING": "led", "OTHER_FUNCTIONS_WORKING": "other",
        "LOAD_RESPONDS": "load", "RELAY_ACTUATED": "relay",
    }.items() if facts.get(code) is True)
    if failed_functions and positive_functions and not (failed_functions <= {"power"} and positive_functions <= {"power"}):
        facts["PARTIAL_FUNCTION_FAILURE"] = True


def _relative_distance(a: dict[str, Any], b: dict[str, Any], diagonal: float) -> float | None:
    box_a, box_b = a.get("bbox_xyxy", a.get("bbox")), b.get("bbox_xyxy", b.get("bbox"))
    if not (isinstance(box_a, (list, tuple)) and len(box_a) == 4
            and isinstance(box_b, (list, tuple)) and len(box_b) == 4 and diagonal > 0):
        return None
    ax, ay = (float(box_a[0]) + float(box_a[2])) / 2, (float(box_a[1]) + float(box_a[3])) / 2
    bx, by = (float(box_b[0]) + float(box_b[2])) / 2, (float(box_b[1]) + float(box_b[3])) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 / diagonal


def _functional_block_candidates(parts: list[dict[str, Any]], taxonomy: dict[str, list[str]]) -> list[dict[str, Any]]:
    boxes = [part.get("bbox_xyxy", part.get("bbox")) for part in parts]
    valid = [box for box in boxes if isinstance(box, (list, tuple)) and len(box) == 4]
    if not valid:
        return [{"component_id": part.get("component_id"), "component_type": part["component_type"],
                 "candidate_blocks": taxonomy.get(part["component_type"], []),
                 "nearby_component_ids": [], "status": "INFERRED_FROM_TYPE_ONLY", "verified": False}
                for part in parts if taxonomy.get(part["component_type"])]
    width = max(float(box[2]) for box in valid) - min(float(box[0]) for box in valid)
    height = max(float(box[3]) for box in valid) - min(float(box[1]) for box in valid)
    diagonal = max((width * width + height * height) ** 0.5, 1.0)
    result = []
    for part in parts:
        nearby = []
        for other in parts:
            if other is part:
                continue
            distance = _relative_distance(part, other, diagonal)
            if distance is not None and distance <= 0.18:
                nearby.append(other)
        role_votes: dict[str, int] = defaultdict(int)
        for related in [part, *nearby]:
            for role in taxonomy.get(related["component_type"], []):
                role_votes[role] += 1 if related is part else 0.5
        if role_votes:
            blocks = sorted(role_votes, key=lambda role: (-role_votes[role], role))[:3]
            result.append({"component_id": part.get("component_id"),
                           "component_type": part["component_type"],
                           "candidate_blocks": blocks,
                           "nearby_component_ids": [item.get("component_id") for item in nearby],
                           "status": "INFERRED_FROM_TYPE_AND_NEIGHBORHOOD" if nearby else "INFERRED_FROM_TYPE_ONLY",
                           "verified": False})
    return result


def diagnose(
    *, symptoms: list[str] | None = None, symptom_text: str = "",
    observations: dict[str, Any] | list[dict[str, Any]] | None = None,
    components: list[dict[str, Any]] | None = None,
    connectivity: dict[str, Any] | None = None, connectivity_status: str = "none",
    knowledge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate ranked, explicitly unverified causes from normalized evidence."""
    kb = knowledge or load_knowledge()
    ontology = kb["symptom_ontology"]["symptoms"]
    observed_facts, provenance = _normalize_observations(observations)
    selected = list(dict.fromkeys([*(symptoms or []), *infer_symptom_codes(symptom_text, kb),
                                   *(code for code in observed_facts if code in ontology)]))
    selected = [code for code in selected if code in ontology]
    facts = dict(observed_facts)
    facts.update({code: True for code in selected})
    normalized_parts = [dict(part, component_type=_normalized_type(part.get("component_type", part.get("label"))))
                        for part in (components or [])]
    detected_types = {part["component_type"] for part in normalized_parts}
    facts.setdefault("FUSE_PRESENT", "fuse" in detected_types)
    _derive_function_facts(facts)

    matched_rules = [rule for rule in kb["symptom_rules"]["rules"]
                     if _condition_matches(rule.get("when", {}), facts)]
    generated: set[str] = {fault_id for rule in matched_rules for fault_id in rule.get("generate", [])}
    modes = kb["fault_modes"]["fault_modes"]
    if selected or any(code in observed_facts for code in ontology):
        generated.update(("UNKNOWN_UNREGISTERED_FAILURE",))
        if facts.get("MULTIPLE_FUNCTIONS_FAILED"):
            generated.add("COMPOUND_FAILURE")
    else:
        return {"status": "NEEDS_SYMPTOMS", "symptom_codes": [], "observations": facts,
                "matched_rules": [], "hypotheses": [], "functional_blocks": [], "next_tests": [],
                "message": "증상을 확인해야 원인 후보를 좁힐 수 있습니다."}

    ranked = []
    for fault_id in generated:
        mode = modes.get(fault_id)
        if not mode:
            continue
        score = 0.0
        evidence_for: list[str] = []
        evidence_against: list[str] = []
        evidence_details: list[dict[str, Any]] = []
        for symptom in selected:
            weight = float(mode.get("supported_symptoms", {}).get(symptom, 0))
            if weight:
                score += weight
                evidence_for.append(ontology[symptom]["label"])
                evidence_details.append({"code": symptom, "source": provenance.get(symptom, Observation(
                    code=symptom, value=True, status=EvidenceStatus.OBSERVED,
                    source="user_report_or_confirmed_symptom")).source,
                    "status": provenance.get(symptom, Observation(code=symptom)).status.value,
                    "weight": weight})
        for code, weight in mode.get("supporting_evidence", {}).items():
            value = facts.get(code)
            if value is True:
                obs = provenance.get(code)
                factor = 1.0 if obs and obs.status == EvidenceStatus.MEASURED else 0.7 if obs else 0.7
                contribution = float(weight) * factor
                score += contribution
                evidence_for.append(code)
                evidence_details.append({"code": code, "source": obs.source if obs else "user_report",
                                         "status": obs.status.value if obs else "observed",
                                         "weight": round(contribution, 2)})
            elif value is False:
                evidence_against.append(code)
                score -= float(weight) * 0.7
        for code, weight in mode.get("opposing_evidence", {}).items():
            value = facts.get(code)
            if value is True:
                obs = provenance.get(code)
                factor = 1.0 if obs and obs.status == EvidenceStatus.MEASURED else 0.7
                contribution = float(weight) * factor
                score -= contribution
                evidence_against.append(code)
                evidence_details.append({"code": code, "source": obs.source if obs else "user_report",
                                         "status": obs.status.value if obs else "observed",
                                         "weight": -round(contribution, 2)})
        applicable_parts = [part for part in normalized_parts
                            if part["component_type"] in mode.get("component_types", [])]
        # Detection gives a small location cue, never a claim that a part is faulty.
        detection_bonus = min(1.0, 0.2 * len(applicable_parts))
        score += detection_bonus
        if applicable_parts:
            evidence_for.append("관련 소자 탐지(약한 위치 단서)")
            evidence_details.append({"code": "RELATED_COMPONENT_DETECTED", "source": "pcb_vision",
                                     "status": "inferred", "weight": round(detection_bonus, 2)})
        ranked.append({"fault_id": fault_id, "title": mode["title"], "level": mode["level"],
                       "score": round(score, 2),
                       "supporting_evidence": list(dict.fromkeys(evidence_for)),
                       "opposing_evidence": list(dict.fromkeys(evidence_against)),
                       "evidence_details": evidence_details,
                       "component_candidates": [part.get("component_id") for part in applicable_parts],
                       "status": "UNVERIFIED_CANDIDATE"})
    ranked.sort(key=lambda row: (-row["score"], row["title"]))

    taxonomy = kb["component_taxonomy"].get("component_types", {})
    functional_blocks = _functional_block_candidates(normalized_parts, taxonomy)
    tests = kb["test_catalog"]["tests"]
    shortlist = ranked[:5]
    test_relevance: dict[str, list[bool]] = defaultdict(list)
    for row in shortlist:
        for test_id in tests:
            test_relevance[test_id].append(test_id in modes[row["fault_id"]].get("recommended_tests", []))
    test_ids = [test_id for test_id, outcomes in test_relevance.items() if any(outcomes)]
    power_off = facts.get("POWER_OFF_CONFIRMED") is True
    recommended = []
    for test_id in test_ids:
        test = tests.get(test_id)
        if not test or test.get("risk_level") not in {"none", "low"}:
            continue
        if test.get("risk_level") == "low" and (not test.get("requires_power_off") or not power_off):
            continue
        outcomes = test_relevance[test_id]
        distinguishes = len(set(outcomes)) > 1
        cost = 0 if test.get("risk_level") == "none" else 1
        utility = (1 if distinguishes else 0) - 0.25 * cost
        recommended.append({"test_id": test_id, **test, "distinguishes_candidates": distinguishes,
                            "utility": utility})
    recommended.sort(key=lambda row: (-row["utility"], row["risk_level"] != "none", row["name"]))

    status = "CANDIDATES_AVAILABLE" if ranked and selected else "ABSTAIN_INSUFFICIENT_EVIDENCE"
    connectivity_status = connectivity_status if connectivity_status in {
        "none", "inferred", "partial_verified", "verified"} else "none"
    # Verified connectivity is accepted only when supplied as scoped evidence;
    # this first rule-engine version deliberately does not consume arbitrary nets.
    verified_edges = connectivity.get("verified_edges", []) if isinstance(connectivity, dict) else []
    return {"status": status, "symptom_codes": selected, "observations": facts,
            "observation_provenance": [item.to_dict() for item in provenance.values()],
            "matched_rules": [{"id": rule["id"], "name": rule["name"]}
                              for rule in sorted(matched_rules, key=lambda item: -item.get("priority", 0))],
            "hypotheses": ranked, "functional_blocks": functional_blocks,
            "next_tests": recommended[:5], "connectivity_status": connectivity_status,
            "verified_connectivity_evidence_count": len(verified_edges) if connectivity_status in {"verified", "partial_verified"} else 0,
            "message": ("후보 원인을 정리했습니다. 순위 점수는 확률이 아니며 검증 전 후보입니다."
                        if status == "CANDIDATES_AVAILABLE" else "정보가 부족해 진단을 보류합니다. 추가 증상이나 확인 결과를 입력해 주세요.")}
