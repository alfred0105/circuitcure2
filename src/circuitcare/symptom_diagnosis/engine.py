"""Deterministic, connection-optional symptom and evidence diagnosis engine."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import re
import time
from typing import Any

import yaml

from circuitcare.symptom_diagnosis.schemas import EvidenceStatus, Observation
from circuitcare.component_taxonomy import normalize_component_class
from circuitcare.symptom_diagnosis.functional_blocks import analyze_functional_blocks
from circuitcare.symptom_diagnosis.functional_blocks import build_functional_block_groups
from circuitcare.symptom_diagnosis.evidence_ledger import build_evidence_ledger
from circuitcare.symptom_diagnosis.evidence_gaps import analyze_evidence_gaps
from circuitcare.symptom_diagnosis.component_fault_ranker import rank_component_candidates, rank_joint_candidates
from circuitcare.symptom_diagnosis.fault_hypothesis_validator import (
    summarize_measurement_verdict, validate_component_hypothesis,
)
from circuitcare.symptom_diagnosis.measurement_evaluator import evaluate_component_measurements
from circuitcare.symptom_diagnosis.measurement_validation import verified_measurements_for_inference
from circuitcare.symptom_diagnosis.active_inspection_planner import plan_active_inspections

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
    names = ("symptom_ontology", "component_taxonomy", "fault_modes", "symptom_rules",
             "test_catalog", "ic_part_catalog", "component_test_profiles")
    knowledge = {name: yaml.safe_load((KNOWLEDGE_DIR / f"{name}.yaml").read_text(encoding="utf-8")) or {}
                 for name in names}
    validate_knowledge_base(knowledge)
    return knowledge


def validate_knowledge_base(knowledge: dict[str, Any]) -> None:
    """Fail early on dangling rule IDs, invalid evidence codes, or duplicate rules."""
    symptoms = knowledge["symptom_ontology"].get("symptoms", {})
    observations = knowledge["symptom_ontology"].get("observations", {})
    fault_modes = knowledge["fault_modes"].get("fault_modes", {})
    component_types = knowledge["component_taxonomy"].get("component_types", {})
    tests = knowledge["test_catalog"].get("tests", {})
    allowed_facts = set(symptoms) | set(observations) | {
        "FUSE_PRESENT", "MULTIPLE_FUNCTIONS_FAILED", "PARTIAL_FUNCTION_FAILURE",
    }
    errors: list[str] = []
    rule_ids: set[str] = set()

    def inspect_condition(condition: Any, rule_id: str) -> None:
        if not isinstance(condition, dict):
            errors.append(f"{rule_id}: condition must be an object")
            return
        if "all" in condition or "any" in condition:
            key = "all" if "all" in condition else "any"
            children = condition.get(key)
            if not isinstance(children, list) or not children:
                errors.append(f"{rule_id}: {key} must have at least one condition")
                return
            for child in children:
                inspect_condition(child, rule_id)
            return
        fact, operator = condition.get("fact"), condition.get("operator")
        if fact not in allowed_facts:
            errors.append(f"{rule_id}: unknown fact {fact!r}")
        if operator not in {"eq", "neq", "exists", "gt", "lt"}:
            errors.append(f"{rule_id}: unknown operator {operator!r}")

    for rule in knowledge["symptom_rules"].get("rules", []):
        rule_id = str(rule.get("id", ""))
        if not rule_id or rule_id in rule_ids:
            errors.append(f"duplicate or empty symptom rule ID: {rule_id!r}")
        rule_ids.add(rule_id)
        inspect_condition(rule.get("when", {}), rule_id)
        for fault_id in rule.get("generate", []):
            if fault_id not in fault_modes:
                errors.append(f"{rule_id}: unknown fault ID {fault_id!r}")
    for fault_id, mode in fault_modes.items():
        if mode.get("level") in {"component", "functional_block"}:
            declared_types = {_normalized_type(value) for value in mode.get("candidate_component_types", [])}
            if not declared_types:
                errors.append(f"{fault_id}: component-scoped modes require candidate_component_types")
            unknown_types = declared_types - set(component_types)
            if unknown_types:
                errors.append(f"{fault_id}: unknown component types {sorted(unknown_types)}")
        required_types = {_normalized_type(value) for value in mode.get("required_component_types", [])}
        unknown_required_types = required_types - set(component_types)
        if unknown_required_types:
            errors.append(f"{fault_id}: unknown required component types {sorted(unknown_required_types)}")
        for symptom_code, raw_types in mode.get("required_component_types_by_symptom", {}).items():
            if symptom_code not in symptoms:
                errors.append(f"{fault_id}: unknown symptom requirement {symptom_code!r}")
            unknown_types = {_normalized_type(value) for value in raw_types} - set(component_types)
            if unknown_types:
                errors.append(f"{fault_id}: unknown symptom-required component types {sorted(unknown_types)}")
        unknown_symptoms = set(mode.get("supported_symptoms", {})) - set(symptoms)
        if unknown_symptoms:
            errors.append(f"{fault_id}: unknown symptom codes {sorted(unknown_symptoms)}")
        for evidence_key in ("supporting_evidence", "supporting_evidence_when_false",
                             "opposing_evidence", "opposing_evidence_when_false"):
            unknown_facts = set(mode.get(evidence_key, {})) - allowed_facts
            if unknown_facts:
                errors.append(f"{fault_id}: unknown {evidence_key} codes {sorted(unknown_facts)}")
        unknown_tests = set(mode.get("recommended_tests", [])) - set(tests)
        if unknown_tests:
            errors.append(f"{fault_id}: unknown test IDs {sorted(unknown_tests)}")
    for test_id, test in tests.items():
        for outcome in test.get("results", {}).values():
            observed_codes = set(outcome.get("observations", {})) if isinstance(outcome, dict) else set()
            unknown_codes = observed_codes - allowed_facts
            if unknown_codes:
                errors.append(f"{test_id}: unknown observation codes {sorted(unknown_codes)}")
    profiles = knowledge.get("component_test_profiles", {}).get("profiles", {})
    if not isinstance(profiles, dict):
        errors.append("component_test_profiles.profiles must be an object")
    else:
        all_test_ids: set[str] = set()
        for component_type, profile in profiles.items():
            tests_for_type = profile.get("tests", {}) if isinstance(profile, dict) else {}
            for test_id, test in tests_for_type.items():
                all_test_ids.add(str(test_id))
                for fault_id, relations in test.get("fault_relations", {}).items():
                    if fault_id not in fault_modes:
                        errors.append(f"{component_type}.{test_id}: unknown fault ID {fault_id!r}")
                    if not isinstance(relations, dict) or any(
                        value not in {"supports", "refutes", "non_diagnostic"}
                        for value in relations.values()
                    ):
                        errors.append(f"{component_type}.{test_id}: invalid fault relation")
        for fault_id, mode in fault_modes.items():
            missing_profiles = set(mode.get("required_measurement_tests", [])) - all_test_ids
            if missing_profiles:
                errors.append(f"{fault_id}: missing measurement profiles {sorted(missing_profiles)}")
    if errors:
        raise ValueError("Invalid symptom diagnosis knowledge base: " + "; ".join(errors))


def infer_symptom_codes(text: str, knowledge: dict[str, Any] | None = None) -> list[str]:
    """Fallback parser for reviewed symptom keywords when the local LLM is unavailable."""
    ontology = (knowledge or load_knowledge())["symptom_ontology"]["symptoms"]
    folded = (text or "").casefold()
    return [code for code, item in ontology.items()
            if any(_keyword_is_asserted(folded, str(word).casefold())
                   for word in item.get("keywords", []))]


def _keyword_is_asserted(text: str, keyword: str) -> bool:
    """Avoid treating hypothetical or explicitly negated clauses as confirmed symptoms."""
    start = text.find(keyword)
    while start >= 0:
        clause_start = max(text.rfind(mark, 0, start) for mark in (".", "!", "?", "\n", ";")) + 1
        clause_end_candidates = [position for mark in (".", "!", "?", "\n", ";")
                                 if (position := text.find(mark, start + len(keyword))) >= 0]
        clause_end = min(clause_end_candidates, default=len(text))
        clause = text[clause_start:clause_end]
        tail = text[start + len(keyword):clause_end]
        if not re.search(r"(?:면|다면|경우|때|가정하면|라면)\s*(?:에는|이라면)?\s*$", tail[-24:]):
            if not re.search(r"(?:지\s*않(?:음|다|고|는|아)|지는\s*않|아님|아니(?:다|고|라|야)|없다고\s*할\s*수\s*없)", tail[:60]):
                return True
        start = text.find(keyword, start + 1)
    return False


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
    return normalize_component_class(value)


def _normalize_observations(
    rows: dict[str, Any] | list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Observation], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str | None], list[Observation]] = defaultdict(list)
    if isinstance(rows, dict):
        items = [Observation(code=str(code), value=value, status=EvidenceStatus.OBSERVED,
                             source="user_report") for code, value in rows.items() if value is not None]
    else:
        items = [Observation.from_dict(row) for row in (rows or []) if isinstance(row, dict)]
    for observation in items:
        if observation.code:
            grouped[(observation.code, observation.component_id)].append(observation)

    facts: dict[str, Any] = {}
    provenance: dict[str, Observation] = {}
    conflicts: list[dict[str, Any]] = []
    usable_statuses = {EvidenceStatus.MEASURED, EvidenceStatus.OBSERVED}
    for (code, component_id), observations in grouped.items():
        contradicted = [row for row in observations if row.status == EvidenceStatus.CONTRADICTED]
        if contradicted:
            conflicts.append({"code": code, "component_id": component_id,
                              "values": list({repr(row.value): row.value for row in contradicted}.values()),
                              "evidence": [{"value": row.value, "status": row.status.value,
                                            "source": row.source} for row in observations],
                              "sources": list(dict.fromkeys(row.source for row in observations)),
                              "resolution": "PARSER_REPORTED_CONTRADICTION"})
            continue
        usable = [row for row in observations if row.status in usable_statuses]
        if not usable:
            continue
        values = {repr(row.value): row.value for row in usable}
        if len(values) > 1:
            conflicts.append({"code": code, "component_id": component_id, "values": list(values.values()),
                              "evidence": [{"value": row.value, "status": row.status.value,
                                            "source": row.source} for row in usable],
                              "sources": list(dict.fromkeys(row.source for row in usable)),
                              "resolution": "UNRESOLVED_CONFLICT"})
            continue
        chosen = usable[-1]
        # Component-specific evidence remains scoped for the component ranker;
        # it must not silently become a board-wide fact.
        if component_id:
            continue
        facts[code] = chosen.value
        provenance[code] = chosen
    return facts, provenance, conflicts


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
    confirmed_component_ids: list[str] | set[str] | tuple[str, ...] | None = None,
    inventory_complete: bool = False,
    measurements: list[dict[str, Any]] | None = None,
    current_context: dict[str, Any] | None = None,
    silkscreen_labels: list[dict[str, Any]] | None = None,
    connectivity: dict[str, Any] | None = None, connectivity_status: str = "none",
    knowledge: dict[str, Any] | None = None,
    inspection_history: list[dict[str, Any]] | None = None,
    allow_retest: bool = False,
    _simulate_questions: bool = True,
) -> dict[str, Any]:
    """Generate ranked, explicitly unverified causes from normalized evidence."""
    diagnosis_started = time.perf_counter()
    kb = knowledge or load_knowledge()
    ontology = kb["symptom_ontology"]["symptoms"]
    observed_facts, provenance, evidence_conflicts = _normalize_observations(observations)
    ledger_observations = (list(observations) if isinstance(observations, list) else
                           [{"code": str(code), "value": value, "status": "observed", "source": "user_report"}
                            for code, value in observations.items() if value is not None]
                           if isinstance(observations, dict) else [])
    requested_symptoms = list(dict.fromkeys([*(symptoms or []), *infer_symptom_codes(symptom_text, kb)]))
    selected = []
    considered: set[str] = set()
    conflicted_codes = {str(row.get("code")) for row in evidence_conflicts
                        if not row.get("component_id")}
    for code in [*requested_symptoms, *(code for code in observed_facts if code in ontology)]:
        if code not in ontology or code in considered:
            continue
        considered.add(code)
        if code in conflicted_codes:
            continue
        # A negative observation contradicting a selected/text symptom is held unresolved.
        if code in observed_facts and observed_facts[code] is False:
            if code in requested_symptoms:
                evidence_conflicts.append({"code": code, "values": [True, False],
                                          "sources": ["selected_symptom", provenance[code].source],
                                          "resolution": "UNRESOLVED_CONFLICT"})
                observed_facts.pop(code, None)
                provenance.pop(code, None)
            continue
        selected.append(code)
    facts = dict(observed_facts)
    for code in selected:
        facts.setdefault(code, True)
    normalized_parts = [dict(part, component_type=_normalized_type(part.get("component_type", part.get("label"))))
                        for part in (components or [])]
    detected_parts = normalized_parts
    # A detection is not proof of presence. Only explicitly reviewed IDs may
    # create component-level hypotheses or scope component measurements.
    supplied_confirmed_ids = {str(value) for value in (confirmed_component_ids or []) if str(value)}
    known_component_ids = {str(part.get("component_id") or "") for part in normalized_parts}
    confirmed_ids = supplied_confirmed_ids & known_component_ids
    confirmed_parts = [part for part in normalized_parts
                       if str(part.get("component_id") or "") in confirmed_ids
                       and part["component_type"] not in {"unknown", "not_component"}]
    unconfirmed_ids = [str(part.get("component_id")) for part in normalized_parts
                       if str(part.get("component_id") or "") not in confirmed_ids]
    normalized_parts = confirmed_parts
    detected_types = {part["component_type"] for part in normalized_parts
                      if part.get("component_type_status") in (None, "USER_CONFIRMED")}
    if "fuse" in detected_types:
        facts.setdefault("FUSE_PRESENT", True)
    elif inventory_complete:
        facts.setdefault("FUSE_PRESENT", False)
    _derive_function_facts(facts)
    taxonomy = kb["component_taxonomy"].get("component_types", {})
    functional_blocks = analyze_functional_blocks(
        detected_parts, taxonomy, silkscreen_labels=silkscreen_labels,
        ic_catalog=kb.get("ic_part_catalog", {}).get("parts", {}),
    )
    confirmed_functional_blocks = [row for row in functional_blocks
                                   if str(row.get("component_id") or "") in confirmed_ids]
    functional_block_groups = build_functional_block_groups(detected_parts, functional_blocks)
    blocks_by_component = {str(row.get("component_id")): row for row in confirmed_functional_blocks}

    matched_rules = [rule for rule in kb["symptom_rules"]["rules"]
                     if _condition_matches(rule.get("when", {}), facts)]
    generated: set[str] = {fault_id for rule in matched_rules for fault_id in rule.get("generate", [])}
    modes = kb["fault_modes"]["fault_modes"]
    confirmed_part_by_id = {str(part.get("component_id")): part for part in normalized_parts}
    contextual_measurements = [
        {**row, "component_type": confirmed_part_by_id[str(row.get("component_id"))]["component_type"]}
        for row in (measurements or []) if isinstance(row, dict)
        and str(row.get("component_id") or "") in confirmed_part_by_id
    ]
    measurement_started = time.perf_counter()
    scoped_measurements = evaluate_component_measurements(
        contextual_measurements,
        profiles=kb.get("component_test_profiles", {}), current_context=current_context,
    )
    ranking_measurements = verified_measurements_for_inference(
        contextual_measurements, profiles=kb.get("component_test_profiles", {}),
        current_context=current_context,
    )
    measurement_validation_ms = (time.perf_counter() - measurement_started) * 1000
    generated.update(fault_id for (_component_id, fault_id, _test, _target, _condition), assessment in scoped_measurements.items()
                     if assessment.get("status") in {"TEST_NORMAL", "TEST_ABNORMAL"}
                     and fault_id in modes)
    # Rule conditions may be deliberately narrow, but every asserted symptom must
    # still receive the reviewed failure modes that explicitly support it.
    generated.update(
        fault_id for fault_id, mode in modes.items()
        if any(float(mode.get("supported_symptoms", {}).get(code, 0)) > 0 for code in selected)
    )
    def has_required_inventory(fault_id: str) -> bool:
        mode = modes.get(fault_id, {})
        required = {_normalized_type(value) for value in mode.get("required_component_types", [])}
        symptom_requirements = mode.get("required_component_types_by_symptom", {})
        if not required.issubset(detected_types):
            return False
        relevant_symptoms = set(mode.get("supported_symptoms", {})) & set(selected)
        for symptom in relevant_symptoms:
            per_symptom = {_normalized_type(value) for value in symptom_requirements.get(symptom, [])}
            if not per_symptom.issubset(detected_types):
                return False
        candidate_types = {_normalized_type(value) for value in mode.get("candidate_component_types", [])}
        if mode.get("level") == "board" and candidate_types and not (candidate_types & detected_types):
            return False
        return True

    generated = {fault_id for fault_id in generated if has_required_inventory(fault_id)}
    # Component and functional-block failure modes are not valid unless at
    # least one reviewed instance has a compatible type and any required role
    # is explicitly confirmed or directly supported by component-scoped facts.
    component_scoped_faults: set[str] = set()
    for fault_id in generated:
        mode = modes.get(fault_id, {})
        if mode.get("level") not in {"component", "functional_block"}:
            continue
        for part in normalized_parts:
            component_id = str(part.get("component_id") or "")
            assessment = scoped_measurements.get((component_id, fault_id), {})
            validation = validate_component_hypothesis(
                mode=mode, component=part, confirmed_component_ids=confirmed_ids,
                confirmed_component_types=detected_types,
                component_block=blocks_by_component.get(component_id),
                component_observations=observations if isinstance(observations, list) else [],
                measurement_verdict=summarize_measurement_verdict(
                    mode=mode, component_id=component_id, fault_id=fault_id,
                    assessments=scoped_measurements,
                ),
            )
            if validation["eligible"]:
                component_scoped_faults.add(fault_id)
                break
    generated = {fault_id for fault_id in generated
                 if modes.get(fault_id, {}).get("level") not in {"component", "functional_block"}
                 or fault_id in component_scoped_faults}
    if selected:
        generated.update(("UNKNOWN_UNREGISTERED_FAILURE",))
        if facts.get("MULTIPLE_FUNCTIONS_FAILED"):
            generated.add("COMPOUND_FAILURE")
    else:
        evidence_ledger = build_evidence_ledger(
            observations=ledger_observations,
            conflicts=evidence_conflicts, components=[dict(part, component_type_status=(
                part.get("component_type_status") or ("USER_CONFIRMED" if str(part.get("component_id") or "") in confirmed_ids else "UNCONFIRMED")),
                presence_status="CONFIRMED" if str(part.get("component_id") or "") in confirmed_ids else "UNCONFIRMED")
                for part in detected_parts], functional_blocks=functional_blocks,
            measurements=measurements or [],
            source_version=str((current_context or {}).get("component_map_version") or "") or None)
        return {"status": "NEEDS_SYMPTOMS", "symptom_codes": [], "observations": facts,
                "evidence_conflicts": evidence_conflicts,
                "matched_rules": [], "hypotheses": [], "functional_blocks": functional_blocks,
                "functional_block_groups": functional_block_groups,
                "evidence_ledger": evidence_ledger, "evidence_gaps": [], "next_tests": [],
                "active_inspection_plan": {"status": "NO_HYPOTHESES", "candidates": [],
                                            "blocked": [], "score_is_probability": False},
                "processing_timings_ms": {"measurement_validation": round(measurement_validation_ms, 3),
                                           "candidate_ranking": 0.0, "recommendation_generation": 0.0,
                                           "total": round((time.perf_counter() - diagnosis_started) * 1000, 3)},
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
        for symptom, raw_weight in mode.get("supported_symptoms", {}).items():
            if facts.get(symptom) is False:
                penalty = float(raw_weight) * 0.7
                score -= penalty
                evidence_against.append(f"{symptom}=false")
                evidence_details.append({"code": symptom, "source": provenance.get(symptom, Observation(
                    code=symptom, value=False, status=EvidenceStatus.OBSERVED,
                    source="user_report")).source,
                    "status": provenance.get(symptom, Observation(code=symptom)).status.value,
                    "weight": -round(penalty, 2)})
        for code, weight in mode.get("supporting_evidence", {}).items():
            value = facts.get(code)
            if value is True:
                if code in selected:
                    continue
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
        for code, weight in mode.get("supporting_evidence_when_false", {}).items():
            if facts.get(code) is False:
                if code in ontology:
                    continue
                obs = provenance.get(code)
                factor = 1.0 if obs and obs.status == EvidenceStatus.MEASURED else 0.7
                contribution = float(weight) * factor
                score += contribution
                evidence_for.append(f"{code}=false")
                evidence_details.append({"code": code, "source": obs.source if obs else "user_report",
                                         "status": obs.status.value if obs else "observed",
                                         "weight": round(contribution, 2)})
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
        for code, weight in mode.get("opposing_evidence_when_false", {}).items():
            if facts.get(code) is False:
                obs = provenance.get(code)
                factor = 1.0 if obs and obs.status == EvidenceStatus.MEASURED else 0.7
                contribution = float(weight) * factor
                score -= contribution
                evidence_against.append(f"{code}=false")
                evidence_details.append({"code": code, "source": obs.source if obs else "user_report",
                                         "status": obs.status.value if obs else "observed",
                                         "weight": -round(contribution, 2)})
        applicable_parts = []
        for part in normalized_parts:
            component_id = str(part.get("component_id") or "")
            assessment = scoped_measurements.get((component_id, fault_id), {})
            validation = validate_component_hypothesis(
                mode=mode, component=part, confirmed_component_ids=confirmed_ids,
                confirmed_component_types=detected_types,
                component_block=blocks_by_component.get(component_id),
                component_observations=observations if isinstance(observations, list) else [],
                measurement_verdict=summarize_measurement_verdict(
                    mode=mode, component_id=component_id, fault_id=fault_id,
                    assessments=scoped_measurements,
                ),
            )
            if validation["eligible"]:
                applicable_parts.append(part)
        # Part count is not evidence of failure. Detection only scopes candidates;
        # score can change only when evidence, role, or an answer supports it.
        ranked.append({"fault_id": fault_id, "title": mode["title"], "level": mode["level"],
                       "score": round(score, 2),
                       "supporting_evidence": list(dict.fromkeys(evidence_for)),
                       "opposing_evidence": list(dict.fromkeys(evidence_against)),
                       "evidence_details": evidence_details,
                       "related_component_ids": [part.get("component_id") for part in applicable_parts],
                       "status": "UNVERIFIED_CANDIDATE"})
    # Keep the legacy flat list for existing callers; new clients receive
    # independently sorted level lists below instead of comparing unlike ranks.
    level_priority = {"component": 0, "functional_block": 1, "board": 2, "system": 3}
    ranked.sort(key=lambda row: (level_priority.get(row["level"], 4), -row["score"], row["title"]))
    rule_ranking_started = time.perf_counter()
    ranked_component_evidence = rank_component_candidates(
        hypotheses=ranked, components=normalized_parts, functional_blocks=confirmed_functional_blocks,
        observations=(observations if isinstance(observations, list) else []),
        measurements=ranking_measurements,
        confirmed_component_ids=confirmed_ids, profiles=kb.get("component_test_profiles", {}),
        current_context=current_context, knowledge=kb,
        limit=max(3, len(normalized_parts) * max(1, len(ranked))),
        include_refuted=True, distinct_components=False,
    )
    verified_component_history = [row for row in ranked_component_evidence
                                  if row.get("visual_status") in {"FAULT_SUPPORTED", "FAULT_REFUTED"}]
    component_candidates = rank_component_candidates(
        hypotheses=ranked, components=normalized_parts, functional_blocks=confirmed_functional_blocks,
        observations=(observations if isinstance(observations, list) else []),
        measurements=ranking_measurements,
        confirmed_component_ids=confirmed_ids, profiles=kb.get("component_test_profiles", {}),
        current_context=current_context, knowledge=kb, limit=3, include_refuted=False,
    )
    # Keep the full fault/component-pair ranking internally. The baseline UI
    # still shows its historical three distinct components, while opt-in
    # strategies and evaluators can inspect every eligible pair before slicing.
    all_component_candidates = ranked_component_evidence
    candidate_groups_by_id: dict[str, dict[str, Any]] = {}
    for row in all_component_candidates:
        group_id = row.get("candidate_group_id")
        if not group_id:
            continue
        group = candidate_groups_by_id.setdefault(str(group_id), {
            "candidate_group_id": str(group_id), "fault_id": row.get("fault_hypothesis"),
            "component_ids": list(row.get("candidate_group_component_ids", [])),
            "status": "UNRESOLVED_AMBIGUITY", "reason": "현재 근거로는 같은 종류 후보를 구별할 수 없습니다.",
            "score_is_probability": False,
        })
        group["component_ids"] = sorted(set(group["component_ids"]) |
                                        {str(row.get("component_id"))})
    joint_candidates = rank_joint_candidates(
        hypotheses=ranked, component_candidates=component_candidates, observations=facts,
    )
    candidate_ranking_ms = (time.perf_counter() - rule_ranking_started) * 1000

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
        if not _simulate_questions:
            break
        test = tests.get(test_id)
        if not test or test.get("risk_level") not in {"none", "low"}:
            continue
        if test.get("risk_level") == "low" and (not test.get("requires_power_off") or not power_off):
            continue
        outcomes = test_relevance[test_id]
        # Re-run the actual diagnosis rules for each answer instead of estimating
        # value with a separate approximation of the scoring logic.
        test_results = test.get("results", {})
        answer_scores: dict[str, list[tuple[str, float]]] = {}
        for answer, outcome in test_results.items():
            updates = outcome.get("observations", {}) if isinstance(outcome, dict) else {}
            simulated_observations = (list(observations) if isinstance(observations, list) else
                                      [{"code": code, "value": value, "status": "observed",
                                        "source": "user_report"}
                                       for code, value in observations.items()]
                                      if isinstance(observations, dict) else [])
            simulated_observations.extend({"code": code, "value": value, "status": "observed",
                                           "source": "question_value_simulation"}
                                          for code, value in updates.items())
            simulated = diagnose(
                symptoms=selected, observations=simulated_observations,
                components=normalized_parts, confirmed_component_ids=confirmed_ids,
                inventory_complete=inventory_complete, measurements=measurements,
                current_context=current_context,
                silkscreen_labels=silkscreen_labels, connectivity=connectivity,
                connectivity_status=connectivity_status, knowledge=kb,
                inspection_history=inspection_history, allow_retest=allow_retest,
                _simulate_questions=False,
            )
            answer_scores[answer] = [(row["fault_id"], float(row["score"]))
                                     for row in simulated.get("hypotheses", [])[:5]]
        rank_orders = [tuple(fault_id for fault_id, _ in sorted(scores, key=lambda pair: (-pair[1], pair[0])))
                       for scores in answer_scores.values()]
        score_profiles = [tuple(round(score, 3) for _, score in sorted(scores))
                          for scores in answer_scores.values()]
        distinguishes = len(set(rank_orders)) > 1 or len(set(score_profiles)) > 1
        cost = 0 if test.get("risk_level") == "none" else 1
        utility = (1 if distinguishes else 0) - 0.25 * cost
        recommended.append({"test_id": test_id, **test, "distinguishes_candidates": distinguishes,
                            "answer_score_simulation": answer_scores, "utility": utility})
    if not _simulate_questions:
        recommended = []
    recommended.sort(key=lambda row: (-row["utility"], row["risk_level"] != "none", row["name"]))
    recommendation_ms = (time.perf_counter() - rule_ranking_started) * 1000 - candidate_ranking_ms

    status = "CANDIDATES_AVAILABLE" if ranked and selected else "ABSTAIN_INSUFFICIENT_EVIDENCE"
    hypothesis_groups = {
        "component": [row for row in ranked if row.get("level") in {"component", "functional_block"}],
        "board": [row for row in ranked if row.get("level") == "board"],
        "system": [row for row in ranked if row.get("level") == "system"],
    }
    connectivity_status = connectivity_status if connectivity_status in {
        "none", "inferred", "partial_verified", "verified"} else "none"
    # Verified connectivity is accepted only when supplied as scoped evidence;
    # this first rule-engine version deliberately does not consume arbitrary nets.
    verified_edges = connectivity.get("verified_edges", []) if isinstance(connectivity, dict) else []
    measurement_assessments = scoped_measurements
    measurement_evidence = [{
        "measurement_id": row.get("measurement_id"),
        "component_id": row.get("component_id"),
        "board_component_id": row.get("board_component_id"),
        "test_id": row.get("test_id"),
        "observed_at": (row.get("measured_at") or row.get("created_at") or row.get("timestamp")),
        "measurement_type": row.get("measurement_type"),
        "value": row.get("value"), "unit": row.get("unit"),
        "source": row.get("source"),
        "verification_status": row.get("verification_status"),
        "reference_min": row.get("reference_min"), "reference_max": row.get("reference_max"),
        "reference_source": row.get("reference_source"),
        "reference_source_id": row.get("reference_source_id"),
        "reference_verification_status": row.get("reference_verification_status"),
        "measurement_conditions": row.get("measurement_conditions"),
        "fault_hypothesis": row.get("fault_hypothesis"),
        "assessment": (measurement_assessments.get((str(row.get("component_id") or ""),
                                                    str(row.get("fault_hypothesis") or ""),
                                                    str(row.get("test_id") or ""),
                                                    " ".join(str(row.get("measurement_target") or "").casefold().split()),
                                                    " ".join(str(row.get("operating_conditions") or "").casefold().split())), {})
                       .get("status", "INCONCLUSIVE")),
        "verdict": (measurement_assessments.get((str(row.get("component_id") or ""),
                                                 str(row.get("fault_hypothesis") or ""),
                                                 str(row.get("test_id") or ""),
                                                 " ".join(str(row.get("measurement_target") or "").casefold().split()),
                                                 " ".join(str(row.get("operating_conditions") or "").casefold().split())), {})
                    .get("verdict", "non_diagnostic")),
        "assessment_status": (measurement_assessments.get((str(row.get("component_id") or ""),
                                                            str(row.get("fault_hypothesis") or ""),
                                                            str(row.get("test_id") or ""),
                                                            " ".join(str(row.get("measurement_target") or "").casefold().split()),
                                                            " ".join(str(row.get("operating_conditions") or "").casefold().split())), {})
                              .get("assessment_status", "INCONCLUSIVE")),
        "decision_scope": "FAULT_HYPOTHESIS_ONLY",
        "used_for_ranking": any(candidate.get("component_id") == row.get("component_id")
                                 and candidate.get("fault_hypothesis") == row.get("fault_hypothesis")
                                 and candidate.get("measurement_verdict") in {"supports", "refutes"}
                                 for candidate in [*component_candidates, *verified_component_history]),
        "reason": measurement_assessments.get((str(row.get("component_id") or ""),
                                                 str(row.get("fault_hypothesis") or ""),
                                                 str(row.get("test_id") or ""),
                                                 " ".join(str(row.get("measurement_target") or "").casefold().split()),
                                                 " ".join(str(row.get("operating_conditions") or "").casefold().split())), {}).get(
                                                     "reason", "검사 프로파일·기준 출처·조건이 검증되지 않아 판정에 사용하지 않았습니다."),
    } for row in measurements or [] if isinstance(row, dict)
        and str(row.get("component_id") or "") in confirmed_ids]
    evidence_ledger = build_evidence_ledger(
        observations=ledger_observations,
        conflicts=evidence_conflicts,
        components=[dict(part, component_type_status=(part.get("component_type_status") or
                         ("USER_CONFIRMED" if str(part.get("component_id") or "") in confirmed_ids else "UNCONFIRMED")),
                         presence_status="CONFIRMED" if str(part.get("component_id") or "") in confirmed_ids else "UNCONFIRMED")
                    for part in detected_parts],
        functional_blocks=functional_block_groups, functional_roles=functional_blocks,
        measurements=measurements or [],
        source_version=str((current_context or {}).get("component_map_version") or "") or None)
    functional_hypotheses = []
    for hypothesis in ranked:
        mode = modes.get(str(hypothesis.get("fault_id") or ""), {})
        if mode.get("level") != "functional_block":
            continue
        required_roles = set(mode.get("functional_roles", []))
        for block in functional_block_groups:
            if required_roles and block.get("role") not in required_roles:
                continue
            functional_hypotheses.append({**hypothesis,
                "hypothesis_id": f"{hypothesis['fault_id']}:{block['block_id']}",
                "level": "functional_block", "functional_block_id": block["block_id"],
                "functional_block": block["role"], "related_component_ids": list(block["component_ids"]),
                "supporting_evidence": list(dict.fromkeys([
                    *hypothesis.get("supporting_evidence", []), *block.get("supporting_evidence", [])])),
                "status": "UNVERIFIED_CANDIDATE", "score_is_probability": False})
    hierarchical_hypotheses = {
        "system": sorted([row for row in ranked if row.get("level") == "system"],
                          key=lambda row: (-row.get("score", 0), row.get("fault_id", ""))),
        "functional_blocks": sorted(functional_hypotheses,
                                    key=lambda row: (-row.get("score", 0), row.get("functional_block_id", ""), row.get("fault_id", ""))),
        "components": sorted(all_component_candidates,
                              key=lambda row: (-row.get("score", 0), row.get("component_id", ""), row.get("fault_hypothesis", ""))),
        "board": sorted([*([row for row in ranked if row.get("level") == "board"]),
                          *[{**row, "level": "board", "source": "direct_board_evidence"}
                            for row in joint_candidates]],
                        key=lambda row: (-row.get("score", 0), row.get("fault_id", ""))),
    }
    evidence_gaps = analyze_evidence_gaps(
        hypotheses=ranked, knowledge=kb, confirmed_types=detected_types,
        functional_blocks=functional_block_groups, component_roles=confirmed_functional_blocks,
        component_candidates=all_component_candidates,
        measurement_evidence=measurement_evidence, evidence_ledger=evidence_ledger,
        connectivity_status=connectivity_status,
    )
    active_inspection_plan = plan_active_inspections(
        hypotheses=ranked, component_candidates=all_component_candidates,
        functional_block_groups=functional_block_groups, evidence_gaps=evidence_gaps,
        measurements=[
            {**row,
             "assessment_status": next((item.get("assessment_status", item.get("verdict"))
                                        for item in measurement_evidence
                                        if item.get("measurement_id") == row.get("measurement_id")), None)}
            for row in (measurements or []) if isinstance(row, dict)
        ], observations=facts,
        inspection_history=inspection_history or [], knowledge=kb,
        allow_retest=allow_retest,
    )
    return {"status": status, "symptom_codes": selected, "observations": facts,
            "component_inventory": {"complete": bool(inventory_complete),
                                    "confirmed_component_ids": sorted(confirmed_ids),
                                    "confirmed_component_types": sorted(detected_types),
                                    "unconfirmed_detection_ids": unconfirmed_ids,
                                    "confirmed_count": len(confirmed_ids)},
            "observation_provenance": [item.to_dict() for item in provenance.values()],
            "evidence_conflicts": evidence_conflicts,
            "matched_rules": [{"id": rule["id"], "name": rule["name"]}
                              for rule in sorted(matched_rules, key=lambda item: -item.get("priority", 0))],
            "hypotheses": ranked, "functional_blocks": functional_blocks,
            "functional_block_groups": functional_block_groups,
            "hierarchical_hypotheses": hierarchical_hypotheses,
            "component_candidate_groups": list(candidate_groups_by_id.values()),
            "evidence_ledger": evidence_ledger, "evidence_gaps": evidence_gaps,
            "active_inspection_plan": active_inspection_plan,
            "hypothesis_groups": hypothesis_groups,
            "component_candidates": component_candidates,
            "all_component_candidates": all_component_candidates,
            "verified_component_history": verified_component_history,
            "joint_candidates": joint_candidates,
            "measurement_evidence": measurement_evidence,
            "next_tests": recommended[:5], "connectivity_status": connectivity_status,
            "processing_timings_ms": {
                "measurement_validation": round(measurement_validation_ms, 3),
                "candidate_ranking": round(candidate_ranking_ms, 3),
                "recommendation_generation": round(max(0.0, recommendation_ms), 3),
                "total": round((time.perf_counter() - diagnosis_started) * 1000, 3),
            },
            "verified_connectivity_evidence_count": len(verified_edges) if connectivity_status in {"verified", "partial_verified"} else 0,
            "message": ("후보 원인을 정리했습니다. 순위 점수는 확률이 아니며 검증 전 후보입니다."
                        if status == "CANDIDATES_AVAILABLE" else "정보가 부족해 진단을 보류합니다. 추가 증상이나 확인 결과를 입력해 주세요.")}
