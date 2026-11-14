"""Rank component/fault pairs from reviewed instances and trusted evidence."""

from __future__ import annotations

import json
import hashlib
from typing import Any

from circuitcare.symptom_diagnosis.fault_hypothesis_validator import (
    summarize_measurement_verdict, validate_component_hypothesis,
)
from circuitcare.symptom_diagnosis.measurement_evaluator import evaluate_component_measurements


def _normalize_component_observations(rows: list[dict[str, Any]] | None) -> tuple[list[dict[str, Any]], int]:
    """Drop untrusted statuses, deduplicate repeats, and suppress contradictory scoped facts."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows or []:
        if not isinstance(row, dict) or not row.get("component_id") or not row.get("code"):
            continue
        status = str(row.get("status", "observed")).casefold()
        if status not in {"observed", "measured"}:
            continue
        conditions = json.dumps(row.get("measurement_conditions") or {}, sort_keys=True,
                                ensure_ascii=False, default=str)
        key = (str(row["component_id"]), str(row["code"]), conditions)
        groups.setdefault(key, []).append(dict(row, status=status))
    normalized: list[dict[str, Any]] = []
    conflicts = 0
    for scoped_rows in groups.values():
        values = {json.dumps(row.get("value"), sort_keys=True, ensure_ascii=False, default=str)
                  for row in scoped_rows}
        if len(values) > 1:
            conflicts += 1
            continue
        normalized.append(max(scoped_rows, key=lambda row: row["status"] == "measured"))
    return normalized, conflicts


def rank_component_candidates(
    *, hypotheses: list[dict[str, Any]], components: list[dict[str, Any]],
    functional_blocks: list[dict[str, Any]], observations: list[dict[str, Any]] | None,
    measurements: list[dict[str, Any]] | None, knowledge: dict[str, Any],
    confirmed_component_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    profiles: dict[str, Any] | None = None, current_context: dict[str, Any] | None = None,
    limit: int = 3, include_refuted: bool = False, distinct_components: bool = True,
) -> list[dict[str, Any]]:
    """Return at most three distinct, present component IDs without forced slots."""
    confirmed_ids = {str(value) for value in (confirmed_component_ids or [])}
    confirmed_types = {
        str(part.get("component_type", part.get("label", "unknown"))).casefold()
        for part in components if str(part.get("component_id") or "") in confirmed_ids
        and part.get("component_type_status") in (None, "USER_CONFIRMED")
    }
    block_by_id = {str(row.get("component_id")): row for row in functional_blocks}
    modes = knowledge.get("fault_modes", {}).get("fault_modes", {})
    tests = knowledge.get("test_catalog", {}).get("tests", {})
    assessments = evaluate_component_measurements(
        measurements, profiles=profiles, current_context=current_context,
    )
    observed, observation_conflicts = _normalize_component_observations(observations)
    results: list[dict[str, Any]] = []

    for hypothesis in hypotheses:
        fault_id = str(hypothesis.get("fault_id", ""))
        mode = modes.get(fault_id, {})
        if mode.get("level") not in {"component", "functional_block"}:
            continue
        eligible: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for part in components:
            component_id = str(part.get("component_id") or "")
            verdict = summarize_measurement_verdict(
                mode=mode, component_id=component_id, fault_id=fault_id, assessments=assessments,
            )
            validation = validate_component_hypothesis(
                mode=mode, component=part, confirmed_component_ids=confirmed_ids,
                confirmed_component_types=confirmed_types,
                component_block=block_by_id.get(component_id),
                component_observations=observed, measurement_verdict=verdict,
            )
            if validation["eligible"]:
                eligible.append((part, validation))
        if not eligible:
            continue

        targeted_ids: set[str] = set()
        for row in observed:
            component_id = str(row.get("component_id") or "")
            code, value = str(row.get("code") or ""), row.get("value")
            supports = code in (mode.get("supporting_evidence", {}) if value is True
                                else mode.get("supporting_evidence_when_false", {}))
            opposes = code in (mode.get("opposing_evidence", {}) if value is True
                               else mode.get("opposing_evidence_when_false", {}))
            if component_id and (supports or opposes):
                targeted_ids.add(component_id)
        targeted_ids.update(
            component_id for (component_id, measured_fault, *_), assessment in assessments.items()
            if measured_fault == fault_id and assessment.get("verdict") in {"supports", "refutes"}
        )
        # Keep indistinguishable instances as an explicit unresolved group. Do
        # not silently drop every candidate or imply that the first ID is best.
        ambiguous_ids = sorted(str(part.get("component_id") or "") for part, _ in eligible)
        candidate_group_id = None
        if len(eligible) > 1 and not targeted_ids:
            candidate_group_id = "candidate-group-" + hashlib.sha256(
                f"{fault_id}|{'|'.join(ambiguous_ids)}".encode()).hexdigest()[:12]
        if targeted_ids:
            eligible = [(part, validation) for part, validation in eligible
                        if str(part.get("component_id")) in targeted_ids]

        for part, validation in eligible:
            component_id = str(part.get("component_id") or "")
            part_type = str(part.get("component_type", part.get("label", "unknown"))).casefold()
            if not component_id:
                continue
            block = block_by_id.get(component_id, {})
            candidate_roles = set(block.get("candidate_blocks", []))
            required_roles = set(mode.get("functional_roles", []))
            role_match = bool(required_roles & set(part.get("confirmed_functional_roles", [])))
            specific = [row for row in observed if str(row.get("component_id") or "") == component_id]
            supporting: list[str] = []
            opposing: list[str] = []
            observation_score = 0.0
            for row in specific:
                code, value = str(row.get("code") or ""), row.get("value")
                trust = 1.0 if row["status"] == "measured" else 0.5
                if value is True and code in mode.get("supporting_evidence", {}):
                    supporting.append(code)
                    observation_score += float(mode["supporting_evidence"][code]) * trust
                elif value is False and code in mode.get("supporting_evidence_when_false", {}):
                    supporting.append(f"{code}=false")
                    observation_score += float(mode["supporting_evidence_when_false"][code]) * trust
                elif value is True and code in mode.get("opposing_evidence", {}):
                    opposing.append(code)
                    observation_score -= float(mode["opposing_evidence"][code]) * trust
                elif value is False and code in mode.get("opposing_evidence_when_false", {}):
                    opposing.append(f"{code}=false")
                    observation_score -= float(mode["opposing_evidence_when_false"][code]) * trust

            verdict = summarize_measurement_verdict(
                mode=mode, component_id=component_id, fault_id=fault_id, assessments=assessments,
            )
            if required_roles and validation.get("role_status") != "CONFIRMED":
                verdict = {**verdict, "verdict": "non_diagnostic",
                           "reason": "기능 역할이 추정 상태라 측정 결과를 이 기능 고장의 확정/반박에 사용할 수 없습니다."}
            symptom_score = float(hypothesis.get("score", 0.0))
            role_score = 0.75 if role_match else 0.25 if validation.get("role_status") == "INFERRED" else 0.0
            measurement_score = {"supports": 2.0, "refutes": -2.0}.get(verdict["verdict"], 0.0)
            raw_score = symptom_score + role_score + observation_score + measurement_score
            # A verified refuting measurement is still a useful, displayable result:
            # retain the component as a green, hypothesis-specific history candidate.
            if verdict["verdict"] == "refutes" and not include_refuted:
                continue
            score = max(0.1 if verdict["verdict"] == "refutes" else 0.0, raw_score)
            if score <= 0:
                continue
            visual_status = {
                "supports": "FAULT_SUPPORTED", "refutes": "FAULT_REFUTED",
                "conflicted": "CONFLICTED", "non_diagnostic": "SUSPECTED",
            }[verdict["verdict"]]
            role_evidence = block.get("role_evidence", {})
            evidence = list(dict.fromkeys([
                *hypothesis.get("supporting_evidence", []),
                *(f"부품별 지지 근거: {code}" for code in sorted(supporting)),
                *(f"부품별 반대 근거: {code}" for code in sorted(opposing)),
                *(text for role in sorted(candidate_roles & required_roles
                                          if validation.get("role_status") == "INFERRED" else set())
                  for text in role_evidence.get(role, [])),
                *(["기능 역할 사용자 확인"] if role_match else []),
                *([verdict["reason"]] if verdict["verdict"] != "non_diagnostic" else []),
            ]))
            results.append({
                "component_id": component_id, "component_type": part_type,
                "candidate_group_id": candidate_group_id,
                "candidate_group_status": "UNRESOLVED_AMBIGUITY" if candidate_group_id else None,
                "candidate_group_component_ids": ambiguous_ids if candidate_group_id else [],
                "candidate_rank": 0, "rank": 0, "fault_hypothesis": fault_id,
                "fault_id": fault_id, "fault_title": hypothesis.get("title", fault_id),
                "score": round(score, 2), "score_breakdown": {
                    "symptom_and_mode": round(symptom_score, 3),
                    "confirmed_role": round(role_score, 3),
                    "component_observations": round(observation_score, 3),
                    "validated_measurement": round(measurement_score, 3),
                },
                "score_is_probability": False,
                "status": visual_status, "visual_status": visual_status,
                "measurement_status": verdict["verdict"].upper(),
                "assessment_status": verdict.get("assessment_status", "INCONCLUSIVE"),
                "decision_scope": "FAULT_HYPOTHESIS_ONLY",
                "measurement_verdict": verdict["verdict"],
                "measurement_reason": verdict["reason"],
                "required_measurement_tests": verdict["required_tests"],
                "completed_measurement_tests": verdict["completed_tests"],
                "missing_measurement_tests": verdict["missing_tests"],
                "alternative_causes": verdict.get("alternative_causes", []),
                "exclusion_tests": list(mode.get("exclusion_tests", [])),
                "role_status": validation.get("role_status", "UNKNOWN"),
                "observation_conflicts_omitted": observation_conflicts,
                "solder_status": "NOT_ASSESSED", "evidence": evidence,
                "supporting_evidence": list(hypothesis.get("supporting_evidence", [])) + supporting,
                "opposing_evidence": list(hypothesis.get("opposing_evidence", [])) + opposing,
                "requires_measurement": verdict["verdict"] in {"non_diagnostic", "conflicted"},
                "recommended_tests": [tests[test_id].get("question", tests[test_id].get("name", test_id))
                                      for test_id in mode.get("recommended_tests", []) if test_id in tests],
                "evidence_ids": list(dict.fromkeys(str(row.get("id") or row.get("observation_id"))
                                                    for row in specific if row.get("id") or row.get("observation_id"))),
                "designator_prefix": component_id.split("-", 1)[0].upper(),
            })

    results.sort(key=lambda row: (-row["score"], row["component_id"], row["fault_hypothesis"]))
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str] | str] = set()
    for row in results:
        unique_key = row["component_id"] if distinct_components else (
            row["component_id"], row["fault_hypothesis"]
        )
        if unique_key in seen:
            continue
        seen.add(unique_key)
        row["candidate_rank"] = len(selected) + 1
        row["rank"] = len(selected) + 1
        selected.append(row)
        if len(selected) >= max(0, limit):
            break
    return selected


def rank_joint_candidates(
    *, hypotheses: list[dict[str, Any]], component_candidates: list[dict[str, Any]],
    observations: dict[str, Any],
) -> list[dict[str, Any]]:
    """Promote solder/board causes only with direct evidence or completed refutation."""
    direct_codes = {"VISIBLE_SOLDER_DAMAGE", "CONNECTOR_MOVEMENT_AFFECTS"}
    direct = [code for code in direct_codes if observations.get(code) is True]
    all_component_faults_refuted = bool(component_candidates) and all(
        row.get("measurement_verdict") == "refutes"
        and row.get("required_measurement_tests")
        and not row.get("missing_measurement_tests")
        for row in component_candidates
    )
    candidates = []
    for row in hypotheses:
        if row.get("level") != "board":
            continue
        fault_id = str(row.get("fault_id", ""))
        if fault_id not in {"SOLDER_JOINT_FAILURE", "PCB_TRACE_OPEN", "UNKNOWN_CONNECTION_FAILURE"}:
            continue
        if not direct and not all_component_faults_refuted:
            continue
        candidates.append({
            "fault_id": fault_id, "title": row.get("title", fault_id),
            "score": row.get("score", 0), "status": "UNVERIFIED_CANDIDATE",
            "evidence": [*(f"직접 관측: {code}" for code in direct),
                         *( ["요구된 부품 검사가 모두 해당 내부 고장 가설을 반박"]
                            if all_component_faults_refuted else [])],
            "related_component_ids": [item["component_id"] for item in component_candidates],
        })
    return candidates
