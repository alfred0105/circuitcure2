"""Deterministic, safety-gated inspection planning from diagnosis evidence.

The planner ranks catalogued checks by how many current hypothesis pairs their
declared outcomes can distinguish. It does not assign probabilities or invent
measurement conditions, costs, durations, or normal values.
"""

from __future__ import annotations

from typing import Any


def plan_active_inspections(
    *, hypotheses: list[dict[str, Any]], component_candidates: list[dict[str, Any]],
    functional_block_groups: list[dict[str, Any]], evidence_gaps: list[dict[str, Any]],
    measurements: list[dict[str, Any]], observations: dict[str, Any],
    inspection_history: list[dict[str, Any]], knowledge: dict[str, Any],
    inspection_conditions: dict[str, dict[str, Any]] | None = None,
    allow_retest: bool = False, limit: int = 5,
) -> dict[str, Any]:
    """Return deterministic inspection candidates and their safety prerequisites."""
    modes = knowledge.get("fault_modes", {}).get("fault_modes", {})
    tests = knowledge.get("test_catalog", {}).get("tests", {})
    profiles = knowledge.get("component_test_profiles", {}).get("profiles", {})
    inspection_conditions = inspection_conditions or {}
    active_hypotheses = [row for row in hypotheses if isinstance(row, dict) and row.get("fault_id")]
    fault_ids = list(dict.fromkeys(str(row["fault_id"]) for row in active_hypotheses))
    if not active_hypotheses:
        return {"status": "NO_HYPOTHESES", "strategy": "deterministic_pair_separation_v1",
                "candidates": [], "blocked": [], "score_is_probability": False}

    component_types_by_fault: dict[str, set[str]] = {}
    components_by_fault: dict[str, set[str]] = {}
    functional_blocks_by_fault: dict[str, list[dict[str, Any]]] = {}
    for hypothesis in active_hypotheses:
        fault_id = str(hypothesis.get("fault_id") or "")
        required_roles = set(modes.get(fault_id, {}).get("functional_roles", []))
        if not required_roles:
            continue
        for block in functional_block_groups:
            if not isinstance(block, dict) or block.get("role") not in required_roles:
                continue
            functional_blocks_by_fault.setdefault(fault_id, []).append(block)
            for member in block.get("members", []):
                if not isinstance(member, dict) or member.get("evidence_status") != "VERIFIED":
                    continue
                component_id = str(member.get("component_id") or "")
                component_type = str(member.get("component_type") or "unknown").casefold()
                if component_id:
                    components_by_fault.setdefault(fault_id, set()).add(component_id)
                if component_type != "unknown":
                    component_types_by_fault.setdefault(fault_id, set()).add(component_type)
    for row in component_candidates:
        if not isinstance(row, dict):
            continue
        fault_id = str(row.get("fault_hypothesis") or row.get("fault_id") or "")
        component_id = str(row.get("board_component_id") or row.get("component_id") or "")
        component_type = str(row.get("component_type") or "unknown").casefold()
        if fault_id:
            component_types_by_fault.setdefault(fault_id, set()).add(component_type)
            if component_id:
                components_by_fault.setdefault(fault_id, set()).add(component_id)

    completed: set[tuple[str, str, str]] = set()
    conflicts: set[tuple[str, str, str]] = set()
    from circuitcare.symptom_diagnosis.measurement_validation import measurement_verification_gaps
    profile_map = knowledge.get("component_test_profiles", {})
    for row in measurements:
        if not isinstance(row, dict) or not row.get("test_id"):
            continue
        if measurement_verification_gaps(row, profiles=profile_map):
            continue
        key = (str(row.get("board_component_id") or row.get("component_id") or ""),
               str(row.get("fault_hypothesis") or ""), str(row.get("test_id")))
        if key[0]:
            completed.add(key)
            if str(row.get("assessment_status") or row.get("verdict") or "").upper() == "CONFLICTED":
                conflicts.add(key)
    for row in inspection_history:
        if not isinstance(row, dict) or row.get("verification_status") in {"STALE", "PENDING"}:
            continue
        test_id = str(row.get("test_id") or "")
        if not test_id or row.get("result") == "확인 불가":
            continue
        target_id = str(row.get("board_component_id") or row.get("target_id") or "")
        related = row.get("related_fault_ids") if isinstance(row.get("related_fault_ids"), list) else []
        for fault_id in related or [row.get("fault_id")]:
            if fault_id:
                key = (target_id, str(fault_id), test_id)
                completed.add(key)
                if row.get("verification_status") == "CONFLICTED":
                    conflicts.add(key)

    answer_signatures = {
        str(test_id): {
            str(answer): outcome.get("observations", {}) if isinstance(outcome, dict) else {}
            for answer, outcome in (test.get("results", {}) or {}).items()
        }
        for test_id, test in tests.items() if isinstance(test, dict)
    }

    def outcome_relation(fault_id: str, outcome: dict[str, Any]) -> str:
        mode = modes.get(fault_id, {})
        supports = set(mode.get("supporting_evidence", {})) | set(mode.get("supporting_evidence_when_false", {}))
        opposes = set(mode.get("opposing_evidence", {})) | set(mode.get("opposing_evidence_when_false", {}))
        support = oppose = False
        for code, value in outcome.items():
            if (value is True and code in mode.get("supporting_evidence", {})) or (
                value is False and code in mode.get("supporting_evidence_when_false", {})):
                support = True
            if (value is True and code in mode.get("opposing_evidence", {})) or (
                value is False and code in mode.get("opposing_evidence_when_false", {})):
                oppose = True
        if support and oppose:
            return "CONFLICTED"
        if support:
            return "SUPPORTS"
        if oppose:
            return "REFUTES"
        return "NO_DIAGNOSTIC_RELATION"

    test_faults: dict[str, set[str]] = {}
    for fault_id in fault_ids:
        mode = modes.get(fault_id, {})
        for test_id in mode.get("recommended_tests", []):
            test_faults.setdefault(str(test_id), set()).add(fault_id)
        for component_type in component_types_by_fault.get(fault_id, set()):
            for test_id, profile in (profiles.get(component_type, {}).get("tests", {}) or {}).items():
                if fault_id in (profile.get("fault_relations", {}) if isinstance(profile, dict) else {}):
                    test_faults.setdefault(str(test_id), set()).add(fault_id)

    gaps_by_fault = {str(row.get("fault_id")): row for row in evidence_gaps if isinstance(row, dict)}
    candidates, blocked = [], []
    for test_id, applies_to in test_faults.items():
        test = tests.get(test_id, {})
        profile_defs = []
        for fault_id in applies_to:
            for component_type in component_types_by_fault.get(fault_id, set()):
                candidate_profile = (profiles.get(component_type, {}).get("tests", {}) or {}).get(test_id)
                if isinstance(candidate_profile, dict) and fault_id in candidate_profile.get("fault_relations", {}):
                    profile_defs.append((fault_id, component_type, candidate_profile))
        scope = str(test.get("target_scope") or ("component" if profile_defs else "board"))
        target_type_set = {value for fault_id in applies_to for value in component_types_by_fault.get(fault_id, set())}
        if test.get("type") == "visual_inspection" and test.get("target_scope") != "board":
            declared = {str(value).casefold() for value in test.get("component_types", [])}
            target_type_set &= declared
            applies_to = {fault_id for fault_id in applies_to
                          if component_types_by_fault.get(fault_id, set()) & declared}
        if not applies_to:
            continue
        scoped_ids = sorted({component_id for fault_id in applies_to
                             for component_id in components_by_fault.get(fault_id, set())})

        separated_pairs = []
        declared_outcomes = answer_signatures.get(test_id, {})
        test_signatures: dict[str, tuple[Any, ...]] = {}
        for fault_id in applies_to:
            if declared_outcomes:
                test_signatures[fault_id] = tuple(
                    outcome_relation(fault_id, outcome) for _answer, outcome in declared_outcomes.items()
                )
                continue
            relations_by_status: dict[str, set[str]] = {"TEST_NORMAL": set(), "TEST_ABNORMAL": set()}
            for profile_fault, component_type, profile in profile_defs:
                if profile_fault != fault_id:
                    continue
                relation_map = profile.get("fault_relations", {}).get(fault_id, {})
                for result_status in relations_by_status:
                    relation = relation_map.get(result_status)
                    if relation:
                        relations_by_status[result_status].add(str(relation).casefold())
            if all(relations_by_status.values()):
                test_signatures[fault_id] = tuple(
                    tuple(sorted(relations_by_status[result_status]))
                    for result_status in ("TEST_NORMAL", "TEST_ABNORMAL")
                )
        for left_index, left_fault in enumerate(fault_ids):
            for right_fault in fault_ids[left_index + 1:]:
                if left_fault not in applies_to or right_fault not in applies_to:
                    continue
                left_signature = test_signatures.get(left_fault)
                right_signature = test_signatures.get(right_fault)
                if left_signature is not None and right_signature is not None and left_signature != right_signature:
                    separated_pairs.append([left_fault, right_fault])

        risk = str(test.get("risk_level") or "unknown").casefold()
        prerequisites = []
        if test.get("requires_power_off"):
            prerequisites.append("전원 분리 확인")
        if test.get("requires_stored_energy_safe"):
            prerequisites.append("잔류 에너지 안전 확인")
        if test.get("type") == "visual_inspection":
            if risk not in {"none", "low"}:
                prerequisites.append("검사 위험 수준 검토")
            safety_blocked = risk not in {"none", "low"}
            power_ok = True
            energy_ok = True
            safety_targets = ["board"] if test.get("target_scope") == "board" else (scoped_ids or ["board"])
            for target_id in safety_targets:
                for fault_id in applies_to:
                    actual = inspection_conditions.get(f"{target_id}|{fault_id}|{test_id}", {})
                    if test.get("requires_power_off") and actual.get("power_state") != "POWER_OFF_CONFIRMED":
                        power_ok = False
                    if test.get("requires_stored_energy_safe") and actual.get("stored_energy_status") != "SAFE_CONFIRMED":
                        energy_ok = False
            safety_blocked = safety_blocked or not power_ok or not energy_ok
            if test.get("requires_power_off"):
                prerequisites.append("전원 분리 확인됨" if power_ok else "전원 분리 확인")
            if test.get("requires_stored_energy_safe"):
                prerequisites.append("잔류 에너지 안전 확인됨" if energy_ok else "잔류 에너지 안전 확인")
        elif profile_defs:
            profile = profile_defs[0][2]
            protocol = profile.get("safety_protocol")
            required = profile.get("required_conditions", {})
            if not protocol or not required or str(profile.get("risk_level", "unknown")).casefold() not in {"low", "none"}:
                prerequisites.extend(["검사 프로파일의 완전한 안전 절차 확인", "실제 소자·측정 단자 확인",
                                      "정상 기준 및 출처 확인"])
                safety_blocked = True
            else:
                # Safety is per component and per fault hypothesis. A free-text
                # note or a generic power-off checkbox cannot satisfy this gate.
                safety_blocked = False
                missing_by_target = []
                missing_required_conditions: set[str] = set()
                missing_location = False
                missing_reference = False
                for target_id in scoped_ids:
                    target_ready = False
                    for fault_id in applies_to:
                        condition_key = f"{target_id}|{fault_id}"
                        actual = inspection_conditions.get(condition_key, {})
                        if (all(actual.get(field) == expected for field, expected in required.items())
                                and actual.get("measurement_location_verified") is True
                                and actual.get("reference_source_verified") is True):
                            target_ready = True
                            break
                    if not target_ready:
                        safety_blocked = True
                        missing_by_target.append(target_id)
                        target_conditions = [inspection_conditions.get(f"{target_id}|{fault_id}", {})
                                             for fault_id in applies_to]
                        missing_required_conditions.update(
                            field for field, expected in required.items()
                            if not any(row.get(field) == expected for row in target_conditions)
                        )
                        missing_location = missing_location or not any(
                            row.get("measurement_location_verified") is True for row in target_conditions)
                        missing_reference = missing_reference or not any(
                            row.get("reference_source_verified") is True for row in target_conditions)
                required_labels = {
                    "power_state": "전원 분리 상태 확인",
                    "stored_energy_status": "잔류 에너지 안전 확인",
                    "component_isolation_status": "측정 대상 분리 확인",
                }
                prerequisites.extend(required_labels[field] for field in required
                                     if field in missing_required_conditions and field in required_labels)
                if missing_location:
                    prerequisites.append("측정 위치·단자를 먼저 확인해야 함")
                if missing_reference:
                    prerequisites.append("정상 기준·출처를 먼저 확인해야 함")
                if missing_by_target:
                    prerequisites.append("해당 소자별 안전 상태가 아직 확인되지 않음")
        else:
            safety_blocked = risk not in {"none", "low"}

        completed_keys = {(target, fault, test_id) for target in (scoped_ids or [""])
                          for fault in applies_to}
        already_done = bool(completed_keys) and completed_keys.issubset(completed)
        has_conflict = bool(completed_keys & conflicts)
        if already_done and not allow_retest and not has_conflict:
            continue

        expected_outcomes = {}
        for answer, outcome in answer_signatures.get(test_id, {}).items():
            expected_outcomes[answer] = {
                fault_id: outcome_relation(fault_id, outcome)
                for fault_id in sorted(applies_to)
            }
        if not expected_outcomes:
            for result_status in ("TEST_NORMAL", "TEST_ABNORMAL"):
                relations = {}
                for profile_fault, _component_type, profile in profile_defs:
                    if profile_fault not in applies_to:
                        continue
                    relation = profile.get("fault_relations", {}).get(profile_fault, {}).get(result_status)
                    if relation:
                        relations.setdefault(profile_fault, set()).add(str(relation).casefold())
                if relations:
                    expected_outcomes[result_status] = {
                        fault_id: sorted(values) for fault_id, values in sorted(relations.items())
                    }
        missing_info = sorted({
            str(item.get("description"))
            for fault_id in applies_to
            for item in gaps_by_fault.get(fault_id, {}).get("missing_evidence", [])
            if isinstance(item, dict) and item.get("description")
        })
        why = (f"현재 가설 {len(applies_to)}개에 적용 가능하며, 결과에 따라 {len(separated_pairs)}개 가설 쌍을 구분할 수 있습니다."
               if separated_pairs else
               f"현재 가설 {len(applies_to)}개와 관련된 확인 항목입니다. 이 검사만으로 후보를 구분할 근거는 아직 없습니다.")
        item = {
            "test_id": test_id, "name": test.get("name", test_id), "type": test.get("type", "measurement"),
            "target_scope": scope, "component_types": sorted(target_type_set),
            "target_component_ids": scoped_ids, "fault_ids": sorted(applies_to),
            "target_functional_block_ids": sorted({
                str(block.get("block_id"))
                for fault_id in applies_to
                for block in functional_blocks_by_fault.get(fault_id, [])
                if block.get("block_id")
            }),
            "separated_hypothesis_pairs": separated_pairs,
            "separation_count": len(separated_pairs), "expected_outcomes": expected_outcomes,
            "risk_level": risk, "prerequisites": list(dict.fromkeys(prerequisites)),
            "blocked_until_safe": safety_blocked,
            "status": "NEEDS_SAFETY_OR_TEST_DETAILS" if safety_blocked else "READY_FOR_USER_REVIEW",
            "missing_evidence": missing_info[:8], "reason": why,
            "estimated_duration_seconds": test.get("estimated_duration_seconds"),
            "estimated_cost": test.get("estimated_cost"),
            "retest_reason": ("측정 결과 충돌" if has_conflict else
                              "사용자가 재검사를 요청함" if already_done and allow_retest else None),
            "score_is_probability": False,
        }
        if safety_blocked:
            blocked.append(item)
        else:
            candidates.append(item)
    candidates.sort(key=lambda row: (
        -row["separation_count"], row["risk_level"] != "none",
        row["estimated_duration_seconds"] if isinstance(row["estimated_duration_seconds"], (int, float)) else float("inf"),
        row["estimated_cost"] if isinstance(row["estimated_cost"], (int, float)) else float("inf"),
        row["test_id"],
    ))
    status = "READY" if candidates else "SAFETY_OR_INFORMATION_REQUIRED" if blocked else "NO_APPLICABLE_CHECKS"
    return {"status": status, "strategy": "deterministic_pair_separation_v1",
            "candidates": candidates[:max(0, limit)], "blocked": blocked[:max(0, limit)],
            "hypothesis_count": len(active_hypotheses), "score_is_probability": False,
            "selection_policy": "pair separation, then risk, stated duration/cost; unspecified values are not imputed"}
