"""Explain what is known, inferred, contradicted, and still needed per hypothesis."""

from __future__ import annotations

from typing import Any


def analyze_evidence_gaps(
    *, hypotheses: list[dict[str, Any]], knowledge: dict[str, Any],
    confirmed_types: set[str] | list[str], functional_blocks: list[dict[str, Any]],
    component_candidates: list[dict[str, Any]], measurement_evidence: list[dict[str, Any]],
    evidence_ledger: list[dict[str, Any]], connectivity_status: str = "none",
    component_roles: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Create system, block, and component evidence gaps without merging their ranks."""
    modes = knowledge.get("fault_modes", {}).get("fault_modes", {})
    tests = knowledge.get("test_catalog", {}).get("tests", {})
    confirmed = {str(value) for value in confirmed_types}
    output = []
    for hypothesis in hypotheses:
        fault_id = str(hypothesis.get("fault_id") or "")
        mode = modes.get(fault_id, {})
        related_ids = [str(value) for value in hypothesis.get("related_component_ids", []) if value]
        related_candidates = [row for row in component_candidates
                              if str(row.get("fault_hypothesis") or row.get("fault_id") or "") == fault_id]
        related_ids = list(dict.fromkeys([*related_ids, *[
            str(row.get("board_component_id") or row.get("component_id") or "")
            for row in related_candidates if row.get("board_component_id") or row.get("component_id")
        ]]))
        relevant_ledger = [item for item in evidence_ledger
                           if not item.get("component_id") or item.get("component_id") in related_ids]
        missing = []
        required_types = {str(value) for value in mode.get("required_component_types", [])}
        missing_types = sorted(required_types - confirmed)
        if missing_types:
            missing.append({"kind": "component_identity", "status": "UNKNOWN",
                            "description": "확인된 소자 종류가 부족합니다: " + ", ".join(missing_types),
                            "verification_methods": ["PCB 소자 검수"]})
        required_roles = {str(value) for value in mode.get("functional_roles", [])}
        available_roles = {str(row.get("role")) for row in functional_blocks if isinstance(row, dict)}
        available_roles.update(str(role) for row in (component_roles or []) if isinstance(row, dict)
                               for role in row.get("candidate_blocks", []))
        verified_roles = {str(row.get("role")) for row in functional_blocks
                          if isinstance(row, dict) and row.get("verification_status") == "VERIFIED"}
        verified_roles.update(str(role) for row in (component_roles or []) if isinstance(row, dict)
                              for role in row.get("confirmed_functional_roles", []))
        if required_roles and not (required_roles & verified_roles):
            inferred_match = required_roles & available_roles
            missing.append({"kind": "functional_role", "status": "INCONCLUSIVE",
                            "description": ("필요 기능 역할이 추정 상태이며 사용자 확인이 필요합니다: "
                                            if inferred_match else "필요 기능 역할을 확인하지 못했습니다: ")
                                           + ", ".join(sorted(required_roles)),
                            "verification_methods": ["각인·실크스크린·부품 정보 확인"]})
        required_tests = {str(value) for value in mode.get("required_measurement_tests", [])}
        completed_tests = {str(row.get("test_id")) for row in measurement_evidence
                           if row.get("test_id") and row.get("used_for_ranking")}
        for test_id in sorted(required_tests - completed_tests):
            test = tests.get(test_id, {})
            missing.append({"kind": "measurement", "status": "UNKNOWN",
                            "description": f"필수 검증 측정 미확인: {test.get('name', test_id)}",
                            "test_id": test_id, "verification_methods": [test.get("question") or test.get("name") or test_id],
                            "risk_level": test.get("risk_level", "unknown"),
                            "requires_power_off": bool(test.get("requires_power_off", False))})
        needs_power_spec = bool(required_roles & {"power", "protection"}) or any(
            str(test_id).casefold() in {"voltage", "power", "supply", "rail"}
            for test_id in required_tests)
        reference_basis_present = any(
            row.get("fault_hypothesis") == fault_id
            and row.get("used_for_ranking") is True
            and row.get("reference_verification_status") in {"DATASHEET_VERIFIED", "BOARD_SPEC_VERIFIED"}
            and row.get("reference_min") is not None and row.get("reference_max") is not None
            and str(row.get("reference_source") or "").strip()
            and str(row.get("reference_source_id") or "").strip()
            for row in measurement_evidence
            if isinstance(row, dict)
            and (not related_ids or str(row.get("board_component_id") or row.get("component_id") or "") in related_ids)
        )
        if needs_power_spec and not reference_basis_present:
            missing.append({"kind": "power_specification", "status": "UNKNOWN",
                            "description": "보드 전원 규격·정상 기준 출처가 확인되지 않았습니다.",
                            "verification_methods": ["제조사 데이터시트 또는 보드 사양 확인"]})
        if mode.get("level") in {"component", "functional_block"} and not related_ids:
            missing.append({"kind": "component_scope", "status": "UNKNOWN",
                            "description": "가설에 연결할 확인된 소자 후보가 없습니다.",
                            "verification_methods": ["소자 종류와 기능 역할 검수"]})
        if required_roles and connectivity_status not in {"verified", "partial_verified"}:
            missing.append({"kind": "electrical_relation", "status": "UNKNOWN",
                            "description": "소자 근접성·기능 역할만으로 전기적 연결을 확정할 수 없습니다.",
                            "verification_methods": ["회로도·검증된 연결 정보 또는 안전한 측정 기록 확인"]})
        known = [item for item in relevant_ledger if item.get("status") in {"VERIFIED", "USER_REPORTED"}]
        inferred = [item for item in relevant_ledger if item.get("status") in {"INFERRED", "AI_SUGGESTED"}]
        opposed = list(hypothesis.get("opposing_evidence", []))
        recommended = []
        for test_id in mode.get("recommended_tests", []):
            test = tests.get(str(test_id), {})
            if test and str(test_id) not in completed_tests:
                recommended.append({"test_id": str(test_id), "name": test.get("name", str(test_id)),
                                    "risk_level": test.get("risk_level", "unknown"),
                                    "requires_power_off": bool(test.get("requires_power_off", False)),
                                    "question": test.get("question")})
        scoped_gaps = []
        matching_blocks = [row for row in functional_blocks if isinstance(row, dict)
                           and (not required_roles or row.get("role") in required_roles)]
        for block in matching_blocks:
            block_ids = {str(value) for value in block.get("component_ids", []) if value}
            block_ledger = [item for item in relevant_ledger
                            if item.get("component_id") in block_ids]
            scoped_gaps.append({"scope_id": str(block.get("block_id") or ""),
                                "level": "functional_block", "fault_id": fault_id,
                                "confirmed_facts": [item for item in block_ledger
                                                    if item.get("status") in {"VERIFIED", "USER_REPORTED"}],
                                "inferred_facts": [item for item in block_ledger
                                                   if item.get("status") in {"INFERRED", "AI_SUGGESTED"}],
                                "opposing_evidence": [],
                                "missing_evidence": ([{"kind": "functional_role", "status": "INCONCLUSIVE",
                                    "description": f"{block.get('role')} 역할은 추정 그룹이며 사용자 확인이 필요합니다.",
                                    "verification_methods": ["기능 역할 확인"]}]
                                    if block.get("verification_status") != "VERIFIED" else []) +
                                    ([{"kind": "electrical_relation", "status": "UNKNOWN",
                                      "description": "기능 블록 소속은 전기적 연결을 의미하지 않습니다.",
                                      "verification_methods": ["검증된 회로 정보 또는 안전한 측정"]}]
                                     if block.get("connectivity_status") not in {"VERIFIED", "PARTIAL_VERIFIED"} else []),
                                "candidate_component_ids": sorted(block_ids),
                                "status": "INCONCLUSIVE" if block.get("verification_status") != "VERIFIED" else "UNVERIFIED_HYPOTHESIS"})
        for candidate in related_candidates:
            component_id = str(candidate.get("board_component_id") or candidate.get("component_id") or "")
            if not component_id:
                continue
            candidate_gaps = []
            component_ledger = [item for item in relevant_ledger if item.get("component_id") == component_id]
            identity_verified = any(item.get("kind") == "component_identity" and
                                    item.get("status") == "VERIFIED" for item in component_ledger)
            if not identity_verified:
                candidate_gaps.append({"kind": "component_identity", "status": "UNKNOWN",
                                       "description": "고장 후보 소자의 실제 존재와 종류를 사용자 검수로 확인해야 합니다.",
                                       "verification_methods": ["PCB 소자 검수"]})
            if candidate.get("missing_measurement_tests"):
                for test_id in candidate["missing_measurement_tests"]:
                    test = tests.get(str(test_id), {})
                    candidate_gaps.append({"kind": "measurement", "status": "UNKNOWN",
                        "test_id": str(test_id), "description": f"소자별 측정 근거 미확인: {test.get('name', test_id)}",
                        "verification_methods": [test.get("question") or test.get("name") or str(test_id)],
                        "risk_level": test.get("risk_level", "unknown")})
            if candidate.get("missing_measurement_tests"):
                candidate_gaps.append({"kind": "measurement_location", "status": "UNKNOWN",
                    "description": "실제 측정 단자·위치와 검사 조건을 확인해야 합니다.",
                    "verification_methods": ["검사 프로파일에 지정된 단자 확인", "측정 위치 직접 확인"]})
            scoped_gaps.append({"scope_id": component_id, "level": "component", "fault_id": fault_id,
                                "confirmed_facts": [item for item in component_ledger
                                                    if item.get("status") in {"VERIFIED", "USER_REPORTED"}],
                                "inferred_facts": [item for item in component_ledger
                                                   if item.get("status") in {"INFERRED", "AI_SUGGESTED"}],
                                "opposing_evidence": list(candidate.get("opposing_evidence", [])),
                                "missing_evidence": candidate_gaps,
                                "candidate_component_ids": [component_id],
                                "candidate_group_id": candidate.get("candidate_group_id"),
                                "status": "INCONCLUSIVE" if candidate_gaps else "UNVERIFIED_HYPOTHESIS"})
        output.append({"fault_id": fault_id, "level": hypothesis.get("level"),
                       "confirmed_facts": known, "inferred_facts": inferred,
                       "opposing_evidence": opposed, "missing_evidence": missing,
                       "verification_methods": list(dict.fromkeys(method for item in missing
                                                                    for method in item.get("verification_methods", []))),
                       "recommended_checks": recommended,
                       "candidate_component_ids": related_ids,
                       "candidate_group_ids": sorted({str(row.get("candidate_group_id"))
                                                       for row in related_candidates if row.get("candidate_group_id")}),
                       "scoped_gaps": scoped_gaps,
                       "status": "INCONCLUSIVE" if missing else "UNVERIFIED_HYPOTHESIS"})
    return output
