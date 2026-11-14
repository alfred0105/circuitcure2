"""Hard safety gates for component-level fault hypotheses."""

from __future__ import annotations

from typing import Any
import json


_INTRINSIC_ROLE_TYPES = {"battery", "buzzer", "fuse", "led", "relay", "switch"}


def summarize_measurement_verdict(
    *, mode: dict[str, Any], component_id: str, fault_id: str,
    assessments: dict[tuple[str, str, str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Combine only profile-matched test results for one component/fault pair."""
    rows = [row for key, row in assessments.items()
            if key[0] == component_id and key[1] == fault_id]
    valid = [row for row in rows if row.get("verdict") in {"supports", "refutes", "conflicted"}]
    verdicts = {str(row.get("verdict")) for row in valid}
    required = {str(value) for value in [*mode.get("required_measurement_tests", []),
                                         *mode.get("exclusion_tests", [])]}
    completed = {str(row.get("test_id")) for row in valid}
    missing = sorted(required - completed)
    if "conflicted" in verdicts or {"supports", "refutes"} <= verdicts:
        verdict = "conflicted"
    elif missing:
        verdict = "non_diagnostic"
    elif "supports" in verdicts:
        verdict = "supports"
    elif "refutes" in verdicts:
        verdict = "refutes"
    else:
        verdict = "non_diagnostic"
    return {"verdict": verdict,
            "assessment_status": {"supports": "SUPPORTED", "refutes": "REFUTED",
                                  "conflicted": "CONFLICTED"}.get(verdict, "INCONCLUSIVE"),
            "decision_scope": "FAULT_HYPOTHESIS_ONLY", "required_tests": sorted(required),
            "completed_tests": sorted(completed), "missing_tests": missing,
            "alternative_causes": list(mode.get("alternative_causes", [])),
            "evidence": valid,
            "reason": ("측정 결과가 고장 가설을 지지합니다." if verdict == "supports" else
                       "검증된 검사가 해당 고장 가설을 반박합니다." if verdict == "refutes" else
                       "측정 결과가 충돌합니다." if verdict == "conflicted" else
                       "가설 판정에 필요한 검사가 부족하거나 진단력이 없습니다.")}


def validate_component_hypothesis(
    *, mode: dict[str, Any], component: dict[str, Any], confirmed_component_ids: set[str],
    confirmed_component_types: set[str] | None = None,
    component_block: dict[str, Any] | None = None,
    component_observations: list[dict[str, Any]] | None = None,
    measurement_verdict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an explicit eligibility decision; detection alone never confirms presence."""
    component_id = str(component.get("component_id") or "")
    component_type = str(component.get("component_type", component.get("label", "unknown"))).casefold()
    allowed_types = {str(value).casefold() for value in mode.get(
        "candidate_component_types", mode.get("component_types", []))}
    reasons: list[str] = []
    if not component_id or component_id not in confirmed_component_ids:
        reasons.append("COMPONENT_PRESENCE_UNCONFIRMED")
    if component.get("component_type_status") not in (None, "USER_CONFIRMED"):
        reasons.append("COMPONENT_TYPE_UNCONFIRMED")
    if allowed_types and component_type not in allowed_types:
        reasons.append("INCOMPATIBLE_COMPONENT_TYPE")
    required_types = {str(value).casefold() for value in mode.get("required_component_types", [])}
    if not required_types.issubset({str(value).casefold() for value in (confirmed_component_types or set())}):
        reasons.append("REQUIRED_COMPONENT_TYPE_MISSING")

    required_roles = {str(value) for value in mode.get("functional_roles", [])}
    role_status = "CONFIRMED"
    if required_roles:
        confirmed_roles = {str(value) for value in component.get("confirmed_functional_roles", [])}
        block = component_block or {}
        if block.get("verified"):
            confirmed_roles.update(str(value) for value in block.get("candidate_blocks", []))
        inferred_roles = {str(value) for value in block.get("candidate_blocks", [])}
        scoped_facts: dict[tuple[str, str], set[str]] = {}
        for row in component_observations or []:
            if (not isinstance(row, dict) or str(row.get("component_id") or "") != component_id
                    or str(row.get("status", "observed")).casefold() not in {"observed", "measured"}
                    or row.get("code") not in mode.get("supporting_evidence", {})):
                continue
            conditions = json.dumps(row.get("measurement_conditions") or {}, sort_keys=True,
                                    ensure_ascii=False, default=str)
            key = (str(row.get("code")), conditions)
            value = json.dumps(row.get("value"), sort_keys=True, ensure_ascii=False, default=str)
            scoped_facts.setdefault(key, set()).add(value)
        directly_supported = any(values == {"true"} for values in scoped_facts.values())
        if required_roles & confirmed_roles:
            role_status = "CONFIRMED"
        elif required_roles & inferred_roles or directly_supported:
            role_status = "INFERRED"
        else:
            role_status = "UNKNOWN"
        intrinsic_role_match = component_type in _INTRINSIC_ROLE_TYPES
        if role_status == "UNKNOWN" and intrinsic_role_match:
            role_status = "CONFIRMED"
        elif role_status == "UNKNOWN":
            reasons.append("FUNCTIONAL_ROLE_UNCONFIRMED")

    measurement_verdict = measurement_verdict or {"verdict": "non_diagnostic"}
    return {"eligible": not reasons, "reasons": reasons,
            "component_id": component_id, "component_type": component_type,
            "role_status": role_status, "measurement_verdict": measurement_verdict.get("verdict"),
            "verdict": measurement_verdict.get("verdict"),
            "measurement_reason": measurement_verdict.get("reason")}
