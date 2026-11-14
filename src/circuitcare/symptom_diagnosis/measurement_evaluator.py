"""Condition-scoped measurement checks tied to validated component test profiles."""

from __future__ import annotations

from math import isfinite
import json
from typing import Any

from circuitcare.symptom_diagnosis.measurement_validation import measurement_verification_gaps


AssessmentKey = tuple[str, str, str, str, str]


def _fold(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def evaluate_component_measurements(
    measurements: list[dict[str, Any]] | None, *, profiles: dict[str, Any] | None = None,
    current_context: dict[str, Any] | None = None,
) -> dict[AssessmentKey, dict[str, Any]]:
    """Evaluate each distinct test/target/condition; never trust a free-text range by itself."""
    profile_map = (profiles or {}).get("profiles", {})
    context = current_context or {}
    evaluations: dict[AssessmentKey, dict[str, Any]] = {}
    for row in measurements or []:
        if not isinstance(row, dict):
            continue
        component_id = str(row.get("component_id") or "")
        fault_id = str(row.get("fault_hypothesis") or "")
        test_id = str(row.get("test_id") or "")
        target = str(row.get("measurement_target") or "").strip()
        condition = row.get("measurement_conditions") or row.get("operating_conditions") or ""
        condition_key = (json.dumps(condition, sort_keys=True, ensure_ascii=False)
                         if isinstance(condition, dict) else _fold(condition))
        key: AssessmentKey = (component_id, fault_id, test_id, _fold(target), condition_key)
        measurement_id = str(row.get("measurement_id") or "")
        reasons = measurement_verification_gaps(row, profiles=profiles, current_context=context)
        reason = " ".join(reasons) if reasons else None
        verdict = "non_diagnostic"
        status = "INCONCLUSIVE"
        if not measurement_id:
            reasons.append("측정 ID 누락")
            reason = " ".join(reasons)
        if not reasons:
            component_type = str(row.get("component_type") or "").casefold()
            test_profile = profile_map.get(component_type, {}).get("tests", {}).get(test_id, {})
            try:
                value = float(row["value"])
                low_value, high_value = float(row["reference_min"]), float(row["reference_max"])
                sigma = float(row["uncertainty"])
                if not all(isfinite(number) for number in (value, low_value, high_value, sigma)):
                    raise ValueError
                if low_value > high_value or sigma <= 0:
                    raise ValueError
            except (KeyError, TypeError, ValueError, OverflowError):
                reason = "측정값·범위·불확실도가 유효하지 않습니다."
            else:
                measured_low, measured_high = value - sigma, value + sigma
                if measured_low >= low_value and measured_high <= high_value:
                    status = "TEST_NORMAL"
                    reason = "해당 검사와 조건에서 측정 범위가 검증 기준 안에 있습니다."
                elif measured_high < low_value or measured_low > high_value:
                    status = "TEST_ABNORMAL"
                    reason = "해당 검사와 조건에서 측정 범위가 검증 기준을 벗어났습니다. 이는 선택한 가설의 근거이며 소자 고장 확정이 아닙니다."
                else:
                    status = "INCONCLUSIVE"
                    reason = "측정 불확실도 범위가 검증 기준의 경계와 겹칩니다."
                verdict = test_profile["fault_relations"][fault_id].get(status, "non_diagnostic")

        item = {
            "component_id": component_id, "fault_id": fault_id, "test_id": test_id,
            "measurement_target": target, "operating_conditions": condition,
            "measurement_id": measurement_id, "status": status, "verdict": verdict,
            "assessment_status": {"supports": "SUPPORTED", "refutes": "REFUTED",
                                  "conflicted": "CONFLICTED"}.get(verdict, "INCONCLUSIVE"),
            "decision_scope": "FAULT_HYPOTHESIS_ONLY",
            "reason": reason, "reference_source_id": row.get("reference_source_id"),
            "reference_verification_status": row.get("reference_verification_status"),
            "evidence_ids": [measurement_id] if measurement_id else [],
        }
        previous = evaluations.get(key)
        if previous is None:
            evaluations[key] = item
        else:
            evidence_ids = list(dict.fromkeys([*previous.get("evidence_ids", []), *item["evidence_ids"]]))
            old_valid = previous.get("verdict") in {"supports", "refutes"}
            new_valid = verdict in {"supports", "refutes"}
            if old_valid and new_valid and previous.get("verdict") != verdict:
                previous.update(status="CONFLICTED", verdict="conflicted",
                                assessment_status="CONFLICTED",
                                reason="동일한 검사·단자·조건의 결과가 서로 충돌합니다.")
            elif previous.get("verdict") == "conflicted":
                previous["evidence_ids"] = evidence_ids
            elif new_valid and not old_valid:
                previous.update(item)
            previous["evidence_ids"] = evidence_ids
    return evaluations
