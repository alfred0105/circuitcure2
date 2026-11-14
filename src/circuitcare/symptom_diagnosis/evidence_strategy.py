"""Opt-in, non-probabilistic evidence re-ranking and safe test ordering.

The production rule engine remains the baseline. This strategy produces a
separate deterministic ranking for controlled comparisons and never presents
its scores as probabilities.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from circuitcare.symptom_diagnosis.component_fault_ranker import rank_joint_candidates


STRATEGIES = {"baseline", "evidence_v1"}


def rank_by_verified_evidence(
    baseline: dict[str, Any], *, components: list[dict[str, Any]],
    measurements: list[dict[str, Any]], observations: list[dict[str, Any]] | dict[str, Any] | None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Return a separate ranking with deduplicated, hypothesis-specific evidence.

    Only measurement verdicts already validated by the production evaluator are
    considered. Contradictory measurements are surfaced but have zero directional
    effect. No confidence or probability is inferred.
    """
    result = deepcopy(baseline)
    evidence_rows = result.get("measurement_evidence", [])
    by_fault: dict[str, list[dict[str, Any]]] = {}
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen_measurements: set[tuple[str, str, str]] = set()
    for row in evidence_rows:
        measurement_id = str(row.get("measurement_id") or "")
        fault_id = str(row.get("fault_hypothesis") or "")
        component_id = str(row.get("component_id") or "")
        key = (measurement_id, fault_id, component_id)
        if not measurement_id or key in seen_measurements:
            continue
        seen_measurements.add(key)
        by_fault.setdefault(fault_id, []).append(row)
        by_pair.setdefault((fault_id, component_id), []).append(row)

    hypotheses = []
    for row in result.get("hypotheses", []):
        fault_id = str(row.get("fault_id") or "")
        unique_facts = _unique_evidence_details(row.get("evidence_details", []))
        evidence_score = _independent_source_score(unique_facts)
        adjustment, states = _measurement_adjustment(by_fault.get(fault_id, []), weight=0.5)
        base_score = evidence_score if unique_facts else float(row.get("score", 0.0))
        copied = {**row, "baseline_score": row.get("score"),
                  "score": round(base_score + adjustment, 3),
                  "evidence_details": unique_facts,
                  "evidence_strategy": {"measurement_adjustment": adjustment,
                                        "measurement_states": states,
                                        "measurement_sources": _unique_values(by_fault.get(fault_id, []), "source"),
                                        "measurement_freshness": _freshness(by_fault.get(fault_id, [])),
                                        "independence_policy": "one contribution per measurement_id and fault",
                                        "independent_evidence_score": round(evidence_score, 3),
                                        "score_semantics": "uncalibrated relative score"}}
        hypotheses.append(copied)
    hypotheses.sort(key=lambda row: (-float(row.get("score", 0.0)), str(row.get("fault_id", ""))))
    for index, row in enumerate(hypotheses, start=1):
        row["rank"] = index

    components_ranked = []
    for row in result.get("all_component_candidates", result.get("component_candidates", [])):
        fault_id = str(row.get("fault_hypothesis") or row.get("fault_id") or "")
        component_id = str(row.get("board_component_id") or row.get("component_id") or "")
        evidence = by_pair.get((fault_id, component_id), [])
        existing_measurement_score = float(row.get("score_breakdown", {}).get("validated_measurement", 0.0))
        adjustment, states = _measurement_adjustment(evidence, weight=2.0)
        base = float(row.get("score", 0.0)) - existing_measurement_score
        score = base + adjustment
        copied = {**row, "baseline_score": row.get("score"), "score": round(score, 3),
                  "evidence_strategy": {"measurement_adjustment": adjustment,
                                        "measurement_states": states,
                                        "measurement_sources": _unique_values(evidence, "source"),
                                        "measurement_freshness": _freshness(evidence),
                                        "independence_policy": "one contribution per measurement_id and fault/component pair",
                                        "score_semantics": "uncalibrated relative score"}}
        components_ranked.append(copied)
    components_ranked.sort(key=lambda row: (-float(row.get("score", 0.0)),
                                             str(row.get("component_id", "")),
                                             str(row.get("fault_hypothesis", ""))))
    for index, row in enumerate(components_ranked, start=1):
        row["rank"] = index

    display_candidates = []
    display_component_ids: set[str] = set()
    for row in components_ranked:
        component_id = str(row.get("board_component_id") or row.get("component_id") or "")
        if not component_id or component_id in display_component_ids:
            continue
        display_component_ids.add(component_id)
        display_candidates.append(row)
        if len(display_candidates) >= top_k:
            break
    result["hypotheses"] = hypotheses
    result["all_component_candidates"] = components_ranked
    result["component_candidates"] = display_candidates
    result["joint_candidates"] = rank_joint_candidates(
        hypotheses=hypotheses, component_candidates=components_ranked,
        observations=baseline.get("observations", {}) if isinstance(baseline.get("observations"), dict) else {},
    )
    result["functional_block_hypotheses"] = summarize_board_roles(
        components, baseline.get("functional_blocks", []))
    result["ranking_strategy"] = "evidence_v1"
    result["score_semantics"] = "uncalibrated relative score; not a probability"
    return result


def optimize_safe_tests(
    tests: list[dict[str, Any]], *, observations: dict[str, Any] | list[dict[str, Any]] | None,
    measurements: list[dict[str, Any]] | None, limit: int = 5,
) -> list[dict[str, Any]]:
    """Order already-approved engine recommendations by deterministic separation.

    It uses answer counterfactuals supplied by the rule engine, not assumed
    answer probabilities. Risky tests, tests with unmet safety prerequisites,
    duplicate tests, and already-completed test IDs are removed.
    """
    observed_codes: set[str] = set()
    completed_tests = {str(row.get("test_id")) for row in measurements or [] if row.get("test_id")}
    completed_tests.update(str(row.get("test_id")) for row in (observations or [])
                           if isinstance(row, dict) and row.get("test_id"))
    if isinstance(observations, dict):
        facts = observations
    else:
        facts = {str(row.get("code")): row.get("value") for row in observations or [] if isinstance(row, dict)}
        observed_codes = set(facts)
    ranked = []
    seen: set[str] = set()
    for row in tests:
        test_id = str(row.get("test_id") or "")
        if not test_id or test_id in seen or test_id in completed_tests:
            continue
        seen.add(test_id)
        risk = str(row.get("risk_level", "unknown"))
        if risk not in {"none", "low"}:
            continue
        if row.get("requires_power_off") and facts.get("POWER_OFF_CONFIRMED") is not True:
            continue
        simulations = row.get("answer_score_simulation", {})
        signatures = []
        for scores in simulations.values():
            ordered = sorted(scores or [], key=lambda item: (-float(item[1]), str(item[0])))
            signatures.append(tuple(fault_id for fault_id, _score in ordered[:3]))
        unique_signatures = set(signatures)
        separation = ((len(unique_signatures) - 1) / (len(signatures) - 1)
                      if len(signatures) > 1 else 0.0)
        if separation <= 0:
            continue
        duration = row.get("expected_duration_seconds")
        cost = row.get("expected_cost")
        ranked.append({**row, "separation_score": round(separation, 4),
            "score_semantics": "counterfactual partition score; not probability",
            "expected_outcomes": list(simulations), "expected_cost": cost,
            "expected_duration_seconds": duration,
            "safety_status": "POWER_OFF_CONFIRMED" if row.get("requires_power_off") else "NO_POWER_ACTION_REQUIRED",
            "assumptions": ["규칙 엔진의 응답 시나리오를 기준으로 계산", "응답 확률은 가정하지 않음"]})
    ranked.sort(key=lambda row: (-row["separation_score"],
                                 row.get("expected_duration_seconds") if isinstance(row.get("expected_duration_seconds"), (int, float)) else float("inf"),
                                 row.get("expected_cost") if isinstance(row.get("expected_cost"), (int, float)) else float("inf"),
                                 row.get("test_id", "")))
    return ranked[:limit]


def summarize_board_roles(components: list[dict[str, Any]],
                          functional_blocks: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Group roles across board identities without inferring electrical edges."""
    role_members: dict[str, dict[str, Any]] = {}
    inferred_roles = {str(row.get("component_id")): set(row.get("candidate_blocks", []))
                      for row in functional_blocks or []}
    for component in components:
        component_id = str(component.get("board_component_id") or component.get("component_id") or "")
        if not component_id or component.get("presence_status") == "EXCLUDED":
            continue
        confirmed = set(component.get("confirmed_functional_roles", []))
        inferred = set(component.get("candidate_functional_roles", inferred_roles.get(component_id, ()))) - confirmed
        for role in confirmed | inferred:
            value = role_members.setdefault(str(role), {"confirmed_component_ids": [], "inferred_component_ids": []})
            value["confirmed_component_ids" if role in confirmed else "inferred_component_ids"].append(component_id)
    return [{"functional_role": role, **value,
             "status": "CONFIRMED_ROLE_PRESENT" if value["confirmed_component_ids"] else "INFERRED_ROLE_CANDIDATES",
             "connectivity": "NOT_INFERRED",
             "note": "기능 역할 그룹은 연결 관계를 뜻하지 않습니다."}
            for role, value in sorted(role_members.items())]


def _unique_evidence_details(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        code = str(row.get("code") or "")
        if not code:
            continue
        grouped.setdefault(code, []).append(row)
    unique = []
    for code, facts in grouped.items():
        weights = [float(row.get("weight", 0.0) or 0.0) for row in facts]
        if any(weight > 0 for weight in weights) and any(weight < 0 for weight in weights):
            unique.append({**facts[0], "code": code, "status": "CONFLICTED", "weight": 0.0,
                           "source": "conflicting_observations"})
            continue
        # A repeated/rephrased fact is one item; retain the strongest trusted source.
        selected = max(facts, key=lambda row: abs(float(row.get("weight", 0.0) or 0.0)) * _source_trust(row))
        unique.append(selected)
    return unique


def _independent_source_score(rows: list[dict[str, Any]]) -> float:
    """Reweight baseline facts by source quality; LLM suggestions have zero weight."""
    total = 0.0
    for row in rows:
        if str(row.get("status") or "").upper() == "CONFLICTED":
            continue
        trust = _source_trust(row)
        try:
            weight = float(row.get("weight", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        total += weight * trust
    return total


def _source_trust(row: dict[str, Any]) -> float:
    source = str(row.get("source") or "").casefold()
    status = str(row.get("status") or "").casefold()
    if any(token in source for token in ("qwen", "ollama", "llm", "model_suggestion")):
        return 0.0
    if status in {"measured", "confirmed", "user_confirmed"} or "measurement" in source:
        return 1.0
    if "user_report" in source or status in {"observed", "user_reported"}:
        return 0.7
    return 0.35


def _measurement_adjustment(rows: list[dict[str, Any]], *, weight: float) -> tuple[float, list[str]]:
    states = []
    total = 0.0
    seen: set[str] = set()
    for row in rows:
        measurement_id = str(row.get("measurement_id") or "")
        if not measurement_id or measurement_id in seen:
            continue
        seen.add(measurement_id)
        verdict = str(row.get("verdict") or "non_diagnostic")
        status = str(row.get("assessment") or row.get("status") or "INCONCLUSIVE")
        states.append(status)
        if status == "CONFLICTED" or verdict == "conflicted":
            continue
        if verdict == "supports":
            total += weight
        elif verdict == "refutes":
            total -= weight
    return total, states


def _unique_values(rows: list[dict[str, Any]], key: str) -> list[str]:
    return sorted({str(row.get(key)) for row in rows if row.get(key)})


def _freshness(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"measurement_id": str(row.get("measurement_id") or ""),
             "status": "TIMESTAMPED" if row.get("observed_at") else "UNKNOWN",
             "observed_at": str(row.get("observed_at")) if row.get("observed_at") else ""}
            for row in rows if row.get("measurement_id")]
