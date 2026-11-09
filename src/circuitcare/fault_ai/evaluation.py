"""Leakage-safe diagnostic inputs and separate proposal/ranking metrics."""

from typing import Any


_COMPONENT_FIELDS = ("ref", "value", "library_id", "nets", "mapping_available", "model_type")
_OBSERVATION_FIELDS = ("node", "value", "unit", "uncertainty", "measurement_mode")


def build_blind_diagnostic_input(case: dict[str, Any], *, components: list[dict[str, Any]],
                                 observations: list[dict[str, Any]],
                                 executable_catalog: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the allowlisted, answer-free payload supplied to the proposal model.

    Deliberately reads only ``llm_case_id`` from the evaluation record. Ground
    truth, descriptive evaluation IDs, paths, and arbitrary case metadata cannot
    flow into the prompt through this function.
    """
    llm_case_id = case.get("llm_case_id")
    if not isinstance(llm_case_id, str) or not llm_case_id.startswith("case_"):
        raise ValueError("An anonymous llm_case_id is required for blinded evaluation")
    public_components = [{key: component[key] for key in _COMPONENT_FIELDS if key in component}
                        for component in components]
    public_observations = [{key: observation[key] for key in _OBSERVATION_FIELDS if key in observation}
                           for observation in observations]
    allowed_pairs = []
    seen = set()
    for row in executable_catalog:
        if row.get("validation_status") != "VALIDATED_EXECUTABLE":
            continue
        pair = (row.get("component_ref"), row.get("fault_type"))
        if pair in seen:
            continue
        seen.add(pair)
        allowed_pairs.append({"component_ref": pair[0], "fault_type": pair[1],
                              "nets": list(row.get("nets", []))})
    symptoms = _observations_summary(public_observations)
    return {"circuit_id": llm_case_id, "components": public_components,
            "observations": public_observations, "symptoms": symptoms,
            "executable_catalog": allowed_pairs}


def _observations_summary(observations: list[dict[str, Any]]) -> str:
    if not observations:
        return "No node measurements were supplied. Use only the circuit and supported fault catalog."
    lines = []
    for row in observations:
        line = f"Observed {row.get('node', 'unknown node')} = {row.get('value')} {row.get('unit', '')}".strip()
        if row.get("uncertainty") is not None:
            line += f" (measurement uncertainty ±{row['uncertainty']} {row.get('unit', '')})"
        if row.get("measurement_mode"):
            line += f"; mode: {row['measurement_mode']}"
        lines.append(line)
    return "\n".join(lines)


def score_fault_proposals(cases: list[dict[str, Any]],
                          predictions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Score candidates before simulation; never mixes proposal and ranking quality."""
    if not cases:
        return _empty("NOT_EVALUATED")
    expected_count = candidate_count = true_positive = executable = unsupported = hallucinated_refs = 0
    per_case = []
    for case in cases:
        rows = predictions.get(case["case_id"], [])
        expected = {_identity(row) for row in case.get("ground_truth", [])}
        found = {_identity(row) for row in rows}
        hits = expected & found
        expected_count += len(expected)
        candidate_count += len(rows)
        true_positive += len(hits)
        executable += sum(row.get("validation_status") == "VALIDATED_EXECUTABLE" for row in rows)
        unsupported += sum(row.get("validation_status") != "VALIDATED_EXECUTABLE" for row in rows)
        hallucinated_refs += sum(row.get("validation_status") == "INVALID_REFERENCE" for row in rows)
        per_case.append({"case_id": case["case_id"], "ground_truth_count": len(expected),
                         "candidate_count": len(rows), "correct_candidate_count": len(hits)})
    precision = true_positive / candidate_count if candidate_count else None
    recall = true_positive / expected_count if expected_count else None
    return {"status": "MEASURED_SYNTHETIC" if any(c.get("truth_source") == "synthetic_fault" for c in cases) else "MEASURED",
            "case_count": len(cases), "ground_truth_count": expected_count, "candidate_count": candidate_count,
            "correct_candidate_count": true_positive, "candidate_precision": precision, "candidate_recall": recall,
            "candidate_f1": _f1(precision, recall),
            "valid_proposal_rate": executable / candidate_count if candidate_count else None,
            "unsupported_proposal_rate": unsupported / candidate_count if candidate_count else None,
            "hallucinated_reference_rate": hallucinated_refs / candidate_count if candidate_count else None,
            "per_case": per_case}


def score_fault_rankings(cases: list[dict[str, Any]],
                         predictions: dict[str, list[dict[str, Any]]], *, top_k=(1, 3)) -> dict[str, Any]:
    """Score only method-generated, simulated candidates; missing ranks are misses."""
    if not cases:
        return {"status": "NOT_EVALUATED", "case_count": 0, "ranking_coverage": None,
                "top_k_accuracy": {str(k): None for k in top_k},
                "precision_at_k": {str(k): None for k in top_k}}
    hits = {k: 0 for k in top_k}
    precision_sums = {k: 0.0 for k in top_k}
    ranked_cases = 0
    per_case = []
    for case in cases:
        expected = {_identity(row) for row in case.get("ground_truth", [])}
        rows = predictions.get(case["case_id"], [])
        ranked = sorted((row for row in rows if isinstance(row.get("evidence_rank"), int)),
                        key=lambda row: row["evidence_rank"])
        has_rank = bool(ranked)
        ranked_cases += int(has_rank)
        case_hits = {}
        for k in top_k:
            top_rows = ranked[:k]
            count = sum(_identity(row) in expected for row in top_rows)
            hits[k] += int(count > 0)
            precision_sums[k] += count / min(k, len(top_rows)) if top_rows else 0.0
            case_hits[str(k)] = int(count > 0)
        per_case.append({"case_id": case["case_id"], "candidate_count": len(rows),
                         "ranked_candidate_count": len(ranked), "has_ranked_result": has_rank,
                         "top_k_hits": case_hits,
                         "correct_rank": next((row["evidence_rank"] for row in ranked if _identity(row) in expected), None)})
    total = len(cases)
    return {"status": "MEASURED_SYNTHETIC" if any(c.get("truth_source") == "synthetic_fault" for c in cases) else "MEASURED",
            "case_count": total, "ranked_case_count": ranked_cases,
            "ranking_coverage": ranked_cases / total,
            "top_k_accuracy": {str(k): hits[k] / total for k in top_k},
            "precision_at_k": {str(k): precision_sums[k] / total for k in top_k},
            "top_k_evaluated_cases": {str(k): total for k in top_k}, "per_case": per_case}


def score_fault_cases(cases: list[dict[str, Any]], predictions: dict[str, list[dict[str, Any]]], *, top_k=(1, 3)) -> dict[str, Any]:
    """Backward-compatible combined record with proposal and ranking fields separated."""
    return {"proposal_quality": score_fault_proposals(cases, predictions),
            "diagnostic_ranking": score_fault_rankings(cases, predictions, top_k=top_k)}


def _identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("component_ref"), row.get("fault_type"), _parameters(row.get("parameters", {})))


def _parameters(value: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    pairs = []
    for key, item in value.items():
        try:
            normalized = f"{float(item):.12g}"
        except (TypeError, ValueError):
            normalized = str(item).strip().casefold()
        pairs.append((str(key), normalized))
    return tuple(sorted(pairs))


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _empty(status: str) -> dict[str, Any]:
    return {"status": status, "case_count": 0, "ground_truth_count": 0, "candidate_count": 0,
            "candidate_precision": None, "candidate_recall": None, "candidate_f1": None,
            "valid_proposal_rate": None, "unsupported_proposal_rate": None,
            "hallucinated_reference_rate": None, "per_case": []}
