"""Input-only baseline runners and bounded simulation orchestration."""

from __future__ import annotations

from typing import Any, Callable

from .integrity import audit_public_payload


def run_input_method(method: Callable[[dict[str, Any]], list[dict[str, Any]]],
                     public_case: dict[str, Any]) -> dict[str, Any]:
    """Run a proposal method with public input only and audit its exact payload."""
    payload = public_case.get("diagnostic_input")
    if not isinstance(payload, dict):
        raise ValueError("Public case is missing its diagnostic_input object")
    audit = audit_public_payload(payload)
    predictions = method(payload)
    if not isinstance(predictions, list):
        raise TypeError("Proposal methods must return a list of hypotheses")
    return {"case_id": public_case["case_id"], "diagnostic_input_hash": audit["payload_sha256"],
            "predictions": predictions}


def apply_simulation_budget(candidates: list[dict[str, Any]], *, max_runs: int,
                            simulate: Callable[[dict[str, Any]], dict[str, Any]]) -> list[dict[str, Any]]:
    if max_runs < 0:
        raise ValueError("max_runs must be non-negative")
    output = []
    attempted = 0
    for candidate in candidates:
        if candidate.get("validation_status") != "VALIDATED_EXECUTABLE":
            output.append({**candidate, "simulation_status": "UNSUPPORTED_HYPOTHESIS"})
        elif attempted >= max_runs:
            output.append({**candidate, "simulation_status": "SIMULATION_BUDGET_EXHAUSTED"})
        else:
            attempted += 1
            output.append(simulate(candidate))
    return output


def merge_hypothesis_proposals(*methods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged, seen = [], set()
    for rows in methods:
        for row in rows:
            params = tuple(sorted((str(k), str(v)) for k, v in row.get("parameters", {}).items()))
            key = (row.get("component_ref"), row.get("fault_type"), params)
            if key in seen:
                continue
            seen.add(key)
            merged.append(dict(row))
    return merged
