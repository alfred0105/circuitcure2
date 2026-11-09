"""Rank simulated hypotheses by measurement residual without inventing priors."""

from typing import Any

from circuitcare.bayesian_engine import update_posteriors
from circuitcare.diagnosis_models import DiagnosisHypothesis, Measurement


def rank_hypotheses(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = []
    for row in rows:
        match = row.get("observation_match", {})
        score = match.get("max_absolute_normalized_residual")
        if row.get("simulation_status") == "SIMULATION_SUCCESS" and score is not None:
            eligible.append((float(score), row))
    eligible.sort(key=lambda pair: (pair[0], pair[1].get("hypothesis_id", "")))
    rank_by_id = {row.get("hypothesis_id"): rank for rank, (_, row) in enumerate(eligible, 1)}
    output = []
    for row in rows:
        item = dict(row)
        match = item.get("observation_match", {})
        if item.get("simulation_status") != "SIMULATION_SUCCESS":
            item["observation_status"] = "NOT_SIMULATED"
        else:
            item["observation_status"] = match.get("status", "INSUFFICIENT_MEASUREMENTS")
        item["evidence_rank"] = rank_by_id.get(item.get("hypothesis_id"))
        item["ranking_basis"] = "minimum maximum absolute standardized residual; no failure-probability claim"
        output.append(item)
    return sorted(output, key=lambda item: (item.get("evidence_rank") is None, item.get("evidence_rank") or 0, item.get("hypothesis_id", "")))


def conditional_bayesian_update(rows: list[dict[str, Any]], observations: list[dict[str, Any]], *,
                                prior_weights: dict[str, float] | None = None,
                                prior_model_version: str | None = None) -> dict[str, Any]:
    """Reuse the existing engine only for a shared, uncertainty-qualified prediction set."""
    measured = []
    for item in observations:
        if item.get("unit", "").casefold() != "v" or item.get("uncertainty") is None:
            return {"status": "INACTIVE", "reason": "Every Bayesian measurement must include a positive uncertainty in volts."}
        measured.append(Measurement(str(item["node"]), float(item["value"]), "V",
                                    float(item["uncertainty"]), str(item.get("source", "unspecified"))))
    eligible = [row for row in rows if row.get("simulation_status") == "SIMULATION_SUCCESS"]
    if not measured or len(eligible) < 2:
        return {"status": "INACTIVE", "reason": "At least two simulated hypotheses and one qualified measurement are required."}
    node_set = {item.node for item in measured}
    if any(not node_set.issubset(row.get("predicted_measurements", {})) for row in eligible):
        return {"status": "INACTIVE", "reason": "Simulated hypotheses do not share predictions for all selected nodes."}
    hypotheses = [DiagnosisHypothesis(
        hypothesis_id=row["hypothesis_id"], component_ref=row["component_ref"], fault_type=row["fault_type"],
        parameters=row.get("parameters", {}), prior_weight=1.0,
        predicted_measurements={node: float(row["predicted_measurements"][node]) for node in node_set},
        simulation_source=row.get("simulation_result_key", "fault_ai_transient"), risk_assessment_source="",
        scenario_id=row.get("hypothesis_id", ""), source_base="fault_ai_session") for row in eligible]
    try:
        if prior_weights is not None:
            from circuitcare.repair_feedback.diagnosis_feedback import apply_prior_map
            hypotheses = apply_prior_map(hypotheses, prior_weights)
        result = update_posteriors(hypotheses, measured)
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        return {"status": "INACTIVE", "reason": f"Existing Bayesian engine rejected this set: {exc}"}
    return {"status": "ACTIVE_CONDITIONAL", "prior_source": "APPROVED_LEARNED" if prior_weights is not None else "DEFAULT_EQUAL",
            "prior_model_version": prior_model_version if prior_weights is not None else None, **result}
