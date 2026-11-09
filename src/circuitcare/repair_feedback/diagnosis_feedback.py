"""Compare the same Bayesian evidence under baseline and approved priors."""

from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
from typing import Any
from uuid import uuid4

from circuitcare.bayesian_engine import update_posteriors
from circuitcare.diagnosis_models import DiagnosisHypothesis, Measurement
from .models import utc_now


def create_diagnosis_snapshot(*, circuit_id: str, circuit_hash: str, diagnosis: dict[str, Any],
                              inspection: dict[str, Any] | None = None,
                              recommendations: dict[str, Any] | None = None,
                              hypothesis_rows: list[dict[str, Any]] | None = None,
                              model_version: str = "unknown", prompt_version: str | None = None,
                              hypothesis_set_id: str | None = None,
                              fault_scenario: dict[str, Any] | None = None,
                              model_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Capture an immutable, provenance-labeled diagnosis at case creation time."""
    posterior = diagnosis.get("posterior", [])
    inferred_set_id = hypothesis_set_id or ";".join(sorted(row.get("hypothesis_id", "") for row in posterior))
    return {"snapshot_id": f"snapshot_{uuid4().hex}", "created_at": utc_now(),
        "circuit_id": circuit_id, "circuit_hash": circuit_hash,
        "hypothesis_set_id": inferred_set_id, "model_version": model_version,
        "prompt_version": prompt_version, "model_info": deepcopy(model_info or {}),
        "fault_scenario": deepcopy(fault_scenario or {}),
        "hypotheses": deepcopy(posterior),
        "hypothesis_definitions": deepcopy(hypothesis_rows or []),
        "observations": deepcopy(diagnosis.get("used_measurements", [])),
        "unsupported_measurements": deepcopy(diagnosis.get("unsupported_measurements", [])),
        "model_mismatch": bool(diagnosis.get("model_mismatch", False)),
        "simulation_sources": sorted({str(row.get("simulation_source", "")) for row in posterior if row.get("simulation_source")}),
        "inspection_rankings": deepcopy(inspection or {}),
        "measurement_recommendations": deepcopy(recommendations or {}),
        "prior_weights": {row["hypothesis_id"]: float(row.get("prior_weight", 1.0)) for row in posterior},
        "snapshot_status": "ORIGINAL_IMMUTABLE"}


def apply_prior_map(hypotheses: list[DiagnosisHypothesis], prior_weights: dict[str, float] | None) -> list[DiagnosisHypothesis]:
    """Apply a validated ID map; absent map preserves every existing weight."""
    if prior_weights is None:
        return list(hypotheses)
    ids = {item.hypothesis_id for item in hypotheses}
    if set(prior_weights) != ids:
        raise ValueError("Prior hypothesis IDs must exactly match the active hypothesis set.")
    weights = list(prior_weights.values())
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not __import__("math").isfinite(item) or item <= 0 for item in weights):
        raise ValueError("Learned prior weights must be finite and strictly positive.")
    total = sum(weights)
    return [replace(item, prior_weight=float(prior_weights[item.hypothesis_id]) / total) for item in hypotheses]


def compare_prior_diagnoses(hypotheses: list[DiagnosisHypothesis], measurements: list[Measurement], *,
                            learned_prior_weights: dict[str, float] | None,
                            learned_prior_version: str | None = None) -> dict[str, Any]:
    if not hypotheses:
        return {"status": "NOT_EVALUATED", "reason": "No compatible hypotheses are available."}
    if learned_prior_weights is None:
        return {"status": "INSUFFICIENT_VERIFIED_REPAIR_DATA", "default": update_posteriors(hypotheses, measurements),
                "learned": None, "learned_prior_version": None}
    baseline = update_posteriors(hypotheses, measurements)
    learned_hypotheses = apply_prior_map(hypotheses, learned_prior_weights)
    adapted = update_posteriors(learned_hypotheses, measurements)
    baseline_by_id = {row["hypothesis_id"]: row for row in baseline["posterior"]}
    learned_by_id = {row["hypothesis_id"]: row for row in adapted["posterior"]}
    component_default: dict[str, float] = {}
    component_learned: dict[str, float] = {}
    for hypothesis in hypotheses:
        component_default[hypothesis.component_ref] = component_default.get(hypothesis.component_ref, 0.0) + baseline_by_id[hypothesis.hypothesis_id]["posterior"]
    for hypothesis in learned_hypotheses:
        component_learned[hypothesis.component_ref] = component_learned.get(hypothesis.component_ref, 0.0) + learned_by_id[hypothesis.hypothesis_id]["posterior"]
    return {"status": "COMPARED", "learned_prior_version": learned_prior_version,
        "default": baseline, "learned": adapted,
        "posterior_comparison": [{"hypothesis_id": key,
            "default_prior": baseline_by_id[key]["prior_weight"],
            "learned_prior": learned_by_id[key]["prior_weight"],
            "default_posterior": baseline_by_id[key]["posterior"],
            "learned_posterior": learned_by_id[key]["posterior"],
            "posterior_delta": learned_by_id[key]["posterior"] - baseline_by_id[key]["posterior"]}
            for key in baseline_by_id],
        "component_posterior_mass_comparison": [{"component_ref": ref,
            "default_posterior_mass": component_default.get(ref, 0.0),
            "learned_posterior_mass": component_learned.get(ref, 0.0),
            "posterior_mass_delta": component_learned.get(ref, 0.0) - component_default.get(ref, 0.0)}
            for ref in sorted(set(component_default) | set(component_learned))],
        "inspection_priority_status": "POSTERIOR_MASS_ONLY; stress/risk rankings require unchanged per-hypothesis risk evidence.",
        "information_gain_recommendation": {"status": "NOT_EVALUATED", "reason": "No compatible safety-qualified shared measurement model was supplied."},
        "improvement_claim": "NOT_EVALUATED: this is a same-evidence prior comparison, not independent performance evidence.",
        "probability_note": baseline["probability_interpretation"]}
