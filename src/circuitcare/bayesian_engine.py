"""Log-space Gaussian Bayesian updates over an explicit fault-hypothesis set."""

import math
from typing import Any

from .diagnosis_models import DiagnosisHypothesis, Measurement


def update_posteriors(
    hypotheses: list[DiagnosisHypothesis],
    measurements: list[Measurement],
    *,
    model_mismatch_threshold_sigma: float = 4.0,
) -> dict[str, Any]:
    """Apply measurements sequentially; probabilities are conditional on these hypotheses."""
    total_prior = sum(item.prior_weight for item in hypotheses)
    log_weights = [math.log(item.prior_weight / total_prior) if item.prior_weight > 0 else -math.inf for item in hypotheses]
    history = []
    supported_measurements = []
    unsupported_measurements = []
    for measurement in measurements:
        if not all(measurement.node in item.predicted_measurements for item in hypotheses):
            unsupported_measurements.append({
                **measurement.to_dict(),
                "status": "unsupported",
                "reason": "At least one hypothesis has no simulated prediction for this node; it was not used to update posterior weights.",
            })
            continue
        sigma = measurement.uncertainty
        increments = []
        for hypothesis in hypotheses:
            mean = hypothesis.predicted_measurements[measurement.node]
            z = min(abs((measurement.value - mean) / sigma), 1e150)
            increments.append(-0.5 * z * z)
        log_weights = [weight + increment for weight, increment in zip(log_weights, increments)]
        posterior = _normalize_log_weights(log_weights)
        supported_measurements.append({
            **measurement.to_dict(),
            "status": "used",
            "likelihood_model": "Gaussian kernel exp(-0.5*((measurement-prediction)/sigma)^2); common normalization cancels across hypotheses",
            "predictions": {item.hypothesis_id: item.predicted_measurements[measurement.node] for item in hypotheses},
            "likelihoods": {item.hypothesis_id: {"log_likelihood": increment, "kernel_likelihood": math.exp(increment)} for item, increment in zip(hypotheses, increments)},
        })
        history.append({
            "measurement": measurement.to_dict(),
            "posterior": _rank(hypotheses, posterior),
        })
    posterior = _normalize_log_weights(log_weights)
    best_standardized_residual = _best_standardized_residual(hypotheses, supported_measurements)
    threshold = model_mismatch_threshold_sigma
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("model mismatch threshold must be finite and positive")
    model_mismatch = best_standardized_residual is not None and best_standardized_residual > threshold
    return {
        "posterior": _rank(hypotheses, posterior),
        "posterior_history": history,
        "used_measurements": supported_measurements,
        "unsupported_measurements": unsupported_measurements,
        "model_mismatch": model_mismatch,
        "model_mismatch_rule": {
            "best_hypothesis_max_standardized_residual": best_standardized_residual,
            "threshold_sigma": threshold,
            "interpretation": "All configured hypotheses poorly explain the used measurements; inspect model coverage or measurement assumptions.",
        },
        "probability_interpretation": "Posterior weights are conditional on this configured hypothesis set, priors, simulated predictions, and stated Gaussian uncertainty. They are not real-world component failure probabilities.",
    }


def _normalize_log_weights(weights: list[float]) -> list[float]:
    maximum = max(weights)
    if maximum == -math.inf:
        return [1.0 / len(weights)] * len(weights)
    values = [0.0 if value == -math.inf else math.exp(value - maximum) for value in weights]
    total = sum(values)
    return [value / total for value in values]


def _rank(hypotheses: list[DiagnosisHypothesis], posterior: list[float]) -> list[dict[str, Any]]:
    ranked = [
        {
            "hypothesis_id": hypothesis.hypothesis_id,
            "component_ref": hypothesis.component_ref,
            "fault_type": hypothesis.fault_type,
            "parameters": hypothesis.parameters,
            "prior_weight": hypothesis.prior_weight,
            "posterior": probability,
            "simulation_source": hypothesis.simulation_source,
            "risk_assessment_source": hypothesis.risk_assessment_source,
        }
        for hypothesis, probability in zip(hypotheses, posterior)
    ]
    return sorted(ranked, key=lambda item: (-item["posterior"], item["hypothesis_id"]))


def _best_standardized_residual(hypotheses, used_measurements):
    if not used_measurements:
        return None
    best = math.inf
    for hypothesis in hypotheses:
        residuals = []
        for measurement in used_measurements:
            residuals.append(abs(measurement["value"] - hypothesis.predicted_measurements[measurement["node"]]) / measurement["uncertainty"])
        best = min(best, max(residuals, default=0.0))
    return best
