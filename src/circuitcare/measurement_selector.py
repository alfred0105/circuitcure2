"""Rank simulated voltage measurements by expected information gain."""

import math
from typing import Any

from .diagnosis_models import DiagnosisHypothesis, Measurement


def recommend_measurements(
    hypotheses: list[DiagnosisHypothesis],
    posterior: list[dict[str, Any]],
    measurement_model: dict[str, Any],
    measurements: list[Measurement],
) -> dict[str, Any]:
    """Approximate mutual information with a deterministic grid quadrature."""
    weights_by_id = {item["hypothesis_id"]: item["posterior"] for item in posterior}
    used_nodes = {measurement.node for measurement in measurements}
    predictions_common = set(hypotheses[0].predicted_measurements) if hypotheses else set()
    for hypothesis in hypotheses[1:]:
        predictions_common.intersection_update(hypothesis.predicted_measurements)
    candidates = measurement_model.get("candidate_nodes", {})
    ranked = []
    excluded = []
    for node, config in candidates.items():
        if node not in predictions_common:
            excluded.append({"node": node, "status": "unsupported", "reason": "No prediction exists in every hypothesis."})
            continue
        if node in used_nodes:
            excluded.append({"node": node, "status": "already_measured", "reason": "This node is already included in the measurement evidence."})
            continue
        if config.get("unit") != "V":
            excluded.append({"node": node, "status": "unsupported", "reason": "Only voltage measurements in V are supported."})
            continue
        if config.get("safety_status") != "low_voltage_demo":
            excluded.append({"node": node, "status": "not_recommended", "reason": "No low-voltage demo safety classification is configured; the engine will not recommend this measurement."})
            continue
        sigma = config.get("uncertainty")
        if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or not math.isfinite(sigma) or sigma <= 0:
            excluded.append({"node": node, "status": "unsupported", "reason": "A positive configured measurement uncertainty is required."})
            continue
        means = [hypothesis.predicted_measurements[node] for hypothesis in hypotheses]
        eig = _expected_information_gain(means, [weights_by_id[item.hypothesis_id] for item in hypotheses], float(sigma))
        ranked.append({
            "node": node,
            "measurement_type": "voltage",
            "unit": "V",
            "uncertainty": float(sigma),
            "uncertainty_source": config.get("uncertainty_source", "configured_measurement_model"),
            "safety_status": config["safety_status"],
            "predicted_range_v": {"minimum": min(means), "maximum": max(means)},
            "expected_information_gain_nats": eig,
            "reason": "Simulated final-time voltage predictions differ across remaining hypotheses; information gain is estimated under the configured Gaussian measurement uncertainty.",
        })
    ranked.sort(key=lambda item: (-item["expected_information_gain_nats"], item["node"]))
    recommended = ranked[0] if ranked and ranked[0]["expected_information_gain_nats"] > 1e-9 else None
    reason = "Maximum information gain among configured low-voltage demo nodes." if recommended else "No unmeasured configured node provides meaningful expected information gain."
    return {
        "recommended_measurement": recommended,
        "recommendation_status": "recommended" if recommended else "none_needed_or_available",
        "reason": reason,
        "ranked_candidates": ranked,
        "excluded_candidates": excluded,
        "safety_notice": "This is an information-priority result for a low-voltage simulated demo only; it is not live-work authorization or an electrical safety approval.",
    }


def _expected_information_gain(means: list[float], prior: list[float], sigma: float, grid_size: int = 301) -> float:
    prior_entropy = _entropy(prior)
    low = min(means) - 5.0 * sigma
    high = max(means) + 5.0 * sigma
    if high <= low:
        return 0.0
    step = (high - low) / grid_size
    masses = []
    conditional_entropies = []
    for index in range(grid_size):
        observation = low + (index + 0.5) * step
        likelihoods = [math.exp(-0.5 * ((observation - mean) / sigma) ** 2) for mean in means]
        weighted = [weight * likelihood for weight, likelihood in zip(prior, likelihoods)]
        predictive = sum(weighted)
        if predictive <= 0:
            masses.append(0.0)
            conditional_entropies.append(0.0)
        else:
            conditional = [value / predictive for value in weighted]
            masses.append(predictive * step)
            conditional_entropies.append(_entropy(conditional))
    total_mass = sum(masses)
    if total_mass <= 0:
        return 0.0
    expected_entropy = sum(mass * entropy for mass, entropy in zip(masses, conditional_entropies)) / total_mass
    return max(0.0, prior_entropy - expected_entropy)


def _entropy(probabilities: list[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0)
