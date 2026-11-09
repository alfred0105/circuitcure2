"""Compare stated observations against explicitly sampled SPICE predictions."""

import math
from typing import Any


def compare_observations(predictions: dict[str, float], observations: list[dict[str, Any]], *,
                         model_mismatch_threshold_sigma: float = 4.0) -> dict[str, Any]:
    if not math.isfinite(model_mismatch_threshold_sigma) or model_mismatch_threshold_sigma <= 0:
        raise ValueError("model mismatch threshold must be finite and positive")
    compared, unsupported = [], []
    for item in observations:
        node = str(item.get("node", ""))
        unit = str(item.get("unit", ""))
        if node not in predictions or unit.casefold() != "v":
            unsupported.append({"observation": item, "reason": "Prediction node unavailable or unit is not volts."})
            continue
        try:
            measured = float(item["value"])
            predicted = float(predictions[node])
            uncertainty = item.get("uncertainty")
            sigma = float(uncertainty) if uncertainty is not None else None
            if not math.isfinite(measured) or not math.isfinite(predicted) or (sigma is not None and (not math.isfinite(sigma) or sigma <= 0)):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            unsupported.append({"observation": item, "reason": "Measurement or uncertainty is not finite numeric data."})
            continue
        residual = measured - predicted
        compared.append({"node": node, "unit": unit, "measured": measured, "predicted": predicted,
                         "residual": residual, "uncertainty": sigma,
                         "normalized_residual": residual / sigma if sigma is not None else None,
                         "source": item.get("source", "unspecified"), "measurement_mode": item.get("measurement_mode", "final_sample")})
    if not compared:
        status = "INSUFFICIENT_MEASUREMENTS"
    elif any(row["normalized_residual"] is None for row in compared):
        status = "INSUFFICIENT_MEASUREMENTS"
    elif max(abs(row["normalized_residual"]) for row in compared) > model_mismatch_threshold_sigma:
        status = "MODEL_MISMATCH"
    else:
        status = "CONSISTENT_WITH_OBSERVATIONS"
    return {"status": status, "comparisons": compared, "unsupported_measurements": unsupported,
            "max_absolute_normalized_residual": max((abs(row["normalized_residual"]) for row in compared if row["normalized_residual"] is not None), default=None),
            "model_mismatch_threshold_sigma": model_mismatch_threshold_sigma,
            "interpretation": "Fit to the selected model and observations only; this does not confirm a physical failure."}
