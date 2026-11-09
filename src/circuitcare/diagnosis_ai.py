"""Offline diagnosis flow: load simulated hypotheses, update posteriors, rank tests."""

import json
from pathlib import Path
from typing import Any

from .bayesian_engine import update_posteriors
from .diagnosis_data import available_measurements, load_diagnosis_data, load_measurements
from .diagnosis_ranking import normalized_prior_rows, rank_inspection_candidates
from .measurement_selector import recommend_measurements


def run_diagnosis(
    hypotheses_path: str | Path,
    measurements_path: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    hypotheses, measurement_model = load_diagnosis_data(hypotheses_path)
    measurements = load_measurements(measurements_path)
    mismatch_threshold = measurement_model.get("model_mismatch_threshold_sigma", 4.0)
    result = update_posteriors(
        hypotheses, measurements,
        model_mismatch_threshold_sigma=mismatch_threshold,
    )
    recommendation = recommend_measurements(hypotheses, result["posterior"], measurement_model, measurements)
    inspection = rank_inspection_candidates(
        hypotheses,
        normalized_prior_rows(hypotheses),
        result["posterior"],
    )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    diagnosis = {
        "assessment_mode": "posterior_updated" if result["used_measurements"] else "prior_only",
        "hypothesis_count": len(hypotheses),
        "hypotheses_source": str(hypotheses_path),
        "available_measurement_nodes": available_measurements(hypotheses),
        "measurements_file": str(measurements_path) if measurements_path else None,
        **result,
        "top_hypothesis": result["posterior"][0] if result["posterior"] else None,
        "recommendation_file": str(directory / "measurement_recommendations.json"),
        "inspection_ranking_file": str(directory / "inspection_ranking.json"),
        "diagnosis_file": str(directory / "diagnosis.json"),
    }
    diagnosis_path = directory / "diagnosis.json"
    recommendation_path = directory / "measurement_recommendations.json"
    inspection_path = directory / "inspection_ranking.json"
    diagnosis_path.write_text(json.dumps(diagnosis, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    recommendation_path.write_text(json.dumps(recommendation, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    inspection_path.write_text(json.dumps(inspection, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return {"diagnosis": diagnosis, "recommendations": recommendation, "inspection_ranking": inspection}
