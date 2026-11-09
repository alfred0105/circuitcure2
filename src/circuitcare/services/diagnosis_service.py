"""Thin local diagnosis orchestration shared by the dashboard and What-if view."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from circuitcare.dashboard.analysis_loader import json_object_if_exists
from circuitcare.dashboard.case_loader import DEMO_HYPOTHESES, PROJECT_ROOT
from circuitcare.diagnosis_ai import run_diagnosis


def load_saved_diagnosis(case: dict) -> dict:
    base = PROJECT_ROOT / case["prior"]
    return {
        "diagnosis": json_object_if_exists(base / "diagnosis.json"),
        "inspection_ranking": json_object_if_exists(base / "inspection_ranking.json"),
        "recommendations": json_object_if_exists(base / "measurement_recommendations.json"),
    }


def advisor_inputs(case: dict) -> tuple[dict, dict, dict]:
    if case.get("prior"):
        return tuple(load_saved_diagnosis(case).values())
    return {}, {}, {}


def run_diagnosis_for_measurements(measurements: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix="circuitcare-ui-") as tmp:
        measurement_path = Path(tmp) / "measurements.json"
        measurement_path.write_text(json.dumps({"measurements": measurements}), encoding="utf-8")
        result = run_diagnosis(DEMO_HYPOTHESES, measurement_path, Path(tmp) / "results")
        return {key: result[key] for key in ("diagnosis", "inspection_ranking", "recommendations")}
