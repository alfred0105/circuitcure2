"""Load diagnosis hypotheses from saved Phase 05/06 JSON output."""

import json
import math
from pathlib import Path
from typing import Any

from .diagnosis_models import DiagnosisHypothesis, Measurement


def load_diagnosis_data(path: str | Path) -> tuple[list[DiagnosisHypothesis], dict[str, Any]]:
    """Hydrate predictions and risk links from referenced simulation artifacts."""
    source_path = Path(path).resolve()
    data = json.loads(source_path.read_text(encoding="utf-8"))
    entries = data.get("hypotheses")
    if not isinstance(entries, list) or not entries:
        raise ValueError("hypotheses JSON must contain at least one hypothesis")
    hypotheses = []
    seen = set()
    for entry in entries:
        hypothesis_id = str(entry.get("hypothesis_id", "")).strip()
        if not hypothesis_id or hypothesis_id in seen:
            raise ValueError("hypothesis_id values must be non-empty and unique")
        seen.add(hypothesis_id)
        comparison_path = _resolve(source_path.parent, entry.get("simulation_source"), "simulation_source")
        risk_path = _resolve(source_path.parent, entry.get("risk_assessment_source"), "risk_assessment_source")
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        risk = json.loads(risk_path.read_text(encoding="utf-8"))
        if comparison.get("simulation_success") is not True or risk.get("simulation_status") != "success":
            raise ValueError(f"Hypothesis {hypothesis_id} does not reference successful simulation and risk results")
        predicted = {}
        for node, record in comparison.get("voltage_nodes", {}).items():
            value = record.get("fault", {}).get("last")
            if isinstance(value, (int, float)) and math.isfinite(value):
                predicted[node] = float(value)
        prior = entry.get("prior_weight", 1.0)
        if isinstance(prior, bool) or not isinstance(prior, (int, float)) or not math.isfinite(prior) or prior < 0:
            raise ValueError(f"prior_weight for {hypothesis_id} must be a finite non-negative number")
        hypotheses.append(DiagnosisHypothesis(
            hypothesis_id=hypothesis_id,
            component_ref=str(entry["component_ref"]),
            fault_type=str(entry["fault_type"]),
            parameters=dict(entry.get("parameters", {})),
            prior_weight=float(prior),
            predicted_measurements=predicted,
            simulation_source=str(entry["simulation_source"]),
            risk_assessment_source=str(entry["risk_assessment_source"]),
            scenario_id=str(entry.get("scenario_id", comparison.get("scenario_id", hypothesis_id))),
            source_base=str(source_path.parent),
        ))
    if not any(item.prior_weight > 0 for item in hypotheses):
        raise ValueError("At least one prior_weight must be positive")
    model = data.get("measurement_model", {})
    if not isinstance(model, dict):
        raise ValueError("measurement_model must be an object")
    threshold = model.get("model_mismatch_threshold_sigma", 4.0)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("model_mismatch_threshold_sigma must be finite and positive")
    if not isinstance(model.get("candidate_nodes", {}), dict):
        raise ValueError("measurement_model.candidate_nodes must be an object")
    return hypotheses, model


def load_measurements(path: str | Path | None) -> list[Measurement]:
    """Load explicit user or simulated-demo voltage observations."""
    if path is None:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data.get("measurements", [])
    if not isinstance(entries, list):
        raise ValueError("measurements must be a list")
    measurements = []
    for entry in entries:
        unit = str(entry.get("unit", ""))
        if unit != "V":
            raise ValueError(f"Unsupported measurement unit {unit!r}; only voltage in V is supported")
        value, uncertainty = entry.get("value"), entry.get("uncertainty")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("measurement value must be a finite number")
        if isinstance(uncertainty, bool) or not isinstance(uncertainty, (int, float)) or not math.isfinite(uncertainty) or uncertainty <= 0:
            raise ValueError("measurement uncertainty sigma must be a finite positive number")
        source = str(entry.get("source", "user_measurement"))
        measurements.append(Measurement(
            node=str(entry.get("node", "")).strip(),
            value=float(value),
            unit=unit,
            uncertainty=float(uncertainty),
            source=source,
        ))
        if not measurements[-1].node:
            raise ValueError("measurement node must not be empty")
    return measurements


def available_measurements(hypotheses: list[DiagnosisHypothesis]) -> list[str]:
    """Return nodes with predictions from every configured hypothesis."""
    if not hypotheses:
        return []
    common = set(hypotheses[0].predicted_measurements)
    for hypothesis in hypotheses[1:]:
        common.intersection_update(hypothesis.predicted_measurements)
    return sorted(common)


def _resolve(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Each hypothesis requires {label}")
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()
