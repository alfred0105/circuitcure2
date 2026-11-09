"""Build compact, traceable facts, assumptions, and recommendations for an advisor."""

import json
from pathlib import Path
from typing import Any


def build_evidence(
    diagnosis: dict[str, Any],
    inspection_ranking: dict[str, Any],
    measurement_recommendations: dict[str, Any],
    risk_assessment: dict[str, Any],
    component_stress: dict[str, Any],
    *,
    source_files: dict[str, str],
) -> dict[str, Any]:
    facts = []
    assumptions = []

    def fact(component, metric, value, unit, source_file, source_type=None, context=None):
        record = {
            "evidence_id": f"E{len(facts) + 1:03d}",
            "component_ref": component,
            "metric": metric,
            "value": value,
            "unit": unit,
            "source_file": source_file,
        }
        if source_type:
            record["source_type"] = source_type
        if context:
            record["context"] = context
        facts.append(record)
        return record["evidence_id"]

    top = diagnosis.get("top_hypothesis") or {}
    failed = diagnosis.get("posterior", [{}])[0].get("component_ref") if diagnosis.get("posterior") else None
    fact(failed, "selected_fault_hypothesis", top.get("hypothesis_id"), "hypothesis_id", "diagnosis.json", context=top.get("fault_type"))
    for row in diagnosis.get("posterior", []):
        fact(row.get("component_ref"), "conditional_posterior", row.get("posterior"), "probability_weight", "diagnosis.json", context=row.get("hypothesis_id"))

    for measurement in diagnosis.get("used_measurements", []):
        node = measurement.get("node")
        meas_id = fact(node, "observed_voltage", measurement.get("value"), measurement.get("unit"), "diagnosis.json", measurement.get("source"), context="measurement uncertainty sigma=" + str(measurement.get("uncertainty")))
        for hypothesis_id, prediction in measurement.get("predictions", {}).items():
            fact(node, "simulated_prediction_for_measurement", prediction, measurement.get("unit"), "diagnosis.json", context=hypothesis_id)
        if measurement.get("source") == "simulated_demo":
            assumptions.append({"text": f"The {node} observation is simulated demo input, not a real instrument measurement.", "evidence_ids": [meas_id]})
    for unsupported in diagnosis.get("unsupported_measurements", []):
        assumptions.append({"text": f"Measurement at {unsupported.get('node')} was not used: {unsupported.get('reason')}", "evidence_ids": []})

    posterior_by_hypothesis = {row.get("hypothesis_id"): row for row in diagnosis.get("posterior", [])}
    hypotheses_source = Path(diagnosis.get("hypotheses_source", ""))
    if not hypotheses_source.is_absolute():
        hypotheses_source = (Path.cwd() / hypotheses_source).resolve()
    hypotheses_root = hypotheses_source.parent
    for candidate in inspection_ranking.get("posterior_weighted_inspection_order", []):
        reference = candidate.get("component_ref")
        for scenario in candidate.get("scenario_evidence", []):
            hypothesis = posterior_by_hypothesis.get(scenario.get("hypothesis_id"), {})
            scenario_source = hypothesis.get("risk_assessment_source", "risk_assessment.json")
            scenario_path = Path(scenario_source)
            if not scenario_path.is_absolute():
                scenario_path = hypotheses_root / scenario_path
            scenario_source = display_path(scenario_path)
            scenario_context = f"{scenario.get('hypothesis_id')}; posterior={scenario.get('posterior')}"
            fact(reference, "scenario_conditional_risk_level", scenario.get("risk_level"), "classification", scenario_source, context=scenario_context)
            fact(reference, "scenario_conditional_priority_index", scenario.get("priority_score"), "index_0_to_100", scenario_source, context=scenario_context)
            for metric, result in scenario.get("metrics", {}).items():
                if result.get("measured_stress") is not None:
                    fact(reference, f"scenario_{metric}_stress", result.get("measured_stress"), result.get("unit"), scenario_source, context=scenario_context)
                if result.get("rated_limit") is not None:
                    fact(reference, f"scenario_{metric}_rating", result.get("rated_limit"), result.get("unit"), scenario_source, result.get("rating_source_type"), context=scenario_context)
                    fact(reference, f"scenario_{metric}_utilization", result.get("utilization_ratio"), "ratio", scenario_source, context=f"{scenario_context}; exceeded={result.get('rating_exceeded')}")
                    if result.get("rating_source_type") == "demo_assumption":
                        assumptions.append({"text": f"The {reference} rating in {scenario.get('hypothesis_id')} is a labeled demo assumption, not a manufacturer-verified value.", "evidence_ids": []})

    for row in risk_assessment.get("secondary_candidates", []):
        reference = row.get("component_ref")
        fact(reference, "risk_level", row.get("risk_level"), "classification", "risk_assessment.json", context=row.get("reason"))
        fact(reference, "inspection_priority_index", row.get("priority_score"), "index_0_to_100", "risk_assessment.json", context="Not a damage probability")
    for item in risk_assessment.get("rating_checks", []):
        for metric in item.get("metrics", {}).values():
            rating_source = metric.get("rating_source") or {}
            note = rating_source.get("note")
            if note:
                assumptions.append({"text": note, "evidence_ids": []})

    for component in component_stress.get("components", []):
        reference = component.get("component_ref")
        if reference in {item.get("component_ref") for item in risk_assessment.get("secondary_candidates", [])}:
            for metric, summary in component.get("fault", {}).items():
                if summary and summary.get("last") is not None:
                    fact(reference, f"fault_last_{metric}", summary["last"], _unit_for_metric(metric), "component_stress.json")
                baseline = component.get("baseline", {}).get(metric)
                if baseline and baseline.get("last") is not None:
                    fact(reference, f"baseline_last_{metric}", baseline["last"], _unit_for_metric(metric), "component_stress.json")

    if diagnosis.get("posterior_history"):
        assumptions.append({"text": "Posterior update history is sequential and conditioned on the configured hypothesis set and Gaussian measurement model.", "evidence_ids": []})
    if top.get("parameters", {}).get("output_voltage") is not None:
        assumptions.append({"text": "The regulator output fault is modeled as an ideal voltage-source setpoint change, not an internal regulator failure mechanism.", "evidence_ids": []})
    if diagnosis.get("used_measurements"):
        assumptions.append({"text": "Bayesian posterior values are conditional on the configured hypotheses, simulated predictions, and stated measurement uncertainty; they are not real-world failure probabilities.", "evidence_ids": []})
    for item in risk_assessment.get("limitations", []):
        assumptions.append({"text": item, "evidence_ids": []})

    recommended = measurement_recommendations.get("recommended_measurement")
    if recommended:
        fact(recommended.get("node"), "next_measurement_expected_information_gain", recommended.get("expected_information_gain_nats"), "nats", "measurement_recommendations.json", context=recommended.get("reason"))
    else:
        fact(None, "next_measurement_recommendation_status", measurement_recommendations.get("recommendation_status"), "status", "measurement_recommendations.json", context=measurement_recommendations.get("reason"))

    return {
        "facts": facts,
        "assumptions": _deduplicate(assumptions),
        "recommendations": [],
        "unknowns": _unknowns(diagnosis, risk_assessment, component_stress),
        "analysis_context": {
            "failed_component": risk_assessment.get("failed_component"),
            "fault_type": risk_assessment.get("fault_type"),
            "scenario_id": risk_assessment.get("scenario_id"),
            "model_mismatch": diagnosis.get("model_mismatch"),
            "model_mismatch_rule": diagnosis.get("model_mismatch_rule"),
            "inspection_order": [
                {
                    "component_ref": item.get("component_ref"),
                    "priority_rank": item.get("priority_rank"),
                    "priority_score": item.get("posterior_weighted_priority_score"),
                    "risk_level": _scenario_risk(item),
                    "any_hypothesis_exceeded": item.get("any_hypothesis_exceeded", False),
                    "scenario_risk_levels": [
                        {"hypothesis_id": row.get("hypothesis_id"), "posterior": row.get("posterior"), "risk_level": row.get("risk_level")}
                        for row in item.get("scenario_evidence", [])
                    ],
                }
                for item in inspection_ranking.get("posterior_weighted_inspection_order", [])
            ],
            "measurement_recommendation": recommended,
            "recommendation_status": measurement_recommendations.get("recommendation_status"),
            "source_files": source_files,
        },
    }


def resolve_related_evidence(diagnosis_path: str | Path, risk_override: str | None = None, stress_override: str | None = None):
    """Load the highest-posterior scenario's compact risk and stress JSON files."""
    diagnosis_path = Path(diagnosis_path)
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    if risk_override:
        risk_path = Path(risk_override)
    else:
        hypotheses_source = Path(diagnosis.get("hypotheses_source", ""))
        if not hypotheses_source.is_absolute():
            hypotheses_source = (Path.cwd() / hypotheses_source).resolve()
        if not hypotheses_source.is_file():
            raise ValueError("Cannot locate the diagnosis hypothesis file; supply --risk-assessment")
        source = (json.loads(hypotheses_source.read_text(encoding="utf-8"))
                  ["hypotheses"])
        top_id = (diagnosis.get("top_hypothesis") or {}).get("hypothesis_id")
        top_config = next((item for item in source if item.get("hypothesis_id") == top_id), None)
        if top_config is None:
            raise ValueError("Top hypothesis is missing from the configured hypothesis file")
        risk_path = Path(top_config["risk_assessment_source"])
        if not risk_path.is_absolute():
            risk_path = hypotheses_source.parent / risk_path
    risk_path = risk_path.resolve()
    stress_path = Path(stress_override).resolve() if stress_override else risk_path.parent / "component_stress.json"
    return (
        json.loads(risk_path.read_text(encoding="utf-8")),
        json.loads(stress_path.read_text(encoding="utf-8")),
        str(risk_path),
        str(stress_path),
    )


def _unknowns(diagnosis, risk, stress):
    results = []
    if diagnosis.get("model_mismatch"):
        results.append("Measurements are poorly explained by all configured scenarios; do not treat the highest posterior as a confirmed diagnosis.")
    if not diagnosis.get("used_measurements"):
        results.append("No usable measurement evidence was provided; hypotheses remain at prior weights.")
    if diagnosis.get("top_hypothesis", {}).get("posterior") == 1.0:
        results.append("The top posterior is conditional on only the configured scenarios and measurement model; it is not certainty about the real circuit.")
    if diagnosis.get("unsupported_measurements"):
        results.append("At least one measurement was not comparable across every hypothesis and was not used.")
    if risk.get("excluded_candidates"):
        results.extend(f"{item.get('component_ref')} is excluded from physical ranking: {item.get('reason')}" for item in risk["excluded_candidates"])
    for item in risk.get("secondary_candidates", []):
        if item.get("risk_level") == "UNKNOWN":
            results.append(f"{item.get('component_ref')} has insufficient ratings or measurements and is not confirmed safe.")
    for item in stress.get("components", []):
        if item.get("status") in {"unavailable", "measured_voltage_only", "voltage_only"}:
            results.append(f"{item.get('component_ref')} has unsupported or incomplete stress metrics ({item.get('status')}).")
    results.append("No component damage has been confirmed by these simulated and configured data.")
    return list(dict.fromkeys(results))


def _scenario_risk(item):
    evidence = item.get("scenario_evidence", [])
    if not evidence:
        return "UNKNOWN"
    top = max(evidence, key=lambda row: row.get("posterior", 0.0))
    winners = [row for row in evidence if abs(row.get("posterior", 0.0) - top.get("posterior", 0.0)) <= 1e-12]
    levels = {row.get("risk_level", "UNKNOWN") for row in winners}
    return next(iter(levels)) if len(levels) == 1 else "HYPOTHESIS_UNRESOLVED"


def _unit_for_metric(metric):
    return {"voltage_v": "V", "current_a": "A", "power_w": "W"}.get(metric, "unknown")


def display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return resolved.name


def _deduplicate(items):
    seen = set()
    result = []
    for item in items:
        key = item["text"]
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result
