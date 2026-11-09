"""Analysis context identity and session result reuse helpers."""

from __future__ import annotations

import json
from pathlib import Path

from circuitcare.workspace.context import file_sha256
from circuitcare.workspace.measurement_provenance import measurement_set_sha256

ROOT = Path(__file__).resolve().parents[3]
DEMO_HYPOTHESES = ROOT / "data/diagnosis/demo_hypotheses.json"

def _analysis_context_key(case_name: str, case: dict, fault: str, fault_type: str, fault_value: str | None) -> tuple:
    files = []
    for name in ("netlist", "base", "mapping", "probes", "ratings", "prior"):
        value = case.get(name)
        if not value:
            continue
        path = ROOT / value
        identity = (str(path.resolve()), _file_sha256(path))
        files.append((name, identity))
    context_paths = [("what_if_hypotheses", DEMO_HYPOTHESES)]
    if case.get("cached"):
        for result_key in ("comparison.json", "component_stress.json", "risk_assessment.json"):
            context_paths.append((f"cached_{result_key}", ROOT / case["cached"] / result_key))
    if case.get("prior"):
        prior_dir = ROOT / case["prior"]
        for result_key in ("diagnosis.json", "inspection_ranking.json", "measurement_recommendations.json"):
            context_paths.append((f"prior_{result_key}", prior_dir / result_key))
    try:
        hypotheses = json.loads(DEMO_HYPOTHESES.read_text(encoding="utf-8")).get("hypotheses", [])
    except (OSError, json.JSONDecodeError):
        hypotheses = []
    for index, hypothesis in enumerate(hypotheses):
        for key, filename in (("simulation_source", "comparison.json"), ("risk_assessment_source", "risk_assessment.json")):
            source = Path(hypothesis[key])
            source = source if source.is_absolute() else DEMO_HYPOTHESES.parent / source
            context_paths.append((f"hypothesis_{index}_{key}", source / filename if source.is_dir() else source))
    for name, path in context_paths:
        identity = (str(path.resolve()), _file_sha256(path))
        files.append((name, identity))
    return (case_name, fault, fault_type, fault_value, tuple(files), _file_sha256(DEMO_HYPOTHESES))


def _supports_bayesian_demo(case_name: str, fault: str, fault_type: str, fault_value: str | None) -> bool:
    return (case_name == "Regulator + divider + switch (cached demo)"
            and fault == "U1" and fault_type == "regulator_output_rise" and fault_value == "8")


def _is_cached_case(case: dict, fault: str, fault_type: str, fault_value: str, needs_value: bool) -> bool:
    return (fault == case["fault"] and fault_type == case["fault_type"]
            and (not needs_value or fault_value == case["value"]))


def _simulation_status(live_matches: bool, live_error: str | None, is_cached_selection: bool) -> str:
    if live_matches:
        return "SUCCESS"
    if live_error:
        return "FAILED"
    return "CACHED" if is_cached_selection else "NOT_RUN"


def _measurement_set_id(measurements: list[dict]) -> str:
    return measurement_set_sha256(measurements)


def _file_sha256(path: Path) -> str | None:
    return file_sha256(path)


def _simulation_result_id(result_dir: Path) -> tuple:
    identity = [str(result_dir.resolve())]
    for name in ("comparison.json", "component_stress.json", "risk_assessment.json", "simulation_context.json"):
        path = result_dir / name
        identity.append((name, _file_sha256(path)))
    return tuple(identity)


def _advisor_matches(saved: dict, context_key: tuple, measurements: list[dict], result_dir: Path) -> bool:
    return (saved.get("analysis_context_key") == context_key
            and saved.get("measurement_set_id") == _measurement_set_id(measurements)
            and saved.get("simulation_result_id") == _simulation_result_id(result_dir))


def _upsert_measurement(measurements: list[dict], measurement: dict) -> list[dict]:
    updated = list(measurements)
    node = str(measurement["node"])
    for index, existing in enumerate(updated):
        if str(existing.get("node", "")).casefold() == node.casefold():
            updated[index] = measurement
            return updated
    updated.append(measurement)
    return updated


