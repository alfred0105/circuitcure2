"""Build evidence snapshots from explicit workflow inputs and a read-only state mapping."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from circuitcare.dashboard.case_loader import DEMO_HYPOTHESES
from circuitcare.dashboard.context import _file_sha256, _measurement_set_id
from circuitcare.graph import direct_candidates
from circuitcare.reporting.report_builder import (
    build_diagnostic_snapshot,
    classify_parameter_grid_coverage,
)


@dataclass(frozen=True)
class ReportSnapshotContext:
    root: Path
    workspace: dict
    circuit: Any
    failed_component: str
    fault_type: str
    fault_value: str | None
    simulation_status: str
    comparison: dict
    risk_assessment: dict
    simulation_context: dict
    analysis_context_key: Any
    bayesian_supported: bool
    include_preview: bool
    what_if_result: dict | None
    what_if_measurements: list[dict]
    connection_paths: list[dict]


def build_report_snapshot_for_dashboard(context: ReportSnapshotContext,
                                        state: Mapping[str, Any]) -> dict:
    hypothesis_set_sha256 = _file_sha256(DEMO_HYPOTHESES)
    fault_ai_state_id = __import__("hashlib").sha256(
        repr(state.get("analysis_context_key")).encode()
    ).hexdigest()[:16]
    fault_ai_state = state.get(f"fault_ai_state_{fault_ai_state_id}", {})
    observations = (state.get("measurements", []) if context.bayesian_supported
                    else fault_ai_state.get("observations", []))
    actual_set_hash = _measurement_set_id(observations)
    analysis_context_id = hashlib.sha256(
        repr(context.analysis_context_key).encode()
    ).hexdigest()[:16]
    fault_hypothesis_context_matches = (
        fault_ai_state.get("context_key") == fault_ai_state_id
        and fault_ai_state.get("measurement_set_sha256") == actual_set_hash
    )
    report_hypotheses = fault_ai_state.get("hypotheses", []) if fault_hypothesis_context_matches else []
    candidate_diagnosis_result = state.get("diagnosis_result") if context.bayesian_supported else None
    diagnosis_context_matches = bool(
        candidate_diagnosis_result
        and candidate_diagnosis_result.get("analysis_context_id") == analysis_context_id
        and candidate_diagnosis_result.get("measurement_set_sha256") == actual_set_hash
    )
    active_diagnosis_result = candidate_diagnosis_result if diagnosis_context_matches else None
    report_diagnosis = (
        active_diagnosis_result.get("diagnosis") if active_diagnosis_result else
        {"status": "CANDIDATES_AVAILABLE", "hypotheses": report_hypotheses,
         "llm_metadata": fault_ai_state.get("llm_metadata")} if report_hypotheses else
        {"status": "NOT_EVALUATED"}
    )
    active_ranking = (active_diagnosis_result.get("inspection_ranking") if active_diagnosis_result
                      else fault_ai_state.get("bayesian_result") if fault_hypothesis_context_matches else None)
    diagnosis_context = {
        "mode": "ACTIVE_DIAGNOSIS", "measurement_set_sha256": actual_set_hash,
        "circuit_hash": context.workspace.get("circuit_hash"),
        "hypothesis_set_sha256": hypothesis_set_sha256 if context.bayesian_supported else None,
        "prior_version": "default",
        "status": "COMPLETE" if active_diagnosis_result else "NOT_EVALUATED",
    }
    preview_context = ({
        "mode": "WHAT_IF_PREVIEW", "measurement_source": "simulated_demo",
        "measurement_set_sha256": _measurement_set_id(context.what_if_measurements),
        "status": "PREVIEW_ONLY",
    } if context.include_preview else None)
    inspection_recommendations = ({
        "status": "CONDITIONAL_INSPECTION_RANKING", "ranking": active_ranking,
        "measurement_recommendations": context.what_if_result.get("recommendations")
        if context.what_if_result else None,
        "interpretation": "Conditional inspection ordering only; separate from electrical stress and not damage probability.",
    } if active_ranking else {"status": "NOT_EVALUATED"})
    rule_candidates = fault_ai_state.get("rule_candidates", []) if fault_hypothesis_context_matches else []
    parameter_identification = classify_parameter_grid_coverage(
        component_ref=context.failed_component, fault_type=context.fault_type,
        fault_parameter=context.fault_value if context.fault_type in {
            "resistance_drift", "regulator_output_rise", "regulator_output_drop"} else None,
        rule_candidates=rule_candidates,
    )
    return build_diagnostic_snapshot(
        root=context.root, workspace=context.workspace, circuit=context.circuit,
        fault={"component_ref": context.failed_component, "fault_type": context.fault_type,
               "parameter": context.fault_value or None},
        simulation_status=context.simulation_status, comparison=context.comparison,
        risk_assessment=context.risk_assessment, observations=observations,
        diagnosis=report_diagnosis, inspection_recommendations=inspection_recommendations,
        parameter_identification=parameter_identification,
        diagnosis_context=diagnosis_context, simulation_context=context.simulation_context,
        preview_context=preview_context,
        preview_measurements=context.what_if_measurements if context.include_preview else [],
        preview_diagnosis=(context.what_if_result.get("diagnosis")
                           if context.include_preview and context.what_if_result else None),
        active_fault_candidates={
            "status": "CONNECTIVITY_CANDIDATES", "failed_component": context.failed_component,
            "direct_candidate_count": len(direct_candidates(context.circuit, context.failed_component)),
            "hypotheses": report_hypotheses, "rule_candidates": rule_candidates,
            "measurement_set_sha256": fault_ai_state.get("measurement_set_sha256")
            if fault_hypothesis_context_matches else None,
            "hypothesis_set_sha256": fault_ai_state.get("hypothesis_set_sha256")
            if fault_hypothesis_context_matches else None,
            "hypothesis_symptom": fault_ai_state.get("symptom") if report_hypotheses else None,
            "paths": context.connection_paths,
            "interpretation": "Potential connectivity only; not confirmed damage.",
        },
    )
