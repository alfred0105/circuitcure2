"""Streamlit-compatible workspace reconciliation without importing Streamlit."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from circuitcare.workspace.state_validator import ANALYSIS_DEPENDENCIES, stale_analysis_records


def reconcile_session_workspace(state: MutableMapping[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    previous = state.get("circuitcare_workspace")
    invalidated: list[str] = []
    if previous and previous.get("workspace_id") != context.get("workspace_id"):
        changed_inputs = {
            name for name in set(previous.get("input_hashes", {})) | set(context.get("input_hashes", {}))
            if previous.get("input_hashes", {}).get(name) != context.get("input_hashes", {}).get(name)
        }
        if previous.get("circuit_hash") != context.get("circuit_hash"):
            changed_inputs.add("netlist")
        if previous.get("active_fault_scenario") != context.get("active_fault_scenario"):
            changed_inputs.add("fault_scenario")
        if previous.get("observation_hash") != context.get("observation_hash"):
            changed_inputs.add("observations")
        old_refs = set(previous.get("component_refs", []))
        new_refs = set(context.get("component_refs", []))
        for analysis_type, dependencies in ANALYSIS_DEPENDENCIES.items():
            if dependencies & changed_inputs:
                invalidated.append(analysis_type)
        if old_refs != new_refs:
            invalidated.append("component_reference_context")
        for key in tuple(state):
            if key.startswith(("fault_ai_state_", "vision_alignment_", "report_snapshot_")):
                state.pop(key, None)
    reconciled = dict(context)
    reconciled["component_refs"] = sorted(context.get("component_refs", []))
    reconciled["invalidated_analyses"] = sorted(set(invalidated))
    reconciled["validation_status"] = "READY" if context.get("circuit_hash") not in (None, "MISSING") else "FAILED"
    if previous and previous.get("analysis_records"):
        changed = set()
        changed.update(name for name in set(previous.get("input_hashes", {})) | set(context.get("input_hashes", {}))
                       if previous.get("input_hashes", {}).get(name) != context.get("input_hashes", {}).get(name))
        if previous.get("circuit_hash") != context.get("circuit_hash"):
            changed.add("netlist")
        if previous.get("active_fault_scenario") != context.get("active_fault_scenario"):
            changed.add("fault_scenario")
        if previous.get("observation_hash") != context.get("observation_hash"):
            changed.add("observations")
        reconciled["analysis_records"] = stale_analysis_records(
            previous["analysis_records"], current_circuit_hash=context.get("circuit_hash", "MISSING"),
            current_input_hashes=context.get("input_hashes", {}), changed_inputs=changed)
    else:
        reconciled["analysis_records"] = []
    state["circuitcare_workspace"] = reconciled
    return reconciled
