"""Dependency-aware staleness checks for reusable analysis records."""

from __future__ import annotations

from typing import Any

ANALYSIS_DEPENDENCIES: dict[str, set[str]] = {
    "circuit_validation": {"netlist"},
    "graph_analysis": {"netlist"},
    "fault_hypotheses": {"netlist", "mapping", "spice_model"},
    "spice_fault_comparison": {"netlist", "mapping", "spice_model", "probes", "fault_scenario"},
    "observation_comparison": {"netlist", "mapping", "spice_model", "probes", "fault_scenario", "observations"},
    "risk_assessment": {"netlist", "mapping", "spice_model", "probes", "fault_scenario", "ratings"},
    "bayesian_diagnosis": {"observations", "prior"},
    "pcb_alignment": {"pcb_image", "component_positions", "confirmed_references"},
}


def stale_analysis_records(records: list[dict[str, Any]], *, current_circuit_hash: str,
                           current_input_hashes: dict[str, str | None],
                           changed_inputs: set[str] | None = None) -> list[dict[str, Any]]:
    changed = changed_inputs or set()
    current_hashes = {**current_input_hashes, "netlist": current_circuit_hash}
    validated = []
    for record in records:
        item = dict(record)
        dependencies = ANALYSIS_DEPENDENCIES.get(item.get("analysis_type", ""), {"netlist"})
        recorded_hashes = item.get("input_hashes", {})
        circuit_changed = item.get("circuit_hash") != current_circuit_hash
        dependency_changed = any(
            name in dependencies and recorded_hashes.get(name) != current_hashes.get(name)
            for name in set(recorded_hashes) | set(current_hashes)
        )
        explicitly_changed = bool(dependencies & changed)
        if circuit_changed or dependency_changed or explicitly_changed:
            item["status"] = "STALE"
            item["stale_reason"] = "Circuit or a declared analysis input changed. Re-run this analysis before using it."
        validated.append(item)
    return validated
