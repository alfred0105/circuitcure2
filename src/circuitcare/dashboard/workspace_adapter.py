"""Translate validated analysis values to the existing workspace record contract."""

from __future__ import annotations


def build_analysis_records(*, workspace_context: dict, active_workspace: dict,
                           comparison: dict, risk: dict, status: str,
                           uploaded: bool = False) -> list[dict]:
    timestamp = active_workspace["generated_at"]
    inputs = workspace_context["input_hashes"]
    circuit_hash = workspace_context["circuit_hash"]
    spice_status = (
        "UNSUPPORTED" if uploaded else
        status if status in {"STALE", "PROVENANCE_UNVERIFIED"} else
        "COMPLETE" if status in {"SUCCESS", "CACHED_SIMULATION"} and comparison else
        "FAILED" if status == "FAILED" else "READY"
    )
    return [
        {"analysis_type": "circuit_validation", "status": "COMPLETE",
         "circuit_hash": circuit_hash, "source": "KiCad XML parser",
         "timestamp": timestamp, "input_hashes": {}},
        {"analysis_type": "graph_analysis", "status": "COMPLETE",
         "circuit_hash": circuit_hash, "source": "Component-Net graph",
         "timestamp": timestamp, "input_hashes": {"netlist": circuit_hash}},
        {"analysis_type": "spice_fault_comparison", "status": spice_status,
         "circuit_hash": circuit_hash, "source": "ngspice" if comparison else "NOT_RUN",
         "timestamp": timestamp, "scenario_id": comparison.get("scenario_id"),
         "input_hashes": inputs},
        {"analysis_type": "risk_assessment", "status": "COMPLETE" if risk else "NOT_EVALUATED",
         "circuit_hash": circuit_hash, "source": "configured ratings" if risk else "NOT_EVALUATED",
         "timestamp": timestamp, "scenario_id": risk.get("scenario_id"),
         "input_hashes": inputs},
    ]
