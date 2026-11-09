"""Regenerate the deterministic example Phase 18 diagnosis exports."""

from __future__ import annotations

import json
from pathlib import Path

from circuitcare.netlist import load_netlist
from circuitcare.reporting.exporters import export_bundle, export_html, export_json, export_markdown
from circuitcare.reporting.report_validator import validate_report, verify_bundle
from circuitcare.reporting.report_builder import build_diagnostic_snapshot, classify_parameter_grid_coverage
from circuitcare.workspace.context import build_workspace_context

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    netlist = ROOT / "tests/fixtures/fault_demo_netlist.xml"
    case = {
        "netlist": "tests/fixtures/fault_demo_netlist.xml",
        "base": "data/circuits/fault_demo.cir",
        "mapping": "data/circuits/fault_demo_mapping.json",
        "probes": "data/circuits/fault_demo_probes.json",
        "ratings": "data/component_specs/fault_demo_ratings.json",
    }
    result = ROOT / "data/fault_scenarios/overvoltage_demo/simulation_results/fault_u1_regulator_output_rise"
    comparison = json.loads((result / "comparison.json").read_text(encoding="utf-8"))
    risk = json.loads((result / "risk_assessment.json").read_text(encoding="utf-8"))
    circuit = load_netlist(netlist)
    context = build_workspace_context(
        circuit_id="Regulator + divider + switch (cached demo)", netlist_path=netlist,
        input_paths={name: ROOT / case[name] for name in ("base", "mapping", "probes", "ratings")},
        fault={"component_ref": "U1", "fault_type": "regulator_output_rise", "parameter": "8"},
    ).to_dict()
    context["validation_status"] = "READY"
    context["component_refs"] = sorted(circuit.components)
    report = build_diagnostic_snapshot(root=ROOT, workspace=context, circuit=circuit,
        fault={"component_ref": "U1", "fault_type": "regulator_output_rise", "parameter": "8"},
        simulation_status="CACHED", comparison=comparison, risk_assessment=risk,
        observations=[{"node": "VOUT", "value": 7.9, "unit": "V", "uncertainty": 0.2,
                       "source": "simulated_demo"}])
    report["reproducibility"]["ngspice_version"] = "ngspice-47"
    errors = validate_report(report, expected_circuit_hash=context["circuit_hash"])
    if errors:
        raise RuntimeError("Report validation failed: " + ", ".join(errors))
    output = ROOT / "examples/phase18_demo"
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnosis.json").write_text(export_json(report), encoding="utf-8")
    (output / "report.md").write_text(export_markdown(report), encoding="utf-8")
    (output / "report.html").write_text(export_html(report), encoding="utf-8")
    (output / "evidence_bundle.zip").write_bytes(export_bundle(report, root=ROOT))

    # A fixed off-grid R1=1.37 kΩ case: report generation consumes saved
    # ngspice/catalog artifacts and does not execute either engine again.
    evidence_dir = output / "offgrid_evidence"
    comparison = json.loads((evidence_dir / "comparison.json").read_text(encoding="utf-8"))
    risk = json.loads((evidence_dir / "risk_assessment.json").read_text(encoding="utf-8"))
    candidates = json.loads((evidence_dir / "fault_hypotheses.json").read_text(encoding="utf-8"))
    offgrid_case = {
        "netlist": "tests/fixtures/fault_demo_netlist.xml",
        "base": "data/circuits/fault_demo.cir",
        "mapping": "data/circuits/fault_demo_mapping.json",
        "probes": "data/circuits/fault_demo_probes.json",
        "ratings": "data/component_specs/fault_demo_ratings.json",
    }
    offgrid_context = build_workspace_context(
        circuit_id="Off-grid resistor demo", netlist_path=ROOT / offgrid_case["netlist"],
        input_paths={key: ROOT / offgrid_case[key] for key in ("base", "mapping", "probes", "ratings")},
        fault={"component_ref": "R1", "fault_type": "resistance_drift", "parameter": "1.37k"},
    ).to_dict()
    offgrid_context["validation_status"] = "READY"
    offgrid_context["component_refs"] = sorted(circuit.components)
    parameter_status = classify_parameter_grid_coverage(
        component_ref="R1", fault_type="resistance_drift", fault_parameter="1.37k",
        rule_candidates=candidates,
    )
    offgrid_report = build_diagnostic_snapshot(
        root=ROOT, workspace=offgrid_context, circuit=circuit,
        fault={"component_ref": "R1", "fault_type": "resistance_drift", "parameter": "1.37k"},
        simulation_status="SUCCESS", comparison=comparison, risk_assessment=risk,
        parameter_identification=parameter_status,
        active_fault_candidates={"status": "GENERATED_CATALOG", "failed_component": "R1",
                                 "rule_candidates": candidates, "hypotheses": [],
                                 "interpretation": "Candidate coverage only; no exact diagnosis selection is implied."},
    )
    offgrid_report["reproducibility"]["ngspice_version"] = "ngspice-47"
    errors = validate_report(offgrid_report, expected_circuit_hash=offgrid_context["circuit_hash"])
    if errors:
        raise RuntimeError("Off-grid report validation failed: " + ", ".join(errors))
    (output / "offgrid_diagnosis.json").write_text(export_json(offgrid_report), encoding="utf-8")
    (output / "offgrid_report.md").write_text(export_markdown(offgrid_report), encoding="utf-8")
    (output / "offgrid_report.html").write_text(export_html(offgrid_report), encoding="utf-8")
    offgrid_bundle = export_bundle(offgrid_report, root=ROOT, selected_artifacts={
        "spice_results.json": evidence_dir / "comparison.json",
        "risk_assessment.json": evidence_dir / "risk_assessment.json",
        "fault_hypotheses.json": evidence_dir / "fault_hypotheses.json",
    })
    (output / "offgrid_evidence_bundle.zip").write_bytes(offgrid_bundle)
    verify_bundle(offgrid_bundle)
    print(f"Generated validated cached and off-grid synthetic report files at {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
