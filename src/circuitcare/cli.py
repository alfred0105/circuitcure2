"""Initial offline CLI. No probabilistic or physical damage claims."""

import argparse
import json
from pathlib import Path

from .fault_models import SUPPORTED_FAULT_TYPES, create_scenario
from .graph import build_graph, direct_candidates, find_paths_result, to_dot
from .impact_analyzer import analyze_fault_impact
from .component_specs import load_component_specs
from .impact_ranking import rank_impacts
from .netlist import load_netlist
from .scenario_generator import generate_scenario_files


def main() -> None:
    parser = argparse.ArgumentParser(description="CircuitCare AI bootstrap CLI")
    parser.add_argument("--netlist", required=True, help="KiCad XML netlist path")
    parser.add_argument("--fault", required=True, help="Known failed component ref (e.g. U1)")
    parser.add_argument("--show-pins", action="store_true", help="Include component metadata and pin-to-net connections")
    parser.add_argument("--show-paths", action="store_true", help="Include direct and indirect connectivity paths")
    parser.add_argument("--max-hops", type=int, default=2, help="Maximum component-to-component path length (default: 2)")
    parser.add_argument("--net-types", type=Path, help="Optional JSON map of net names to ground, power, signal or unknown")
    parser.add_argument("--exclude-net-types", help="Comma-separated net types to exclude from path traversal")
    parser.add_argument("--export-dot", type=Path, help="Write the circuit connectivity graph as a Graphviz DOT file")
    parser.add_argument("--fault-type", choices=sorted(SUPPORTED_FAULT_TYPES), help="Generate a faulted SPICE scenario")
    parser.add_argument("--fault-value", help="Explicit drift resistance or regulator output voltage")
    parser.add_argument("--spice-base", type=Path, help="Baseline SPICE .cir file")
    parser.add_argument("--spice-mapping", type=Path, help="JSON mapping from KiCad references to SPICE elements")
    parser.add_argument("--output", type=Path, help="Directory for baseline, faulted .cir, and scenario metadata")
    parser.add_argument("--simulate", action="store_true", help="Run baseline and faulted circuits using ngspice")
    parser.add_argument("--probes", type=Path, help="JSON voltage-node and component probe configuration")
    parser.add_argument("--ratings", type=Path, help="JSON component ratings with explicit sources")
    parser.add_argument("--assess-risk", action="store_true", help="Classify simulated secondary stress against configured ratings")
    args = parser.parse_args()

    spice_options = (args.fault_value, args.spice_base, args.spice_mapping, args.output)
    if args.fault_type:
        if not all((args.spice_base, args.spice_mapping, args.output)):
            parser.error("--fault-type requires --spice-base, --spice-mapping, and --output")
    elif any(value is not None for value in spice_options):
        parser.error("SPICE scenario options require --fault-type")
    if args.simulate and (not args.fault_type or not args.probes):
        parser.error("--simulate requires --fault-type and --probes")
    if args.probes and not args.simulate:
        parser.error("--probes requires --simulate")
    if args.assess_risk and (not args.simulate or not args.ratings):
        parser.error("--assess-risk requires --simulate and --ratings")
    if args.ratings and not args.assess_risk:
        parser.error("--ratings requires --assess-risk")

    circuit = load_netlist(args.netlist)
    candidates = direct_candidates(circuit, args.fault)
    net_types = json.loads(args.net_types.read_text(encoding="utf-8")) if args.net_types else None
    if net_types is not None and not isinstance(net_types, dict):
        parser.error("--net-types must contain a JSON object mapping net names to types")
    graph = build_graph(circuit, net_types)
    excluded_types = {
        item.strip().lower()
        for item in (args.exclude_net_types or "").split(",")
        if item.strip()
    }
    if excluded_types and not (args.show_paths or args.export_dot):
        parser.error("--exclude-net-types requires --show-paths or --export-dot")
    path_search = None
    if args.show_paths or args.export_dot:
        path_search = find_paths_result(
            circuit,
            args.fault,
            max_hops=args.max_hops,
            net_types=net_types,
            exclude_net_types=excluded_types,
        )

    output = {
        "failed_component": args.fault,
        "analysis_stage": "connectivity_only",
        "warning": "Connected components are not confirmed damaged. Fault type, electrical ratings and simulation are not yet evaluated.",
        "components_loaded": len(circuit.components),
        "candidates": [
            {
                "reference": candidate.target_ref,
                "shared_nets": list(candidate.shared_nets),
                "shared_net_types": {
                    net: str((net_types or {}).get(net, "unknown")).strip().lower()
                    for net in candidate.shared_nets
                },
                "evidence_level": candidate.evidence_level,
            }
            for candidate in candidates
        ],
    }
    if args.show_paths:
        output["max_hops"] = args.max_hops
        output["path_candidates"] = [
            {
                "reference": candidate.target_ref,
                "connection_type": candidate.connection_type,
                "hops": candidate.hops,
                "path": list(candidate.path),
                "nets": list(candidate.nets),
                "net_types": dict(candidate.net_types),
                "ground_only_path": candidate.ground_only_path,
                "evidence_level": candidate.evidence_level,
            }
            for candidate in (path_search.paths if path_search else [])
        ]
        output["path_search"] = {"status": path_search.status,
                                 "expanded_states": path_search.expanded_states,
                                 "peak_queue_size": path_search.peak_queue_size,
                                 "total_paths": path_search.total_paths,
                                 "limit_reason": path_search.limit_reason}
    if excluded_types:
        output["excluded_net_types"] = sorted(excluded_types)
        output["path_filter_note"] = (
            "Excluded net types only limit displayed traversal; this does not establish "
            "that excluded connections cannot carry fault effects."
        )
    if args.show_pins:
        output["component_details"] = [
            {
                "reference": component.ref,
                "value": component.value,
                "library_id": component.library_id,
                "footprint": component.footprint,
                "datasheet": component.datasheet,
                "nets": list(circuit.component_nets[component.ref]),
            }
            for component in circuit.components.values()
        ]
        output["pin_connections"] = [
            {"reference": item.ref, "pin": item.pin, "net": item.net, "net_code": item.net_code}
            for item in circuit.connections
        ]
    if args.export_dot:
        args.export_dot.write_text(to_dot(graph, args.fault, list(path_search.paths) if path_search else None), encoding="utf-8")

    if args.fault_type:
        mapping = json.loads(args.spice_mapping.read_text(encoding="utf-8"))
        if not isinstance(mapping, dict):
            parser.error("--spice-mapping must contain a JSON object keyed by component reference")
        parameters = {}
        if args.fault_value is not None:
            if args.fault_type == "resistance_drift":
                parameters["fault_resistance"] = args.fault_value
            elif args.fault_type in {"regulator_output_rise", "regulator_output_drop"}:
                parameters["output_voltage"] = args.fault_value
            else:
                parser.error(f"--fault-value is not used by {args.fault_type}")
        scenario = create_scenario(args.fault, args.fault_type, parameters)
        generated = generate_scenario_files(args.spice_base, [scenario], mapping, args.output)
        record = generated["scenarios"][0]
        output["analysis_stage"] = "fault_scenario_generated"
        output["fault_scenario"] = {
            **record["scenario"],
            "baseline_file": generated["baseline_path"],
            "faulted_netlist": generated["scenario_paths"][0],
            "metadata_file": generated["metadata_path"],
            "changes": record["changes"],
            "affected_candidates": [candidate.target_ref for candidate in candidates],
            "simulation_status": "not_run",
        }
        if args.simulate:
            probes = json.loads(args.probes.read_text(encoding="utf-8"))
            if not isinstance(probes, dict):
                parser.error("--probes must contain a JSON object")
            simulation_dir = args.output / "simulation_results" / scenario.scenario_id.lower()
            try:
                analysis = analyze_fault_impact(
                    circuit,
                    record["scenario"],
                    generated["baseline_path"],
                    generated["scenario_paths"][0],
                    mapping,
                    probes,
                    simulation_dir,
                )
            except Exception as exc:
                _update_scenario_metadata(
                    Path(generated["metadata_path"]), scenario.scenario_id, "failed", error=str(exc)
                )
                raise
            output["analysis_stage"] = "fault_impact_simulated"
            output["warning"] = "Simulated changes are not a damage assessment; component ratings were not evaluated."
            output["fault_scenario"]["simulation_status"] = "success"
            output["fault_scenario"]["simulation_analysis"] = analysis["report"]
            _update_scenario_metadata(
                Path(generated["metadata_path"]), scenario.scenario_id, "success",
                report=analysis["report"]["impact_report_json"],
            )
            if args.assess_risk:
                specifications = load_component_specs(args.ratings)
                risk_report = rank_impacts(
                    circuit,
                    args.fault,
                    scenario.fault_type,
                    scenario.scenario_id,
                    analysis["component_stress"],
                    specifications,
                    mapping,
                    simulation_dir,
                )
                output["analysis_stage"] = "secondary_stress_risk_assessed"
                output["warning"] = "Risk levels are deterministic inspection priorities based on configured ratings; they do not confirm damage or estimate failure probability."
                output["fault_scenario"]["risk_assessment"] = risk_report
                scenario_metadata = json.loads(Path(generated["metadata_path"]).read_text(encoding="utf-8"))
                scenario_metadata["scenarios"][0]["risk_assessment"] = risk_report["risk_assessment_json"]
                Path(generated["metadata_path"]).write_text(
                    json.dumps(scenario_metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
    print(json.dumps(output, ensure_ascii=False, indent=2))


def _update_scenario_metadata(path: Path, scenario_id: str, status: str, *, report: str | None = None, error: str | None = None) -> None:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    found = False
    for item in metadata.get("scenarios", []):
        if item.get("scenario", {}).get("scenario_id") == scenario_id:
            item["simulation_status"] = status
            if report:
                item["simulation_report"] = report
            if error:
                item["simulation_error"] = error
            found = True
    if not found:
        raise ValueError(f"Scenario {scenario_id} is missing from {path}")
    statuses = [item.get("simulation_status", "not_run") for item in metadata.get("scenarios", [])]
    if statuses and all(value == "success" for value in statuses):
        metadata["simulation_status"] = "success"
    elif not statuses or all(value == "not_run" for value in statuses):
        metadata["simulation_status"] = "not_run"
    elif all(value in {"success", "failed"} for value in statuses):
        metadata["simulation_status"] = "failed" if all(value == "failed" for value in statuses) else "partial_failure"
    else:
        metadata["simulation_status"] = "partial"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
