#!/usr/bin/env python3
"""Create a new, topology-grouped synthetic holdout after benchmark-code freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

from circuitcare.fault_models import create_scenario
from circuitcare.fault_injector import inject_fault
from circuitcare.benchmark.freeze import benchmark_code_fingerprint, benchmark_sources_match_commit
from circuitcare.spice_runner import ngspice_available, run_ngspice

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evaluation/phase17/holdout"

TOPOLOGIES = [
    {"name": "topology_01", "components": [("J1", "5V source"), ("R1", "1k"), ("R2", "2k"), ("R3", "1k"), ("R4", "2k"), ("C1", "10uF")],
     "nets": {"VIN": [("J1", "1"), ("R1", "1")], "N1": [("R1", "2"), ("R2", "1"), ("R3", "1")],
       "VOUT": [("R3", "2"), ("R4", "1"), ("C1", "1")], "GND": [("J1", "2"), ("R2", "2"), ("R4", "2"), ("C1", "2")]},
     "spice": "V1 VIN 0 DC 5\nR1 VIN N1 1k\nR2 N1 0 2k\nR3 N1 VOUT 1k\nR4 VOUT 0 2k\nC1 VOUT 0 10u\n.tran 10u 10m\n.end\n",
     "mapping": {"R1": {"spice_element": "R1", "model_type": "resistor"}, "R2": {"spice_element": "R2", "model_type": "resistor"}, "R3": {"spice_element": "R3", "model_type": "resistor"}, "R4": {"spice_element": "R4", "model_type": "resistor"}},
     "observations": ["VOUT"]},
    {"name": "topology_02", "components": [("J1", "5V pulse"), ("R1", "2.2k"), ("C1", "4.7uF"), ("R2", "3.3k"), ("C2", "10uF"), ("R3", "100k")],
     "nets": {"VIN": [("J1", "1"), ("R1", "1")], "N1": [("R1", "2"), ("C1", "1"), ("R2", "1")],
       "VOUT": [("R2", "2"), ("C2", "1"), ("R3", "1")], "GND": [("J1", "2"), ("C1", "2"), ("C2", "2"), ("R3", "2")]},
     "spice": "V1 VIN 0 PULSE(0 5 0 1u 1u 10m 20m)\nR1 VIN N1 2.2k\nC1 N1 0 4.7u\nR2 N1 VOUT 3.3k\nC2 VOUT 0 10u\nR3 VOUT 0 100k\n.tran 20u 8m\n.end\n",
     "mapping": {"R1": {"spice_element": "R1", "model_type": "resistor"}, "R2": {"spice_element": "R2", "model_type": "resistor"}, "R3": {"spice_element": "R3", "model_type": "resistor"}},
     "observations": ["N1", "VOUT"]},
    {"name": "topology_03", "components": [("J1", "5V pulse"), ("L1", "10mH"), ("R1", "330"), ("C1", "22uF"), ("RLOAD", "1k")],
     "nets": {"VIN": [("J1", "1"), ("L1", "1")], "N1": [("L1", "2"), ("R1", "1")],
       "VOUT": [("R1", "2"), ("C1", "1"), ("RLOAD", "1")], "GND": [("J1", "2"), ("C1", "2"), ("RLOAD", "2")]},
     "spice": "V1 VIN 0 PULSE(0 5 0 1u 1u 20m 40m)\nL1 VIN N1 10m\nR1 N1 VOUT 330\nC1 VOUT 0 22u\nRLOAD VOUT 0 1k\n.tran 20u 8m\n.end\n",
     "mapping": {"R1": {"spice_element": "R1", "model_type": "resistor"}, "RLOAD": {"spice_element": "RLOAD", "model_type": "resistor"}},
     "observations": ["N1", "VOUT"]},
    {"name": "topology_04", "components": [("J1", "5V source"), ("R1", "1k"), ("R2", "2.2k"), ("R3", "1.5k"), ("R4", "2.7k"), ("R5", "820")],
     "nets": {"VIN": [("J1", "1"), ("R1", "1"), ("R3", "1")], "LEFT": [("R1", "2"), ("R2", "1"), ("R5", "1")],
       "RIGHT": [("R3", "2"), ("R4", "1"), ("R5", "2")], "GND": [("J1", "2"), ("R2", "2"), ("R4", "2")]},
     "spice": "V1 VIN 0 DC 5\nR1 VIN LEFT 1k\nR2 LEFT 0 2.2k\nR3 VIN RIGHT 1.5k\nR4 RIGHT 0 2.7k\nR5 LEFT RIGHT 820\n.tran 10u 5m\n.end\n",
     "mapping": {key: {"spice_element": key, "model_type": "resistor"} for key in ("R1", "R2", "R3", "R4", "R5")},
     "observations": ["LEFT", "RIGHT"]},
    {"name": "topology_05", "components": [("J1", "12V source"), ("U1", "5V regulator"), ("R1", "470"), ("R2", "1k"), ("C1", "100uF")],
     "nets": {"VIN": [("J1", "1"), ("U1", "1")], "VOUT": [("U1", "3"), ("R1", "1"), ("C1", "1")],
       "LOAD": [("R1", "2"), ("R2", "1")], "GND": [("J1", "2"), ("U1", "2"), ("R2", "2"), ("C1", "2")]},
     "spice": "V1 VIN 0 DC 12\nVREG VOUT 0 DC 5\nR1 VOUT LOAD 470\nR2 LOAD 0 1k\nC1 VOUT 0 100u\n.tran 10u 5m\n.end\n",
     "mapping": {"U1": {"spice_element": "VREG", "model_type": "regulator_output"}, "R1": {"spice_element": "R1", "model_type": "resistor"}, "R2": {"spice_element": "R2", "model_type": "resistor"}},
     "observations": ["VOUT", "LOAD"]},
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace only this generated holdout directory.")
    parser.add_argument("--output", default="evaluation/phase17/holdout")
    args = parser.parse_args()
    if not ngspice_available():
        raise SystemExit("ngspice is required to generate this synthetic holdout; no data was written.")
    code_sha = _git_sha()
    if not code_sha or not benchmark_sources_match_commit(ROOT, code_sha):
        raise SystemExit("Benchmark source files must be committed and unchanged before holdout generation.")
    output_dir = ROOT / args.output
    generated_files = ("public_cases.json", "private_ground_truth.json", "split_manifest.json")
    if output_dir.exists() and any((output_dir / name).exists() for name in generated_files) and not args.force:
        raise SystemExit(f"Refusing to overwrite existing evaluation data at {output_dir}; choose a new output directory.")
    if output_dir == DEFAULT_OUTPUT and args.force:
        raise SystemExit("The published holdout is immutable. Choose a new output directory to rebuild it.")
    output_dir.mkdir(parents=True, exist_ok=True)
    public_cases, private_truth = [], {}
    for topology_index, topology in enumerate(TOPOLOGIES, 1):
        folder = output_dir / "circuits" / topology["name"]
        folder.mkdir(parents=True, exist_ok=True)
        netlist_path = folder / "netlist.xml"
        spice_path = folder / "baseline.cir"
        mapping_path = folder / "mapping.json"
        probes_path = folder / "probes.json"
        netlist_path.write_bytes(_netlist_xml(topology["components"], topology["nets"]))
        spice_path.write_text(topology["spice"], encoding="utf-8")
        mapping_path.write_text(json.dumps(topology["mapping"], indent=2) + "\n", encoding="utf-8")
        probes = {"voltage_nodes": topology["observations"], "components": []}
        probes_path.write_text(json.dumps(probes, indent=2) + "\n", encoding="utf-8")
        from circuitcare.netlist import load_netlist
        circuit = load_netlist(netlist_path)
        component_rows = [{"ref": item.ref, "value": item.value, "library_id": item.library_id,
            "nets": list(circuit.component_nets.get(item.ref, ())), "mapping_available": item.ref in topology["mapping"],
            "model_type": topology["mapping"].get(item.ref, {}).get("model_type")}
            for item in circuit.components.values()]
        for case_offset, (reference, fault_type, params) in enumerate(_faults(topology), 1):
            case_number = (topology_index - 1) * 2 + case_offset
            case_id = f"p17h_{case_number:04d}"
            scenario = create_scenario(reference, fault_type, params)
            injected = inject_fault(topology["spice"], scenario, topology["mapping"])
            with tempfile.TemporaryDirectory(prefix="circuitcare-p17-truth-") as work:
                faulted = Path(work) / "faulted.cir"
                faulted.write_text(injected.faulted_netlist, encoding="utf-8")
                sim_output = Path(work) / "out"
                sim = run_ngspice(faulted, probes, topology["mapping"], sim_output, "truth")
                observations = []
                for node in topology["observations"]:
                    values = sim.columns.get(f"voltage:{node}")
                    if not values:
                        raise RuntimeError(f"Missing simulated observation for node {node} in {case_id}")
                    observations.append({"node": node, "value": float(values[-1]), "unit": "V",
                        "uncertainty": 0.03, "source": "synthetic_fault_simulation", "measurement_mode": "final_sample"})
            relative = lambda path: str(path.relative_to(ROOT))
            public_cases.append({"case_id": case_id, "llm_case_id": f"case_{1000 + case_number:04d}",
                "topology_group": topology["name"], "split": "independent_synthetic_holdout",
                "circuit": {"netlist": relative(netlist_path), "spice": relative(spice_path),
                    "mapping": relative(mapping_path), "probes": relative(probes_path)},
                "diagnostic_input": {"circuit_id": f"case_{1000 + case_number:04d}",
                    "components": component_rows, "observations": observations,
                    "symptoms": "Measured circuit response differs from normal behavior."}})
            private_truth[case_id] = [{"component_ref": reference, "fault_type": fault_type, "parameters": params}]
    public_bytes = _encode({"dataset_id": "circuitcare-phase17-independent-synthetic-v1", "cases": public_cases})
    truth_bytes = _encode({"cases": private_truth})
    public_path, truth_path = output_dir / "public_cases.json", output_dir / "private_ground_truth.json"
    public_path.write_bytes(public_bytes)
    truth_path.write_bytes(truth_bytes)
    group_ids = sorted({case["topology_group"] for case in public_cases})
    asset_paths = sorted({path for case in public_cases for path in case["circuit"].values()})
    manifest = {"dataset_id": "circuitcare-phase17-independent-synthetic-v1", "version": "1.0.0",
        "split": "independent_synthetic_holdout", "independent_holdout": True,
        "data_type": "synthetic_spice_injected_not_physical_repair", "case_count": len(public_cases),
        "topology_group_count": len(group_ids), "topology_groups": group_ids,
        "case_ids": [case["case_id"] for case in public_cases],
        "code_freeze_sha": code_sha, "builder": "scripts/build_phase17_synthetic_holdout.py",
        "benchmark_code_fingerprint": benchmark_code_fingerprint(ROOT),
        "asset_hashes": {path: _sha((ROOT / path).read_bytes()) for path in asset_paths},
        "hashes": {"public_cases": _sha(public_bytes), "private_ground_truth": _sha(truth_bytes)}}
    (output_dir / "split_manifest.json").write_bytes(_encode(manifest))
    print(json.dumps({"output": str(output_dir.relative_to(ROOT)), "cases": len(public_cases),
        "topology_groups": len(group_ids), "independent_synthetic_holdout": True,
        "physical_repair_data": False, "code_freeze_sha": manifest["code_freeze_sha"]}, indent=2))
    return 0


def _faults(topology):
    if topology["name"] == "topology_05":
        return [("U1", "regulator_output_rise", {"output_voltage": 6}),
                ("U1", "regulator_output_drop", {"output_voltage": 4})]
    resistors = [ref for ref, value in topology["components"] if ref.startswith("R")]
    ref = resistors[0]
    baseline = _spice_value(next(line.split()[3] for line in topology["spice"].splitlines() if line.split()[0] == ref))
    return [(ref, "resistance_drift", {"fault_resistance": baseline * factor}) for factor in (2, 0.5)]


def _spice_value(value):
    import re
    match = re.fullmatch(r"([0-9.]+)([A-Za-z]*)", value)
    if not match:
        return float(value)
    scale = {"": 1, "K": 1e3, "MEG": 1e6, "M": 1e-3, "U": 1e-6, "N": 1e-9}
    return float(match.group(1)) * scale[match.group(2).upper()]


def _netlist_xml(components, nets):
    root = Element("export", version="E")
    comps_el = SubElement(root, "components")
    for ref, value in components:
        comp = SubElement(comps_el, "comp", ref=ref)
        SubElement(comp, "value").text = value
    nets_el = SubElement(root, "nets")
    for code, (name, nodes) in enumerate(nets.items(), 1):
        net = SubElement(nets_el, "net", code=str(code), name=name)
        for ref, pin in nodes:
            SubElement(net, "node", ref=ref, pin=pin)
    return b'<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(root, encoding="utf-8") + b"\n"


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
