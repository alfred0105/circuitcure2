#!/usr/bin/env python3
"""Build a new off-grid synthetic dataset on previously used topology templates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from circuitcare.benchmark.freeze import benchmark_code_fingerprint, benchmark_sources_match_commit
from circuitcare.benchmark.integrity import audit_public_payload
from circuitcare.fault_ai.evaluation import build_blind_diagnostic_input
from circuitcare.fault_injector import inject_fault
from circuitcare.fault_models import create_scenario
from circuitcare.netlist import load_netlist
from circuitcare.spice_runner import ngspice_available, run_ngspice
from build_phase17_synthetic_holdout import TOPOLOGIES, _netlist_xml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="evaluation/phase17_1/off_grid_v2")
    args = parser.parse_args()
    if not ngspice_available():
        raise SystemExit("ngspice is required; no off-grid dataset was written.")
    output = ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite an existing evaluation dataset: {output}")
    freeze_sha = _git_sha()
    if not freeze_sha or not benchmark_sources_match_commit(ROOT, freeze_sha):
        raise SystemExit("Benchmark code must be committed and unchanged before off-grid data generation.")
    output.mkdir(parents=True, exist_ok=True)
    public, private = [], {}
    for topology_index, topology in enumerate(TOPOLOGIES, 1):
        folder = output / "circuits" / topology["name"]
        folder.mkdir(parents=True, exist_ok=True)
        netlist_path, spice_path = folder / "netlist.xml", folder / "baseline.cir"
        mapping_path, probes_path = folder / "mapping.json", folder / "probes.json"
        netlist_path.write_bytes(_netlist_xml(topology["components"], topology["nets"]))
        spice_path.write_text(topology["spice"], encoding="utf-8")
        mapping_path.write_text(json.dumps(topology["mapping"], indent=2) + "\n", encoding="utf-8")
        probes = {"voltage_nodes": topology["observations"], "components": []}
        probes_path.write_text(json.dumps(probes, indent=2) + "\n", encoding="utf-8")
        circuit = load_netlist(netlist_path)
        components = [{"ref": item.ref, "value": item.value, "library_id": item.library_id,
                       "nets": list(circuit.component_nets.get(item.ref, ())),
                       "mapping_available": item.ref in topology["mapping"],
                       "model_type": topology["mapping"].get(item.ref, {}).get("model_type")}
                      for item in circuit.components.values()]
        faults = _off_grid_faults(topology)
        for offset, (ref, fault_type, params) in enumerate(faults, 1):
            number = (topology_index - 1) * 2 + offset
            case_id = f"p17o_{number:04d}"
            injected = inject_fault(topology["spice"], create_scenario(ref, fault_type, params), topology["mapping"])
            with tempfile.TemporaryDirectory(prefix="circuitcare-p17-offgrid-") as work:
                faulted_path = Path(work) / "off-grid-fault.cir"
                faulted_path.write_text(injected.faulted_netlist, encoding="utf-8")
                sim = run_ngspice(faulted_path, probes, topology["mapping"], Path(work) / "output", case_id)
                observations = []
                for node in topology["observations"]:
                    values = sim.columns.get(f"voltage:{node}")
                    if not values:
                        raise RuntimeError(f"Missing simulated observation {node} for {case_id}")
                    observations.append({"node": node, "value": float(values[-1]), "unit": "V",
                                         "uncertainty": 0.03, "source": "synthetic_off_grid_fault_simulation",
                                         "measurement_mode": "final_sample"})
            rel = lambda path: str(path.relative_to(ROOT))
            llm_case_id = f"case_{9000 + number:04d}"
            diagnostic_input = {"circuit_id": llm_case_id, "components": components,
                               "observations": observations,
                               "symptoms": "Measured circuit response differs from nominal behavior."}
            blind = build_blind_diagnostic_input({"llm_case_id": llm_case_id},
                components=components, observations=observations, executable_catalog=[])
            audit_public_payload(blind)
            public.append({"case_id": case_id, "llm_case_id": llm_case_id,
                "topology_group": topology["name"], "split": "off_grid_parameter_reused_topology",
                "circuit": {"netlist": rel(netlist_path), "spice": rel(spice_path),
                            "mapping": rel(mapping_path), "probes": rel(probes_path)},
                "diagnostic_input": diagnostic_input})
            private[case_id] = [{"component_ref": ref, "fault_type": fault_type, "parameters": params}]

    dataset_id = "circuitcare-phase17-1-off-grid-v2"
    public_bytes = _encode({"dataset_id": dataset_id, "cases": public})
    truth_bytes = _encode({"cases": private})
    public_path, truth_path = output / "public_cases.json", output / "private_ground_truth.json"
    public_path.write_bytes(public_bytes)
    truth_path.write_bytes(truth_bytes)
    assets = sorted({path for case in public for path in case["circuit"].values()})
    groups = sorted({case["topology_group"] for case in public})
    manifest = {"dataset_id": dataset_id, "version": "2.0.0",
        "split": "off_grid_parameter_reused_topology", "off_grid": True, "independent_holdout": False,
        "data_type": "synthetic_spice_injected_off_grid_parameters",
        "topology_source": "Phase 17 synthetic topology templates; topology groups reused, not independently novel",
        "case_count": len(public), "topology_group_count": len(groups), "topology_groups": groups,
        "case_ids": [case["case_id"] for case in public], "code_freeze_sha": freeze_sha,
        "builder": "scripts/build_phase17_1_off_grid.py", "benchmark_code_fingerprint": benchmark_code_fingerprint(ROOT),
        "asset_hashes": {path: _sha((ROOT / path).read_bytes()) for path in assets},
        "hashes": {"public_cases": _sha(public_bytes), "private_ground_truth": _sha(truth_bytes)},
        "fault_grid_policy": "resistor drift factors 1.37 and 0.73; regulator output offsets +0.7 V and -0.7 V, outside Phase 17 rule grid"}
    (output / "split_manifest.json").write_bytes(_encode(manifest))
    print(json.dumps({"dataset_id": manifest["dataset_id"], "cases": len(public),
        "topology_groups": len(groups), "new_topologies": 0, "off_grid": True,
        "code_freeze_sha": freeze_sha, "public_hash": manifest["hashes"]["public_cases"],
        "ground_truth_hash": manifest["hashes"]["private_ground_truth"]}, indent=2))
    return 0


def _off_grid_faults(topology):
    if topology["name"] == "topology_05":
        return [("U1", "regulator_output_rise", {"output_voltage": 5.7}),
                ("U1", "regulator_output_drop", {"output_voltage": 4.3})]
    reference = next(ref for ref, _ in topology["components"] if ref.startswith("R"))
    baseline = _resistance(topology["spice"], reference)
    return [(reference, "resistance_drift", {"fault_resistance": baseline * factor})
            for factor in (1.37, 0.73)]


def _resistance(spice, reference):
    for line in spice.splitlines():
        parts = line.split()
        if parts and parts[0].casefold() == reference.casefold():
            text = parts[3]
            scale = {"": 1, "K": 1e3, "MEG": 1e6, "M": 1e-3, "U": 1e-6, "N": 1e-9}
            import re
            match = re.fullmatch(r"([0-9.]+)([A-Za-z]*)", text)
            if not match:
                return float(text)
            return float(match.group(1)) * scale[match.group(2).upper()]
    raise ValueError(f"Missing resistor {reference}")


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


if __name__ == "__main__":
    raise SystemExit(main())
