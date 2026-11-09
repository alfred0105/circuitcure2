#!/usr/bin/env python3
"""Split the existing Phase 15 development manifest into public and truth files.

This migration does not relabel the three reused examples as independent holdout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from circuitcare.netlist import load_netlist

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evaluation/fault_hypothesis/manifest.json"
OUTPUT = ROOT / "evaluation/phase17/development"


def main() -> int:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    public_cases, ground_truth = [], {}
    topologies = {}
    for index, case in enumerate(source.get("cases", []), 1):
        case_id = f"p17d_{index:04d}"
        paths = case["circuit"]
        topology = _topology_group(paths["spice"])
        topologies[case_id] = topology
        public_cases.append({"case_id": case_id, "llm_case_id": f"case_{index:04d}",
            "topology_group": topology, "split": "development_synthetic",
            "circuit": paths,
            "diagnostic_input": {"circuit_id": f"case_{index:04d}", "components": _public_components(paths),
                "observations": case.get("observations", []),
                "symptoms": "Circuit output measurements differ from expected behavior."}})
        ground_truth[case_id] = case.get("ground_truth", [])
    out_public = OUTPUT / "public_cases.json"
    out_truth = OUTPUT / "private_ground_truth.json"
    out_manifest = OUTPUT / "split_manifest.json"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    public_bytes = _encode({"dataset_id": "circuitcare-phase17-reused-development-v1", "cases": public_cases})
    truth_bytes = _encode({"cases": ground_truth})
    out_public.write_bytes(public_bytes)
    out_truth.write_bytes(truth_bytes)
    manifest = {"dataset_id": "circuitcare-phase17-reused-development-v1", "version": "1.0.0",
        "split": "development_synthetic", "independent_holdout": False,
        "provenance": "Transformed from the already used Phase 15 three-case development manifest.",
        "case_ids": [case["case_id"] for case in public_cases], "topology_groups": topologies,
        "asset_hashes": {path: _hash((ROOT / path).read_bytes())
            for path in sorted({path for case in public_cases for path in case["circuit"].values()})},
        "hashes": {"public_cases": _hash(public_bytes), "private_ground_truth": _hash(truth_bytes)}}
    out_manifest.write_bytes(_encode(manifest))
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), "cases": len(public_cases),
        "topology_groups": len(set(topologies.values())), "independent_holdout": False,
        "public_sha256": manifest["hashes"]["public_cases"],
        "ground_truth_sha256": manifest["hashes"]["private_ground_truth"]}, indent=2))
    return 0


def _public_components(paths):
    circuit = load_netlist(ROOT / paths["netlist"])
    mapping = json.loads((ROOT / paths["mapping"]).read_text(encoding="utf-8"))
    return [{"ref": item.ref, "value": item.value, "library_id": item.library_id,
             "nets": list(circuit.component_nets.get(item.ref, ())),
             "mapping_available": item.ref in mapping,
             "model_type": mapping.get(item.ref, {}).get("model_type")}
            for item in circuit.components.values()]


def _topology_group(spice_path):
    name = Path(spice_path).stem.casefold()
    return {"fault_demo": "phase15_regulator_demo", "divider": "phase15_divider_demo",
            "rc_charge": "phase15_rc_demo"}.get(name, f"reused_{name}")


def _encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _hash(value):
    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
