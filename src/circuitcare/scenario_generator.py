"""Write immutable baseline and faulted SPICE scenario files."""

import json
import re
from pathlib import Path
from typing import Any

from .fault_injector import inject_fault
from .fault_models import FaultScenario


def generate_scenario_files(
    base_netlist: str | Path,
    scenarios: list[FaultScenario],
    component_mapping: dict[str, dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Copy the baseline and generate one deterministic .cir per scenario."""
    base_path = Path(base_netlist)
    baseline = base_path.read_text(encoding="utf-8")
    directory = Path(output_dir)
    baseline_path = directory / "baseline.cir"
    if base_path.resolve() == baseline_path.resolve():
        raise ValueError("Output baseline.cir must not overwrite the input SPICE circuit")
    directory.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(baseline, encoding="utf-8")

    scenario_records = []
    used_filenames: set[str] = set()
    for scenario in scenarios:
        filename = _scenario_filename(scenario.scenario_id)
        if filename in used_filenames:
            raise ValueError(f"Scenario IDs produce a duplicate output filename: {filename}")
        used_filenames.add(filename)
        result = inject_fault(baseline, scenario, component_mapping)
        scenario_path = directory / f"{filename}.cir"
        scenario_path.write_text(result.faulted_netlist, encoding="utf-8")
        scenario_data = scenario.to_dict()
        if scenario.fault_type in {"stuck_open", "stuck_on"}:
            switch_mapping = component_mapping[scenario.component_ref]
            control_key = "open_control_voltage" if scenario.fault_type == "stuck_open" else "closed_control_voltage"
            scenario_data["parameters"] = {
                **scenario.parameters,
                "control_source": switch_mapping["control_source"],
                "switch_threshold": switch_mapping["switch_threshold"],
                "control_voltage": switch_mapping[control_key],
            }
        scenario_records.append({
            "scenario": scenario_data,
            "netlist_file": scenario_path.name,
            "changes": [change.to_dict() for change in result.changes],
            "simulation_status": "not_run",
        })

    metadata = {
        "baseline_file": baseline_path.name,
        "simulation_status": "not_run",
        "scenarios": scenario_records,
    }
    (directory / "scenarios.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "baseline_path": str(baseline_path),
        "scenario_paths": [str(directory / record["netlist_file"]) for record in scenario_records],
        "metadata_path": str(directory / "scenarios.json"),
        "scenarios": scenario_records,
    }


def _scenario_filename(scenario_id: str) -> str:
    filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", scenario_id).strip("._-").lower()
    if not filename:
        raise ValueError("scenario_id must contain a letter or number")
    return filename
