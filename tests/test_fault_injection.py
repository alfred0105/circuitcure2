import json
from pathlib import Path

import pytest

from circuitcare.fault_injector import inject_fault
from circuitcare.fault_models import create_scenario
from circuitcare.scenario_generator import generate_scenario_files


ROOT = Path(__file__).resolve().parents[1]
BASE_CIR = ROOT / "data" / "circuits" / "fault_demo.cir"
MAPPING_PATH = ROOT / "data" / "circuits" / "fault_demo_mapping.json"
MAPPING = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))


def test_resistor_faults_generate_copies_and_preserve_baseline(tmp_path):
    original = BASE_CIR.read_text(encoding="utf-8")
    scenarios = [
        ("resistor_open", {}, "R1 VOUT LOAD 1T"),
        ("resistor_short", {}, "R1 VOUT LOAD 1u"),
        ("resistance_drift", {"fault_resistance": "10k"}, "R1 VOUT LOAD 10k"),
    ]

    for fault_type, parameters, expected_line in scenarios:
        result = inject_fault(BASE_CIR, create_scenario("R1", fault_type, parameters), MAPPING)
        assert result.baseline_netlist == original
        assert expected_line in result.faulted_netlist
        assert result.faulted_netlist != original
        assert len(result.changes) == 1

    assert BASE_CIR.read_text(encoding="utf-8") == original


def test_regulator_and_switch_faults_are_explicit_and_reproducible():
    baseline = BASE_CIR.read_text(encoding="utf-8")
    cases = [
        ("U1", "regulator_output_rise", {"output_voltage": "8"}, "VREG VOUT 0 DC 8"),
        ("U1", "regulator_output_drop", {"output_voltage": "3"}, "VREG VOUT 0 DC 3"),
        ("U1", "regulator_output_loss", {}, "RFAULT_VREG VOUT 0 1T"),
        ("Q1", "stuck_open", {}, "VCTRL CTRL 0 DC 0"),
        ("Q1", "stuck_on", {}, "VCTRL CTRL 0 DC 5"),
    ]

    for reference, fault_type, parameters, expected_line in cases:
        scenario = create_scenario(reference, fault_type, parameters)
        first = inject_fault(baseline, scenario, MAPPING)
        second = inject_fault(baseline, scenario, MAPPING)
        assert expected_line in first.faulted_netlist
        assert first.faulted_netlist == second.faulted_netlist
        assert first.baseline_netlist == baseline
        assert first.changes


def test_mapping_is_required_and_scenario_files_include_metadata(tmp_path):
    scenarios = [
        create_scenario("R1", "resistor_open"),
        create_scenario("R1", "resistor_short"),
        create_scenario("R1", "resistance_drift", {"fault_resistance": "10k"}),
    ]
    with pytest.raises(ValueError, match="No explicit SPICE mapping"):
        inject_fault(BASE_CIR, scenarios[-1], {})

    generated = generate_scenario_files(BASE_CIR, scenarios, MAPPING, tmp_path / "generated")
    baseline = Path(generated["baseline_path"])
    metadata_path = Path(generated["metadata_path"])
    assert baseline.read_text(encoding="utf-8") == BASE_CIR.read_text(encoding="utf-8")
    assert len(generated["scenario_paths"]) == 3
    assert any("R1 VOUT LOAD 10k" in Path(path).read_text(encoding="utf-8") for path in generated["scenario_paths"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert len(metadata["scenarios"]) == 3
    assert all(record["simulation_status"] == "not_run" for record in metadata["scenarios"])
    opened = next(record for record in metadata["scenarios"] if record["scenario"]["fault_type"] == "resistor_open")
    assert opened["scenario"]["parameters"]["fault_resistance"] == "1T"
    drift = next(record for record in metadata["scenarios"] if record["scenario"]["fault_type"] == "resistance_drift")
    assert drift["scenario"]["assumptions"]
    assert drift["changes"][0]["spice_element"] == "R1"
