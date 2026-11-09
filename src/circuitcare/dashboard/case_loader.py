"""Static demo circuit selections and their supported fault options."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEMO_HYPOTHESES = PROJECT_ROOT / "data/diagnosis/demo_hypotheses.json"

DEMO_CASES = {
    "Regulator + divider + switch (cached demo)": {
        "netlist": "tests/fixtures/fault_demo_netlist.xml", "fault": "U1", "fault_type": "regulator_output_rise", "value": "8",
        "fault_types": ["regulator_output_rise", "regulator_output_drop"],
        "schematic": "tests/fixtures/regulator.kicad_sch",
        "base": "data/circuits/fault_demo.cir", "mapping": "data/circuits/fault_demo_mapping.json", "probes": "data/circuits/fault_demo_probes.json", "ratings": "data/component_specs/fault_demo_ratings.json",
        "cached": "data/fault_scenarios/overvoltage_demo/simulation_results/fault_u1_regulator_output_rise", "prior": "data/diagnosis/prior_results",
    },
    "Independent resistor divider (evaluation circuit A)": {
        "netlist": "tests/fixtures/divider_netlist.xml", "fault": "R1", "fault_type": "resistance_drift", "value": "10k",
        "fault_types": ["resistance_drift", "resistor_open", "resistor_short"],
        "schematic": "tests/fixtures/divider.kicad_sch",
        "fingerprint_ngspice_version": True,
        "base": "data/circuits/divider.cir", "mapping": "data/circuits/divider_mapping.json", "probes": "data/circuits/divider_probes.json", "ratings": "data/component_specs/divider_ratings.json",
        "cached": "data/evaluations/divider/phase19_3_4_run/simulation_results/fault_r1_resistance_drift", "prior": None,
    },
    "Independent RC charging circuit (evaluation circuit B)": {
        "netlist": "tests/fixtures/rc_netlist.xml", "fault": "RCHARGE", "fault_type": "resistance_drift", "value": "100k",
        "fault_types": ["resistance_drift", "resistor_open", "resistor_short"],
        "base": "data/circuits/rc_charge.cir", "mapping": "data/circuits/rc_charge_mapping.json", "probes": "data/circuits/rc_charge_probes.json", "ratings": "data/component_specs/rc_charge_ratings.json",
        "cached": "data/evaluations/rc_charge/simulation_results/fault_rcharge_resistance_drift", "prior": None,
    },
}


UPLOAD_CHOICE = "KiCad XML 업로드 (연결 그래프 전용)"


def fault_options_for(case_name: str, case: dict, component_refs: list[str]) -> list[str]:
    configured = {
        "Regulator + divider + switch (cached demo)": ["U1"],
        "Independent resistor divider (evaluation circuit A)": ["R1", "R2"],
        "Independent RC charging circuit (evaluation circuit B)": ["RCHARGE", "RBLEED"],
    }
    return configured.get(case_name, sorted(component_refs))
