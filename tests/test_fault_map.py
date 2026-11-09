from __future__ import annotations

import json
from pathlib import Path

import pytest

from circuitcare.netlist import load_netlist
from circuitcare.models import Circuit, Component, Connection
from circuitcare.ui import ROOT, _run_diagnosis
from circuitcare.vision.overlay import render_detection_risk_overlay
from circuitcare.visualization.fault_map import build_fault_map, fault_map_json, render_fault_map_dot


def test_fault_map_keeps_connectivity_and_stress_evidence_separate() -> None:
    circuit = load_netlist(ROOT / "tests/fixtures/fault_demo_netlist.xml")
    risk = json.loads((ROOT / "data/fault_scenarios/overvoltage_demo/simulation_results/fault_u1_regulator_output_rise/risk_assessment.json").read_text())
    diagnosis = json.loads((ROOT / "data/diagnosis/prior_results/inspection_ranking.json").read_text())
    payload = build_fault_map(circuit, "U1", risk_assessment=risk, inspection_ranking=diagnosis,
                              selected_component="C1", view_mode="Electrical Stress View", max_hops=3)
    by_ref = {row["component_ref"]: row for row in payload["nodes"]}
    assert by_ref["U1"]["risk_level"] == "KNOWN_FAILED"
    assert by_ref["C1"]["risk_level"] == "EXCEEDED"
    assert by_ref["C1"]["connectivity"]["evidence_level"] == "connectivity_only"
    assert by_ref["J1"]["risk_level"] == "UNKNOWN"
    assert by_ref["C1"]["damage_confirmed"] is False
    dot = render_fault_map_dot(payload)
    assert "#E74C3C" in dot and "#9B59B6" in dot and "#D35400" in dot
    assert "Connectivity candidate map" in dot
    assert json.loads(fault_map_json(payload))["map_type"] == "fault_influence_candidate_map"


def test_indirect_selected_route_highlights_all_component_net_edges() -> None:
    circuit = Circuit(
        components={ref: Component(ref=ref, value=ref) for ref in ("F1", "A1", "B1")},
        connections=[Connection("F1", "1", "N1"), Connection("A1", "1", "N1"),
                     Connection("A1", "2", "N2"), Connection("B1", "1", "N2")],
    )
    payload = build_fault_map(circuit, "F1", selected_component="B1", view_mode="Connection View")
    route = next(row for row in payload["paths"] if row["target_ref"] == "B1")
    assert route["connection_type"] == "indirect"
    assert route["hops"] == 2
    assert route["path_nodes"] == ["F1", "N1", "A1", "N2", "B1"]
    dot = render_fault_map_dot(payload)
    assert dot.count('color="#D35400"') == 4


def test_what_if_is_virtual_and_changes_posterior_without_changing_saved_stress() -> None:
    stress_path = ROOT / "data/fault_scenarios/overvoltage_demo/simulation_results/fault_u1_regulator_output_rise/component_stress.json"
    original_stress = stress_path.read_bytes()
    high = _run_diagnosis([{"node": "VOUT", "value": 7.9, "unit": "V", "uncertainty": 0.2, "source": "simulated_demo"}])
    low = _run_diagnosis([{"node": "VOUT", "value": 2.0, "unit": "V", "uncertainty": 0.2, "source": "simulated_demo"}])
    assert high["diagnosis"]["used_measurements"][0]["source"] == "simulated_demo"
    assert high["diagnosis"]["top_hypothesis"]["hypothesis_id"] == "U1_OUTPUT_8V"
    assert low["diagnosis"]["top_hypothesis"]["hypothesis_id"] == "U1_OUTPUT_3V"
    assert low["diagnosis"]["model_mismatch"] is True
    assert high["inspection_ranking"]["rank_changes_after_measurements"]
    assert stress_path.read_bytes() == original_stress

    circuit = load_netlist(ROOT / "tests/fixtures/fault_demo_netlist.xml")
    risk = json.loads((ROOT / "data/fault_scenarios/overvoltage_demo/simulation_results/fault_u1_regulator_output_rise/risk_assessment.json").read_text())
    high_map = build_fault_map(circuit, "U1", risk_assessment=risk,
                               inspection_ranking=high["inspection_ranking"], selected_component="C1",
                               view_mode="Inspection Priority View")
    low_map = build_fault_map(circuit, "U1", risk_assessment=risk,
                              inspection_ranking=low["inspection_ranking"], selected_component="C1",
                              view_mode="Inspection Priority View")
    high_c1 = next(line for line in render_fault_map_dot(high_map).splitlines() if '"c:C1"' in line)
    low_c1 = next(line for line in render_fault_map_dot(low_map).splitlines() if '"c:C1"' in line)
    assert high_c1 != low_c1


def test_photo_overlay_keeps_electrical_color_and_what_if_priority_separate() -> None:
    from PIL import Image

    risk = json.loads((ROOT / "data/fault_scenarios/overvoltage_demo/simulation_results/fault_u1_regulator_output_rise/risk_assessment.json").read_text())
    ranking = _run_diagnosis([{"node": "VOUT", "value": 7.9, "unit": "V", "uncertainty": 0.2, "source": "simulated_demo"}])["inspection_ranking"]
    priorities = {row["component_ref"]: row for row in ranking["posterior_weighted_inspection_order"]}
    detections = [{"detection_id": "det_1", "class_name": "capacitor", "confidence": 0.9,
                   "bbox_xyxy": [10, 10, 40, 40]},
                  {"detection_id": "manual_1", "class_name": "resistor", "confidence": None,
                   "bbox_xyxy": [42, 10, 55, 30], "source": "user_manual_addition"}]
    image, meta = render_detection_risk_overlay(
        Image.new("RGB", (60, 60), "white"), detections,
        confirmed_references={"det_1": "C1"}, risk_assessment=risk,
        expected_fault="U1", expected_fault_type="regulator_output_rise",
        inspection_priority_by_reference=priorities,
    )
    assert image.getpixel((10, 10)) == (220, 38, 38)  # Existing electrical EXCEEDED color.
    assert meta["risk_mapped_count"] == 1
    assert meta["inspection_priority_count"] == 1
