from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from circuitcare.dashboard.analysis_loader import load_analysis_snapshot
from circuitcare.dashboard.case_loader import DEMO_CASES
from circuitcare.models import Circuit, Component, Connection
from circuitcare.schematic.electrical_equivalence import verify_schematic_netlist
from circuitcare.schematic.exporter import export_schematic, find_kicad_cli
from circuitcare.schematic.geometry import build_symbol_geometry
from circuitcare.schematic.measurement_geometry import build_measurement_anchors
from circuitcare.services.simulation_service import _current_simulation_fingerprint
from circuitcare.workspace.simulation_context import verify_cached_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def test_missing_engine_fingerprint_fails_closed_without_raising():
    status, reasons = verify_cached_fingerprint(
        {"fingerprint_sha256": "a" * 64}, {"fingerprint_sha256": None})
    assert status == "PROVENANCE_UNVERIFIED"
    assert reasons == ["UNVERIFIED_SIMULATION_CACHE"]


@pytest.mark.integration
def test_divider_phase19_3_4_simulation_manifest_is_verified_and_tamper_is_stale(tmp_path):
    case = DEMO_CASES["Independent resistor divider (evaluation circuit A)"]
    result = ROOT / case["cached"]
    fingerprint = _current_simulation_fingerprint(case, "R1", "resistance_drift", "10k")
    manifest = json.loads((result / "simulation_context.json").read_text(encoding="utf-8"))
    expected_engine = manifest["fingerprint"]["simulation_settings"].get("ngspice_version")
    current_engine = fingerprint["simulation_settings"].get("ngspice_version")
    if current_engine != expected_engine:
        pytest.skip("NOT_VERIFIED: cached simulation is pinned to a different ngspice version")
    snapshot = load_analysis_snapshot(result_dir=result, current_fingerprint=fingerprint,
                                     live_used=False, live_error=None, cached_selection=True)
    assert snapshot.status == "CACHED_SIMULATION"
    assert snapshot.integrity == "VERIFIED"
    assert (snapshot.manifest["fingerprint"]["simulation_settings"]["ngspice_version"]
            == fingerprint["simulation_settings"]["ngspice_version"])
    assert snapshot.risk["scenario_id"] == snapshot.comparison["scenario_id"] == "FAULT_R1_RESISTANCE_DRIFT"
    levels = {item["component_ref"]: item["risk_level"]
              for item in snapshot.risk["secondary_candidates"]}
    assert levels["R2"] == "WITHIN_LIMIT"
    assert levels["J1"] == "UNKNOWN"

    copied = tmp_path / "stale"
    shutil.copytree(result, copied)
    target = copied / "risk_assessment.json"
    target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
    stale = load_analysis_snapshot(result_dir=copied, current_fingerprint=fingerprint,
                                   live_used=False, live_error=None, cached_selection=True)
    assert stale.status == stale.integrity == "STALE"
    assert stale.risk == {}


@pytest.mark.integration
def test_regulator_schematic_is_electrically_equivalent_and_has_verified_measurement_anchors(tmp_path):
    if not find_kicad_cli():
        pytest.skip("NOT_VERIFIED: KiCad CLI unavailable; fixture acceptance requires actual CLI validation")
    schematic = ROOT / "tests/fixtures/regulator.kicad_sch"
    source_text = schematic.read_text(encoding="utf-8")
    assert source_text.count("(wire ") >= 19
    assert source_text.count("(junction ") >= 7
    xml = ROOT / "tests/fixtures/fault_demo_netlist.xml"
    cache_base = ROOT / "data" / f"phase19_3_4_integration_{tmp_path.name}"
    equivalence = verify_schematic_netlist(schematic, xml, project_root=ROOT,
                                           cache_root=cache_base / "equivalence")
    assert equivalence.status == "VERIFIED_EQUIVALENT", equivalence.message

    exported = export_schematic(schematic, project_root=ROOT, cache_root=cache_base / "svg")
    assert exported.status == "EXPORTED", exported.message
    geometry = build_symbol_geometry(schematic, exported.svg_paths, project_root=ROOT)
    assert geometry.status == "VERIFIED_GEOMETRY", geometry.message
    assert {region.component_ref for region in geometry.regions} == {"J1", "U1", "R1", "RLOAD", "C1", "Q1"}

    circuit = Circuit(
        {"J1": Component("J1", "12V Input"), "U1": Component("U1", "Regulator"),
         "R1": Component("R1", "1k"), "RLOAD": Component("RLOAD", "1k"),
         "C1": Component("C1", "100uF"), "Q1": Component("Q1", "NMOS switch")},
        [Connection("J1", "1", "VIN"), Connection("U1", "1", "VIN"),
         Connection("Q1", "1", "VIN"), Connection("U1", "3", "VOUT"),
         Connection("R1", "1", "VOUT"), Connection("C1", "1", "VOUT"),
         Connection("R1", "2", "LOAD"), Connection("RLOAD", "1", "LOAD"),
         Connection("U1", "2", "GND"), Connection("C1", "2", "GND"),
         Connection("Q1", "2", "GND"), Connection("RLOAD", "2", "GND"),
         Connection("J1", "2", "GND")],
    )
    vout = build_measurement_anchors(node="VOUT", circuit=circuit, schematic_path=schematic,
                                    project_root=ROOT, geometry=geometry)
    load = build_measurement_anchors(node="LOAD", circuit=circuit, schematic_path=schematic,
                                    project_root=ROOT, geometry=geometry)
    assert vout.status == load.status == "VERIFIED_LOGICAL_NET_CANDIDATES"
    assert {(row["component_ref"], row["pin"]) for row in vout.anchors} == {
        ("U1", "3"), ("R1", "1"), ("C1", "1")}
    assert {(row["component_ref"], row["pin"]) for row in load.anchors} == {
        ("R1", "2"), ("RLOAD", "1")}
    shutil.rmtree(cache_base, ignore_errors=True)
