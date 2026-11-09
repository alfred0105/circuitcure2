from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from circuitcare.models import Circuit, Component, Connection
from circuitcare.schematic.exporter import export_schematic, find_kicad_cli
from circuitcare.schematic.geometry import GeometryResult, build_symbol_geometry
from circuitcare.schematic.measurement_geometry import build_measurement_anchors

ROOT = Path(__file__).resolve().parents[1]


def _divider() -> Circuit:
    return Circuit(
        {"J1": Component("J1", "5V input"), "R1": Component("R1", "1k"),
         "R2": Component("R2", "1k")},
        [Connection("J1", "1", "VIN"), Connection("R1", "1", "VIN"),
         Connection("R1", "2", "VOUT"), Connection("R2", "1", "VOUT"),
         Connection("J1", "2", "GND"), Connection("R2", "2", "GND")],
    )


@pytest.mark.integration
def test_divider_net_anchors_match_real_kicad_svg_pin_endpoints(tmp_path):
    if not find_kicad_cli():
        pytest.skip("NOT_VERIFIED: KiCad CLI is unavailable")
    source = ROOT / "tests/fixtures/divider.kicad_sch"
    exported = export_schematic(source, project_root=ROOT, cache_root=ROOT / "data/schematic_svg_cache")
    assert exported.status == "EXPORTED", exported.message
    geometry = build_symbol_geometry(source, exported.svg_paths, project_root=ROOT)
    result = build_measurement_anchors(node="VOUT", circuit=_divider(), schematic_path=source,
                                       project_root=ROOT, geometry=geometry)
    assert result.status == "VERIFIED_LOGICAL_NET_CANDIDATES", result.message
    anchors = {(item["component_ref"], item["pin"]): (item["x"], item["y"])
               for item in result.anchors}
    assert anchors[("R1", "2")] == pytest.approx((100.0, 77.62))
    assert anchors[("R2", "1")] == pytest.approx((120.0, 77.62))

    svg = ET.parse(exported.svg_paths[0]).getroot()
    path_data = [element.attrib.get("d", "") for element in svg.iter()
                 if element.tag.rsplit("}", 1)[-1] == "path"]
    assert any("M100.0000 76.3500 L100.0000 77.6200" in data for data in path_data)
    assert any("M120.0000 78.8900 L120.0000 77.6200" in data for data in path_data)


def test_unverified_geometry_and_missing_or_ambiguous_pin_fail_closed(tmp_path):
    source = ROOT / "tests/fixtures/divider.kicad_sch"
    circuit = _divider()
    unverified = build_measurement_anchors(node="VOUT", circuit=circuit,
                                           schematic_path=source, project_root=ROOT,
                                           geometry=build_symbol_geometry(source, [], project_root=ROOT))
    assert unverified.status == "UNVERIFIED_MEASUREMENT_LOCATION"
    assert unverified.anchors == ()

    fake_geometry = GeometryResult("VERIFIED_GEOMETRY", "synthetic test geometry", (0, 0, 297, 210))
    missing = build_measurement_anchors(node="NO_NODE", circuit=circuit,
                                        schematic_path=source, project_root=ROOT,
                                        geometry=fake_geometry)
    assert missing.status == "UNVERIFIED_MEASUREMENT_LOCATION"
    bad_pin = Circuit(circuit.components, [Connection("R1", "99", "VOUT")])
    ambiguous = build_measurement_anchors(node="VOUT", circuit=bad_pin,
                                          schematic_path=source, project_root=ROOT,
                                          geometry=fake_geometry)
    assert ambiguous.status == "UNVERIFIED_MEASUREMENT_LOCATION"
