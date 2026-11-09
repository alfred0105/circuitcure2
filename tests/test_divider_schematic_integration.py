"""KiCad CLI integration checks for the checked-in Divider schematic fixture."""

import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from circuitcare.schematic.exporter import export_schematic, find_kicad_cli
from circuitcare.schematic.geometry import build_symbol_geometry

ROOT = Path(__file__).resolve().parents[1]


def _net_map(path: Path) -> dict[str, frozenset[tuple[str, str]]]:
    root = ET.parse(path).getroot()
    result = {}
    for net in root.findall("./nets/net"):
        name = (net.get("name") or "").removeprefix("/")
        pins = frozenset((node.get("ref", ""), node.get("pin", ""))
                         for node in net.findall("node"))
        result[name] = pins
    return result


@pytest.mark.integration
def test_divider_schematic_matches_xml_netlist(tmp_path):
    cli = find_kicad_cli()
    if not cli:
        pytest.skip("NOT_VERIFIED: KiCad CLI is unavailable")
    source = ROOT / "tests/fixtures/divider.kicad_sch"
    expected_xml = ROOT / "tests/fixtures/divider_netlist.xml"
    actual_xml = tmp_path / "divider-export.xml"
    process = subprocess.run(
        [cli, "sch", "export", "netlist", "--format", "kicadxml",
         "--output", str(actual_xml), str(source)],
        check=False, capture_output=True, text=True, timeout=30, close_fds=False,
    )
    assert process.returncode == 0, process.stderr
    assert actual_xml.is_file()

    expected = ET.parse(expected_xml).getroot()
    actual = ET.parse(actual_xml).getroot()
    expected_components = {item.get("ref"): item for item in expected.findall("./components/comp")}
    actual_components = {item.get("ref"): item for item in actual.findall("./components/comp")}
    assert set(actual_components) == set(expected_components)
    expected_libraries = {
        "J1": ("Connector_Generic", "Conn_01x02"),
        "R1": ("Device", "R"),
        "R2": ("Device", "R"),
    }
    for ref, old_component in expected_components.items():
        new_component = actual_components[ref]
        assert new_component.findtext("value") == old_component.findtext("value"), ref
        # The legacy XML has no libsource metadata, so that field is explicitly unknown there.
        old_lib = old_component.find("libsource")
        if old_lib is not None:
            assert new_component.find("libsource").attrib == old_lib.attrib, ref
        else:
            # The legacy XML omitted Library IDs; verify the intended fixture libraries explicitly.
            new_lib = new_component.find("libsource")
            assert (new_lib.get("lib"), new_lib.get("part")) == expected_libraries[ref]

    assert _net_map(actual_xml) == _net_map(expected_xml)
    cache_root = ROOT / "data" / f"schematic_integration_cache_{tmp_path.name}"
    try:
        exported = export_schematic(source, project_root=ROOT, cache_root=cache_root)
        assert exported.status == "EXPORTED", exported.message
        assert len(exported.svg_paths) == 1
        assert exported.svg_paths[0].is_file()
        geometry = build_symbol_geometry(source, exported.svg_paths, project_root=ROOT)
        assert geometry.status == "VERIFIED_GEOMETRY", geometry.message
        assert geometry.view_box == (0.0, 0.0, 297.0022, 210.0072)
        region_refs = {region.component_ref for region in geometry.regions}
        assert region_refs == {"J1", "R1", "R2"}
        svg_root = ET.parse(exported.svg_paths[0]).getroot()
        svg_rects = [element.attrib for element in svg_root.iter()
                     if element.tag.rsplit("}", 1)[-1] == "rect"]
        assert {("98.984000", "71.270000", "2.032000", "5.080000"),
                ("118.984000", "78.890000", "2.032000", "5.080000")} <= {
                    (rect.get("x"), rect.get("y"), rect.get("width"), rect.get("height"))
                    for rect in svg_rects
                }
        svg_paths = [element.get("d", "") for element in svg_root.iter()
                     if element.tag.rsplit("}", 1)[-1] == "path"]
        assert any("M100.0000 71.2700 L100.0000 70.0000" in path for path in svg_paths)
        assert any("M120.0000 78.8900 L120.0000 77.6200" in path for path in svg_paths)
    finally:
        if cache_root.exists():
            shutil.rmtree(cache_root)
