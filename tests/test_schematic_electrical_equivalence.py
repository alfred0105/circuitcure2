from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from circuitcare.schematic import electrical_equivalence as ee

ROOT = Path(__file__).resolve().parents[1]
VALID_XML = '''<export><components>
<comp ref="J1"><value>5V input</value></comp><comp ref="R1"><value>1k</value></comp><comp ref="R2"><value>1k</value></comp>
</components><nets>
<net code="1" name="VIN"><node ref="J1" pin="1"/><node ref="R1" pin="1"/></net>
<net code="2" name="VOUT"><node ref="R1" pin="2"/><node ref="R2" pin="1"/></net>
<net code="3" name="GND"><node ref="J1" pin="2"/><node ref="R2" pin="2"/></net>
</nets></export>'''


def _project(tmp_path: Path, xml: str = VALID_XML) -> tuple[Path, Path]:
    source = tmp_path / "divider.kicad_sch"
    shutil.copyfile(ROOT / "tests/fixtures/divider.kicad_sch", source)
    netlist = tmp_path / "analysis.xml"
    netlist.write_text(xml, encoding="utf-8")
    return source, netlist


def _mock_cli(monkeypatch, generated_xml: str):
    calls: list[list[str]] = []
    monkeypatch.setattr(ee, "find_kicad_cli", lambda: "/fake/kicad-cli")

    def run(args, **kwargs):
        calls.append(list(args))
        if args[1] == "version":
            return SimpleNamespace(stdout="KiCad 10.0.6", stderr="", returncode=0)
        output = Path(args[args.index("--output") + 1])
        output.write_text(generated_xml, encoding="utf-8")
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(ee.subprocess, "run", run)
    return calls


def test_equivalent_connectivity_normalizes_only_kicad_root_net_separator(tmp_path, monkeypatch):
    source, xml = _project(tmp_path)
    generated = VALID_XML.replace('name="VIN"', 'name="/VIN"').replace('name="VOUT"', 'name="/VOUT"').replace('name="GND"', 'name="/GND"')
    calls = _mock_cli(monkeypatch, generated)
    result = ee.verify_schematic_netlist(source, xml, project_root=tmp_path, cache_root=tmp_path / "cache")
    assert result.status == "VERIFIED_EQUIVALENT"
    assert len(result.unknown_metadata) == 6
    cached = ee.verify_schematic_netlist(source, xml, project_root=tmp_path, cache_root=tmp_path / "cache")
    assert cached == result
    assert sum(call[1] == "sch" for call in calls) == 1
    assert all("Library ID missing" in item or "Pin-number set missing" in item
               for item in result.unknown_metadata)


def test_pin_number_set_mismatch_is_rejected_when_both_exports_supply_it(tmp_path, monkeypatch):
    def add_pins(xml: str, ref: str, pins: str) -> str:
        marker = f'<comp ref="{ref}"><value>'
        start = xml.index(marker)
        end = xml.index("</comp>", start)
        return xml[:end] + f"<units><unit><pins>{pins}</pins></unit></units>" + xml[end:]

    expected = VALID_XML
    actual = VALID_XML
    for ref in ("J1", "R1", "R2"):
        pins = '<pin num="1"/><pin num="2"/>'
        expected = add_pins(expected, ref, pins)
        actual = add_pins(actual, ref, pins)
    expected = expected.replace('<pin num="1"/><pin num="2"/>',
                                '<pin num="1"/><pin num="2"/><pin num="3"/>', 1)
    source, netlist = _project(tmp_path, expected)
    _mock_cli(monkeypatch, actual)
    result = ee.verify_schematic_netlist(source, netlist, project_root=tmp_path,
                                         cache_root=tmp_path / "cache")
    assert result.status == "MISMATCH"
    assert "Pin-number sets differ for J1" in result.message


@pytest.mark.parametrize(("xml", "reason"), [
    (VALID_XML.replace('ref="R1" pin="1"', 'ref="R1" pin="X"').replace(
        'ref="R2" pin="2"', 'ref="R1" pin="1"').replace('ref="R1" pin="X"', 'ref="R2" pin="2"'),
     "Pin-to-net connectivity"),
    (VALID_XML.replace("<value>1k</value>", "<value>2k</value>", 1), "value differs"),
    (VALID_XML.replace('ref="J1"', 'ref="J9"'), "Reference sets"),
    (VALID_XML.replace('name="VIN"', 'name="/ALT"'), "Net identity differs"),
])
def test_connectivity_value_reference_and_net_identity_mismatches_are_rejected(tmp_path, monkeypatch, xml, reason):
    source, supplied = _project(tmp_path, xml)
    _mock_cli(monkeypatch, VALID_XML)
    result = ee.verify_schematic_netlist(source, supplied, project_root=tmp_path, cache_root=tmp_path / "cache")
    assert result.status == "MISMATCH"
    assert reason.lower() in result.message.lower()


def test_xml_change_invalidates_cached_verdict_and_cli_failure_is_not_cached(tmp_path, monkeypatch):
    source, xml = _project(tmp_path)
    calls = _mock_cli(monkeypatch, VALID_XML)
    first = ee.verify_schematic_netlist(source, xml, project_root=tmp_path, cache_root=tmp_path / "cache")
    xml.write_text(VALID_XML.replace('name="VIN"', 'name="SOURCE"'), encoding="utf-8")
    second = ee.verify_schematic_netlist(source, xml, project_root=tmp_path, cache_root=tmp_path / "cache")
    assert second.cache_key != first.cache_key
    assert second.status == "MISMATCH"
    assert sum(call[1] == "sch" for call in calls) == 2

    def failure(args, **kwargs):
        if args[1] == "version":
            return SimpleNamespace(stdout="KiCad 10", stderr="", returncode=0)
        return SimpleNamespace(stdout="", stderr="export failed", returncode=2)
    monkeypatch.setattr(ee.subprocess, "run", failure)
    failed = ee.verify_schematic_netlist(source, xml, project_root=tmp_path, cache_root=tmp_path / "failed-cache")
    assert failed.status == "NOT_VERIFIED"
    assert not list((tmp_path / "failed-cache").glob("*.json"))


def test_missing_cli_incomplete_project_and_export_timeout_fail_closed(tmp_path, monkeypatch):
    source, xml = _project(tmp_path)
    monkeypatch.setattr(ee, "find_kicad_cli", lambda: None)
    assert ee.verify_schematic_netlist(source, xml, project_root=tmp_path,
                                       cache_root=tmp_path / "cache").status == "NOT_VERIFIED"
    monkeypatch.setattr(ee, "find_kicad_cli", lambda: "/fake/kicad-cli")
    source.write_text(source.read_text(encoding="utf-8").replace('(sheet_instances',
                        '(sheet (property "Sheet file" "missing.kicad_sch") (uuid "sheet")) (sheet_instances'),
                        encoding="utf-8")
    result = ee.verify_schematic_netlist(source, xml, project_root=tmp_path, cache_root=tmp_path / "cache")
    assert result.status == "INCOMPLETE_PROJECT"


def test_reused_normalized_net_name_for_different_pin_groups_is_not_verified(tmp_path, monkeypatch):
    conflicting = VALID_XML.replace('name="GND"', 'name="/VIN"')
    source, netlist = _project(tmp_path, conflicting)
    _mock_cli(monkeypatch, conflicting)
    result = ee.verify_schematic_netlist(source, netlist, project_root=tmp_path,
                                         cache_root=tmp_path / "cache")
    assert result.status == "NOT_VERIFIED"
    assert "multiple pin groups" in result.message


@pytest.mark.integration
def test_real_kicad_cli_divider_equivalence(tmp_path):
    if not ee.find_kicad_cli():
        pytest.skip("NOT_VERIFIED: KiCad CLI is unavailable")
    source, netlist = _project(tmp_path)
    netlist.write_text((ROOT / "tests/fixtures/divider_netlist.xml").read_text(encoding="utf-8"), encoding="utf-8")
    result = ee.verify_schematic_netlist(source, netlist, project_root=tmp_path,
                                         cache_root=tmp_path / "equivalence-cache")
    assert result.status == "VERIFIED_EQUIVALENT", result.message
    assert all("Library ID missing" in item or "Pin-number set missing" in item
               for item in result.unknown_metadata)
