from pathlib import Path

import pytest

from circuitcare.dashboard.views.schematic_workspace import spice_verification_label
from circuitcare.schematic.identity import map_component_refs, read_symbol_identities

ROOT = Path(__file__).resolve().parents[1]


def test_reads_symbol_identity_and_marks_reference_mapping():
    identities, missing = read_symbol_identities(
        ROOT / "tests/fixtures/identity_minimal.kicad_sch", project_root=ROOT)
    assert not missing
    mapped = map_component_refs(["R1", "R2"], identities)
    assert mapped["R1"]["status"] == "VERIFIED_REFERENCE"
    assert mapped["R1"]["symbol_instances"][0]["symbol_uuid"] == "ed693198-4ec1-4cbe-a345-ec57389d31e4"
    assert mapped["R1"]["symbol_instances"][0]["unit"] == 1
    assert mapped["R2"]["match_count"] == 0


def test_rejects_schematic_outside_project(tmp_path):
    outside = tmp_path / "outside.kicad_sch"
    outside.write_text("(kicad_sch)", encoding="utf-8")
    with pytest.raises(ValueError, match="inside the project root"):
        read_symbol_identities(outside, project_root=ROOT)


def _symbol(ref: str, uid: str, path: str = "/root", unit: int = 1) -> str:
    return f'''(symbol (lib_id "Device:R") (at 10 10 0) (unit {unit}) (uuid "{uid}")
      (property "Reference" "{ref}" (at 10 10 0) (effects (font (size 1 1))))
      (property "Value" "1k" (at 10 12 0) (effects (font (size 1 1))))
      (instances (project "test" (path "{path}" (reference "{ref}") (unit {unit})))))'''


def _root(*blocks: str, uid: str = "root", tail: str = "") -> str:
    return f'(kicad_sch (version 20250114) (generator "eeschema") (uuid "{uid}") (paper "A4") (lib_symbols) {" ".join(blocks)} {tail} (sheet_instances (path "/" (page "1"))))'


def test_multiunit_symbol_and_reused_uuid_on_distinct_sheet_paths(tmp_path):
    path = tmp_path / "multi.kicad_sch"
    path.write_text(_root(_symbol("U1", "same", "/root/sheetA", 1),
                          _symbol("U1", "same", "/root/sheetB", 2)), encoding="utf-8")
    identities, missing = read_symbol_identities(path, project_root=tmp_path)
    assert not missing
    assert {(item.sheet_path, item.unit) for item in identities} == {
        ("/root/sheetA", 1), ("/root/sheetB", 2)}
    assert map_component_refs(["U1"], identities)["U1"]["status"] == "VERIFIED_REFERENCE"


def test_rejects_uuid_conflict_but_accepts_distinct_units(tmp_path):
    path = tmp_path / "conflict.kicad_sch"
    path.write_text(_root(_symbol("U1", "shared", "/root", 1),
                          _symbol("U2", "shared", "/root", 1)), encoding="utf-8")
    with pytest.raises(ValueError, match="Conflicting symbol UUID"):
        read_symbol_identities(path, project_root=tmp_path)
    path.write_text(_root(_symbol("U1", "unit1", "/root", 1),
                          _symbol("U1", "unit2", "/root", 2)), encoding="utf-8")
    identities, _ = read_symbol_identities(path, project_root=tmp_path)
    assert {item.unit for item in identities} == {1, 2}
    path.write_text(_root(_symbol("U1", "id1", "/root", 1),
                          _symbol("U1", "id2", "/root", 1)), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate symbol reference/unit"):
        read_symbol_identities(path, project_root=tmp_path)


def test_nested_missing_sheet_and_external_sheet_are_reported(tmp_path):
    root = tmp_path / "root.kicad_sch"
    child = tmp_path / "child.kicad_sch"
    grandchild = tmp_path / "grandchild.kicad_sch"
    sheet_child = '(sheet (property "Sheet file" "child.kicad_sch") (uuid "s1"))'
    sheet_grand = '(sheet (property "Sheet file" "grandchild.kicad_sch") (uuid "s2"))'
    root.write_text(_root(tail=sheet_child), encoding="utf-8")
    child.write_text(_root(_symbol("R3", "r3", "/root/s1"), tail=sheet_grand, uid="child"), encoding="utf-8")
    grandchild.write_text(_root(_symbol("R4", "r4", "/root/s1/s2"), uid="grand"), encoding="utf-8")
    identities, missing = read_symbol_identities(root, project_root=tmp_path)
    assert {item.component_ref for item in identities} == {"R3", "R4"}
    assert not missing

    grandchild.unlink()
    _, missing = read_symbol_identities(root, project_root=tmp_path)
    assert any("INCOMPLETE_PROJECT" in item and "grandchild" in item for item in missing)


def test_sheet_path_traversal_and_malformed_sexpr_fail_closed(tmp_path):
    path = tmp_path / "root.kicad_sch"
    path.write_text(_root(tail='(sheet (property "Sheet file" "../outside.kicad_sch") (uuid "s1"))'), encoding="utf-8")
    _, missing = read_symbol_identities(path, project_root=tmp_path)
    assert any("external sheet blocked" in item for item in missing)
    path.write_text('(kicad_sch (property "bad\" (uuid "x"))', encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed|Incomplete"):
        read_symbol_identities(path, project_root=tmp_path)


def test_escaped_quotes_and_parentheses_in_properties_are_preserved(tmp_path):
    path = tmp_path / "escaped.kicad_sch"
    symbol = _symbol("R1", "r1").replace('"1k"', r'"1k (\"trim\")"')
    path.write_text(_root(symbol), encoding="utf-8")
    identities, _ = read_symbol_identities(path, project_root=tmp_path)
    assert identities[0].value == '1k ("trim")'


@pytest.mark.parametrize(("status", "integrity", "expected"), [
    ("SUCCESS", "VERIFIED", "VERIFIED · Live Simulation"),
    ("CACHED_SIMULATION", "VERIFIED", "VERIFIED · Cached Simulation"),
    ("PROVENANCE_UNVERIFIED", "PROVENANCE_UNVERIFIED", "NOT VERIFIED"),
    ("STALE", "STALE", "NOT VERIFIED"),
    ("FAILED", "VERIFIED", "NOT VERIFIED"),
    ("NOT_RUN", "NOT_PRESENT", "NOT VERIFIED"),
])
def test_spice_inspector_requires_success_status_and_verified_manifest(status, integrity, expected):
    assert spice_verification_label(status, integrity).startswith(expected)
