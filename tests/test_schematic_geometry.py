import shutil
from pathlib import Path

from circuitcare.schematic.geometry import build_symbol_geometry
from circuitcare.schematic.identity import read_symbol_identities
from circuitcare.schematic.interaction import validate_selection_event

ROOT = Path(__file__).resolve().parents[1]


def _fixture_project(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "divider.kicad_sch"
    shutil.copyfile(ROOT / "tests/fixtures/divider.kicad_sch", source)
    svg = tmp_path / "divider.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 297.0022 210.0072"/>', encoding="utf-8")
    return source, svg


def test_divider_symbol_geometry_maps_to_kicad_svg_viewbox_and_identity(tmp_path):
    source, svg = _fixture_project(tmp_path)
    result = build_symbol_geometry(source, [svg], project_root=tmp_path)
    regions = {region.component_ref: region for region in result.regions}
    assert result.status == "VERIFIED_GEOMETRY"
    assert result.view_box == (0, 0, 297.0022, 210.0072)
    assert regions["J1"].symbol_uuid == "e3d45606-99ef-4d4a-b4b6-4b279ea1f2e2"
    assert regions["R1"].polygon == ((98.984, 77.62), (101.016, 77.62), (101.016, 70.0), (98.984, 70.0))
    assert regions["R2"].polygon == ((118.984, 85.24), (121.016, 85.24), (121.016, 77.62), (118.984, 77.62))
    assert regions["J1"].unit == 1 and regions["R1"].sheet_path.endswith("56ab210d-4b30-4193-84cc-4f5c5af24f5f")


def test_geometry_fails_closed_for_bad_viewbox_unknown_primitives_or_multi_sheet(tmp_path):
    source, svg = _fixture_project(tmp_path)
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="1 0 297 210"/>', encoding="utf-8")
    assert build_symbol_geometry(source, [svg], project_root=tmp_path).status == "UNVERIFIED_GEOMETRY"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 297.0022 210.0072"/>', encoding="utf-8")
    assert build_symbol_geometry(source, [svg, svg], project_root=tmp_path).status == "MULTI_SHEET_MAPPING_PARTIAL"
    contents = source.read_text(encoding="utf-8").replace('(rectangle\n\t\t\t\t(start -1.016 -2.54)', '(arc (start 0 0) (mid 1 1) (end 2 0))\n\t\t\t(rectangle\n\t\t\t\t(start -1.016 -2.54)', 1)
    source.write_text(contents, encoding="utf-8")
    assert build_symbol_geometry(source, [svg], project_root=tmp_path).status == "UNVERIFIED_GEOMETRY"


def test_selection_event_requires_exact_context_reference_uuid_sheet_and_unit(tmp_path):
    source, svg = _fixture_project(tmp_path)
    geometry = build_symbol_geometry(source, [svg], project_root=tmp_path)
    region = next(item for item in geometry.regions if item.component_ref == "R1")
    event = {"component_ref": region.component_ref, "symbol_uuid": region.symbol_uuid,
             "sheet_path": region.sheet_path, "unit": region.unit,
             "context_key": "active", "asset_key": "active-svg", "event_seq": 1}
    refs = {"J1", "R1", "R2"}
    assert validate_selection_event(event, component_refs=refs, geometry=geometry,
                                    context_key="active", asset_key="active-svg")[0]
    assert not validate_selection_event({**event, "component_ref": "Q9"}, component_refs=refs,
                                       geometry=geometry, context_key="active", asset_key="active-svg")[0]
    assert not validate_selection_event({**event, "symbol_uuid": "forged"}, component_refs=refs,
                                       geometry=geometry, context_key="active", asset_key="active-svg")[0]
    assert not validate_selection_event(event, component_refs=refs, geometry=geometry,
                                        context_key="other", asset_key="active-svg")[0]
    assert not validate_selection_event(event, component_refs=refs, geometry=geometry,
                                        context_key="active", asset_key="old-svg")[0]


def test_multi_sheet_project_has_no_click_geometry_even_when_references_are_known(tmp_path):
    source, svg = _fixture_project(tmp_path)
    child = tmp_path / "child.kicad_sch"
    child.write_text('(kicad_sch (version 20250114) (generator "eeschema") (uuid "child") (paper "A4") (lib_symbols))', encoding="utf-8")
    contents = source.read_text(encoding="utf-8").replace('(sheet_instances', '(sheet (property "Sheet file" "child.kicad_sch") (uuid "sheet-child"))\n(sheet_instances', 1)
    source.write_text(contents, encoding="utf-8")
    result = build_symbol_geometry(source, [svg], project_root=tmp_path)
    assert result.status == "MULTI_SHEET_MAPPING_PARTIAL"
    identities, missing = read_symbol_identities(source, project_root=tmp_path)
    assert not missing
    assert identities
