from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from circuitcare.schematic.exporter import export_schematic, find_kicad_cli
from circuitcare.schematic.geometry import build_symbol_geometry

ROOT = Path(__file__).resolve().parents[1]
TRANSFORMS = [(0, None), (90, None), (180, None), (270, None),
              (0, "x"), (0, "y"), (90, "x"), (90, "y")]


def _transform(x: float, y: float, angle: int, mirror: str | None) -> tuple[float, float]:
    radians = math.radians(angle)
    rx = x * math.cos(radians) - y * math.sin(radians)
    ry = x * math.sin(radians) + y * math.cos(radians)
    page_x, page_y = 100 + rx, 73.81 - ry
    if mirror == "x":
        page_y = 2 * 73.81 - page_y
    elif mirror == "y":
        page_x = 2 * 100 - page_x
    return page_x, page_y


@pytest.mark.integration
@pytest.mark.parametrize(("angle", "mirror"), TRANSFORMS)
def test_rotated_and_mirrored_symbol_geometry_matches_kicad_cli(tmp_path, angle, mirror):
    if not find_kicad_cli():
        pytest.skip("NOT_VERIFIED: KiCad CLI is unavailable")
    source = tmp_path / "divider.kicad_sch"
    raw = (ROOT / "tests/fixtures/divider.kicad_sch").read_text(encoding="utf-8")
    # Asymmetric embedded resistor body makes X/Y mirroring observable in SVG bounds.
    raw = raw.replace("(end 1.016 2.54)", "(end 4 2.54)", 1)
    old = "(at 100 73.81 0) (unit 1)"
    replacement = f"(at 100 73.81 {angle}) " + (f"(mirror {mirror}) " if mirror else "") + "(unit 1)"
    raw = raw.replace(old, replacement, 1)
    source.write_text(raw, encoding="utf-8")

    exported = export_schematic(source, project_root=tmp_path, cache_root=tmp_path / "svg-cache")
    assert exported.status == "EXPORTED", exported.message
    geometry = build_symbol_geometry(source, exported.svg_paths, project_root=tmp_path)
    assert geometry.status == "VERIFIED_GEOMETRY", geometry.message
    region = next(item for item in geometry.regions if item.component_ref == "R1")
    points = [_transform(x, y, angle, mirror)
              for x in (-1.016, 4.0) for y in (-3.81, 3.81)]
    expected = (min(x for x, _ in points), min(y for _, y in points),
                max(x for x, _ in points), max(y for _, y in points))
    xs, ys = zip(*region.polygon)
    assert (min(xs), min(ys), max(xs), max(ys)) == pytest.approx(expected, abs=0.0002)

    svg_root = ET.parse(exported.svg_paths[0]).getroot()
    body_points = [_transform(x, y, angle, mirror)
                   for x in (-1.016, 4.0) for y in (-2.54, 2.54)]
    body_bounds = (min(x for x, _ in body_points), min(y for _, y in body_points),
                   max(x for x, _ in body_points), max(y for _, y in body_points))
    rects = []
    for element in svg_root.iter():
        if element.tag.rsplit("}", 1)[-1] != "rect":
            continue
        try:
            x, y, width, height = [float(element.attrib[key]) for key in ("x", "y", "width", "height")]
        except (KeyError, ValueError):
            continue
        cx, cy = (x + x + width) / 2, (y + y + height) / 2
        if abs(cx - (body_bounds[0] + body_bounds[2]) / 2) < 0.002 and abs(cy - (body_bounds[1] + body_bounds[3]) / 2) < 0.002:
            rects.append((x, y, width, height))
    # The rendered asymmetric body must be contained by the verified hit polygon.
    assert rects, f"No KiCad SVG body rectangle found for R1 at transform {(angle, mirror)}"
    rect = rects[0]
    assert body_bounds[0] == pytest.approx(rect[0], abs=0.002)
    assert body_bounds[1] == pytest.approx(rect[1], abs=0.002)
    assert body_bounds[2] == pytest.approx(rect[0] + rect[2], abs=0.002)
    assert body_bounds[3] == pytest.approx(rect[1] + rect[3], abs=0.002)
