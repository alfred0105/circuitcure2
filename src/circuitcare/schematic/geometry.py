"""Fail-closed symbol hit geometry tied to KiCad's exported SVG page units."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from circuitcare.schematic.identity import (
    SExpr,
    _atom,
    _child,
    _children,
    _prop,
    discover_schematic_files,
    parse_sexpr,
    read_symbol_identities,
)


@dataclass(frozen=True)
class SymbolHitRegion:
    component_ref: str
    symbol_uuid: str
    sheet_path: str
    unit: int
    library_id: str
    polygon: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class GeometryResult:
    status: str
    message: str
    view_box: tuple[float, float, float, float] | None = None
    regions: tuple[SymbolHitRegion, ...] = ()
    page_sources: tuple[tuple[str, str], ...] = ()


def _number(node: list[SExpr] | None, index: int, what: str) -> float:
    if node is None or len(node) <= index or not isinstance(node[index], str):
        raise ValueError(f"Missing {what}")
    value = float(cast(str, node[index]))
    if not math.isfinite(value):
        raise ValueError(f"Invalid {what}")
    return value


def _points(node: list[SExpr]) -> list[tuple[float, float]]:
    result = []
    for xy in _children(node, "xy"):
        result.append((_number(xy, 1, "x"), _number(xy, 2, "y")))
    return result


def _library_bounds(library: list[SExpr], unit: int) -> tuple[float, float, float, float]:
    """Get exact bounds from supported KiCad symbol primitives; unknown shapes fail closed."""
    points: list[tuple[float, float]] = []
    child_symbols = _children(library, "symbol")
    selected = []
    for child in child_symbols:
        name = str(child[1]) if len(child) > 1 else ""
        match = re.search(r"_(\d+)_(\d+)$", name)
        if not match or (int(match.group(1)) in (0, unit) and int(match.group(2)) == 1):
            selected.append(child)
    if not selected:
        raise ValueError("Library symbol has no matching unit geometry")
    supported = {"rectangle", "polyline", "circle", "pin"}
    for child in selected:
        for primitive in child[1:]:
            if not isinstance(primitive, list) or not primitive:
                continue
            kind = primitive[0]
            if kind not in supported:
                raise ValueError(f"Unsupported symbol geometry: {kind}")
            if kind == "rectangle":
                start, end = _child(primitive, "start"), _child(primitive, "end")
                x1, y1 = _number(start, 1, "rectangle x"), _number(start, 2, "rectangle y")
                x2, y2 = _number(end, 1, "rectangle x"), _number(end, 2, "rectangle y")
                points.extend(((x1, y1), (x2, y2)))
            elif kind == "polyline":
                pts = _child(primitive, "pts")
                if pts:
                    points.extend(_points(pts))
            elif kind == "circle":
                center = _child(primitive, "center")
                cx, cy = _number(center, 1, "circle x"), _number(center, 2, "circle y")
                radius = _number(_child(primitive, "radius"), 1, "circle radius")
                points.extend(((cx - radius, cy - radius), (cx + radius, cy + radius)))
            else:
                at = _child(primitive, "at")
                x, y = _number(at, 1, "pin x"), _number(at, 2, "pin y")
                length = _number(_child(primitive, "length"), 1, "pin length")
                # Pin angle is in KiCad's local coordinate system; extend to its endpoint.
                angle = math.radians(_number(at, 3, "pin angle"))
                points.extend(((x, y), (x + length * math.cos(angle), y + length * math.sin(angle))))
    if not points:
        raise ValueError("Symbol has no supported graphical bounds")
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _symbol_mirror(placed: list[SExpr]) -> tuple[bool, bool]:
    """Return page X/Y sign flips for KiCad's mirror x/y/xy option."""
    mirror = _child(placed, "mirror")
    value = _atom(mirror)
    if not value:
        return False, False
    if value == "x":
        return True, False
    if value == "y":
        return False, True
    if value == "xy":
        return True, True
    raise ValueError(f"Unsupported KiCad mirror mode: {value}")


def transform_symbol_point(x: float, y: float, *, cx: float, cy: float,
                           angle_degrees: float, mirror_x: bool = False,
                           mirror_y: bool = False) -> tuple[float, float]:
    """Map a KiCad library point to sheet coordinates using the verified transform."""
    angle = math.radians(angle_degrees)
    rotated_x = x * math.cos(angle) - y * math.sin(angle)
    rotated_y = -(x * math.sin(angle) + y * math.cos(angle))
    if mirror_y:
        rotated_x = -rotated_x
    if mirror_x:
        rotated_y = -rotated_y
    return round(cx + rotated_x, 4), round(cy + rotated_y, 4)


def read_svg_view_box(svg: str | Path) -> tuple[float, float, float, float] | None:
    """Read a validated numeric page viewBox for pan/zoom, independent of hit geometry."""
    try:
        root = ET.parse(svg).getroot()
        values = root.attrib.get("viewBox", "").replace(",", " ").split()
        if len(values) != 4:
            return None
        parsed = (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
        if not all(math.isfinite(value) for value in parsed) or parsed[0] != 0 or parsed[1] != 0 or parsed[2] <= 0 or parsed[3] <= 0:
            return None
        return parsed
    except (OSError, ET.ParseError, ValueError):
        return None


def _load_page(source: Path, svg: Path, project_root: Path, identities: list[Any]) -> GeometryResult:
    svg_root = ET.parse(svg).getroot()
    view = svg_root.attrib.get("viewBox", "").replace(",", " ").split()
    if len(view) != 4:
        return GeometryResult("UNVERIFIED_GEOMETRY", "SVG has no four-value viewBox")
    try:
        view_box = (float(view[0]), float(view[1]), float(view[2]), float(view[3]))
    except ValueError:
        return GeometryResult("UNVERIFIED_GEOMETRY", "SVG viewBox is not numeric")
    if not all(math.isfinite(value) for value in view_box) or view_box[0] != 0 or view_box[1] != 0 or view_box[2] <= 0 or view_box[3] <= 0:
        return GeometryResult("UNVERIFIED_GEOMETRY", "SVG page coordinate system is unsupported")

    tree = parse_sexpr(source.read_text(encoding="utf-8"))
    root_path = "/" + _atom(_child(tree, "uuid"))
    library_root = _child(tree, "lib_symbols")
    libraries: dict[str, list[SExpr]] = {}
    if library_root:
        for library in _children(library_root, "symbol"):
            libraries[_atom(library, 1)] = library

    regions: list[SymbolHitRegion] = []
    for placed in _children(tree, "symbol"):
        ref = _prop(placed, "Reference") or ""
        if not ref or ref.startswith("#"):
            continue
        lib_id = _atom(_child(placed, "lib_id"))
        definition = libraries.get(lib_id)
        if definition is None:
            return GeometryResult("UNVERIFIED_GEOMETRY", f"Missing embedded library definition for {ref}")
        at = _child(placed, "at")
        cx, cy = _number(at, 1, f"{ref} x"), _number(at, 2, f"{ref} y")
        angle_degrees = _number(at, 3, f"{ref} angle")
        unit = int(_atom(_child(placed, "unit"), default="1"))
        try:
            mirror_x, mirror_y = _symbol_mirror(placed)
        except ValueError as exc:
            return GeometryResult("UNVERIFIED_GEOMETRY", f"{ref}: {exc}")
        try:
            x1, y1, x2, y2 = _library_bounds(definition, unit)
        except ValueError as exc:
            return GeometryResult("UNVERIFIED_GEOMETRY", f"{ref}: {exc}")
        corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        # KiCad library y is up; SVG page y is down. The tested page transform is 1 mm to 1 viewBox unit.
        polygon = []
        for x, y in corners:
            polygon.append(transform_symbol_point(x, y, cx=cx, cy=cy,
                                                  angle_degrees=angle_degrees,
                                                  mirror_x=mirror_x, mirror_y=mirror_y))
        symbol_uuid = _atom(_child(placed, "uuid"))
        matches = [item for item in identities if item.component_ref == ref and item.symbol_uuid == symbol_uuid
                   and item.library_id == lib_id and item.unit == unit and item.sheet_path == root_path]
        if len(matches) != 1:
            return GeometryResult("UNVERIFIED_GEOMETRY", f"{ref}: reference, UUID, unit, or sheet path is ambiguous")
        regions.append(SymbolHitRegion(ref, matches[0].symbol_uuid, matches[0].sheet_path, unit, lib_id, tuple(polygon)))

    # Refuse overlapping hit boxes: the click target would not identify one symbol uniquely.
    boxes = []
    for region in regions:
        xs, ys = zip(*region.polygon)
        boxes.append((region, min(xs), min(ys), max(xs), max(ys)))
    for i, (_, ax1, ay1, ax2, ay2) in enumerate(boxes):
        for _, bx1, by1, bx2, by2 in boxes[i + 1:]:
            if min(ax2, bx2) - max(ax1, bx1) > 0.05 and min(ay2, by2) - max(ay1, by1) > 0.05:
                return GeometryResult("AMBIGUOUS_GEOMETRY", "Symbol hit regions overlap")
    if not regions:
        return GeometryResult("UNVERIFIED_GEOMETRY", "No placed symbol geometry was verified")
    return GeometryResult("VERIFIED_GEOMETRY", "Single-sheet KiCad mm-to-SVG viewBox mapping validated", view_box, tuple(regions), ((svg.name, source.relative_to(project_root).as_posix()),))


def build_symbol_geometry(path: str | Path, svg_paths: tuple[Path, ...] | list[Path], *, project_root: str | Path) -> GeometryResult:
    """Map a single KiCad sheet to a single named SVG page; all ambiguous projects fail closed."""
    root = Path(project_root).resolve(strict=True)
    source = Path(path).resolve(strict=True)
    try:
        files, incomplete = discover_schematic_files(source, project_root=root)
        identities, identity_incomplete = read_symbol_identities(source, project_root=root)
    except (OSError, ValueError) as exc:
        return GeometryResult("UNVERIFIED_GEOMETRY", str(exc))
    if incomplete or identity_incomplete:
        return GeometryResult("INCOMPLETE_PROJECT", "; ".join(incomplete + identity_incomplete))
    if len(files) != 1 or len(svg_paths) != 1:
        return GeometryResult("MULTI_SHEET_MAPPING_PARTIAL", "Multi-sheet SVG-to-sheet identity is not verified; click mapping disabled")
    svg = Path(svg_paths[0]).resolve()
    if svg.stem != source.stem:
        return GeometryResult("UNVERIFIED_GEOMETRY", "SVG page name does not match the schematic source")
    try:
        return _load_page(source, svg, root, identities)
    except (OSError, ET.ParseError, ValueError, OverflowError) as exc:
        return GeometryResult("UNVERIFIED_GEOMETRY", f"Geometry parsing failed: {exc}")
