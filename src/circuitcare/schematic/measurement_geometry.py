"""Conservative logical-net to symbol-pin anchor mapping for schematic overlays."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from circuitcare.models import Circuit
from circuitcare.schematic.geometry import (
    GeometryResult,
    _atom,
    _child,
    _children,
    _number,
    _prop,
    _symbol_mirror,
    parse_sexpr,
    transform_symbol_point,
)


@dataclass(frozen=True)
class MeasurementGeometryResult:
    status: str
    message: str
    node: str
    anchors: tuple[dict[str, str | float], ...] = ()


def build_measurement_anchors(*, node: str, circuit: Circuit,
                              schematic_path: str | Path,
                              project_root: str | Path,
                              geometry: GeometryResult) -> MeasurementGeometryResult:
    """Return every verified symbol pin on a named net; never choose one probe point."""
    if geometry.status != "VERIFIED_GEOMETRY":
        return MeasurementGeometryResult("UNVERIFIED_MEASUREMENT_LOCATION",
                                         f"Symbol geometry is {geometry.status}.", node)
    try:
        source = Path(schematic_path).resolve(strict=True)
        root = Path(project_root).resolve(strict=True)
        if not source.is_relative_to(root):
            raise ValueError("Schematic path is outside the project root")
        tree = parse_sexpr(source.read_text(encoding="utf-8"))
        libraries = {_atom(item, 1): item for item in _children(_child(tree, "lib_symbols") or [], "symbol")}
    except (OSError, ValueError) as exc:
        return MeasurementGeometryResult("UNVERIFIED_MEASUREMENT_LOCATION",
                                         f"Schematic could not be read safely: {exc}", node)

    pin_coordinates: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for placed in _children(tree, "symbol"):
        ref = _prop(placed, "Reference") or ""
        if not ref or ref.startswith("#"):
            continue
        lib_id = _atom(_child(placed, "lib_id"))
        library = libraries.get(lib_id)
        if library is None:
            return MeasurementGeometryResult("UNVERIFIED_MEASUREMENT_LOCATION",
                                             f"Embedded library definition missing for {ref}.", node)
        try:
            unit = int(_atom(_child(placed, "unit"), default="1"))
            at = _child(placed, "at")
            cx, cy = _number(at, 1, f"{ref} x"), _number(at, 2, f"{ref} y")
            angle = _number(at, 3, f"{ref} angle")
            mirror_x, mirror_y = _symbol_mirror(placed)
        except (TypeError, ValueError) as exc:
            return MeasurementGeometryResult("UNVERIFIED_MEASUREMENT_LOCATION",
                                             f"Placement for {ref} is invalid: {exc}", node)
        for sub_symbol in _children(library, "symbol"):
            name = str(sub_symbol[1]) if len(sub_symbol) > 1 else ""
            match = re.search(r"_(\d+)_(\d+)$", name)
            if match and int(match.group(1)) not in (0, unit):
                continue
            for pin in _children(sub_symbol, "pin"):
                pin_number = _atom(_child(pin, "number"))
                pin_at = _child(pin, "at")
                if not pin_number or pin_at is None:
                    continue
                try:
                    local_x = _number(pin_at, 1, f"{ref}.{pin_number} x")
                    local_y = _number(pin_at, 2, f"{ref}.{pin_number} y")
                    x, y = transform_symbol_point(local_x, local_y, cx=cx, cy=cy,
                                                  angle_degrees=angle,
                                                  mirror_x=mirror_x, mirror_y=mirror_y)
                except ValueError as exc:
                    return MeasurementGeometryResult("UNVERIFIED_MEASUREMENT_LOCATION",
                                                     f"Pin geometry for {ref}.{pin_number} is invalid: {exc}", node)
                pin_coordinates.setdefault((ref, pin_number), []).append((x, y))

    anchors: list[dict[str, str | float]] = []
    for connection in circuit.connections:
        if connection.net != node:
            continue
        candidates = pin_coordinates.get((connection.ref, connection.pin), [])
        if len(candidates) != 1:
            return MeasurementGeometryResult(
                "UNVERIFIED_MEASUREMENT_LOCATION",
                f"Pin {connection.ref}.{connection.pin} has {len(candidates)} verified schematic anchors.", node)
        x, y = candidates[0]
        if not (math.isfinite(x) and math.isfinite(y)):
            return MeasurementGeometryResult("UNVERIFIED_MEASUREMENT_LOCATION",
                                             f"Pin {connection.ref}.{connection.pin} is not finite.", node)
        anchors.append({"component_ref": connection.ref, "pin": connection.pin,
                        "net": node, "x": x, "y": y})
    if not anchors:
        return MeasurementGeometryResult("UNVERIFIED_MEASUREMENT_LOCATION",
                                         f"Net {node!r} has no mapped component pins in the current circuit.", node)
    return MeasurementGeometryResult(
        "VERIFIED_LOGICAL_NET_CANDIDATES",
        "All listed symbol pin anchors match the current netlist. They are logical schematic candidates, not PCB probe points.",
        node, tuple(anchors),
    )
