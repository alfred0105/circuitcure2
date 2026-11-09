"""Parser for KiCad PCB position CSV files."""

from __future__ import annotations

import csv
import io
import math
import re
from typing import Iterable

from circuitcare.vision.models import ComponentPosition

_COLUMN_ALIASES = {
    "reference": {"ref", "reference", "designator"},
    "value": {"val", "value"},
    "footprint": {"package", "footprint"},
    "x": {"posx", "x", "xposition", "xmm"},
    "y": {"posy", "y", "yposition", "ymm"},
    "rotation": {"rot", "rotation", "rotationdeg"},
    "side": {"side", "layer"},
}


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.strip().lstrip("#").lower())


def parse_position_csv(text: str, *, units: str, board_side: str,
                       origin_description: str = "KiCad export origin; see export options") -> tuple[list[ComponentPosition], dict]:
    if units not in {"mm", "in"}:
        raise ValueError("Position CSV 단위를 mm 또는 in으로 명시해야 합니다.")
    if board_side not in {"front", "back", "both"}:
        raise ValueError("PCB 면은 front, back, both 중 하나여야 합니다.")
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if not reader.fieldnames:
        raise ValueError("CSV 헤더를 찾을 수 없습니다.")
    normalized = {_normalize_header(name): name for name in reader.fieldnames if name}
    columns = {}
    for target, aliases in _COLUMN_ALIASES.items():
        match = next((normalized[name] for name in aliases if name in normalized), None)
        if match:
            columns[target] = match
    required = {"reference", "x", "y", "rotation", "side"}
    if missing := required - columns.keys():
        raise ValueError(f"KiCad position CSV 필수 열이 없습니다: {', '.join(sorted(missing))}")
    rows: list[ComponentPosition] = []
    seen: set[str] = set()
    factor = 25.4 if units == "in" else 1.0
    for line, row in enumerate(reader, start=2):
        ref = (row.get(columns["reference"]) or "").strip()
        if not ref:
            continue
        if ref in seen:
            raise ValueError(f"중복 Reference {ref!r} (CSV {line}행)")
        seen.add(ref)
        try:
            x = float((row.get(columns["x"]) or "").strip()) * factor
            y = float((row.get(columns["y"]) or "").strip()) * factor
            rotation = float((row.get(columns["rotation"]) or "").strip())
        except ValueError as exc:
            raise ValueError(f"CSV {line}행의 좌표/회전 값이 숫자가 아닙니다.") from exc
        if not all(math.isfinite(value) for value in (x, y, rotation)):
            raise ValueError(f"CSV {line}행의 좌표/회전 값은 유한한 숫자여야 합니다.")
        raw_side = (row.get(columns["side"]) or "").strip().lower()
        side = "front" if raw_side in {"front", "f", "top", "f.cu"} else "back" if raw_side in {"back", "b", "bottom", "b.cu"} else ""
        if not side:
            raise ValueError(f"CSV {line}행의 PCB 면을 front/back으로 판독할 수 없습니다: {raw_side!r}")
        if board_side == "both" or board_side == side:
            rows.append(ComponentPosition(
                reference=ref,
                value=(row.get(columns.get("value", ""), "") or "").strip(),
                footprint=(row.get(columns.get("footprint", ""), "") or "").strip(),
                x_mm=x, y_mm=y, rotation_deg=rotation, side=side,
            ))
    if not rows:
        raise ValueError("선택한 PCB 면에 해당하는 부품 행이 없습니다.")
    return rows, {
        "source": "KiCad Component Position CSV",
        "units_in_file": units,
        "coordinates_normalized_to": "mm",
        "board_side_filter": board_side,
        "origin_description": origin_description,
        "row_count": len(rows),
    }


def match_netlist_references(positions: Iterable[ComponentPosition], component_refs: Iterable[str]) -> dict:
    known = set(component_refs)
    position_map = {item.reference: item for item in positions}
    return {
        "matched": sorted(known & position_map.keys()),
        "in_netlist_without_position": sorted(known - position_map.keys()),
        "position_without_netlist_component": sorted(position_map.keys() - known),
    }
