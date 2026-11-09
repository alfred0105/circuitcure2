"""Validation boundary for browser-originated schematic selection events."""

from __future__ import annotations

from typing import Any

from circuitcare.schematic.geometry import GeometryResult


def validate_selection_event(event: Any, *, component_refs: set[str], geometry: GeometryResult,
                             context_key: str, asset_key: str) -> tuple[bool, str]:
    """Re-check every identifier against current trusted geometry and analysis context."""
    if not isinstance(event, dict):
        return False, "Selection event is missing or invalid"
    if event.get("context_key") != context_key:
        return False, "Selection event belongs to another circuit context"
    if event.get("asset_key") != asset_key:
        return False, "Selection event belongs to an old schematic page or SVG"
    if geometry.status != "VERIFIED_GEOMETRY":
        return False, "Selection is disabled because geometry is not verified"
    ref, uuid = event.get("component_ref"), event.get("symbol_uuid")
    sheet, unit = event.get("sheet_path"), event.get("unit")
    if not isinstance(ref, str) or ref not in component_refs:
        return False, "Unknown component Reference"
    if not isinstance(uuid, str) or not isinstance(sheet, str) or type(unit) is not int:
        return False, "Selection identity is incomplete"
    found = any(region.component_ref == ref and region.symbol_uuid == uuid
                and region.sheet_path == sheet and region.unit == unit
                for region in geometry.regions)
    if not found:
        return False, "Selection identity does not match the current schematic"
    return True, ""
