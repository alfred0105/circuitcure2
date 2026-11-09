"""Streamlit Components v2 wrapper for the self-contained schematic viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

ASSET_DIR = Path(__file__).with_name("assets")
_VIEWER = None


def render_interactive_viewer(st: Any, *, svg: str, asset_key: str, view_box: tuple[float, float, float, float],
                              fit_view_box: tuple[float, float, float, float] | None = None,
                              geometry_status: str, regions: list[dict], overlay: dict | None,
                              selected_ref: str,
                              navigation_id: int, context_key: str, event_sequence_seed: int, key: str,
                              on_selection_change: Any) -> Any | None:
    """Mount a v2 viewer, returning None when this Streamlit build lacks that API."""
    global _VIEWER
    if not hasattr(st, "components") or not hasattr(st.components, "v2"):
        return None
    if _VIEWER is None:
        _VIEWER = st.components.v2.component(
            name="circuitcare_interactive_schematic",
            html=(ASSET_DIR / "viewer.html").read_text(encoding="utf-8"),
            css=(ASSET_DIR / "viewer.css").read_text(encoding="utf-8"),
            js=(ASSET_DIR / "viewer.js").read_text(encoding="utf-8"),
        )
    return _VIEWER(
        data={"svg": svg, "asset_key": asset_key, "view_box": list(view_box),
              "fit_view_box": list(fit_view_box or view_box),
              "geometry_status": geometry_status, "regions": regions,
              "overlay": overlay,
              "selected_ref": selected_ref, "navigation_id": navigation_id,
              "navigation_ref": selected_ref, "context_key": context_key,
              "event_sequence_seed": event_sequence_seed},
        default={"selection": None}, on_selection_change=on_selection_change,
        key=key, width="stretch", height=520,
    )
