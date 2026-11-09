"""Streamlit Components v2 PCB image and component overlay viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

ASSET_DIR = Path(__file__).with_name("assets")
_VIEWER = None


def render_pcb_viewer(
    st: Any, *, image_png_base64: str, image_size_px: tuple[int, int],
    components: list[dict[str, Any]], project_id: str, board_id: str,
    board_side: str, frame_id: str, selected_component_id: str | None,
    asset_key: str, event_sequence_seed: int, key: str,
    on_selection_change: Any, on_viewer_status_change: Any,
) -> Any | None:
    """Render image and vector overlays under one SVG camera transform."""
    if not hasattr(st, "components") or not hasattr(st.components, "v2"):
        return None
    global _VIEWER
    if _VIEWER is None:
        _VIEWER = st.components.v2.component(
            name="circuitcare_interactive_pcb",
            html=(ASSET_DIR / "pcb_viewer.html").read_text(encoding="utf-8"),
            css=(ASSET_DIR / "pcb_viewer.css").read_text(encoding="utf-8"),
            js=(ASSET_DIR / "pcb_viewer.js").read_text(encoding="utf-8"),
        )
    return _VIEWER(
        data={
            "image_png_base64": image_png_base64,
            "image_size_px": list(image_size_px),
            "components": components,
            "project_id": project_id,
            "board_id": board_id,
            "board_side": board_side,
            "frame_id": frame_id,
            "asset_key": asset_key,
            "selected_component_id": selected_component_id,
            "event_sequence_seed": event_sequence_seed,
        },
        default={"selection": None, "viewer_status": None},
        on_selection_change=on_selection_change,
        on_viewer_status_change=on_viewer_status_change,
        key=key, width="stretch", height=600,
    )
