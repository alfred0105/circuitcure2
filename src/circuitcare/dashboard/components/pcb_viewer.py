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
    selected_component_ids: list[str],
    asset_key: str, overlay_version: str, debug_telemetry: bool,
    event_sequence_seed: int, event_epoch: str, key: str,
    selection_ack_event_id: str | None, selection_rejected_event_id: str | None,
    on_selection_change: Any, on_selection_event_change: Any,
    on_viewer_status_change: Any, photo_id: str | None = None, analysis_loading: bool = False, viewer_ready_ack: bool = False,
) -> Any | None:
    """Render image and vector overlays under one SVG camera transform."""
    if not hasattr(st, "components") or not hasattr(st.components, "v2"):
        return None
    global _VIEWER
    if _VIEWER is None:
        _VIEWER = st.components.v2.component(
            name="circuitcare_interactive_pcb_v4",
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
            "photo_id": photo_id or frame_id,
            "analysis_loading": analysis_loading,
            "viewer_ready_ack": viewer_ready_ack,
            "viewer_key": key,
            "asset_key": asset_key,
            "overlay_version": overlay_version,
            "debug_telemetry": debug_telemetry,
            "selected_component_id": selected_component_id,
            "selected_component_ids": selected_component_ids,
            "selection_ack_event_id": selection_ack_event_id,
            "selection_rejected_event_id": selection_rejected_event_id,
            "event_sequence_seed": event_sequence_seed,
            "event_epoch": event_epoch,
        },
        default={"selection_event": None, "viewer_status": None},
        on_selection_change=None,
        on_selection_event_change=on_selection_event_change,
        on_viewer_status_change=on_viewer_status_change,
        # Keep enough room for toolbar, legend, and the CSS-sized viewport while
        # avoiding a large empty iframe tail below the PCB.
        key=key, width="stretch", height=680,
    )
