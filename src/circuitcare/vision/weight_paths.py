"""Resolve local PCB model weights consistently across app checkouts."""

from __future__ import annotations

import os
from pathlib import Path

WEIGHT_ENV = "CIRCUITCARE_PCB_WEIGHTS"
WEIGHT_CONFIG_PATH = Path.home() / ".config" / "circuitcare-ai" / "pcb_weights_path"


def resolve_pcb_weights_root(app_dir: Path) -> Path:
    """Return the configured local weights folder, independent of clone location.

    An explicit environment variable wins, followed by a per-user config path,
    the legacy user model folder, then the weights directory beside app_server.py.
    """
    configured = os.environ.get(WEIGHT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()

    try:
        configured_path = WEIGHT_CONFIG_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        configured_path = ""
    if configured_path:
        return Path(configured_path).expanduser().resolve()

    local_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share"))
    user_weights = local_data / "circuitcare-ai" / "models" / "pcb_component_pipeline"
    if user_weights.is_dir():
        return user_weights.resolve()
    return (Path(app_dir).resolve() / "weights")
