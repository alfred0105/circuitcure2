"""Pinned third-party PCB detector artifact metadata and integrity-checked download."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

CHECKPOINT_URL = (
    "https://github.com/wangshnegqi/pcb-component-detection-dataset-yolo/raw/"
    "0d7ae7310fa607389779b6132285336ad73e96e1/models/yolov8s_pcb_best.pt"
)
CHECKPOINT_SHA256 = "ad21e33f0962e3a23119223e0f82ef7d189ef118a8d4b79d51ab62187612e93c"
CHECKPOINT_SIZE_BYTES = 89_552_854
CHECKPOINT_LICENSE = "Apache-2.0 (as declared by the model repository)"
DATASET_LICENSE = "CC BY 4.0 (as declared in LICENSE-DATASET)"
MODEL_SOURCE_COMMIT = "0d7ae7310fa607389779b6132285336ad73e96e1"


def default_model_dir() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache")
    return root / "circuitcare-ai" / "models"


def download_detector_checkpoint(destination: str | Path | None = None) -> Path:
    """Download the pinned checkpoint only after an explicit caller action."""
    target = Path(destination) if destination else default_model_dir() / "yolov8s_pcb_best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    temporary_path = None
    try:
        with urllib.request.urlopen(CHECKPOINT_URL, timeout=60) as response:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temp:
                temporary_path = Path(temp.name)
                while block := response.read(1024 * 1024):
                    size += len(block)
                    digest.update(block)
                    temp.write(block)
        if size != CHECKPOINT_SIZE_BYTES or digest.hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("모델 파일 크기 또는 SHA-256이 고정된 출처 정보와 일치하지 않습니다.")
        temporary_path.replace(target)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
    return target
