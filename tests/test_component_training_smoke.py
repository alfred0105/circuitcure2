from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

from PIL import Image

from scripts import component_training


def test_generated_smoke_dataset_has_distinct_train_and_validation_images(tmp_path, monkeypatch):
    class FakeYOLO:
        def __init__(self, model):
            self.model = model

        def train(self, **kwargs):
            weights = Path(kwargs["project"]) / kwargs["name"] / "weights"
            weights.mkdir(parents=True)
            (weights / "last.pt").write_bytes(b"smoke-checkpoint")

        def export(self, **kwargs):
            return "unused.onnx"

    monkeypatch.setattr(component_training, "DATA_ROOT", tmp_path / "component_training")
    monkeypatch.setattr(component_training, "probe_mps", lambda: {})
    monkeypatch.setattr(component_training, "select_device", lambda _: "cpu")
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))

    assert component_training.smoke(None) == 0
    base = tmp_path / "component_training" / "raw" / "smoke"
    image_paths = sorted((base / "images").glob("*/*.jpg"))
    image_hashes = {
        hashlib.sha256(path.read_bytes()).hexdigest(): path.relative_to(base).as_posix()
        for path in image_paths
    }
    assert len(image_paths) == 3
    assert len(image_hashes) == 3, image_hashes
    for image_path in image_paths:
        label_path = base / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
        rows = label_path.read_text().strip().splitlines()
        assert len(rows) == 2
        assert {row.split()[0] for row in rows} == {"0", "1"}
        with Image.open(image_path) as image:
            assert image.size == (512, 512)
