import json
from pathlib import Path

import numpy as np
import pytest

from circuitcare.vision.reconstruction.metrics import binary_mask_metrics, greedy_point_match
from circuitcare.vision.reconstruction.prism_dataset import SAMPLES, load_references, load_sample, validate_inventory


def _make_dataset(root: Path) -> None:
    for sample in SAMPLES:
        board = root / "Dataset" / sample
        project = board / sample
        ref = root / "evaluation" / "reference" / sample.lower()
        project.mkdir(parents=True)
        ref.mkdir(parents=True)
        for filename in ("TOP.png", "BOTTOM.png"):
            (board / filename).write_bytes(b"fixture")
        for filename in (f"{sample}.kicad_pcb", f"{sample}.kicad_sch"):
            (project / filename).write_text("fixture", encoding="utf-8")
        for filename in ("expected_geometry.json", "expected_components.json", "expected_netlist.json",
                         "expected_stats.json", "expected_schematic_components.json"):
            (ref / filename).write_text("{}", encoding="utf-8")


def test_prism_inventory_and_hashing_are_deterministic(tmp_path):
    _make_dataset(tmp_path)
    report = validate_inventory(tmp_path)
    assert report["expected_samples"] == 12
    assert report["ready_samples"] == 12
    first = load_sample(tmp_path, "simple")
    second = load_sample(tmp_path, "Simple")
    assert first.sample_id == "Simple"
    assert first.dataset_hash == second.dataset_hash
    assert not hasattr(first, "references")
    assert set(load_references(tmp_path, "Simple")) == {"expected_geometry", "expected_components", "expected_netlist",
                                                         "expected_stats", "expected_schematic_components"}


def test_prism_inventory_reports_missing_reference_file(tmp_path):
    _make_dataset(tmp_path)
    (tmp_path / "evaluation/reference/simple/expected_netlist.json").unlink()
    report = validate_inventory(tmp_path)
    simple = next(row for row in report["samples"] if row["sample"] == "Simple")
    assert simple["status"] == "INVALID"
    with pytest.raises(FileNotFoundError):
        load_sample(tmp_path, "Simple")


def test_binary_mask_metrics_and_one_to_one_point_matching():
    mask = np.array([[1, 1], [0, 0]], dtype=bool)
    assert binary_mask_metrics(mask, mask)["iou"] == 1.0
    metrics = greedy_point_match([(0, 0), (0.5, 0)], [(0.1, 0)], tolerance_px=1)
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 0
