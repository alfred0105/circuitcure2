from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluate_pcb_detector", ROOT / "scripts/evaluate_pcb_detector.py")
EVALUATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EVALUATOR)


def _fixture(tmp_path: Path):
    image_path = tmp_path / "board.png"
    Image.new("RGB", (40, 30), "white").save(image_path)
    image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    label_path = tmp_path / "board.txt"
    label_path.write_text("", encoding="utf-8")
    label_sha = hashlib.sha256(label_path.read_bytes()).hexdigest()
    manifest = {"schema_version": "circuitcare.component-annotation.v1", "dataset_id": "fixture",
        "dataset_name": "Small fixture", "dataset_source": "local", "dataset_source_url": "https://example.org",
        "source_version": "v1", "dataset_license": "CC0", "annotation_source": "fixture",
        "annotation_review_status": "SOURCE_PROVIDED_UNADJUDICATED", "split_name": "validation",
        "split_method": "source-provided", "board_grouping_policy": "unknown",
        "split_policy": "source-provided", "class_names": ["resistor"],
        "images": [{"image_id": "board.png", "image_path": "board.png", "label_path": "board.txt",
            "board_id": "BOARD_ID_UNRESOLVED", "board_revision": "BOARD_REVISION_UNRESOLVED",
            "board_side": "UNKNOWN", "split": "validation", "image_sha256": image_sha,
            "label_sha256": label_sha, "image_size_px": [40, 30], "source": "fixture",
            "label_state": "UNANNOTATED", "annotations": []}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest, image_path


def test_manifest_loader_uses_metadata_hashes_and_keeps_unannotated_images(tmp_path):
    manifest_path, manifest, _ = _fixture(tmp_path)
    rows, dataset = EVALUATOR.load_manifest_inputs(manifest_path, split="validation")
    assert len(rows) == 1 and rows[0]["ground_truth"] == []
    assert dataset["dataset_id"] == "fixture"
    assert dataset["annotation_review_status"] == "SOURCE_PROVIDED_UNADJUDICATED"
    assert dataset["manifest_image_count"] == 1
    assert dataset["unannotated_image_count"] == 1
    manifest["images"][0]["image_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        EVALUATOR.load_manifest_inputs(manifest_path)
    except ValueError as exc:
        assert "Image hash mismatch" in str(exc)
    else:
        raise AssertionError("Manifest hash mismatch was accepted")


def test_review_export_does_not_promote_unreviewed_labels(tmp_path):
    manifest_path, manifest, _ = _fixture(tmp_path)
    row = manifest["images"][0]
    row["label_state"] = "LABELED"
    row["annotations"] = [{"component_id": "annotation:0001", "class": "resistor",
        "bbox_xyxy": [5, 5, 15, 15], "occlusion_state": "UNKNOWN", "small_component": None,
        "annotation_status": "DRAFT_ANNOTATION", "reviewer": None, "source": "fixture",
        "provenance": {"origin": "fixture"}}]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "review"
    subprocess.run([sys.executable, str(ROOT / "scripts/export_annotation_review.py"),
        "--manifest", str(manifest_path), "--output-dir", str(output)], check=True, capture_output=True, text=True)
    exported = json.loads((output / "review_manifest.json").read_text(encoding="utf-8"))
    assert exported["images"][0]["annotations"][0]["annotation_status"] == "DRAFT_ANNOTATION"
    assert exported["images"][0]["review_preview"].startswith("images/")
    assert len(list((output / "images").glob("*.jpg"))) == 1


def test_failure_report_links_false_positive_and_negative_to_review_preview(tmp_path):
    manifest_path, manifest, _ = _fixture(tmp_path)
    row = manifest["images"][0]
    row["label_state"] = "LABELED"
    row["annotations"] = [{"component_id": "annotation:0001", "class": "resistor",
        "bbox_xyxy": [5, 5, 15, 15], "occlusion_state": "UNKNOWN", "small_component": None,
        "annotation_status": "DRAFT_ANNOTATION", "reviewer": None, "source": "fixture"}]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps({"evaluation_status": "SOURCE_LABEL_COMPARISON",
        "dataset": {"dataset_id": "fixture"},
        "ground_truth": {"board.png": [{"class_id": 0, "bbox_xyxy": [5, 5, 15, 15]}]},
        "predictions": {"board.png": [{"class_name": "capacitor", "confidence": 0.9,
            "bbox_xyxy": [5, 5, 15, 15], "detection_id": "det-1"}]}}), encoding="utf-8")
    output = tmp_path / "failures"
    subprocess.run([sys.executable, str(ROOT / "scripts/report_detection_failures.py"),
        "--evaluation", str(evaluation_path), "--manifest", str(manifest_path),
        "--output-dir", str(output)], check=True, capture_output=True, text=True)
    report = json.loads((output / "failure_report.json").read_text(encoding="utf-8"))
    assert report["failure_count"] == 2
    assert {row["suggested_error_type"] for row in report["failures"]} == {"CLASS_CONFUSION"}
    assert len(list((output / "images").glob("*.jpg"))) == 1
