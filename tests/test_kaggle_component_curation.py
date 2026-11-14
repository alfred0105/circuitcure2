import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "curate_kaggle_component_dataset", ROOT / "scripts" / "curate_kaggle_component_dataset.py")
curator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curator)


def _fixture_dataset(tmp_path, *, exact_duplicate=False, shared_stem=False):
    repo = tmp_path
    rows = []
    for split, name, color, objects in (
        ("train", "one", (20, 30, 40), [{"id": 1, "classTitle": "Cap1", "labelerLogin": "source",
            "points": {"exterior": [[2, 2], [10, 10]]}}]),
        ("test", "two", (50, 60, 70), []),
    ):
        image = repo / "data/component_training/raw/external/pcb-component-detection/extracted" / split / "img" / f"{name}.jpg"
        annotation = repo / "data/component_training/raw/external/pcb-component-detection/extracted" / split / "ann" / f"{name}.jpg.json"
        image.parent.mkdir(parents=True, exist_ok=True)
        annotation.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), color).save(image)
        annotation.write_text(json.dumps({"size": {"height": 16, "width": 16}, "objects": objects}))
        rows.append({"image": f"pcb-component-detection/extracted/{split}/img/{name}.jpg",
            "annotation": f"pcb-component-detection/extracted/{split}/ann/{name}.jpg.json",
            "source_split": split, "source_stem_group": "family" if shared_stem else name})
    if exact_duplicate:
        first = repo / "data/component_training/raw/external/pcb-component-detection/extracted/train/img/one.jpg"
        second = repo / "data/component_training/raw/external/pcb-component-detection/extracted/test/img/two.jpg"
        second.write_bytes(first.read_bytes())
    acquisition = repo / "data/component_training/manifests/acquisition.json"
    acquisition.parent.mkdir(parents=True, exist_ok=True)
    acquisition.write_text(json.dumps({"images": rows}))
    return acquisition


def test_domain_policy_excludes_non_pcb_and_keeps_single_mounted_component():
    assert curator.training_domain_eligible("POPULATED_PCB")
    assert curator.training_domain_eligible("PARTIALLY_POPULATED_PCB")
    assert curator.training_domain_eligible("PCB_COMPONENT_CLOSEUP")
    assert not curator.training_domain_eligible("ISOLATED_COMPONENT")
    assert not curator.training_domain_eligible("LOOSE_COMPONENT_COLLECTION")
    assert not curator.training_domain_eligible("BREADBOARD")
    assert not curator.training_domain_eligible("BARE_PCB")
    assert curator.training_domain_eligible("POPULATED_PCB")  # mounting count is not a filter


def test_class_mapping_keeps_source_labels_separate():
    mapping = {row["source_class"]: row for row in curator._class_mapping(
        __import__("collections").Counter({"Cap1": 1, "Resestor": 3}))}
    assert mapping["Cap1"]["target_class"] is None
    assert mapping["Cap2"]["target_class"] is None
    assert mapping["Resestor"]["target_class"] is None
    assert mapping["MOSFET"]["target_class"] is None
    assert all(not row["approved_for_training"] for row in mapping.values())


def test_annotation_completeness_never_assumes_nonempty_is_complete():
    assert curator.annotation_state(0) == "EMPTY_UNREVIEWED"
    assert curator.annotation_state(0, empty_reviewed=True) == "EMPTY_REVIEWED"
    assert curator.annotation_state(0, visible_target_without_label=True) == "PARTIAL_ANNOTATION"
    assert curator.annotation_state(3) == "ANNOTATION_STATUS_UNKNOWN"
    assert curator.annotation_state(3, completeness_reviewed=True) == "ANNOTATION_COMPLETE"
    with pytest.raises(ValueError):
        curator.annotation_state(-1)


def test_curation_manifest_preserves_source_and_blocks_unsafe_training(tmp_path):
    acquisition = _fixture_dataset(tmp_path)
    source_image = tmp_path / "data/component_training/raw/external/pcb-component-detection/extracted/train/img/one.jpg"
    original_hash = curator._sha256(source_image)
    output = tmp_path / "data/component_training/manifests/curation.json"
    result = curator.curate(acquisition, output)
    manifest = json.loads(output.read_text())
    assert result["image_count"] == 2
    assert result["training_ready_image_count"] == 0
    assert result["source_only_training_candidate_count"] == 0
    assert result["canonical_10_class_training_candidate_count"] == 0
    assert manifest["images"][0]["training_include"] is False
    assert manifest["images"][0]["board_id"] == "BOARD_ID_UNRESOLVED"
    assert manifest["images"][0]["assigned_split"] is None
    assert manifest["images"][1]["annotation_status"] == "EMPTY_UNREVIEWED"
    assert manifest["images"][1]["annotations"] == []
    assert curator._sha256(source_image) == original_hash
    yaml_path = output.with_name("pcb-component-detection-source-only-blocked.yaml")
    assert "BLOCKED_NO_VERIFIED_TRAINING_DATA" in yaml_path.read_text()
    assert "train: []" in yaml_path.read_text() and "val: []" in yaml_path.read_text()
    assert output.with_name("pcb-component-detection-duplicate-report.json").is_file()
    assert output.with_name("pcb-component-detection-annotation-completeness-report.json").is_file()
    assert output.with_name("pcb-component-detection-board-leakage-report.json").is_file()


def test_invalid_box_is_rejected(tmp_path):
    acquisition = _fixture_dataset(tmp_path)
    label = tmp_path / "data/component_training/raw/external/pcb-component-detection/extracted/train/ann/one.jpg.json"
    data = json.loads(label.read_text())
    data["objects"][0]["points"]["exterior"] = [[-1, 2], [10, 10]]
    label.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="out-of-bounds"):
        curator.curate(acquisition, tmp_path / "curation.json")


def test_duplicate_stem_and_board_split_leakage_are_reported(tmp_path):
    acquisition = _fixture_dataset(tmp_path, exact_duplicate=True, shared_stem=True)
    output = tmp_path / "data/component_training/manifests/curation.json"
    result = curator.curate(acquisition, output)
    leakage = json.loads(output.with_name("pcb-component-detection-board-leakage-report.json").read_text())
    assert result["exact_duplicate_group_count"] == 1
    assert result["source_stem_groups_crossing_splits"] == 1
    assert result["visual_board_group_candidate_crosses_source_splits"] == 1
    assert leakage["resolved_source_board_group_count"] == 0
    assert leakage["independent_test_available"] is False
