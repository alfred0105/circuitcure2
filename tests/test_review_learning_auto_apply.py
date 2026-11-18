import json
import io
import os
import hashlib

from PIL import Image

from circuitcare.vision import review_learning


def _training_fixture(tmp_path, board_rows):
    images, boxes, source_groups = [], [], {}
    for board_id, image_id in board_rows:
        image_path = tmp_path / f"{image_id}.png"
        Image.new("RGB", (32, 32), (40, 80, 120)).save(image_path)
        images.append({"image_sha256": image_id, "project_id": board_id,
                       "image_path": str(image_path)})
        component_id = f"box-{image_id}"
        boxes.append({"image_sha256": image_id, "component_id": component_id,
                      "class_name": "led", "x1": 2, "y1": 2, "x2": 28, "y2": 28})
        source_groups[hashlib.sha256(f"{image_id}:{component_id}".encode()).hexdigest()] = board_id
    return images, boxes, source_groups


def test_training_validation_split_keeps_each_board_project_in_one_split(tmp_path, monkeypatch):
    monkeypatch.setattr(review_learning, "PRIVATE_ROOT", tmp_path / "private")
    images, boxes, source_groups = _training_fixture(tmp_path, [
        ("board-a", "photo-a1"), ("board-a", "photo-a2"), ("board-b", "photo-b1"),
    ])
    data_yaml = review_learning._export_yolo_dataset(
        images, boxes, "board-group-split", model_key="test-model", training_mode="component_type",
        source_class_names=["led"],
    )
    content = data_yaml.read_text()
    assert "unique_source_images: 3" in content
    assert "unique_source_boards: 2" in content
    output = data_yaml.parent
    train_sources = {source_groups[p.stem] for p in (output / "labels/train").glob("*.txt")}
    val_sources = {source_groups[p.stem] for p in (output / "labels/val").glob("*.txt")}
    assert train_sources and val_sources
    assert train_sources.isdisjoint(val_sources)


def test_single_board_training_marks_validation_as_in_sample_without_losing_train_photos(tmp_path, monkeypatch):
    monkeypatch.setattr(review_learning, "PRIVATE_ROOT", tmp_path / "private")
    images, boxes, source_groups = _training_fixture(tmp_path, [
        ("one-board", "photo-a"), ("one-board", "photo-b"),
    ])
    data_yaml = review_learning._export_yolo_dataset(
        images, boxes, "one-board-split", model_key="test-model", training_mode="component_type",
        source_class_names=["led"],
    )
    content = data_yaml.read_text()
    assert "unique_source_images: 2" in content
    assert "unique_source_boards: 1" in content
    output = data_yaml.parent
    train_names = {p.stem for p in (output / "labels/train").glob("*.txt")}
    val_names = {p.stem for p in (output / "labels/val").glob("*.txt")}
    assert len(train_names) == 2
    assert len(val_names) == 1
    assert train_names >= val_names
    assert {source_groups[name] for name in train_names | val_names} == {"one-board"}


def test_board_group_change_invalidates_cached_training_dataset_identity(tmp_path, monkeypatch):
    private_root = tmp_path / "review-learning"
    private_root.mkdir()
    monkeypatch.setattr(review_learning, "PRIVATE_ROOT", private_root)
    monkeypatch.setattr(review_learning, "DB_PATH", private_root / "state.sqlite3")
    with review_learning._db() as db:
        db.execute(
            "INSERT INTO reviewed_images(image_sha256,project_id,photo_id,image_path,width,height,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            ("photo-hash", "board-a", "photo-1", "/tmp/photo.png", 32, 32, "now", "now"),
        )
        db.execute(
            "INSERT INTO reviewed_boxes(image_sha256,component_id,class_name,x1,y1,x2,y2) "
            "VALUES(?,?,?,?,?,?,?)",
            ("photo-hash", "box-1", "led", 1, 1, 20, 20),
        )

    original_digest = review_learning._dataset_snapshot()[0]
    with review_learning._db() as db:
        db.execute("UPDATE reviewed_images SET project_id=? WHERE image_sha256=?", ("board-b", "photo-hash"))
    assert review_learning._dataset_snapshot()[0] != original_digest


def test_successful_finetune_is_automatically_selected_for_following_inference(tmp_path, monkeypatch):
    private_root = tmp_path / "review-learning"
    private_root.mkdir()
    db_path = private_root / "state.sqlite3"
    monkeypatch.setattr(review_learning, "PRIVATE_ROOT", private_root)
    monkeypatch.setattr(review_learning, "DB_PATH", db_path)
    monkeypatch.delenv("CIRCUITCARE_USE_TRAINED_CHECKPOINTS", raising=False)

    candidate_dir = private_root / "candidate_weights" / "job-1"
    candidate_dir.mkdir(parents=True)
    checkpoint = candidate_dir / "candidate_best.pt"
    checkpoint.write_bytes(b"trained-yolo")
    (candidate_dir / "candidate.json").write_text(json.dumps({
        "job_id": "job-1", "model_key": "parsa_locator", "training_mode": "locator",
        "candidate_weights": str(checkpoint), "dataset_sha256": "dataset-1",
        "independent_validation_available": False,
    }), encoding="utf-8")
    with review_learning._db() as db:
        db.execute(
            "INSERT INTO training_jobs(job_id,dataset_sha256,status,output_dir,log_path,started_at,finished_at,"
            "model_key,training_mode) VALUES(?,?,?,?,?,?,?,?,?)",
            ("job-1", "dataset-1", "SUCCEEDED", str(candidate_dir), str(candidate_dir / "training.log"),
             "2026-01-01T00:00:00+00:00", "2026-01-01T00:01:00+00:00", "parsa_locator", "locator"),
        )

    selected = review_learning.active_training_checkpoint("parsa_locator")
    assert selected["checkpoint"] == checkpoint.resolve()
    assert selected["job_id"] == "job-1"
    assert selected["independent_validation_available"] is False


def test_baseline_evaluation_can_explicitly_disable_trained_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("CIRCUITCARE_USE_TRAINED_CHECKPOINTS", "0")
    assert review_learning.active_training_checkpoint("parsa_locator") is None


def test_external_board_holdout_cannot_be_saved_to_training_or_enter_snapshot(tmp_path, monkeypatch):
    private_root = tmp_path / "review-learning"
    private_root.mkdir()
    monkeypatch.setattr(review_learning, "PRIVATE_ROOT", private_root)
    monkeypatch.setattr(review_learning, "DB_PATH", private_root / "state.sqlite3")
    image_buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (16, 80, 140)).save(image_buffer, format="JPEG")
    image_bytes = image_buffer.getvalue()
    fingerprint, _, _ = review_learning._pixel_fingerprint(image_bytes)
    rows = [{"component_id": "box-1", "confirmed_base_class": "ic", "training_label_user_confirmed": True,
             "상태": "확인", "x1": 2, "y1": 2, "x2": 30, "y2": 20}]
    monkeypatch.delenv("CIRCUITCARE_VALIDATION_HOLDOUT_MANIFEST", raising=False)
    assert review_learning.save_reviewed_boxes(image_bytes, project_id="p", photo_id="holdout", rows=rows)["added_boxes"] == 1
    holdout = tmp_path / "holdout.json"
    holdout.write_text(json.dumps({"cases": [{"training_exclusion_pixel_sha256": fingerprint}]}), encoding="utf-8")
    monkeypatch.setenv("CIRCUITCARE_VALIDATION_HOLDOUT_MANIFEST", str(holdout))

    saved = review_learning.save_reviewed_boxes(image_bytes, project_id="p", photo_id="holdout", rows=rows)
    assert saved["excluded_from_training"] is True
    digest, image_count, images, boxes = review_learning._dataset_snapshot()
    assert image_count == 0
    assert images == [] and boxes == []


def test_multiple_external_validation_manifests_are_all_excluded_from_training(tmp_path, monkeypatch):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_text(json.dumps({"cases": [{"training_exclusion_pixel_sha256": "pixel-a"}]}), encoding="utf-8")
    second.write_text(json.dumps({"cases": [{"training_exclusion_pixel_sha256": "pixel-b"}]}), encoding="utf-8")
    monkeypatch.setenv("CIRCUITCARE_VALIDATION_HOLDOUT_MANIFEST", f"{first}{os.pathsep}{second}")
    assert review_learning._holdout_pixel_fingerprints() == {"pixel-a", "pixel-b"}
