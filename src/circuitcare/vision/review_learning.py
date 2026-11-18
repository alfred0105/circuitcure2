"""Private store and isolated training workflow for user-reviewed PCB boxes."""
from __future__ import annotations

import hashlib
import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from circuitcare.component_taxonomy import COMPONENT_CLASSES, normalize_component_class

ROOT = Path(__file__).resolve().parents[3]
USER_DATA_HOME = Path(os.environ.get("CIRCUITCARE_REVIEW_LEARNING_DATA") or
                      os.environ.get("LOCALAPPDATA") or
                      os.environ.get("XDG_DATA_HOME") or
                      (Path.home() / ".local" / "share"))
PRIVATE_ROOT = USER_DATA_HOME / "circuitcare-ai" / "review_learning"
DB_PATH = PRIVATE_ROOT / "review_learning.sqlite3"
MIN_UNIQUE_IMAGES = 1
_SAMPLE_WRITER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pcb-reviewed-sample")
_AUTO_TRAINER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pcb-yolo-auto-train")
_PENDING_SAVES: list[Future] = []
_SAVE_ERRORS: list[str] = []
_SAMPLE_LOCK = threading.Lock()
_AUTO_LOCK = threading.Lock()
_AUTO_FUTURE: Future | None = None
_AUTO_TRAIN_REQUESTED = False
_AUTO_PENDING_TARGETS: dict[str, dict[str, Any]] = {}

_YOLO_TYPE_CLASSES = (
    "battery", "button", "buzzer", "capacitor", "clock", "connector", "diode",
    "display", "fuse", "heatsink", "ic", "inductor", "led", "pads", "pins",
    "potentiometer", "relay", "resistor", "switch", "transformer", "transistor",
)


def yolo_training_targets(weights_root: Path) -> list[dict[str, Any]]:
    root = Path(weights_root).expanduser().resolve()
    return [
        {"model_key": "parsa_locator", "training_mode": "locator",
         "checkpoint": str(root / "any_best.pt"), "source_class_names": ["component"]},
        {"model_key": "pcb_ai_doctor_v12n_locator", "training_mode": "locator",
         "checkpoint": str(root / "external" / "pcb-ai-doctor" / "best_v12n.pt"),
         "source_class_names": ["component"]},
        {"model_key": "arshia_pcb_yolov8s_type", "training_mode": "component_type",
         "checkpoint": str(root / "external" / "pcb-yolov8s-detection" / "best.pt"),
         "source_class_names": list(_YOLO_TYPE_CLASSES)},
    ]


def active_training_checkpoint(model_key: str) -> dict[str, Any] | None:
    """Return the latest successful user-trained checkpoint for automatic inference.

    Successful fine-tunes are applied to subsequent inference automatically; they
    remain private and are not mixed into the reviewed-box dataset. The baseline
    can be selected explicitly for controlled evaluation with
    CIRCUITCARE_USE_TRAINED_CHECKPOINTS=0.
    """
    if os.environ.get("CIRCUITCARE_USE_TRAINED_CHECKPOINTS", "1").strip().lower() in {"0", "false", "no", "off"}:
        return None
    with _db() as db:
        row = db.execute(
            "SELECT job_id,training_mode,output_dir,finished_at FROM training_jobs "
            "WHERE model_key=? AND status='SUCCEEDED' ORDER BY finished_at DESC LIMIT 1",
            (str(model_key),),
        ).fetchone()
    if not row:
        return None
    output_dir = Path(row["output_dir"])
    metadata_path = output_dir / "candidate.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint = Path(metadata["candidate_weights"]).expanduser().resolve()
        if (metadata.get("job_id") != row["job_id"]
                or metadata.get("model_key") != model_key
                or metadata.get("training_mode") != row["training_mode"]
                or not checkpoint.is_file()):
            return None
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return {"job_id": str(row["job_id"]), "model_key": str(model_key),
            "training_mode": str(row["training_mode"]), "checkpoint": checkpoint,
            "finished_at": str(row["finished_at"] or ""),
            "independent_validation_available": bool(metadata.get("independent_validation_available")),
            "dataset_sha256": str(metadata.get("dataset_sha256") or "")}


def _db() -> sqlite3.Connection:
    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS reviewed_images (
            image_sha256 TEXT PRIMARY KEY, project_id TEXT NOT NULL, photo_id TEXT NOT NULL,
            image_path TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reviewed_boxes (
            image_sha256 TEXT NOT NULL, component_id TEXT NOT NULL, class_name TEXT NOT NULL,
            x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
            PRIMARY KEY (image_sha256, component_id),
            FOREIGN KEY (image_sha256) REFERENCES reviewed_images(image_sha256)
        );
        CREATE TABLE IF NOT EXISTS component_knowledge (
            knowledge_id INTEGER PRIMARY KEY AUTOINCREMENT, component_type TEXT NOT NULL,
            marking_key TEXT NOT NULL, part_number TEXT NOT NULL, subtype TEXT NOT NULL,
            functional_role TEXT NOT NULL, notes TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE(component_type, marking_key, part_number)
        );
        CREATE TABLE IF NOT EXISTS user_component_types (
            type_id TEXT PRIMARY KEY, display_name TEXT NOT NULL UNIQUE,
            parent_class TEXT NOT NULL, description TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS training_jobs (
            job_id TEXT PRIMARY KEY, dataset_sha256 TEXT NOT NULL, status TEXT NOT NULL,
            pid INTEGER, output_dir TEXT NOT NULL, log_path TEXT NOT NULL,
            started_at TEXT NOT NULL, finished_at TEXT, error TEXT,
            model_key TEXT NOT NULL DEFAULT 'arshia_pcb_yolov8s',
            training_mode TEXT NOT NULL DEFAULT 'component_type',
            source_class_names_json TEXT NOT NULL DEFAULT '[]'
        );
        CREATE INDEX IF NOT EXISTS training_jobs_dataset_status
            ON training_jobs(dataset_sha256, status);
    """)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(training_jobs)")}
    for name, declaration in (
        ("model_key", "TEXT NOT NULL DEFAULT 'arshia_pcb_yolov8s'"),
        ("training_mode", "TEXT NOT NULL DEFAULT 'component_type'"),
        ("source_class_names_json", "TEXT NOT NULL DEFAULT '[]'"),
    ):
        if name not in columns:
            connection.execute(f"ALTER TABLE training_jobs ADD COLUMN {name} {declaration}")
    connection.executescript("""
        DROP INDEX IF EXISTS training_jobs_one_active_dataset;
        DROP INDEX IF EXISTS training_jobs_one_successful_dataset;
        CREATE UNIQUE INDEX IF NOT EXISTS training_jobs_one_active_dataset_model
            ON training_jobs(dataset_sha256, model_key) WHERE status IN ('QUEUED','RUNNING');
        CREATE UNIQUE INDEX IF NOT EXISTS training_jobs_one_successful_dataset_model
            ON training_jobs(dataset_sha256, model_key) WHERE status='SUCCEEDED';
        CREATE TABLE IF NOT EXISTS app_settings (
            setting_key TEXT PRIMARY KEY, setting_value TEXT NOT NULL, updated_at TEXT NOT NULL
        );
    """)
    return connection


def auto_training_enabled() -> bool:
    with _db() as db:
        row = db.execute("SELECT setting_value FROM app_settings WHERE setting_key='auto_yolo_training'").fetchone()
    return bool(row and row[0] == "true")


def set_auto_training_enabled(enabled: bool) -> None:
    with _db() as db:
        db.execute("INSERT INTO app_settings(setting_key,setting_value,updated_at) VALUES('auto_yolo_training',?,?) "
                   "ON CONFLICT(setting_key) DO UPDATE SET setting_value=excluded.setting_value, "
                   "updated_at=excluded.updated_at",
                   ("true" if enabled else "false", datetime.now(timezone.utc).isoformat()))


def _pixel_fingerprint(image_bytes: bytes) -> tuple[str, int, int]:
    from PIL import Image
    import io

    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        digest = hashlib.sha256(f"{width}x{height}:RGB:".encode() + rgb.tobytes()).hexdigest()
    return digest, width, height


def _holdout_pixel_fingerprints() -> set[str]:
    """Return pixel hashes reserved for external validation, never for training.

    A caller can override the validation list with one or more manifest paths
    separated by the platform path separator. The default includes both the
    original 50-photo holdout and the multi-class EPID validation photos.
    """
    configured = os.environ.get("CIRCUITCARE_VALIDATION_HOLDOUT_MANIFEST")
    if configured:
        paths = [Path(value).expanduser() for value in configured.split(os.pathsep) if value.strip()]
    else:
        root = Path.home() / ".local/share/circuitcare-ai/external_datasets"
        paths = [root / "pcb-dslr" / "holdout-50.json",
                 root / "epid" / "validation" / "holdout-101.json"]
    fingerprints: set[str] = set()
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        fingerprints.update(str(case.get("training_exclusion_pixel_sha256"))
                            for case in payload.get("cases", [])
                            if isinstance(case, dict) and case.get("training_exclusion_pixel_sha256"))
    return fingerprints


def save_reviewed_boxes(image_bytes: bytes, *, project_id: str, photo_id: str,
                        rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Upsert explicitly user-labeled boxes; duplicate pixels count as one image."""
    fingerprint, width, height = _pixel_fingerprint(image_bytes)
    if fingerprint in _holdout_pixel_fingerprints():
        return {"image_sha256": fingerprint, "unique_images": 0, "box_count": 0,
                "added_boxes": 0, "excluded_from_training": True,
                "reason": "외부 PCB 평가 홀드아웃 사진은 학습 데이터에 포함하지 않습니다."}
    image_dir = PRIVATE_ROOT / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{fingerprint}.png"
    if not image_path.exists():
        from PIL import Image
        import io

        with Image.open(io.BytesIO(image_bytes)) as source:
            source.convert("RGB").save(image_path, format="PNG", optimize=True)
    now = datetime.now(timezone.utc).isoformat()
    annotations = []
    with _db() as db:
        custom_types = {row["type_id"] for row in db.execute(
            "SELECT type_id FROM user_component_types").fetchall()}
    for row in rows:
        custom_type_id = str(row.get("user_component_type_id") or "")
        kind = (custom_type_id if custom_type_id in custom_types else
                normalize_component_class(row.get("confirmed_base_class") or row.get("effective_class")
                                          or row.get("소자 종류") or "unknown", allow_not_component=False))
        explicitly_labeled = row.get("training_label_user_confirmed") is True
        if (row.get("상태") == "제외" or (kind not in COMPONENT_CLASSES and kind not in custom_types)
                or not explicitly_labeled):
            continue
        try:
            x1, y1, x2, y2 = (float(row[key]) for key in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError, ValueError):
            continue
        x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
        y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        annotations.append((fingerprint, str(row.get("component_id") or uuid.uuid4().hex),
                            kind, x1, y1, x2, y2))
    with _db() as db:
        db.execute("INSERT INTO reviewed_images VALUES(?,?,?,?,?,?,?,?) "
                   "ON CONFLICT(image_sha256) DO UPDATE SET project_id=excluded.project_id, "
                   "photo_id=excluded.photo_id, updated_at=excluded.updated_at",
                   (fingerprint, str(project_id), str(photo_id), str(image_path), width, height, now, now))
        # A reviewed image is a current snapshot: remove labels that were removed or excluded.
        db.execute("DELETE FROM reviewed_boxes WHERE image_sha256=?", (fingerprint,))
        db.executemany("INSERT INTO reviewed_boxes VALUES(?,?,?,?,?,?,?)", annotations)
        unique_images = int(db.execute("SELECT COUNT(*) FROM reviewed_images WHERE EXISTS "
                                       "(SELECT 1 FROM reviewed_boxes WHERE reviewed_boxes.image_sha256="
                                       "reviewed_images.image_sha256)").fetchone()[0])
        box_count = int(db.execute("SELECT COUNT(*) FROM reviewed_boxes").fetchone()[0])
    return {"image_sha256": fingerprint, "unique_images": unique_images,
            "box_count": box_count, "added_boxes": len(annotations)}


def user_component_types() -> list[dict[str, str]]:
    with _db() as db:
        return [dict(row) for row in db.execute(
            "SELECT type_id,display_name,parent_class,description FROM user_component_types "
            "ORDER BY display_name COLLATE NOCASE").fetchall()]


def save_user_component_type(*, display_name: str, parent_class: str, description: str = "") -> dict[str, str]:
    name = " ".join(str(display_name or "").split())[:80]
    parent = normalize_component_class(parent_class, allow_not_component=False)
    if not name:
        raise ValueError("추가할 소자 종류 이름을 입력해 주세요.")
    if parent not in COMPONENT_CLASSES:
        raise ValueError("기존 18종 중 진단에 사용할 상위 종류를 선택해 주세요.")
    if name.casefold() in set(COMPONENT_CLASSES) | {"unknown", "not_component"}:
        raise ValueError("기존 분류와 겹치는 이름은 추가할 수 없습니다.")
    type_id = "user_" + hashlib.sha256(name.casefold().encode("utf-8")).hexdigest()[:12]
    item = {"type_id": type_id, "display_name": name, "parent_class": parent,
            "description": " ".join(str(description or "").split())[:300]}
    try:
        with _db() as db:
            db.execute("INSERT INTO user_component_types(type_id,display_name,parent_class,description,created_at) "
                       "VALUES(?,?,?,?,?)", (*item.values(), datetime.now(timezone.utc).isoformat()))
    except sqlite3.IntegrityError as exc:
        raise ValueError("이미 등록된 소자 종류입니다.") from exc
    return item


def queue_reviewed_boxes(image_bytes: bytes, *, project_id: str, photo_id: str,
                         rows: list[dict[str, Any]]) -> None:
    """Persist a copy of a reviewed photo asynchronously and without UI state."""
    with _SAMPLE_LOCK:
        for future in list(_PENDING_SAVES):
            if future.done():
                _PENDING_SAVES.remove(future)
                try:
                    future.result()
                except Exception as exc:
                    _SAVE_ERRORS.append(f"{type(exc).__name__}: {exc}")
        _PENDING_SAVES.append(_SAMPLE_WRITER.submit(
            save_reviewed_boxes, bytes(image_bytes), project_id=str(project_id),
            photo_id=str(photo_id), rows=json.loads(json.dumps(rows, default=str)),
        ))


def flush_review_samples() -> list[str]:
    """Wait for pending background sample writes before starting model training."""
    with _SAMPLE_LOCK:
        errors = list(_SAVE_ERRORS)
        _SAVE_ERRORS.clear()
    while True:
        with _SAMPLE_LOCK:
            pending = list(_PENDING_SAVES)
            _PENDING_SAVES.clear()
        if not pending:
            break
        for future in pending:
            try:
                future.result()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
    return errors


def save_component_knowledge(*, component_type: str, marking: str, part_number: str,
                             subtype: str, functional_role: str, notes: str) -> None:
    kind = normalize_component_class(component_type, allow_not_component=False)
    if kind not in COMPONENT_CLASSES:
        raise ValueError("기존 18종 중 하나로 소자 종류를 지정해야 합니다.")
    marking = " ".join(str(marking or "").split())[:100]
    part_number = " ".join(str(part_number or "").split())[:100]
    marking_key = "".join(character for character in (marking or part_number).upper() if character.isalnum())
    if not marking_key:
        raise ValueError("다음 분류에서 찾을 각인 또는 부품번호를 입력해 주세요.")
    with _db() as db:
        db.execute("INSERT INTO component_knowledge(component_type,marking_key,part_number,subtype,"
                   "functional_role,notes,updated_at) VALUES(?,?,?,?,?,?,?) "
                   "ON CONFLICT(component_type,marking_key,part_number) DO UPDATE SET subtype=excluded.subtype, "
                   "functional_role=excluded.functional_role,notes=excluded.notes,updated_at=excluded.updated_at",
                   (kind, marking_key, part_number, str(subtype or "")[:100],
                    str(functional_role or "")[:120], str(notes or "")[:500],
                    datetime.now(timezone.utc).isoformat()))


def matching_component_knowledge(markings: list[str], component_type: str = "") -> list[dict[str, str]]:
    values = []
    for value in markings:
        if isinstance(value, dict):
            values.extend(value.get(key) for key in ("text", "value", "marking", "candidate") if value.get(key))
        elif value:
            values.append(value)
    keys = {"".join(character for character in str(value).upper() if character.isalnum())
            for value in values if value}
    keys.discard("")
    with _db() as db:
        rows = db.execute("SELECT component_type, marking_key, part_number, subtype, functional_role, notes "
                          "FROM component_knowledge ORDER BY updated_at DESC").fetchall()
    exact = [dict(row) for row in rows if row["marking_key"] in keys
             or "".join(character for character in str(row["part_number"]).upper()
                        if character.isalnum()) in keys]
    if exact:
        return exact[:8]
    normalized_type = normalize_component_class(component_type, allow_not_component=False)
    return [dict(row) for row in rows if row["component_type"] == normalized_type][:4]


def _dataset_snapshot() -> tuple[str, int, list[sqlite3.Row], list[sqlite3.Row]]:
    excluded = _holdout_pixel_fingerprints()
    with _db() as db:
        images = db.execute("SELECT * FROM reviewed_images WHERE EXISTS (SELECT 1 FROM reviewed_boxes "
                            "WHERE reviewed_boxes.image_sha256=reviewed_images.image_sha256) "
                            "ORDER BY image_sha256").fetchall()
        images = [image for image in images if image["image_sha256"] not in excluded]
        active_hashes = {image["image_sha256"] for image in images}
        boxes = [box for box in db.execute("SELECT * FROM reviewed_boxes ORDER BY image_sha256,component_id").fetchall()
                 if box["image_sha256"] in active_hashes]
    digest = hashlib.sha256()
    for image in images:
        digest.update(image["image_sha256"].encode())
        # Board grouping changes the train/validation assignment. Include it in
        # the dataset identity so a cached export cannot retain an old split.
        digest.update(str(image["project_id"] or "").strip().encode())
    for box in boxes:
        digest.update(json.dumps(dict(box), sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest(), len(images), images, boxes


def training_summary() -> dict[str, Any]:
    digest, image_count, _, boxes = _dataset_snapshot()
    with _db() as db:
        last = db.execute("SELECT * FROM training_jobs ORDER BY started_at DESC LIMIT 1").fetchone()
        same = db.execute("SELECT * FROM training_jobs WHERE dataset_sha256=? AND status='SUCCEEDED' "
                          "ORDER BY started_at DESC LIMIT 1", (digest,)).fetchone()
        active = db.execute("SELECT * FROM training_jobs WHERE dataset_sha256=? AND status IN "
                            "('QUEUED','RUNNING') ORDER BY started_at DESC LIMIT 1", (digest,)).fetchone()
    return {"dataset_sha256": digest, "unique_images": image_count, "box_count": len(boxes),
            "last_job": dict(last) if last else None,
            "already_trained": dict(same) if same else None,
            "active_job": dict(active) if active else None}


def _export_yolo_dataset(images: list[sqlite3.Row], boxes: list[sqlite3.Row], dataset_sha256: str,
                         *, model_key: str, training_mode: str,
                         source_class_names: list[str]) -> Path:
    from PIL import Image
    import random

    output = PRIVATE_ROOT / "datasets" / f"{dataset_sha256[:12]}-{model_key}"
    if (output / "data.yaml").is_file():
        return output / "data.yaml"
    if output.exists():
        shutil.rmtree(output)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    image_by_hash = {row["image_sha256"]: row for row in images}
    box_by_image: dict[str, list[sqlite3.Row]] = {}
    for box in boxes:
        box_by_image.setdefault(box["image_sha256"], []).append(box)
    custom_ids = sorted({str(box["class_name"]) for box in boxes
                         if str(box["class_name"]) not in COMPONENT_CLASSES})
    class_names = ["component"] if training_mode == "locator" else [*source_class_names, *custom_ids]
    class_to_id = {name: index for index, name in enumerate(class_names)}
    image_hashes = [str(row["image_sha256"]) for row in images]
    board_images: dict[str, list[str]] = {}
    for row in images:
        board_group = str(row["project_id"] or "").strip() or f"image:{row['image_sha256']}"
        board_images.setdefault(board_group, []).append(str(row["image_sha256"]))
    board_groups = sorted(board_images)
    random.Random(21).shuffle(board_groups)
    if len(board_groups) > 1:
        val_board_count = min(len(board_groups) - 1, max(1, round(len(board_groups) * 0.2)))
        validation_boards = set(board_groups[:val_board_count])
    else:
        # One board cannot provide independent validation. Keep its photos in
        # train and copy one source photo's examples into val for Ultralytics;
        # the candidate metadata will explicitly mark this validation in-sample.
        validation_boards = set()
    validation = {image_hash for group in validation_boards for image_hash in board_images[group]}
    exported_files_by_source: dict[str, list[str]] = {image_hash: [] for image_hash in image_hashes}
    for image_hash in image_hashes:
        image_row = image_by_hash[image_hash]
        split = "val" if image_hash in validation else "train"
        source_path = Path(image_row["image_path"])
        with Image.open(source_path) as source:
            source = source.convert("RGB")
            width, height = source.size
            if training_mode == "locator":
                target_name = f"{image_hash}.png"
                source.save(output / "images" / split / target_name, format="PNG", optimize=True)
                exported_files_by_source[image_hash].append(target_name)
                label_lines = []
                for box in box_by_image.get(image_hash, []):
                    x1, y1, x2, y2 = map(float, (box["x1"], box["y1"], box["x2"], box["y2"]))
                    center_x, center_y = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
                    box_width, box_height = (x2 - x1) / width, (y2 - y1) / height
                    label_lines.append(f"0 {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}")
                (output / "labels" / split / f"{image_hash}.txt").write_text("\n".join(label_lines) + "\n")
                continue

            for box in box_by_image.get(image_hash, []):
                stored_name = str(box["class_name"])
                if stored_name in custom_ids:
                    label_name = stored_name
                else:
                    canonical = normalize_component_class(stored_name, allow_not_component=False)
                    label_name = next((name for name in source_class_names if name == canonical), None)
                    if label_name is None:
                        label_name = next((name for name in source_class_names
                                           if normalize_component_class(name, allow_not_component=False) == canonical), None)
                class_id = class_to_id.get(label_name or "")
                if class_id is None:
                    raise ValueError(f"학습 표본에 등록되지 않은 소자 종류가 있습니다: {stored_name}")
                x1, y1, x2, y2 = map(int, (box["x1"], box["y1"], box["x2"], box["y2"]))
                x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
                y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                crop = source.crop((x1, y1, x2, y2))
                crop_key = hashlib.sha256(f"{image_hash}:{box['component_id']}".encode()).hexdigest()
                target_name = f"{crop_key}.png"
                crop.save(output / "images" / split / target_name, format="PNG", optimize=True)
                exported_files_by_source[image_hash].append(target_name)
                # The crop itself is the labeled object; train only the type head here.
                (output / "labels" / split / f"{crop_key}.txt").write_text(
                    f"{class_id} 0.5 0.5 0.95 0.95\n")
    if len(board_groups) == 1:
        source_hash = sorted(image_hashes)[0]
        for filename in exported_files_by_source[source_hash]:
            shutil.copy2(output / "images" / "train" / filename,
                         output / "images" / "val" / filename)
            label_name = Path(filename).with_suffix(".txt").name
            shutil.copy2(output / "labels" / "train" / label_name,
                         output / "labels" / "val" / label_name)
    yaml_text = (f"path: {output.resolve()}\ntrain: images/train\nval: images/val\n"
                 f"unique_source_images: {len(image_hashes)}\n"
                 f"unique_source_boards: {len(board_groups)}\nnames:\n")
    yaml_text += "".join(f"  {index}: {name}\n" for index, name in enumerate(class_names))
    (output / "data.yaml").write_text(yaml_text)
    return output / "data.yaml"


def start_retraining(*, checkpoint: Path, epochs: int = 10, model_key: str = "arshia_pcb_yolov8s",
                     training_mode: str = "component_type",
                     source_class_names: list[str] | None = None) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"로컬 초기 가중치를 찾을 수 없습니다: {checkpoint}")
    if training_mode not in {"locator", "component_type"}:
        raise ValueError("지원하지 않는 YOLO 학습 종류입니다.")
    if source_class_names is None:
        source_class_names = list(_YOLO_TYPE_CLASSES) if training_mode == "component_type" else ["component"]
    dataset_sha, image_count, images, boxes = _dataset_snapshot()
    if image_count < MIN_UNIQUE_IMAGES:
        raise ValueError(f"재학습에는 중복 제거 후 {MIN_UNIQUE_IMAGES}장 이상 필요합니다. 현재 {image_count}장입니다.")
    training_sha = hashlib.sha256(json.dumps({"dataset": dataset_sha, "model": model_key,
                                              "mode": training_mode,
                                              "checkpoint": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                              "classes": source_class_names}, sort_keys=True).encode()).hexdigest()
    with _db() as db:
        if db.execute("SELECT 1 FROM training_jobs WHERE dataset_sha256=? AND model_key=? AND status='SUCCEEDED'",
                       (training_sha, model_key)).fetchone():
            return {"status": "ALREADY_TRAINED", "dataset_sha256": training_sha, "model_key": model_key}
        active = db.execute("SELECT job_id FROM training_jobs WHERE dataset_sha256=? AND model_key=? AND status IN "
                            "('QUEUED','RUNNING')", (training_sha, model_key)).fetchone()
        if active:
            return {"status": "RUNNING", "job_id": active[0], "dataset_sha256": training_sha,
                    "model_key": model_key}
    data_yaml = _export_yolo_dataset(images, boxes, training_sha, model_key=model_key,
                                     training_mode=training_mode, source_class_names=source_class_names)
    job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    output_dir = PRIVATE_ROOT / "candidate_weights" / job_id
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "training.log"
    command = [sys.executable, str(ROOT / "scripts" / "component_training.py"), "train",
               "--data", str(data_yaml), "--init-weights", str(checkpoint),
               "--profile", "baseline", "--epochs", str(max(1, min(500, int(epochs)))),
               "--workers", "0", "--batch", "2", "--output-root", str(output_dir / "runs")]
    try:
        with _db() as db:
            db.execute("INSERT INTO training_jobs(job_id,dataset_sha256,status,pid,output_dir,log_path,started_at,"
                       "model_key,training_mode,source_class_names_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (job_id, training_sha, "QUEUED", None,
                                                  str(output_dir), str(log_path),
                                                  datetime.now(timezone.utc).isoformat(), model_key,
                                                  training_mode, json.dumps(source_class_names)))
    except sqlite3.IntegrityError:
        with _db() as db:
            active = db.execute("SELECT job_id FROM training_jobs WHERE dataset_sha256=? AND model_key=? AND status IN "
                                "('QUEUED','RUNNING')", (training_sha, model_key)).fetchone()
            success = db.execute("SELECT job_id FROM training_jobs WHERE dataset_sha256=? AND model_key=? AND status='SUCCEEDED'",
                                 (training_sha, model_key)).fetchone()
        if active:
            return {"status": "RUNNING", "job_id": active[0], "dataset_sha256": training_sha,
                    "model_key": model_key}
        if success:
            return {"status": "ALREADY_TRAINED", "dataset_sha256": training_sha, "model_key": model_key}
        raise
    worker_env = os.environ.copy()
    worker_env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + worker_env.get("PYTHONPATH", "")
    try:
        process = subprocess.Popen([sys.executable, "-m", "circuitcare.vision.review_learning",
                                    "_worker", job_id, *command[1:]], cwd=ROOT,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   start_new_session=True, env=worker_env)
    except OSError as exc:
        with _db() as db:
            db.execute("UPDATE training_jobs SET status='FAILED',finished_at=?,error=? WHERE job_id=?",
                       (datetime.now(timezone.utc).isoformat(), str(exc), job_id))
        raise
    with _db() as db:
        db.execute("UPDATE training_jobs SET status='RUNNING',pid=? WHERE job_id=?",
                   (process.pid, job_id))
    return {"status": "RUNNING", "job_id": job_id, "dataset_sha256": training_sha,
            "pid": process.pid, "output_dir": str(output_dir), "log_path": str(log_path)}


def _auto_training_worker(targets: list[dict[str, Any]], epochs: int) -> list[dict[str, Any]]:
    global _AUTO_TRAIN_REQUESTED, _AUTO_PENDING_TARGETS
    results: list[dict[str, Any]] = []
    while True:
        if not auto_training_enabled():
            return results
        save_errors = flush_review_samples()
        if save_errors:
            results.append({"status": "SAMPLE_SAVE_FAILED", "errors": save_errors})
            return results
        for target in targets:
            if not auto_training_enabled():
                return results
            checkpoint = Path(target["checkpoint"])
            if not checkpoint.is_file():
                results.append({"status": "UNAVAILABLE", "model_key": target["model_key"],
                                "reason": f"가중치 파일 없음: {checkpoint}"})
                continue
            try:
                outcome = start_retraining(checkpoint=checkpoint, epochs=epochs,
                                           model_key=target["model_key"],
                                           training_mode=target["training_mode"],
                                           source_class_names=list(target["source_class_names"]))
                if outcome.get("status") == "RUNNING":
                    job_id = outcome["job_id"]
                    while True:
                        refresh_training_jobs()
                        with _db() as db:
                            row = db.execute("SELECT status,error,output_dir FROM training_jobs WHERE job_id=?",
                                             (job_id,)).fetchone()
                        if not row or row["status"] not in {"QUEUED", "RUNNING"}:
                            outcome = {"status": row["status"] if row else "FAILED", "job_id": job_id,
                                       "model_key": target["model_key"],
                                       "error": row["error"] if row else "학습 작업을 찾을 수 없습니다.",
                                       "output_dir": row["output_dir"] if row else None}
                            break
                        time.sleep(3)
                results.append({**outcome, "model_key": target["model_key"]})
            except Exception as exc:
                results.append({"status": "FAILED", "model_key": target["model_key"],
                                "error": f"{type(exc).__name__}: {exc}"})
        with _AUTO_LOCK:
            if not _AUTO_TRAIN_REQUESTED:
                return results
            targets = list(_AUTO_PENDING_TARGETS.values())
            _AUTO_PENDING_TARGETS = {}
            _AUTO_TRAIN_REQUESTED = False


def queue_auto_retraining(targets: list[dict[str, Any]], *, epochs: int = 10) -> dict[str, Any]:
    """Queue one sequential, idempotent training pass after an explicit review completion."""
    global _AUTO_FUTURE, _AUTO_TRAIN_REQUESTED, _AUTO_PENDING_TARGETS
    if not auto_training_enabled():
        return {"status": "DISABLED"}
    with _AUTO_LOCK:
        if _AUTO_FUTURE is not None and not _AUTO_FUTURE.done():
            _AUTO_PENDING_TARGETS.update({str(target.get("model_key") or target.get("checkpoint")): dict(target)
                                          for target in targets})
            _AUTO_TRAIN_REQUESTED = True
            return {"status": "QUEUED"}
        _AUTO_TRAIN_REQUESTED = False
        _AUTO_PENDING_TARGETS = {}
        _AUTO_FUTURE = _AUTO_TRAINER.submit(_auto_training_worker, list(targets), max(1, int(epochs)))
        return {"status": "QUEUED"}


def latest_training_job() -> dict[str, Any] | None:
    with _db() as db:
        row = db.execute("SELECT job_id,model_key,training_mode,status,started_at,finished_at,error,output_dir "
                         "FROM training_jobs ORDER BY started_at DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def refresh_training_jobs() -> None:
    with _db() as db:
        jobs = db.execute("SELECT * FROM training_jobs WHERE status='RUNNING'").fetchall()
        for job in jobs:
            try:
                os.kill(int(job["pid"]), 0)
                continue
            except (OSError, TypeError, ValueError):
                pass
            log = Path(job["log_path"])
            text = log.read_text(errors="replace") if log.exists() else ""
            if job["status"] in {"RUNNING", "QUEUED"}:
                db.execute("UPDATE training_jobs SET status='FAILED',finished_at=?,error=? WHERE job_id=?",
                           (datetime.now(timezone.utc).isoformat(), text[-1200:] or "Training worker ended unexpectedly.", job["job_id"]))


def _run_training_worker(job_id: str, command: list[str]) -> int:
    with _db() as db:
        job = db.execute("SELECT * FROM training_jobs WHERE job_id=?", (job_id,)).fetchone()
    if not job:
        return 2
    output_dir, log_path = Path(job["output_dir"]), Path(job["log_path"])
    try:
        from ultralytics import YOLO
        import yaml

        model = YOLO(command[command.index("--init-weights") + 1])
        names = model.names
        ordered_names = [names[index] for index in sorted(names)] if isinstance(names, dict) else list(names)
        expected_source_names = json.loads(job["source_class_names_json"] or "[]")
        training_mode = str(job["training_mode"] or "component_type")
        if ordered_names != expected_source_names:
            raise ValueError(f"초기 가중치 클래스가 등록된 설정과 다릅니다: {ordered_names}")
        if training_mode == "locator" and (len(ordered_names) != 1 or ordered_names[0] != "component"):
            raise ValueError("위치 탐지 재학습에는 단일 component 클래스 가중치만 사용할 수 있습니다.")
        if training_mode == "component_type" and not ordered_names:
            raise ValueError("소자 분류용 YOLO 클래스가 비어 있습니다.")
        dataset_config = yaml.safe_load(Path(command[command.index("--data") + 1]).read_text()) or {}
        dataset_names = dataset_config.get("names", {})
        class_names = ([dataset_names[index] for index in sorted(dataset_names)]
                       if isinstance(dataset_names, dict) else list(dataset_names))
        expected_dataset_prefix = ["component"] if training_mode == "locator" else expected_source_names
        if class_names[:len(expected_dataset_prefix)] != expected_dataset_prefix:
            raise ValueError("학습 데이터셋 클래스 순서가 해당 YOLO 가중치와 일치하지 않습니다.")
        if len(class_names) != len(set(class_names)):
            raise ValueError("학습 클래스 이름이 중복되어 있습니다.")
        with log_path.open("wb") as log:
            result = subprocess.run([sys.executable, *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env=os.environ.copy(), check=False)
        record_line = next((line for line in log_path.read_text(errors="replace").splitlines()
                            if line.startswith("Experiment record: ")), None)
        if result.returncode != 0 or not record_line:
            raise RuntimeError(f"학습 프로세스가 실패했습니다 (exit={result.returncode}).")
        experiment = Path(record_line.split(": ", 1)[1])
        best = experiment.parent / "weights" / "best.pt"
        if not best.is_file():
            raise RuntimeError("학습은 종료했지만 후보 best.pt를 만들지 못했습니다.")
        candidate = output_dir / "candidate_best.pt"
        candidate.write_bytes(best.read_bytes())
        metadata = {"job_id": job_id, "dataset_sha256": job["dataset_sha256"],
                    "model_key": job["model_key"], "training_mode": training_mode,
                    "source_checkpoint": command[command.index("--init-weights") + 1],
                    "source_experiment": str(experiment), "candidate_weights": str(candidate),
                    "class_names": class_names, "source_class_names": ordered_names,
                    "class_head_reinitialized": len(class_names) != len(ordered_names),
                    "independent_validation_available": int(dataset_config.get("unique_source_boards", 0)) > 1,
                    "created_at": datetime.now(timezone.utc).isoformat()}
        (output_dir / "candidate.json").write_text(json.dumps(metadata, indent=2))
        with _db() as db:
            db.execute("UPDATE training_jobs SET status='SUCCEEDED',finished_at=?,error=NULL WHERE job_id=?",
                       (datetime.now(timezone.utc).isoformat(), job_id))
        return 0
    except Exception as exc:
        with log_path.open("a") as log:
            log.write(f"\n{type(exc).__name__}: {exc}\n")
        with _db() as db:
            db.execute("UPDATE training_jobs SET status='FAILED',finished_at=?,error=? WHERE job_id=?",
                       (datetime.now(timezone.utc).isoformat(), f"{type(exc).__name__}: {exc}", job_id))
        return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("_worker",))
    parser.add_argument("job_id")
    parser.add_argument("train_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    raise SystemExit(_run_training_worker(args.job_id, args.train_command))


if __name__ == "__main__":
    main()
