#!/usr/bin/env python3
"""Convert YOLO text labels into a provenance-rich, explicitly unreviewed annotation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from PIL import Image
from circuitcare.vision.detection_postprocess import PCB_MODEL_CLASSES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-source", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--annotation-source", required=True)
    parser.add_argument("--split", choices=("development", "training", "validation", "sealed_test"), required=True)
    parser.add_argument("--split-method", required=True)
    parser.add_argument("--board-grouping-policy", required=True)
    parser.add_argument("--class-names", nargs="+", default=list(PCB_MODEL_CLASSES))
    parser.add_argument("--board-map", type=Path, help="optional reviewed JSON map keyed by image filename")
    args = parser.parse_args()
    if len(set(args.class_names)) != len(args.class_names) or not args.class_names:
        parser.error("class names must be unique and non-empty")
    board_map = json.loads(args.board_map.read_text(encoding="utf-8")) if args.board_map else {}
    images = sorted(path for path in args.images.iterdir()
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not images:
        parser.error(f"No images found in {args.images}")
    records = []
    for image_path in images:
        label_path = args.labels / f"{image_path.stem}.txt"
        if not label_path.is_file():
            parser.error(f"Missing label file (not interpreted as a negative image): {label_path}")
        image_bytes, label_bytes = image_path.read_bytes(), label_path.read_bytes()
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        annotations = []
        for line_number, line in enumerate(label_bytes.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                parser.error(f"Invalid YOLO row at {label_path}:{line_number}")
            try:
                class_id, cx, cy, box_width, box_height = int(parts[0]), *map(float, parts[1:])
            except ValueError:
                parser.error(f"Invalid numeric YOLO row at {label_path}:{line_number}")
            if not 0 <= class_id < len(args.class_names):
                parser.error(f"Class index outside supplied taxonomy at {label_path}:{line_number}")
            x1, y1, x2, y2 = cx - box_width / 2, cy - box_height / 2, cx + box_width / 2, cy + box_height / 2
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                parser.error(f"YOLO bounding box outside image at {label_path}:{line_number}")
            annotations.append({"component_id": f"annotation:{line_number:04d}",
                "class": args.class_names[class_id],
                "bbox_xyxy": [x1 * width, y1 * height, x2 * width, y2 * height],
                "occlusion_state": "UNKNOWN", "small_component": None,
                "annotation_status": "DRAFT_ANNOTATION", "reviewer": None,
                "source": f"{args.dataset_source}:{label_path.name}",
                "provenance": {"origin": "source_label_file_not_independently_reviewed",
                               "line_number": line_number}})
        board = board_map.get(image_path.name, {})
        records.append({"image_id": image_path.name,
            "image_path": Path(os.path.relpath(image_path.resolve(), args.output.parent.resolve())).as_posix(),
            "label_path": Path(os.path.relpath(label_path.resolve(), args.output.parent.resolve())).as_posix(),
            "board_id": board.get("board_id", "BOARD_ID_UNRESOLVED"),
            "board_revision": board.get("board_revision", "BOARD_REVISION_UNRESOLVED"),
            "board_side": board.get("board_side", "UNKNOWN"), "split": args.split,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "label_sha256": hashlib.sha256(label_bytes).hexdigest(),
            "image_size_px": [width, height], "source": f"{args.dataset_source}:{image_path.name}",
            "label_state": "LABELED" if annotations else "UNANNOTATED", "annotations": annotations})
    manifest = {"schema_version": "circuitcare.component-annotation.v1",
        "dataset_id": args.dataset_id, "dataset_name": args.dataset_name,
        "dataset_source": args.dataset_source, "dataset_source_url": args.source_url,
        "source_version": args.source_version, "dataset_license": args.license,
        "annotation_source": args.annotation_source,
        "annotation_review_status": "SOURCE_PROVIDED_UNADJUDICATED",
        "split_name": args.split, "split_method": args.split_method,
        "board_grouping_policy": args.board_grouping_policy,
        "split_policy": "Source-provided split; board-level grouping not verified.",
        "class_names": args.class_names, "images": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "image_count": len(records),
        "annotation_count": sum(len(row["annotations"]) for row in records),
        "empty_label_files": sum(row["label_state"] == "UNANNOTATED" for row in records),
        "board_id_unresolved_count": sum(row["board_id"] == "BOARD_ID_UNRESOLVED" for row in records),
        "annotation_status": manifest["annotation_review_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
