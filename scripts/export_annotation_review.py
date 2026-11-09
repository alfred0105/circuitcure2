#!/usr/bin/env python3
"""Export a local, image-based review packet without promoting annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from PIL import Image, ImageDraw, ImageFont
from circuitcare.vision.detection_evaluation import validate_annotation_manifest
from circuitcare.vision.detection_postprocess import PCB_MODEL_CLASSES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validation = validate_annotation_manifest(manifest, class_names=list(PCB_MODEL_CLASSES))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    queue = []
    output_manifest = json.loads(json.dumps(manifest))
    for row, output_row in zip(manifest.get("images", []), output_manifest.get("images", [])):
        source = Path(row["image_path"])
        if not source.is_absolute():
            source = (args.manifest.parent / source).resolve()
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        digest = hashlib.sha256(row["image_id"].encode()).hexdigest()[:12]
        filename = f"{digest}.jpg"
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        scale = max(1, round(max(canvas.size) / 1000))
        for annotation in row.get("annotations", []):
            x1, y1, x2, y2 = map(float, annotation["bbox_xyxy"])
            color = "#13a36f" if annotation.get("annotation_status") == "REVIEWED_GROUND_TRUTH" else "#f09b25"
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2 * scale)
            label = f"{annotation.get('component_id')} | {annotation.get('class')} | {annotation.get('annotation_status')}"
            draw.text((x1, max(0, y1 - 14 * scale)), label, fill=color, font=ImageFont.load_default())
        canvas.save(image_dir / filename, quality=90)
        output_row["review_preview"] = f"images/{filename}"
        queue.append({"image_id": row["image_id"], "review_preview": f"images/{filename}",
            "board_id": row.get("board_id"), "board_side": row.get("board_side"),
            "label_state": row.get("label_state"), "annotation_count": len(row.get("annotations", [])),
            "review_status": "PENDING_HUMAN_REVIEW"})
    (args.output_dir / "review_manifest.json").write_text(
        json.dumps(output_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "review_queue.json").write_text(
        json.dumps({"manifest_validation": validation, "images": queue}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    guide = """# Component annotation review packet

Open each file listed in `review_queue.json` alongside its preview under `images/`.

In `review_manifest.json`, correct boxes and classes, add missed components, remove false boxes, and distinguish `REVIEWED_EMPTY` from `UNANNOTATED`. Promote an annotation to `REVIEWED_GROUND_TRUTH` only after a person has inspected it; set a real reviewer and `provenance.reviewed_at` / `provenance.review_method`. Set `small_component` from visual review. Record board identity/side only when independently known. Keep unresolved identities as `BOARD_ID_UNRESOLVED` / `UNKNOWN`. Re-run `scripts/validate_component_dataset.py` before using it for scoreable evaluation.
"""
    (args.output_dir / "README.md").write_text(guide, encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "images_exported": len(queue),
        "annotation_count": sum(row["annotation_count"] for row in queue),
        "ground_truth_promoted": False, "scoreable_status_before_review": validation["scoreable_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
