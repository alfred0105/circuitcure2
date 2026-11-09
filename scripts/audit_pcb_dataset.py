#!/usr/bin/env python3
"""Audit source-split inventory and local image duplicate candidates without assigning board IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def _phash(path: Path) -> int:
    with Image.open(path) as source:
        gray = np.asarray(ImageOps.exif_transpose(source).convert("L").resize((32, 32)), dtype=np.float32)
    transformed = cv2.dct(gray)
    values = transformed[:8, :8].reshape(-1)
    median = float(np.median(values[1:]))
    bits = values > median
    return sum(int(value) << index for index, value in enumerate(bits))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phash-distance", type=int, default=6)
    args = parser.parse_args()
    repo = args.dataset_repo.resolve()
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    paths = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", commit],
                           capture_output=True, text=True, check=True).stdout.splitlines()
    split_images: dict[str, list[str]] = defaultdict(list)
    split_labels: dict[str, list[str]] = defaultdict(list)
    for name in paths:
        match = re.fullmatch(r"dataset/(images|labels)/(train|val)/(.+)", name)
        if match:
            target = split_images if match.group(1) == "images" else split_labels
            target[match.group(2)].append(name)
    class_counts: dict[str, Counter] = defaultdict(Counter)
    empty_labels: dict[str, list[str]] = defaultdict(list)
    for split, names in split_labels.items():
        for name in names:
            content = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{name}"],
                                     capture_output=True, check=True).stdout.decode("utf-8", errors="replace")
            object_count = 0
            for line in content.splitlines():
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) == 5 and fields[0].isdigit():
                    class_counts[split][int(fields[0])] += 1
                    object_count += 1
            if object_count == 0:
                empty_labels[split].append(name)
    local_images = []
    for paths_in_split in split_images.values():
        for name in paths_in_split:
            path = repo / name
            if not path.is_file():
                continue
            raw = path.read_bytes()
            with Image.open(path) as image:
                width, height = ImageOps.exif_transpose(image).size
            local_images.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest(),
                                 "size_px": [width, height], "phash": _phash(path)})
    duplicates = []
    for index, first in enumerate(local_images):
        for second in local_images[index + 1:]:
            hamming = int(first["phash"] ^ second["phash"]).bit_count()
            if first["sha256"] == second["sha256"] or hamming <= args.phash_distance:
                duplicates.append({"first": first["path"], "second": second["path"],
                    "exact_hash_match": first["sha256"] == second["sha256"],
                    "phash_hamming_distance": hamming,
                    "status": "DUPLICATE_CANDIDATE_REQUIRES_HUMAN_BOARD_REVIEW"})
    derivative_groups: dict[str, list[str]] = defaultdict(list)
    for name in split_images.get("train", []):
        stem = Path(name).stem
        base = re.sub(r"__(?:flip|rot_[a-z0-9]+|crop_[a-z0-9]+|resize_[a-z0-9]+)$", "", stem, flags=re.I)
        if base != stem:
            derivative_groups[base].append(name)
    output = {"schema_version": "circuitcare.pcb-dataset-audit.v1",
        "source_repository": "https://github.com/wangshnegqi/pcb-component-detection-dataset-yolo",
        "source_commit": commit, "dataset_license": "CC BY 4.0 (source-declared for images/annotations)",
        "model_repository_license": "Apache-2.0 (source-declared)",
        "source_split_inventory": {split: {"image_count": len(split_images[split]),
            "label_file_count": len(split_labels[split]), "object_count": sum(class_counts[split].values()),
            "class_id_counts": {str(key): value for key, value in sorted(class_counts[split].items())},
            "empty_label_file_count": len(empty_labels[split])} for split in sorted(split_images)},
        "source_board_count": None, "board_identity_status": "BOARD_ID_UNRESOLVED",
        "source_side_metadata": "NOT_PROVIDED", "source_split_method": "IMAGE_LEVEL_SOURCE_SPLIT",
        "same_board_split_leakage": "NOT_TESTABLE_WITHOUT_BOARD_IDS",
        "model_validation_use": "SOURCE_REPOSITORY_INCLUDES_TRAINING_RESULTS_AND_VALIDATION_COMMAND; source val is not a sealed independent holdout",
        "local_image_file_count": len(local_images),
        "local_resolution_summary": {"min_width_px": min((row["size_px"][0] for row in local_images), default=None),
            "max_width_px": max((row["size_px"][0] for row in local_images), default=None),
            "min_height_px": min((row["size_px"][1] for row in local_images), default=None),
            "max_height_px": max((row["size_px"][1] for row in local_images), default=None)},
        "local_exact_or_phash_duplicate_candidates": duplicates,
        "filename_derivative_families": {key: sorted(value) for key, value in sorted(derivative_groups.items())},
        "warnings": ["Image hashes and pHash are only duplicate-review aids; they do not establish physical Board ID.",
            "Filename suffix families can indicate derived images but require source/image confirmation.",
            "Unresolved board identity blocks board-grouped train/validation/test claims."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "source_commit": commit,
        "split_inventory": output["source_split_inventory"], "local_image_file_count": len(local_images),
        "board_id_status": output["board_identity_status"],
        "duplicate_review_candidates": len(duplicates),
        "filename_derivative_families": len(derivative_groups)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
