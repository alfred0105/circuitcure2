"""Provenance-aware PCB dataset bundle metadata.

The manifest is intentionally permissive: missing artifacts are represented as
NOT_AVAILABLE, never synthesized. Images and source design files remain outside
the application repository unless a caller explicitly chooses another root.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BundleFile:
    path: str | None
    status: str
    sha256: str | None = None


@dataclass(frozen=True)
class CalibrationRecord:
    coordinate_frames: tuple[str, ...]
    correspondence_points: tuple[dict[str, Any], ...]
    transform: dict[str, Any] | None
    validation_results: dict[str, Any] | None
    status: str = "REGISTRATION_UNVERIFIED"


@dataclass(frozen=True)
class PCBDatasetBundle:
    board_id: str
    board_revision: str
    dataset_source: str
    source_commit: str
    license_metadata: dict[str, Any]
    images: dict[str, BundleFile]
    design: dict[str, BundleFile]
    calibration: CalibrationRecord = field(default_factory=lambda: CalibrationRecord(
        ("IMAGE_RAW", "IMAGE_RECTIFIED", "BOARD_CANONICAL", "KICAD_DESIGN", "LAYER_LOCAL"),
        (), None, None,
    ))
    references: dict[str, BundleFile] = field(default_factory=dict)
    schema_version: str = "pcb-dataset-bundle-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def file_record(root: Path, candidate: Path | None) -> BundleFile:
    if candidate is None or not candidate.is_file():
        return BundleFile(None, "NOT_AVAILABLE")
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("Bundle files must be inside the selected dataset root") from exc
    return BundleFile(relative, "AVAILABLE", _sha256(resolved))


def build_prism_bundle(root: str | Path, sample_id: str) -> PCBDatasetBundle:
    """Describe a pinned PRISM sample without loading reference labels as inference input."""
    from circuitcare.vision.reconstruction.prism_dataset import (
        PRISM_COMMIT, load_sample, locate_prism_root,
    )

    prism_root = locate_prism_root(root)
    sample = load_sample(prism_root, sample_id)
    image_records = {
        "TOP": file_record(prism_root, sample.top_image),
        "BOTTOM": file_record(prism_root, sample.bottom_image),
    }
    design_records = {
        "kicad_pcb": file_record(prism_root, sample.reference_pcb),
        "kicad_schematic": file_record(prism_root, sample.reference_schematic),
        "netlist": BundleFile(None, "NOT_AVAILABLE"),
        "position_csv": BundleFile(None, "NOT_AVAILABLE"),
    }
    # PRISM repository uses AGPL-3.0, while its README does not provide a
    # separate dataset-specific grant. Preserve that ambiguity explicitly.
    license_metadata = {
        "repository_license_file": "LICENSE.txt",
        "repository_license": "AGPL-3.0",
        "dataset_specific_license": "NOT_STATED",
        "redistribution_status": "REVIEW_REQUIRED",
        "evidence": "Pinned PRISM repository LICENSE.txt and Dataset/README.md",
    }
    refs = {name: BundleFile(None, "AVAILABLE_SEPARATE_EVALUATION_STAGE")
            for name in ("geometry", "components", "netlist")}
    return PCBDatasetBundle(
        board_id=sample.sample_id,
        board_revision="NOT_AVAILABLE",
        dataset_source="https://github.com/KamMalinski/PRISM",
        source_commit=PRISM_COMMIT,
        license_metadata=license_metadata,
        images=image_records,
        design=design_records,
        references=refs,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
