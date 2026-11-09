"""Read-only adapter for the pinned PRISM PCB reconstruction dataset.

Ground-truth records are deliberately loaded separately from image inference.
Callers should pass only ``top_image``/``bottom_image`` to a reconstruction
pipeline, then load ``references`` after predictions have been produced.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PRISM_COMMIT = "9189c87bbec6995b745b809db99cef6d3269401f"
SAMPLES = ("Cascade", "Connector", "Crossover", "Divider", "Elements", "Filter",
           "Floating", "Groundplane", "Meander", "Round", "Simple", "Transistors")
REFERENCE_FILES = ("expected_geometry.json", "expected_components.json", "expected_netlist.json",
                   "expected_stats.json", "expected_schematic_components.json")


@dataclass(frozen=True)
class PrismSample:
    sample_id: str
    top_image: Path
    bottom_image: Path
    reference_pcb: Path
    reference_schematic: Path
    source_commit: str
    input_hashes: dict[str, str]

    @property
    def dataset_hash(self) -> str:
        payload = "\n".join(f"{name}:{digest}" for name, digest in sorted(self.input_hashes.items()))
        return hashlib.sha256(payload.encode()).hexdigest()


def locate_prism_root(root: str | Path) -> Path:
    """Accept either the sparse checkout root or its Dataset directory."""
    path = Path(root).expanduser().resolve()
    if (path / "Dataset").is_dir() and (path / "evaluation" / "reference").is_dir():
        return path
    if path.name.lower() == "dataset" and path.is_dir():
        candidate = path.parent
        if (candidate / "evaluation" / "reference").is_dir():
            return candidate
    raise FileNotFoundError(f"PRISM root must contain Dataset/ and evaluation/reference/: {path}")


def discover_samples(root: str | Path) -> list[str]:
    prism = locate_prism_root(root)
    return [name for name in SAMPLES if (prism / "Dataset" / name).is_dir()]


def load_sample(root: str | Path, sample_id: str) -> PrismSample:
    prism = locate_prism_root(root)
    name = next((item for item in SAMPLES if item.lower() == sample_id.lower()), None)
    if name is None:
        raise ValueError(f"Unknown PRISM sample: {sample_id}")
    board = prism / "Dataset" / name
    ref_dir = prism / "evaluation" / "reference" / name.lower()
    candidates = {
        "top_image": board / "TOP.png", "bottom_image": board / "BOTTOM.png",
        "reference_pcb": board / name / f"{name}.kicad_pcb",
        "reference_schematic": board / name / f"{name}.kicad_sch",
    }
    missing = [str(path.relative_to(prism)) for path in candidates.values() if not path.is_file()]
    missing.extend(str((ref_dir / file).relative_to(prism)) for file in REFERENCE_FILES
                   if not (ref_dir / file).is_file())
    if missing:
        raise FileNotFoundError(f"{name}: required PRISM files missing: {', '.join(missing)}")
    # Hashing verifies provenance only; expected annotations are intentionally
    # not parsed here so inference receives no Ground Truth data.
    # Only image/design input files are hashed here. Reference annotations are
    # not read or hashed until the separate evaluation stage.
    hashes = {path.relative_to(prism).as_posix(): sha256_file(path) for path in candidates.values()}
    return PrismSample(name, candidates["top_image"], candidates["bottom_image"],
                       candidates["reference_pcb"], candidates["reference_schematic"],
                       PRISM_COMMIT, hashes)


def load_references(root: str | Path, sample_id: str) -> dict[str, dict[str, Any]]:
    """Load PRISM annotations separately, only after image predictions exist."""
    prism = locate_prism_root(root)
    name = next((item for item in SAMPLES if item.lower() == sample_id.lower()), None)
    if name is None:
        raise ValueError(f"Unknown PRISM sample: {sample_id}")
    ref_dir = prism / "evaluation" / "reference" / name.lower()
    refs = {}
    for filename in REFERENCE_FILES:
        with (ref_dir / filename).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"{filename} must contain a JSON object")
        refs[filename.removesuffix(".json")] = payload
    return refs


def hash_reference_files(root: str | Path, sample_id: str) -> dict[str, str]:
    """Hash annotation bytes after evaluation-stage access is allowed."""
    prism = locate_prism_root(root)
    name = next((item for item in SAMPLES if item.lower() == sample_id.lower()), None)
    if name is None:
        raise ValueError(f"Unknown PRISM sample: {sample_id}")
    ref_dir = prism / "evaluation" / "reference" / name.lower()
    return {(ref_dir / file).relative_to(prism).as_posix(): sha256_file(ref_dir / file)
            for file in REFERENCE_FILES}


def validate_inventory(root: str | Path, *, parse_references: bool = True) -> dict[str, Any]:
    prism = locate_prism_root(root)
    found = discover_samples(prism)
    results = []
    for name in SAMPLES:
        if name not in found:
            results.append({"sample": name, "status": "MISSING_SAMPLE"})
            continue
        try:
            sample = load_sample(prism, name)
            if parse_references:
                load_references(prism, name)  # Fetch validation only; inference uses parse_references=False.
            results.append({"sample": name, "status": "READY", "input_file_count": len(sample.input_hashes),
                            "dataset_hash": sample.dataset_hash})
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            results.append({"sample": name, "status": "INVALID", "error": str(exc)})
    return {"source_commit": PRISM_COMMIT, "expected_samples": len(SAMPLES),
            "available_samples": len(found), "ready_samples": sum(row["status"] == "READY" for row in results),
            "samples": results}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
