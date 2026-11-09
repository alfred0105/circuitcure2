"""Dataset loading, separation, and split integrity checks."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetBundle:
    public_cases: tuple[dict[str, Any], ...]
    ground_truth: dict[str, tuple[dict[str, Any], ...]]
    manifest: dict[str, Any]
    public_hash: str
    ground_truth_hash: str
    manifest_hash: str


@dataclass(frozen=True)
class PublicDatasetBundle:
    public_cases: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]
    public_hash: str
    manifest_hash: str
    directory: Path


def load_public_dataset(directory: str | Path) -> PublicDatasetBundle:
    """Load and validate only public inputs; never opens the ground-truth file."""
    root = Path(directory)
    public_bytes = (root / "public_cases.json").read_bytes()
    manifest_bytes = (root / "split_manifest.json").read_bytes()
    public, manifest = json.loads(public_bytes), json.loads(manifest_bytes)
    bundle = PublicDatasetBundle(tuple(public.get("cases", [])), manifest, _sha(public_bytes),
                                 _sha(manifest_bytes), root)
    validate_public_dataset(bundle, repository_root=root)
    return bundle


def load_private_ground_truth(directory: str | Path, manifest: dict[str, Any], *,
                              expected_case_ids: set[str]) -> tuple[dict[str, tuple[dict[str, Any], ...]], str]:
    """Read private labels only when the caller enters the scoring stage."""
    truth_bytes = (Path(directory) / "private_ground_truth.json").read_bytes()
    actual_hash = _sha(truth_bytes)
    expected_hash = manifest.get("hashes", {}).get("private_ground_truth")
    if expected_hash and expected_hash != actual_hash:
        raise ValueError("Private ground-truth hash does not match the manifest.")
    truth = json.loads(truth_bytes)
    rows = {key: tuple(items) for key, items in truth.get("cases", {}).items()}
    if set(rows) != expected_case_ids or any(not items for items in rows.values()):
        raise ValueError("Private ground truth must contain non-empty labels for every public case and no extra IDs.")
    return rows, actual_hash


def load_dataset_bundle(directory: str | Path) -> DatasetBundle:
    """Convenience loader for offline scoring tools and tests."""
    public = load_public_dataset(directory)
    truths, truth_hash = load_private_ground_truth(directory, public.manifest,
        expected_case_ids={str(case["case_id"]) for case in public.public_cases})
    bundle = DatasetBundle(public.public_cases, truths, public.manifest, public.public_hash,
                           truth_hash, public.manifest_hash)
    validate_dataset_bundle(bundle, repository_root=Path(directory))
    return bundle


def validate_public_dataset(bundle: PublicDatasetBundle, *, repository_root: str | Path | None = None) -> None:
    cases = bundle.public_cases
    ids = [case.get("case_id") for case in cases]
    if not ids or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("Benchmark public cases must have non-empty case IDs.")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate benchmark case IDs detected.")
    if set(bundle.manifest.get("case_ids", [])) != set(ids):
        raise ValueError("Dataset manifest case IDs do not match the public cases.")
    expected = bundle.manifest.get("hashes", {}).get("public_cases")
    if expected and expected != bundle.public_hash:
        raise ValueError("Dataset public_cases hash does not match the manifest.")
    _validate_public_cases(cases, bundle.manifest, repository_root or bundle.directory)


def validate_dataset_bundle(bundle: DatasetBundle, *, repository_root: str | Path | None = None) -> None:
    public = PublicDatasetBundle(bundle.public_cases, bundle.manifest, bundle.public_hash,
                                 bundle.manifest_hash, Path(repository_root or Path.cwd()))
    validate_public_dataset(public, repository_root=repository_root)
    if set(bundle.public_cases[index]["case_id"] for index in range(len(bundle.public_cases))) != set(bundle.ground_truth):
        raise ValueError("Public cases and private ground truth must have identical case ID sets.")
    expected = bundle.manifest.get("hashes", {}).get("private_ground_truth")
    if expected and expected != bundle.ground_truth_hash:
        raise ValueError("Dataset private_ground_truth hash does not match the manifest.")
    for case_id, rows in bundle.ground_truth.items():
        if not isinstance(rows, tuple) or not rows:
            raise ValueError(f"Ground truth for {case_id} must contain at least one hypothesis.")


def _validate_public_cases(cases, manifest, repository_root):
    splits: dict[str, set[str]] = {}
    for case in cases:
        if any(key in case for key in ("ground_truth", "fault_type", "fault_parameter", "injected_fault")):
            raise ValueError(f"Public case {case['case_id']} contains a private answer field.")
        split, group = str(case.get("split", "")), str(case.get("topology_group", ""))
        if not split or not group:
            raise ValueError(f"Public case {case['case_id']} needs split and topology_group metadata.")
        splits.setdefault(group, set()).add(split)
    if any(len(values) > 1 for values in splits.values()):
        raise ValueError("Cases derived from one topology group cross dataset splits.")
    asset_hashes = manifest.get("asset_hashes", {})
    if asset_hashes:
        search_root = Path(repository_root).resolve()
        repo_root = next((candidate for candidate in (search_root, *search_root.parents)
                          if (candidate / ".git").exists()), None)
        if repo_root is None:
            raise ValueError("Cannot locate repository root to verify circuit asset hashes.")
        for relative, expected in asset_hashes.items():
            asset = repo_root / relative
            if not asset.is_file() or _sha(asset.read_bytes()) != expected:
                raise ValueError(f"Circuit asset hash mismatch: {relative}")


def hash_file(path: str | Path) -> str:
    return _sha(Path(path).read_bytes())


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
