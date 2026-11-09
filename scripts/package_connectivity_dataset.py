#!/usr/bin/env python3
"""Package the pin-link graphs and provenance manifests for offline training."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = ROOT / "data/benchmarks/phase22_connectivity"
PAIR_DIR = ROOT / "data/external/connectivity_pairs-5000-complete"
SOURCE_DIR = ROOT / "data/external/open-schematics-5000"

FILES = {
    "pin_candidate_graphs.jsonl": GRAPH_DIR / "pin_candidate_graphs.jsonl",
    "extraction_summary.json": GRAPH_DIR / "extraction_summary.json",
    "learning_curve.json": GRAPH_DIR / "learning_curve.json",
    "model_comparison.json": GRAPH_DIR / "model_comparison.json",
    "records.jsonl": PAIR_DIR / "records.jsonl",
    "source_inventory.jsonl": SOURCE_DIR / "source_inventory.jsonl",
    "fetch_summary.json": SOURCE_DIR / "fetch_summary.json",
}

README = """# CircuitCare pin-connectivity training data

This archive contains the ready-to-train pin candidate graphs for the
CircuitCare-AI connectivity link-prediction experiments.

## Contents

- `pin_candidate_graphs.jsonl`: 4,948 schematic graphs. Each line includes
  pin nodes, cKDTree candidate edges, same-net labels, and the project-grouped
  train/validation/test split. The test split has 513 projects.
- `records.jsonl`: paired-example metadata, class labels, source URLs, and
  source SHA-256 values for 4,957 wire-removed/original pairs.
- `source_inventory.jsonl` and `fetch_summary.json`: acquisition provenance
  for the 5,046 source schematics.
- `extraction_summary.json`, `learning_curve.json`, and
  `model_comparison.json`: extraction and model-evaluation records.

Original KiCad schematic files and images are not included. The graphs are
derived from [bshada/open-schematics](https://huggingface.co/datasets/bshada/open-schematics),
whose dataset metadata declares CC BY 4.0. Individual upstream projects may
carry additional or different terms. Use `records.jsonl` and
`source_inventory.jsonl` to retain attribution and inspect each source.

## Train on macOS

From a clone of CircuitCare-AI, install the optional training dependencies,
then extract this archive into the benchmark data directory:

```bash
python3 -m pip install -e ".[connectivity]"
mkdir -p data/benchmarks/phase22_connectivity
tar -xzf circuitcare-connectivity-training-2026-11-08.tar.gz \\
  -C data/benchmarks/phase22_connectivity
python3 scripts/benchmark_connectivity_link_prediction.py sweep \\
  --sizes 200 500 1000 2000 3920
```

KiCad CLI is only needed to regenerate graph labels from original schematics;
training from this archive does not require KiCad.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="output .tar.gz archive path")
    parser.add_argument("--compresslevel", type=int, default=6, choices=range(1, 10))
    parser.add_argument("--force", action="store_true", help="replace an existing archive")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing archive: {output} (pass --force)")
    missing = [str(path) for path in FILES.values() if not path.is_file()]
    if missing:
        raise SystemExit("Missing package inputs:\n" + "\n".join(missing))
    if output in FILES.values():
        raise SystemExit("Output path must not replace a dataset input")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, mode="w:gz", compresslevel=args.compresslevel) as archive:
        for name, path in FILES.items():
            print(f"Adding {name} ({path.stat().st_size:,} bytes)", flush=True)
            archive.add(path, arcname=name, recursive=False)
        info = tarfile.TarInfo("README.md")
        readme_bytes = README.encode("utf-8")
        info.size = len(readme_bytes)
        info.mode = 0o644
        import io
        archive.addfile(info, io.BytesIO(readme_bytes))

    digest = sha256(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(json.dumps({"archive": str(output), "size_bytes": output.stat().st_size,
                      "sha256": digest, "sha256_file": str(sidecar)}, indent=2))


if __name__ == "__main__":
    main()
