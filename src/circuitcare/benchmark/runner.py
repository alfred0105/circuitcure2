"""Versioned result persistence; benchmark execution stays explicit."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .integrity import sha256_json


def reproducibility_metadata(*, dataset_id: str, public_hash: str, ground_truth_hash: str,
                             split: str, prompt_version: str | None = None, model_name: str | None = None,
                             model_digest: str | None = None, simulator_version: str | None = None,
                             seed: int | None = None) -> dict[str, Any]:
    return {"git_sha": _git_sha(), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "dataset_id": dataset_id, "dataset_hash": public_hash,
            "ground_truth_hash": ground_truth_hash, "split": split,
            "prompt_version": prompt_version, "model_name": model_name,
            "model_digest": model_digest, "simulator_version": simulator_version,
            "seed": seed}


def save_result(results_dir: str | Path, payload: dict[str, Any]) -> Path:
    root = Path(results_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_id = str(payload.get("run_id") or f"phase17_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}")
    record = {**payload, "run_id": run_id}
    record.setdefault("result_hash", sha256_json({key: value for key, value in record.items() if key != "result_hash"}))
    target = root / f"{run_id}.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark result: {target}")
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return target


def load_result_records(results_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(results_dir)
    if not root.exists():
        return []
    records = []
    for path in sorted(root.glob("*.json")):
        if path.name == "summary.json":
            continue
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, timeout=3).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
