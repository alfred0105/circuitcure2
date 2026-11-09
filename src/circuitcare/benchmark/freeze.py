"""Hash code that can affect Phase 17 proposal, validation, or ranking results."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


BENCHMARK_SOURCE_FILES = (
    "scripts/run_phase17_benchmark.py",
    "scripts/build_phase17_synthetic_holdout.py",
    "scripts/build_phase17_1_off_grid.py",
    "src/circuitcare/fault_ai/evaluation.py",
    "src/circuitcare/local_llm.py",
    "src/circuitcare/benchmark/dataset.py",
    "src/circuitcare/benchmark/integrity.py",
    "src/circuitcare/benchmark/methods.py",
    "src/circuitcare/benchmark/scoring.py",
    "src/circuitcare/benchmark/statistics.py",
    "src/circuitcare/fault_ai/fault_catalog.py",
    "src/circuitcare/fault_ai/hypothesis_validator.py",
    "src/circuitcare/fault_ai/simulation_bridge.py",
    "src/circuitcare/fault_ai/hypothesis_ranking.py",
    "src/circuitcare/fault_ai/observation_matcher.py",
    "src/circuitcare/fault_ai/llm_hypothesis.py",
    "src/circuitcare/fault_models.py",
    "src/circuitcare/fault_injector.py",
    "src/circuitcare/spice_runner.py",
)


def benchmark_code_fingerprint(root: str | Path) -> dict:
    root = Path(root)
    parts = []
    for relative in BENCHMARK_SOURCE_FILES:
        content = (root / relative).read_bytes()
        parts.append((relative, hashlib.sha256(content).hexdigest()))
    combined = "\n".join(f"{path}:{digest}" for path, digest in parts).encode()
    return {"sha256": hashlib.sha256(combined).hexdigest(),
            "files": {path: digest for path, digest in parts}}


def benchmark_sources_match_commit(root: str | Path, git_sha: str) -> bool:
    paths = [*BENCHMARK_SOURCE_FILES, "src/circuitcare/benchmark/freeze.py"]
    try:
        result = subprocess.run(["git", "diff", "--quiet", git_sha, "--", *paths],
                                cwd=root, timeout=5, check=False)
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", *paths],
                                 cwd=root, capture_output=True, text=True, timeout=5, check=False)
        return result.returncode == 0 and tracked.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
