"""Portable wall-time and Python-allocation measurements."""

from __future__ import annotations

import statistics
import time
import tracemalloc
from typing import Any, Callable


def measure_callable(function: Callable[[], Any], *, warmups: int = 0, repeats: int = 5) -> dict[str, Any]:
    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    for _ in range(warmups):
        function()
    timings = []
    peak_bytes = []
    for _ in range(repeats):
        tracemalloc.start()
        started = time.perf_counter()
        try:
            function()
            timings.append(time.perf_counter() - started)
            _, peak = tracemalloc.get_traced_memory()
            peak_bytes.append(peak)
        finally:
            tracemalloc.stop()
    return {"status": "MEASURED", "warmups": warmups, "repeats": repeats,
            "wall_seconds": {"mean": statistics.mean(timings), "median": statistics.median(timings),
                             "standard_deviation": statistics.stdev(timings) if len(timings) > 1 else None,
                             "min": min(timings), "max": max(timings), "samples": timings},
            "python_tracemalloc_peak_bytes": max(peak_bytes),
            "memory_note": "Python-managed allocations only; excludes native libraries and total process RSS."}


def efficiency_metrics(*, correct_cases: int, elapsed_seconds: float, spice_runs: int) -> dict[str, Any]:
    if correct_cases < 0 or spice_runs < 0 or elapsed_seconds < 0:
        raise ValueError("counts and elapsed_seconds must be non-negative")
    return {"correct_diagnoses_per_second": correct_cases / elapsed_seconds if elapsed_seconds > 0 else None,
            "spice_runs_per_correct_diagnosis": spice_runs / correct_cases if correct_cases > 0 else None,
            "status": "MEASURED" if elapsed_seconds > 0 else "NOT_EVALUATED"}
