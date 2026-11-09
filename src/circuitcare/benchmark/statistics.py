"""Small-sample summaries and topology-cluster bootstrap intervals."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"status": "NOT_EVALUATED", "count": 0, "mean": None, "median": None,
                "standard_deviation": None, "min": None, "max": None}
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Statistics input must be finite")
    return {"status": "MEASURED", "count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values),
            "standard_deviation": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values)}


def cluster_bootstrap_interval(rows: list[dict[str, Any]], *, value_key: str, cluster_key: str = "topology_group",
                               iterations: int = 2000, seed: int = 1701, confidence: float = 0.95) -> dict[str, Any]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        cluster = row.get(cluster_key)
        if cluster and isinstance(value, (int, float)) and math.isfinite(value):
            clusters[str(cluster)].append(float(value))
    names = sorted(clusters)
    if len(names) < 2:
        return {"status": "NOT_EVALUATED", "reason": "At least two independent clusters are required.",
                "cluster_count": len(names), "lower": None, "upper": None}
    if iterations < 100 or not 0 < confidence < 1:
        raise ValueError("iterations must be >=100 and confidence strictly between zero and one")
    rng = random.Random(seed)
    distribution = []
    for _ in range(iterations):
        sampled = [rng.choice(names) for _ in names]
        values = [value for name in sampled for value in clusters[name]]
        distribution.append(sum(values) / len(values))
    distribution.sort()
    tail = (1 - confidence) / 2
    return {"status": "MEASURED_EXPLORATORY", "method": "cluster bootstrap by topology group",
            "cluster_count": len(names), "iterations": iterations, "confidence": confidence,
            "lower": distribution[int(tail * (iterations - 1))],
            "upper": distribution[int((1 - tail) * (iterations - 1))]}
