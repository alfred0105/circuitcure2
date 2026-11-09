"""Compare observed before/after measurements without claiming causation."""

from __future__ import annotations

import math
from typing import Any


def compare_before_after(before: list[dict[str, Any]], after: list[dict[str, Any]], *,
                         expected: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    """Compare matching measurements; simulated and virtual values are not physical evidence."""
    expected = expected or {}
    by_after = {row.get("node"): row for row in after if isinstance(row, dict)}
    comparisons = []
    for pre in before:
        post = by_after.get(pre.get("node"))
        if not post:
            continue
        same_conditions = pre.get("conditions", {}) == post.get("conditions", {})
        same_unit = pre.get("unit") == post.get("unit")
        measured_sources = {"user_measurement", "instrument_recorded"}
        physical = pre.get("source") in measured_sources and post.get("source") in measured_sources
        valid = (same_conditions and same_unit and physical and
                 all(isinstance(pre.get(key), (int, float)) and math.isfinite(pre[key]) for key in ("value", "uncertainty")) and
                 all(isinstance(post.get(key), (int, float)) and math.isfinite(post[key]) for key in ("value", "uncertainty")) and
                 pre["uncertainty"] > 0 and post["uncertainty"] > 0)
        target = expected.get(pre.get("node"), {})
        in_range = valid and isinstance(target.get("minimum"), (int, float)) and isinstance(target.get("maximum"), (int, float)) and target["minimum"] <= post["value"] <= target["maximum"]
        comparisons.append({"node": pre.get("node"), "unit": pre.get("unit"),
            "before": pre.get("value"), "after": post.get("value"),
            "delta_after_minus_before": post.get("value") - pre.get("value") if valid else None,
            "conditions_match": same_conditions, "unit_matches": same_unit,
            "physical_measurements": physical, "comparable": valid,
            "expected_range": target or None,
            "post_repair_measurement_matched_expected": in_range if target else None,
            "interpretation": "Measurement change only; not proof that a single intervention caused recovery."})
    comparable_count = sum(row["comparable"] for row in comparisons)
    return {"status": "COMPARED" if comparable_count else "NOT_COMPARABLE",
            "comparison_count": len(comparisons), "comparable_count": comparable_count,
            "comparisons": comparisons,
            "causality_note": "A post-repair measurement match does not identify a unique root cause, especially after multiple interventions."}
