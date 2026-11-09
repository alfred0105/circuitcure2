"""Merge rule and local-model proposals without promoting either to diagnoses."""

import json
from typing import Any


def merge_candidates(rule_rows: list[dict[str, Any]], llm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    seen = set()
    for row in [*rule_rows, *llm_rows]:
        key = (row.get("component_ref"), row.get("fault_type"),
               json.dumps(row.get("parameters", {}), sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(row))
    return merged
