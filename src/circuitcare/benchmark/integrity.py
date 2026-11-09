"""Audit public inference payloads and reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
from typing import Any


FORBIDDEN_PUBLIC_FIELDS = frozenset({"ground_truth", "fault_parameter", "fault_parameters",
                                     "injected_fault", "truth_source", "seed", "seed_metadata",
                                     "private_case_id", "scenario_id"})


def audit_public_payload(payload: dict[str, Any]) -> dict[str, Any]:
    found = sorted(_forbidden_paths(payload))
    if found:
        raise ValueError(f"Answer-bearing fields are forbidden in diagnostic input: {', '.join(found)}")
    return {"status": "PASSED", "forbidden_fields_found": [], "payload_sha256": sha256_json(payload)}


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def audit_holdout_splits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    case_splits: dict[str, set[str]] = {}
    board_splits: dict[str, set[str]] = {}
    for row in rows:
        case_splits.setdefault(str(row.get("case_id", "")), set()).add(str(row.get("split", "")))
        board = row.get("board_instance_id")
        if board:
            board_splits.setdefault(str(board), set()).add(str(row.get("split", "")))
    case_leaks = sorted(key for key, splits in case_splits.items() if len(splits) > 1)
    board_leaks = sorted(key for key, splits in board_splits.items() if len(splits) > 1)
    group_split: dict[str, set[str]] = {}
    for row in rows:
        group_split.setdefault(str(row.get("topology_group", "")), set()).add(str(row.get("split", "")))
    group_leaks = sorted(key for key, splits in group_split.items() if key and len(splits) > 1)
    return {"status": "PASSED" if not (case_leaks or board_leaks or group_leaks) else "FAILED",
            "case_leaks": case_leaks, "board_instance_leaks": board_leaks,
            "topology_group_leaks": group_leaks}


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key in value} | set().union(*(_keys(item) for item in value.values())) if value else set()
    if isinstance(value, (tuple, list)):
        return set().union(*(_keys(item) for item in value)) if value else set()
    return set()


def _forbidden_paths(value: Any, path: tuple[str, ...] = ()) -> set[str]:
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key)
            nested_path = path + (name,)
            if name in FORBIDDEN_PUBLIC_FIELDS:
                found.add(name)
            if name == "fault_type" and not (len(path) >= 1 and path[-1] == "executable_catalog"):
                found.add(name)
            found.update(_forbidden_paths(item, nested_path))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_forbidden_paths(item, path))
    return found
