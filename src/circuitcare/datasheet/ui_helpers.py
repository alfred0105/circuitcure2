"""Pure data preparation helpers used by the datasheet review screen."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _attach_document(candidate: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    candidate["source_document_sha256"] = document["sha256"]
    return candidate


def _mpn_in_pdf(mpn: str, document: dict[str, Any]) -> bool:
    return bool(mpn.strip()) and any(mpn.casefold() in page["text"].casefold() for page in document["pages"])


def _specs_for_case(root: Path, case: dict[str, Any]) -> dict[str, Any]:
    return json.loads((root / case["ratings"]).read_text(encoding="utf-8"))


def _mapping_for_case(root: Path, case: dict[str, Any]) -> dict[str, Any]:
    return json.loads((root / case["mapping"]).read_text(encoding="utf-8"))


def _risk_row(report: dict[str, Any], reference: str) -> dict[str, Any]:
    if not report:
        return {"component_ref": reference, "risk_level": "UNKNOWN", "priority_rank": None, "priority_score": None, "dominant_metric": None}
    return next((row for row in report.get("secondary_candidates", []) if row.get("component_ref") == reference),
                {"component_ref": reference, "risk_level": "UNKNOWN", "priority_rank": None, "priority_score": None, "dominant_metric": None})


def _comparison_row(label: str, row: dict[str, Any]) -> dict[str, Any]:
    metric = {"voltage": "max_abs_voltage_v", "current": "max_current_a", "power": "max_power_w"}.get(row.get("dominant_metric"))
    values = row.get("metrics", {}).get(metric, {}) if metric else {}
    return {"평가": label, "등급": row.get("risk_level"), "사용률": values.get("utilization_ratio"),
            "정격": values.get("rated_limit"), "단위": values.get("unit"), "점수": row.get("priority_score"),
            "순위": row.get("priority_rank"), "정격 출처": (values.get("rating_source") or {}).get("type")}


