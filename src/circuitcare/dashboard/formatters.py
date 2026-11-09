"""Dashboard table and explanation formatting helpers."""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

from circuitcare.dashboard.downloads import MAX_DOWNLOAD_CSV_BYTES


def csv_rows(path: Path, *, expected_sha256: str | None = None) -> list[dict]:
    try:
        if path.stat().st_size > MAX_DOWNLOAD_CSV_BYTES:
            return []
        raw = path.read_bytes()
        if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
            return []
        with io.StringIO(raw.decode("utf-8"), newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error):
        return []


def risk_table(risk: dict) -> list[dict]:
    result = []
    for row in risk.get("secondary_candidates", []):
        metric = row.get("dominant_metric")
        metric_key = {"voltage": "max_abs_voltage_v", "current": "max_current_a", "power": "max_power_w"}.get(metric)
        detail = row.get("metrics", {}).get(metric_key, {}) if metric_key else {}
        result.append({"부품": row.get("component_ref"), "등급": row.get("risk_level"), "사용률": detail.get("utilization_ratio"), "스트레스": detail.get("measured_stress"), "정격": detail.get("rated_limit"), "단위": detail.get("unit"), "점검순위": row.get("priority_rank"), "점수 (확률 아님)": row.get("priority_score"), "정격 출처": (detail.get("rating_source") or {}).get("note"), "미평가": "; ".join(v.get("reason", "") for v in row.get("metrics", {}).values() if v.get("reason"))})
    for row in risk.get("excluded_candidates", []):
        result.append({"부품": row.get("component_ref"), "등급": "EXCLUDED", "미평가": row.get("reason")})
    return result


def stress_table(stress: dict) -> list[dict]:
    rows = []
    for item in stress.get("components", []):
        delta = item.get("delta_fault_minus_baseline", {})
        rows.append({"부품": item.get("component_ref"), "상태": item.get("status"), "전압 변화 (V)": (delta.get("voltage_v") or {}).get("last"), "전류 변화 (A)": (delta.get("current_a") or {}).get("last"), "전력 변화 (W)": (delta.get("power_w") or {}).get("last"), "미지원 항목": "; ".join((item.get("unsupported_metrics") or {}).values())})
    return rows


def fallback_text(evidence: dict) -> str:
    order = evidence.get("analysis_context", {}).get("inspection_order", [])
    if order:
        return f"결정론적 결과에서 {order[0].get('component_ref')}는 우선 점검 후보입니다. 실제 손상은 확인되지 않았습니다. 수치와 가정은 아래 Evidence 자료를 확인하세요."
    return "저장된 분석으로 특정 부품의 상태를 판정할 수 없습니다. 정격과 측정 근거를 확인하세요."


