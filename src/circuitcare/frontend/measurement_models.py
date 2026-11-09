"""Frontend-owned measurement records; independent from diagnosis model Nodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import uuid
from typing import Any


MEASUREMENT_UNITS: dict[str, dict[str, float]] = {
    "voltage": {"V": 1.0, "mV": 1e-3, "µV": 1e-6},
    "resistance": {"Ω": 1.0, "kΩ": 1e3, "MΩ": 1e6},
    "current": {"A": 1.0, "mA": 1e-3, "µA": 1e-6},
}
BASE_UNITS = {"voltage": "V", "resistance": "Ω", "current": "A"}
MEASUREMENT_TYPE_LABELS = {"voltage": "전압", "resistance": "저항", "current": "전류"}
SOURCE_LABELS = {
    "user_measurement": "실제 측정 · 사용자 입력",
    "simulation": "시뮬레이션",
    "simulated_demo": "가상 데모",
    "imported_measurement": "가져온 측정값",
    "unknown": "출처 미확인",
}


@dataclass(frozen=True)
class MeasurementRecord:
    measurement_id: str
    project_id: str
    board_id: str
    component_id: str
    component_map_version: str | None
    component_identity_status: str
    probe_id: str | None
    node_id: str | None
    mapping_status: str
    measurement_type: str
    value: float
    unit: str
    entered_value: float
    entered_unit: str
    uncertainty: float | None
    source: str
    verification_status: str
    timestamp: str
    analysis_context_id: str | None = None
    measurement_set_id: str | None = None
    changed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_finite_number(value: Any, *, label: str = "측정값") -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label}을(를) 숫자로 입력해 주세요.")
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}을(를) 숫자로 입력해 주세요.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label}은(는) 유한한 숫자여야 합니다.")
    return parsed


def create_measurement_record(
    *, project_id: str, board_id: str, component_id: str,
    component_identity_status: str | None, measurement_type: str,
    value: Any, unit: str, uncertainty: Any = None, source: str = "user_measurement",
    component_map_version: str | None = None,
    analysis_context_id: str | None = None, measurement_id: str | None = None,
    timestamp: str | None = None,
) -> MeasurementRecord:
    if not project_id.strip() or not board_id.strip() or not component_id.strip():
        raise ValueError("프로젝트, 보드, 부품 정보가 필요합니다.")
    if measurement_type not in MEASUREMENT_UNITS:
        raise ValueError("지원하지 않는 측정 종류입니다.")
    if unit not in MEASUREMENT_UNITS[measurement_type]:
        raise ValueError("측정 종류와 단위가 일치하지 않습니다.")
    if source not in SOURCE_LABELS:
        raise ValueError("측정 출처를 확인해 주세요.")
    entered = parse_finite_number(value)
    multiplier = MEASUREMENT_UNITS[measurement_type][unit]
    converted = entered * multiplier
    if not math.isfinite(converted):
        raise ValueError("단위 변환 결과가 유효한 범위를 벗어났습니다.")
    sigma = None
    if uncertainty is not None and str(uncertainty).strip():
        entered_sigma = parse_finite_number(uncertainty, label="불확실도")
        if entered_sigma <= 0:
            raise ValueError("불확실도는 0보다 커야 합니다.")
        sigma = entered_sigma * multiplier
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("불확실도 단위 변환 결과가 유효하지 않습니다.")
    verification = {
        "user_measurement": "USER_REPORTED",
        "simulation": "MODEL_OUTPUT",
        "simulated_demo": "SIMULATED",
        "imported_measurement": "UNVERIFIED",
        "unknown": "UNVERIFIED",
    }[source]
    return MeasurementRecord(
        measurement_id=measurement_id or str(uuid.uuid4()),
        project_id=project_id, board_id=board_id, component_id=component_id,
        component_map_version=component_map_version,
        component_identity_status=component_identity_status or "UNKNOWN",
        probe_id=None, node_id=None, mapping_status="NODE_MAPPING_UNAVAILABLE",
        measurement_type=measurement_type, value=converted, unit=BASE_UNITS[measurement_type],
        entered_value=entered, entered_unit=unit, uncertainty=sigma,
        source=source, verification_status=verification,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        analysis_context_id=analysis_context_id,
    )


def measurement_set_sha256(records: list[MeasurementRecord], *, project_id: str, board_id: str) -> str:
    payload = {
        "project_id": project_id,
        "board_id": board_id,
        "measurements": [
            {key: value for key, value in row.to_dict().items() if key != "measurement_set_id"}
            for row in sorted(records, key=lambda item: item.measurement_id)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def assign_measurement_set(records: list[MeasurementRecord], *, project_id: str, board_id: str) -> list[MeasurementRecord]:
    set_id = measurement_set_sha256(records, project_id=project_id, board_id=board_id)
    return [replace(record, measurement_set_id=set_id) for record in records]
