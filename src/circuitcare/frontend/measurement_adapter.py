"""Build safe frontend records and narrowly project them into supported engine inputs."""

from __future__ import annotations

from typing import Any

from circuitcare.frontend.measurement_models import MeasurementRecord, create_measurement_record


def adapt_measurement_input(**values: Any) -> MeasurementRecord:
    return create_measurement_record(**values)


def to_diagnosis_measurement(record: MeasurementRecord) -> tuple[dict[str, Any] | None, str | None]:
    if record.measurement_type != "voltage" or record.unit != "V":
        return None, "진단 입력은 현재 V 단위 전압만 지원합니다."
    if record.mapping_status != "MAPPED" or not record.probe_id or not record.node_id:
        return None, "전기적 Probe/Node Mapping이 확인되지 않았습니다."
    if record.uncertainty is None or record.uncertainty <= 0:
        return None, "양의 측정 불확실도를 입력해야 합니다."
    if record.source != "user_measurement" or record.verification_status != "USER_REPORTED":
        return None, "검증되지 않은 출처는 진단 입력으로 사용할 수 없습니다."
    return {
        "node": record.node_id, "value": record.value, "unit": "V",
        "uncertainty": record.uncertainty, "source": record.source,
        "measurement_id": record.measurement_id,
        "component_id": record.component_id, "probe_id": record.probe_id,
    }, None
