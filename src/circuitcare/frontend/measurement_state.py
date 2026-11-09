"""Project and board scoped measurement history and diagnosis invalidation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from circuitcare.frontend.measurement_models import (
    MeasurementRecord, assign_measurement_set, measurement_set_sha256,
)

MEASUREMENTS_KEY = "circuitcare_v5_measurements"
DIAGNOSIS_STATE_KEY = "circuitcare_v5_diagnosis_state"
DIAGNOSIS_PROVIDER_KEY = "circuitcare_v5_diagnosis_provider"
DIAGNOSIS_CONTEXT_KEY = "circuitcare_v5_diagnosis_context"


def owner_key(project_id: str, board_id: str) -> str:
    return f"{len(project_id)}:{project_id}{board_id}"


def get_measurements(state: dict[str, Any], *, project_id: str, board_id: str,
                     component_id: str | None = None) -> list[MeasurementRecord]:
    scoped = state.get(MEASUREMENTS_KEY, {})
    records = list(scoped.get(owner_key(project_id, board_id), [])) if isinstance(scoped, dict) else []
    if component_id is not None:
        records = [record for record in records if record.component_id == component_id]
    return records


def current_measurement_set_sha256(state: dict[str, Any], *, project_id: str, board_id: str) -> str:
    return measurement_set_sha256(get_measurements(state, project_id=project_id, board_id=board_id),
                                   project_id=project_id, board_id=board_id)


def save_measurement(state: dict[str, Any], record: MeasurementRecord) -> None:
    scoped = dict(state.get(MEASUREMENTS_KEY, {}))
    key = owner_key(record.project_id, record.board_id)
    records = list(scoped.get(key, []))
    if any(item.measurement_id == record.measurement_id for item in records):
        raise ValueError("이미 저장된 Measurement ID입니다.")
    records.append(record)
    scoped[key] = assign_measurement_set(records, project_id=record.project_id, board_id=record.board_id)
    state[MEASUREMENTS_KEY] = scoped
    invalidate_diagnosis(state, project_id=record.project_id, board_id=record.board_id)


def update_measurement(state: dict[str, Any], updated: MeasurementRecord) -> None:
    scoped = dict(state.get(MEASUREMENTS_KEY, {}))
    key = owner_key(updated.project_id, updated.board_id)
    records = list(scoped.get(key, []))
    index = next((i for i, item in enumerate(records) if item.measurement_id == updated.measurement_id), None)
    if index is None:
        raise ValueError("현재 프로젝트에서 측정 기록을 찾을 수 없습니다.")
    original = records[index]
    if original.component_id != updated.component_id:
        raise ValueError("측정 기록의 부품 연결을 임의로 변경할 수 없습니다.")
    if (original.component_map_version != updated.component_map_version
            or original.component_identity_status != updated.component_identity_status):
        raise ValueError("측정 기록의 Component Map 또는 Identity를 수정할 수 없습니다.")
    records[index] = replace(updated, probe_id=original.probe_id, node_id=original.node_id,
                             mapping_status=original.mapping_status,
                             changed_at=updated.changed_at or updated.timestamp,
                             measurement_set_id=None)
    scoped[key] = assign_measurement_set(records, project_id=updated.project_id, board_id=updated.board_id)
    state[MEASUREMENTS_KEY] = scoped
    invalidate_diagnosis(state, project_id=updated.project_id, board_id=updated.board_id)


def delete_measurement(state: dict[str, Any], *, project_id: str, board_id: str,
                       measurement_id: str) -> bool:
    scoped = dict(state.get(MEASUREMENTS_KEY, {}))
    key = owner_key(project_id, board_id)
    records = list(scoped.get(key, []))
    remaining = [record for record in records if record.measurement_id != measurement_id]
    if len(remaining) == len(records):
        return False
    scoped[key] = assign_measurement_set(remaining, project_id=project_id, board_id=board_id)
    state[MEASUREMENTS_KEY] = scoped
    invalidate_diagnosis(state, project_id=project_id, board_id=board_id)
    return True


def invalidate_diagnosis(state: dict[str, Any], *, project_id: str, board_id: str) -> None:
    scoped = dict(state.get(DIAGNOSIS_STATE_KEY, {}))
    scoped[owner_key(project_id, board_id)] = {
        "status": "STALE_RESULT", "result": None, "context": None,
        "measurement_set_sha256": None,
    }
    state[DIAGNOSIS_STATE_KEY] = scoped


def diagnosis_state(state: dict[str, Any], *, project_id: str, board_id: str) -> dict[str, Any]:
    scoped = state.get(DIAGNOSIS_STATE_KEY, {})
    if not isinstance(scoped, dict):
        return {}
    return dict(scoped.get(owner_key(project_id, board_id), {}))


def clear_measurement_workspace_state(state: Any) -> None:
    """Retire temporary measurement, diagnosis, and injected test-provider state."""
    for key in (MEASUREMENTS_KEY, DIAGNOSIS_STATE_KEY, DIAGNOSIS_PROVIDER_KEY,
                DIAGNOSIS_CONTEXT_KEY):
        state.pop(key, None)
    for key in tuple(state):
        if str(key).startswith("cc_v5_"):
            state.pop(key, None)
