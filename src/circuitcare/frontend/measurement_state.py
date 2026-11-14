"""Project and board scoped measurement history and diagnosis invalidation."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import uuid
from typing import Any

from circuitcare.frontend.measurement_models import (
    MeasurementRecord, assign_measurement_set, measurement_set_sha256,
)
from circuitcare.analysis.fingerprints import build_measurement_fingerprint
from circuitcare.analysis.state_manager import invalidate_analysis

MEASUREMENTS_KEY = "circuitcare_v5_measurements"
MEASUREMENT_HISTORY_KEY = "circuitcare_v8_measurement_history"
DIAGNOSIS_STATE_KEY = "circuitcare_v5_diagnosis_state"
DIAGNOSIS_PROVIDER_KEY = "circuitcare_v5_diagnosis_provider"
DIAGNOSIS_CONTEXT_KEY = "circuitcare_v5_diagnosis_context"


def owner_key(project_id: str, board_id: str) -> str:
    return f"{len(project_id)}:{project_id}{board_id}"


def get_measurements(state: dict[str, Any], *, project_id: str, board_id: str,
                     component_id: str | None = None,
                     board_component_id: str | None = None) -> list[MeasurementRecord]:
    scoped = state.get(MEASUREMENTS_KEY, {})
    records = list(scoped.get(owner_key(project_id, board_id), [])) if isinstance(scoped, dict) else []
    if component_id is not None:
        records = [record for record in records if record.component_id == component_id]
    if board_component_id is not None:
        records = [record for record in records if record.board_component_id == board_component_id]
    return records


def _append_measurement_history(state: dict[str, Any], record: MeasurementRecord,
                                event_type: str, *, previous: MeasurementRecord | None = None) -> None:
    history = list(state.get(MEASUREMENT_HISTORY_KEY, []))
    history.append({
        "event_id": str(uuid.uuid4()), "event_type": event_type,
        "project_id": record.project_id, "board_id": record.board_id,
        "measurement_id": record.measurement_id,
        "record": record.to_dict(),
        "previous_record": previous.to_dict() if previous is not None else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    state[MEASUREMENT_HISTORY_KEY] = history


def current_measurement_set_sha256(state: dict[str, Any], *, project_id: str, board_id: str) -> str:
    return measurement_set_sha256(get_measurements(state, project_id=project_id, board_id=board_id),
                                   project_id=project_id, board_id=board_id)


def save_measurement(state: dict[str, Any], record: MeasurementRecord) -> None:
    scoped = dict(state.get(MEASUREMENTS_KEY, {}))
    key = owner_key(record.project_id, record.board_id)
    records = list(scoped.get(key, []))
    if any(item.measurement_id == record.measurement_id for item in records):
        raise ValueError("이미 저장된 Measurement ID입니다.")
    if record.board_component_id and (not record.source_photo_id or not record.source_component_id):
        raise ValueError("보드 소자 측정에는 원본 사진과 사진별 소자 ID가 필요합니다.")
    before = build_measurement_fingerprint(records)
    records.append(record)
    scoped[key] = assign_measurement_set(records, project_id=record.project_id, board_id=record.board_id)
    state[MEASUREMENTS_KEY] = scoped
    _append_measurement_history(state, scoped[key][-1], "CREATED")
    if before != build_measurement_fingerprint(scoped[key]):
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
    if (original.component_type, original.board_component_id, original.source_photo_id, original.source_photo_fingerprint,
            original.source_observation_fingerprint, original.source_component_id,
            original.identity_resolution_status) != (
            updated.component_type, updated.board_component_id, updated.source_photo_id, updated.source_photo_fingerprint,
            updated.source_observation_fingerprint, updated.source_component_id,
            updated.identity_resolution_status):
        raise ValueError("측정 기록의 보드 소자 연결과 원본 식별 정보는 수정할 수 없습니다.")
    before = build_measurement_fingerprint(records)
    records[index] = replace(updated, probe_id=original.probe_id, node_id=original.node_id,
                             mapping_status=original.mapping_status,
                             changed_at=updated.changed_at or updated.timestamp,
                             measurement_set_id=None)
    scoped[key] = assign_measurement_set(records, project_id=updated.project_id, board_id=updated.board_id)
    state[MEASUREMENTS_KEY] = scoped
    _append_measurement_history(state, records[index], "UPDATED", previous=original)
    if before != build_measurement_fingerprint(scoped[key]):
        invalidate_diagnosis(state, project_id=updated.project_id, board_id=updated.board_id)


def reconfirm_measurement_identity(
    state: dict[str, Any], *, measurement_id: str, project_id: str, board_id: str,
    board_component_id: str, source_photo_id: str, source_component_id: str,
    source_photo_fingerprint: str, source_observation_fingerprint: str,
    component_type: str, component_map_version: str | None,
) -> None:
    """Explicit user action to reconnect an archived reading to a current observation."""
    records = get_measurements(state, project_id=project_id, board_id=board_id)
    original = next((item for item in records if item.measurement_id == measurement_id), None)
    if original is None:
        raise ValueError("현재 보드에서 측정 기록을 찾을 수 없습니다.")
    if not all((board_component_id, source_photo_id, source_component_id, source_photo_fingerprint,
                source_observation_fingerprint)):
        raise ValueError("현재 사진과 소자 관측의 식별 정보가 충분하지 않습니다.")
    updated = replace(
        original, board_component_id=board_component_id, component_id=board_component_id,
        source_photo_id=source_photo_id, source_component_id=source_component_id,
        source_photo_fingerprint=source_photo_fingerprint,
        source_observation_fingerprint=source_observation_fingerprint,
        component_type=component_type, component_map_version=component_map_version,
        identity_resolution_status="USER_RECONFIRMED_VERIFIED",
        component_identity_status="USER_CONFIRMED",
        changed_at=datetime.now(timezone.utc).isoformat(), measurement_set_id=None,
    )
    scoped = dict(state.get(MEASUREMENTS_KEY, {}))
    key = owner_key(project_id, board_id)
    current = list(scoped.get(key, []))
    index = next(i for i, item in enumerate(current) if item.measurement_id == measurement_id)
    before = build_measurement_fingerprint(current)
    current[index] = updated
    scoped[key] = assign_measurement_set(current, project_id=project_id, board_id=board_id)
    state[MEASUREMENTS_KEY] = scoped
    _append_measurement_history(state, scoped[key][index], "IDENTITY_RECONFIRMED", previous=original)
    if before != build_measurement_fingerprint(scoped[key]):
        invalidate_diagnosis(state, project_id=project_id, board_id=board_id)


def delete_measurement(state: dict[str, Any], *, project_id: str, board_id: str,
                       measurement_id: str) -> bool:
    scoped = dict(state.get(MEASUREMENTS_KEY, {}))
    key = owner_key(project_id, board_id)
    records = list(scoped.get(key, []))
    remaining = [record for record in records if record.measurement_id != measurement_id]
    if len(remaining) == len(records):
        return False
    removed = next(record for record in records if record.measurement_id == measurement_id)
    before = build_measurement_fingerprint(records)
    scoped[key] = assign_measurement_set(remaining, project_id=project_id, board_id=board_id)
    state[MEASUREMENTS_KEY] = scoped
    _append_measurement_history(state, removed, "DELETED")
    invalidate_diagnosis(state, project_id=project_id, board_id=board_id)
    return True


def invalidate_legacy_diagnosis(state: dict[str, Any], *, project_id: str, board_id: str) -> None:
    scoped = dict(state.get(DIAGNOSIS_STATE_KEY, {}))
    owner = owner_key(project_id, board_id)
    previous = dict(scoped.get(owner, {}))
    history = list(previous.get("history", []))
    if previous.get("result") is not None:
        history.append({"result": previous["result"], "context": previous.get("context"),
                        "measurement_set_sha256": previous.get("measurement_set_sha256")})
    scoped[owner] = {
        **previous, "status": "STALE_RESULT", "result": None, "context": None,
        "measurement_set_sha256": None,
        "history": history[-100:],
    }
    state[DIAGNOSIS_STATE_KEY] = scoped


def invalidate_diagnosis(state: dict[str, Any], *, project_id: str, board_id: str) -> None:
    invalidate_legacy_diagnosis(state, project_id=project_id, board_id=board_id)
    invalidate_analysis(state, project_id=project_id, board_id=board_id,
                        reason="측정 기록이 변경되었습니다.")


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
