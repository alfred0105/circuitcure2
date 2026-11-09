"""Strict guard between uploaded-board measurements and supported diagnosis providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from circuitcare.frontend.measurement_models import MeasurementRecord
from circuitcare.frontend.measurement_adapter import to_diagnosis_measurement
from circuitcare.frontend.measurement_models import measurement_set_sha256
from circuitcare.frontend.measurement_state import DIAGNOSIS_CONTEXT_KEY, owner_key


class DiagnosisStatus:
    NO_MEASUREMENT = "NO_MEASUREMENT"
    MEASUREMENT_RECORDED = "MEASUREMENT_RECORDED"
    NODE_MAPPING_UNAVAILABLE = "NODE_MAPPING_UNAVAILABLE"
    HYPOTHESIS_MODEL_UNAVAILABLE = "HYPOTHESIS_MODEL_UNAVAILABLE"
    DIAGNOSIS_READY = "DIAGNOSIS_READY"
    DIAGNOSIS_RUNNING = "DIAGNOSIS_RUNNING"
    DIAGNOSIS_COMPLETE = "DIAGNOSIS_COMPLETE"
    DIAGNOSIS_FAILED = "DIAGNOSIS_FAILED"
    MODEL_MISMATCH = "MODEL_MISMATCH"
    STALE_RESULT = "STALE_RESULT"


@dataclass(frozen=True)
class DiagnosisContext:
    project_id: str
    board_id: str
    analysis_context_id: str | None = None
    circuit_model_id: str | None = None
    circuit_model_version: str | None = None
    hypothesis_set_id: str | None = None
    component_map_version: str | None = None
    mapping_status: str = "UNAVAILABLE"


_CONTEXT_FIELDS = (
    "project_id", "board_id", "analysis_context_id", "circuit_model_id",
    "circuit_model_version", "hypothesis_set_id", "component_map_version",
)


def current_diagnosis_context(
    state: Any, *, project_id: str, board_id: str,
    analysis_context_id: str | None, component_map_version: str | None,
) -> DiagnosisContext:
    """Build the live context from current selection/map and explicitly configured model data."""
    configured = state.get(DIAGNOSIS_CONTEXT_KEY, {})
    if isinstance(configured, dict) and owner_key(project_id, board_id) in configured:
        configured = configured[owner_key(project_id, board_id)]
    if isinstance(configured, DiagnosisContext):
        configured = {name: getattr(configured, name) for name in DiagnosisContext.__dataclass_fields__}
    if not isinstance(configured, dict):
        configured = {}
    return DiagnosisContext(
        project_id=project_id,
        board_id=board_id,
        analysis_context_id=analysis_context_id,
        circuit_model_id=configured.get("circuit_model_id"),
        circuit_model_version=configured.get("circuit_model_version"),
        hypothesis_set_id=configured.get("hypothesis_set_id"),
        component_map_version=component_map_version,
        mapping_status=configured.get("mapping_status", "UNAVAILABLE"),
    )


def is_diagnosis_result_current(
    *, stored_result: Any, current_context: DiagnosisContext,
    current_measurement_set: str,
) -> bool:
    """Use one strict predicate for both the diagnosis panel and PCB fault overlay."""
    if not isinstance(stored_result, dict):
        return False
    result = stored_result.get("result", stored_result)
    if not isinstance(result, dict):
        return False
    if any(not getattr(current_context, field) for field in _CONTEXT_FIELDS):
        return False
    if result.get("status") not in {DiagnosisStatus.DIAGNOSIS_COMPLETE, DiagnosisStatus.MODEL_MISMATCH}:
        return False
    if any(result.get(field) != getattr(current_context, field) for field in _CONTEXT_FIELDS):
        return False
    stored_set = stored_result.get("measurement_set_sha256", result.get("measurement_set_sha256"))
    if stored_set != current_measurement_set or result.get("measurement_set_sha256") != current_measurement_set:
        return False
    stored_context = stored_result.get("context")
    if isinstance(stored_context, DiagnosisContext) and any(
        getattr(stored_context, field) != getattr(current_context, field) for field in _CONTEXT_FIELDS
    ):
        return False
    return isinstance(result.get("candidates"), list)


@dataclass(frozen=True)
class DiagnosisReadiness:
    status: str
    blockers: tuple[str, ...]


class DiagnosisProvider(Protocol):
    def refresh_diagnosis(
        self, *, project_id: str, board_id: str, analysis_context_id: str,
        circuit_model_id: str, circuit_model_version: str, hypothesis_set_id: str,
        component_map_version: str, measurement_set_sha256: str,
        measurements: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


def assess_diagnosis_readiness(
    *, context: DiagnosisContext, measurements: list[MeasurementRecord],
    provider_available: bool = False,
) -> DiagnosisReadiness:
    if not measurements:
        return DiagnosisReadiness(DiagnosisStatus.NO_MEASUREMENT, ("측정값이 아직 저장되지 않았습니다.",))
    blockers = []
    if any(row.project_id != context.project_id or row.board_id != context.board_id for row in measurements):
        blockers.append("측정값의 프로젝트 또는 보드가 현재 작업과 다릅니다.")
    if any(row.mapping_status != "MAPPED" or not row.node_id or not row.probe_id for row in measurements):
        blockers.append("전기적 Probe/Node Mapping이 확인되지 않았습니다.")
    if (not context.component_map_version
            or any(row.component_map_version != context.component_map_version for row in measurements)):
        blockers.append("측정 기록이 현재 Component Map에 연결되지 않았습니다.")
    for row in measurements:
        if row.measurement_type != "voltage" or row.unit != "V":
            blockers.append("현재 진단 엔진은 V 단위 전압 측정만 지원합니다.")
            break
        if row.uncertainty is None or row.uncertainty <= 0:
            blockers.append("진단에 사용할 검증된 양의 측정 불확실도가 없습니다.")
            break
        if row.source != "user_measurement" or row.verification_status != "USER_REPORTED":
            blockers.append("진단 입력으로 사용할 실제 측정 출처가 확인되지 않았습니다.")
            break
    if (not context.analysis_context_id or not context.circuit_model_id
            or not context.circuit_model_version or not context.hypothesis_set_id
            or not context.component_map_version):
        blockers.append("이 보드에 연결된 회로 모델과 가설 집합이 없습니다.")
    if context.mapping_status != "MAPPED":
        blockers.append("현재 보드의 측정 지점 매핑이 제공되지 않았습니다.")
    if not provider_available:
        blockers.append("보드별 진단 Provider가 연결되지 않았습니다.")
    unique = tuple(dict.fromkeys(blockers))
    if not unique:
        return DiagnosisReadiness(DiagnosisStatus.DIAGNOSIS_READY, ())
    if any("Probe/Node Mapping" in item or "측정 지점 매핑" in item for item in unique):
        status = DiagnosisStatus.NODE_MAPPING_UNAVAILABLE
    elif any("모델과 가설" in item or "Provider" in item for item in unique):
        status = DiagnosisStatus.HYPOTHESIS_MODEL_UNAVAILABLE
    else:
        status = DiagnosisStatus.MEASUREMENT_RECORDED
    return DiagnosisReadiness(status, unique)


def validate_diagnosis_result(
    result: Any, *, context: DiagnosisContext, measurement_set_sha256: str,
) -> tuple[bool, str]:
    if not isinstance(result, dict):
        return False, "진단 결과 형식이 올바르지 않습니다."
    expected = {
        "project_id": context.project_id, "board_id": context.board_id,
        "analysis_context_id": context.analysis_context_id,
        "circuit_model_id": context.circuit_model_id,
        "circuit_model_version": context.circuit_model_version,
        "hypothesis_set_id": context.hypothesis_set_id,
        "component_map_version": context.component_map_version,
        "measurement_set_sha256": measurement_set_sha256,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        return False, "현재 프로젝트·모델·측정값과 다른 오래된 진단 결과입니다."
    if result.get("status") not in {DiagnosisStatus.DIAGNOSIS_COMPLETE, DiagnosisStatus.MODEL_MISMATCH}:
        return False, "완료 또는 검증된 모델 불일치 결과가 아닙니다."
    if not isinstance(result.get("candidates"), list):
        return False, "진단 후보 데이터가 없습니다."
    return True, ""


def refresh_diagnosis(
    provider: DiagnosisProvider, *, context: DiagnosisContext,
    measurements: list[MeasurementRecord],
) -> tuple[dict[str, Any] | None, DiagnosisReadiness, str | None]:
    readiness = assess_diagnosis_readiness(context=context, measurements=measurements,
                                           provider_available=provider is not None)
    if readiness.status != DiagnosisStatus.DIAGNOSIS_READY:
        return None, readiness, None
    adapted = []
    for record in measurements:
        row, reason = to_diagnosis_measurement(record)
        if row is None:
            return None, DiagnosisReadiness(DiagnosisStatus.MEASUREMENT_RECORDED,
                                            (reason or "측정값을 진단 입력으로 사용할 수 없습니다.",)), None
        adapted.append(row)
    set_sha = measurement_set_sha256(measurements, project_id=context.project_id, board_id=context.board_id)
    try:
        result = provider.refresh_diagnosis(
            project_id=context.project_id, board_id=context.board_id,
            analysis_context_id=context.analysis_context_id or "",
            circuit_model_id=context.circuit_model_id or "",
            circuit_model_version=context.circuit_model_version or "",
            hypothesis_set_id=context.hypothesis_set_id or "",
            component_map_version=context.component_map_version or "",
            measurement_set_sha256=set_sha, measurements=adapted,
        )
    except Exception as exc:
        return None, DiagnosisReadiness(DiagnosisStatus.DIAGNOSIS_FAILED,
                                        ("진단 Provider 실행 중 오류가 발생했습니다.",)), str(exc)
    valid, reason = validate_diagnosis_result(result, context=context, measurement_set_sha256=set_sha)
    if not valid:
        return None, DiagnosisReadiness(DiagnosisStatus.STALE_RESULT, (reason,)), None
    return result, DiagnosisReadiness(result["status"], ()), None


def adapt_fault_overlay(
    envelope: Any, *, context: DiagnosisContext, measurement_set_sha256: str,
    board_side: str, canonical_frame_id: str, component_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], str | None]:
    """Accept only a context-bound diagnosis envelope and mapped current-frame candidates."""
    if not isinstance(envelope, dict):
        return {}, "진단 근거가 연결되지 않았습니다."
    valid, reason = validate_diagnosis_result(
        envelope.get("diagnosis"), context=context,
        measurement_set_sha256=measurement_set_sha256,
    )
    if not valid:
        return {}, reason
    overlay = envelope.get("fault_overlay")
    if (not isinstance(overlay, dict) or overlay.get("schema_version") != "circuitcare.pcb-fault-overlay.v1"
            or overlay.get("board_id") != context.board_id or overlay.get("board_side") != board_side
            or envelope.get("canonical_frame_id") != canonical_frame_id
            or envelope.get("component_map_version") != context.component_map_version):
        return {}, "고장 후보 이미지 좌표 또는 Component Map 버전이 일치하지 않습니다."
    confirmed_ids = set(envelope.get("confirmed_component_ids", []))
    result = {}
    for candidate in overlay.get("candidates", []):
        if (not isinstance(candidate, dict) or candidate.get("board_id") != context.board_id
                or candidate.get("board_side") != board_side
                or candidate.get("canonical_image_reference") != canonical_frame_id
                or candidate.get("mapping_status") != "MAPPED"):
            continue
        component_id = candidate.get("component_id")
        status = candidate.get("fault_candidate_status")
        if component_id not in component_ids or status not in {
            "CONFIRMED_FAULT", "SUSPECTED_FAULT", "RELATED_COMPONENT", "UNDETERMINED",
        }:
            continue
        if envelope["diagnosis"].get("status") == DiagnosisStatus.MODEL_MISMATCH and status != "UNDETERMINED":
            continue
        if status == "CONFIRMED_FAULT" and component_id not in confirmed_ids:
            status = "UNDETERMINED"
        result[component_id] = {**candidate, "fault_candidate_status": status}
    return result, None
