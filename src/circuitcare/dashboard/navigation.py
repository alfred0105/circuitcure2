"""Korean-first navigation labels for the EDA workbench."""

from __future__ import annotations

WORKBENCH_VIEW = "회로도 Workbench"
WORKSPACE_VIEWS = [
    WORKBENCH_VIEW, "분석 개요", "SPICE 파형", "2차 손상 위험", "Bayesian 진단",
    "로컬 LLM 수리 지원", "PCB 사진 인식", "데이터시트 AI", "AI 고장 가설 분석",
    "수리 이력 및 AI 학습", "AI 성능 평가", "다중 사진 PCB 복원",
]

AREAS = ("회로도", "분석", "측정·진단", "보고서")
ANALYSIS_VIEWS = ("분석 개요", "SPICE 파형", "2차 손상 위험")
ADVANCED_VIEWS = (
    "로컬 LLM 수리 지원", "PCB 사진 인식", "데이터시트 AI", "AI 고장 가설 분석",
    "수리 이력 및 AI 학습", "AI 성능 평가", "다중 사진 PCB 복원",
)
VIEW_TO_AREA = {
    WORKBENCH_VIEW: "회로도", "분석 개요": "분석", "SPICE 파형": "분석",
    "2차 손상 위험": "분석", "Bayesian 진단": "측정·진단",
    **{view: "고급 도구" for view in ADVANCED_VIEWS},
}
CASE_LABELS = {
    "Regulator + divider + switch (cached demo)": "레귤레이터·분압기·스위치 (저장 데모)",
    "Independent resistor divider (evaluation circuit A)": "저항 분압 회로 (평가 회로 A)",
    "Independent RC charging circuit (evaluation circuit B)": "RC 충전 회로 (평가 회로 B)",
    "KiCad XML 업로드 (연결 그래프 전용)": "KiCad XML 업로드 (연결 그래프 전용)",
}


def korean_status(value: object) -> str:
    """Translate stable internal status codes at the presentation boundary."""
    return {
        "UNKNOWN": "미평가", "NOT_EVALUATED": "미평가",
        "WITHIN_LIMIT": "평가 항목 정격 이내", "NEAR_LIMIT": "정격 근접",
        "EXCEEDED": "설정 정격 초과", "SUCCESS": "완료",
        "CACHED_SIMULATION": "검증된 저장 결과", "NOT_RUN": "미실행",
        "FAILED": "실패", "STALE": "입력 변경됨", "RUNNING": "실행 중",
        "PROVENANCE_UNVERIFIED": "출처 확인 필요",
        "UNVERIFIED_GEOMETRY": "도형 미검증",
        "UNVERIFIED_MEASUREMENT_LOCATION": "측정 위치 미검증",
        "UNVERIFIED_MAPPING": "부품 연결 미검증",
        "NOT_EVALUATED": "미평가", "NOT_AVAILABLE": "자료 없음",
        "VERIFIED": "검증됨", "VERIFIED_EQUIVALENT": "전기적 정합 확인",
        "VERIFIED_GEOMETRY": "도형 위치 검증",
        "regulator_output_rise": "레귤레이터 출력 상승",
        "regulator_output_drop": "레귤레이터 출력 하락",
        "resistance_drift": "저항값 변화", "resistor_open": "저항 단선",
        "resistor_short": "저항 단락", "regulator_output_loss": "레귤레이터 출력 손실",
        "unspecified": "미지정",
        "NOT_VERIFIED": "검증되지 않음", "VERIFIED_DESIGN": "설계 기준 확인",
        "VERIFIED_PHYSICAL": "실물 확인", "VERIFIED_SYNTHETIC": "합성 자료 확인",
        "SVG_EXPORTED": "회로도 표시 준비됨", "EXPORT_FAILED": "회로도 표시 실패",
        "REGISTRATION_UNVERIFIED": "사진 정합 미검증",
        "OBSERVED_TOPOLOGY_ONLY": "사진 접촉 후보만 확인",
        "PARTIAL_DEPTH_LIMIT": "경로 일부만 표시",
        "INCOMPLETE_PROJECT": "프로젝트 일부 누락",
        "MODEL_MISMATCH": "모델과 측정 차이 큼",
        "UNSUPPORTED": "지원되지 않음", "PARTIAL": "일부 완료",
        "BLOCKED": "선행 확인 필요", "READY": "준비됨", "EXPORTED": "내보내기 완료",
    }.get(str(value), "상태 확인 필요")
