"""Short Korean labels for stable frontend status codes."""
STATUS_LABELS = {
    "UNANALYZED": "미분석", "INSPECT": "측정 필요", "SUSPECTED": "고장 의심",
    "CONFIRMED": "고장 확인", "UNVERIFIED": "미검증", "UNAVAILABLE": "분석 불가",
    "DEMO_ONLY": "데모 전용",
}

def status_label(status: str) -> str:
    return STATUS_LABELS.get(status, "상태 확인 필요")
