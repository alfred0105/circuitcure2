# 기존 앱에 진단 화면 연결

이 패키지에는 진단 엔진 및 선택형 증상 패널이 있습니다. 기존 CircuitCare 앱에서는 측정 기반 진단을 그대로 유지하면서 같은 `전체 진단` 패널에 증상 진단을 추가합니다.

1. `src/circuitcare/dashboard/views/symptom_diagnosis_panel.py`를 호스트 앱의 동일한 경로에 복사합니다.
2. `inspection_panel.py`에 패널을 import하고 기존 `render_diagnosis_panel(...)` 호출 뒤에 `render_symptom_diagnosis_panel(st, project, components=components, preview=preview)`를 추가합니다.
3. 보드/사진/부품/측정 입력이 바뀌면 증상 패널의 입력 fingerprint가 달라져 이전 결과와 부품 강조가 초기화됩니다.

최소 추가 예시는 `inspection_panel.patch`에 있습니다. 기존 진단이나 측정 패널을 제거하지 않습니다.

IC 각인의 접두사 해석, 배치 기반 기능 블록은 미확정 힌트입니다. 기본 데이터시트 카탈로그는 비어 있으므로 공식 제조사 데이터시트로 확인한 정보만 카탈로그에 넣어야 합니다.
