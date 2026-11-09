# CircuitCare AI 화면 연결

이 저장소에는 진단 엔진과 화면 모듈이 분리되어 있습니다.

- 진단 규칙/상태/저장/LLM 보조: `src/circuitcare/symptom_diagnosis/`
- 로컬 Ollama 클라이언트: `src/circuitcare/local_llm.py`
- Streamlit 화면 모듈: `src/circuitcare/dashboard/views/symptom_diagnosis_panel.py`

기존 CircuitCare-AI 앱에 연결할 때는 이 화면 파일을 같은 경로에 두고
`inspection_panel.py`에서 아래 두 작업을 합니다.

1. `render_symptom_diagnosis_panel`을 import합니다.
2. 전체 진단 영역에서 `render_symptom_diagnosis_panel(st, project, components=components, preview=preview)`를 호출합니다.

최소 변경 예시는 `inspection_panel.patch`에 있습니다. 기존의 측정 패널 등은 유지하고,
기존 단일 진단 패널 호출을 새 증상 기반 진단 패널 호출로 교체합니다.

UI 실행에는 프로젝트의 Streamlit 의존성이 필요합니다. Ollama가 연결되지 않아도
규칙 기반 진단은 실행되며, 로컬 LLM 보조는 사용 가능한 로컬 모델이 있을 때만 켤 수 있습니다.
