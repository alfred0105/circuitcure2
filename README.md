# CircuitCare 3

CircuitCare 3는 PCB 사진과 사용자가 입력한 증상을 바탕으로 부품 후보를 표시하고, 사용자가 검수한 결과를 측정 및 고장 진단 작업공간에 연결하는 로컬 Streamlit 앱입니다. 탐지 결과는 검수 후보이며 전기적 고장 확정으로 표시하지 않습니다.

## 실행

Python 3.11 이상을 사용합니다. macOS Apple Silicon에서는 MPS를 사용하고 지원되지 않는 연산은 CPU 경로로 처리합니다. 모델 가중치는 별도로 제공되는 로컬 파일을 사용하며 이 저장소에는 포함하지 않습니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-macos-arm64.txt
streamlit run src/circuitcare/ui.py
```

Windows 또는 별도 환경에서는 필요한 UI·이미지·탐지 의존성을 설치한 뒤 같은 Streamlit 진입 파일을 실행합니다. 로컬 가중치 경로와 탐지 설정은 `apps/pcb-component-recognition/`의 앱 설정을 따릅니다. 가중치는 다운로드하거나 학습하지 않습니다.

## 앱 흐름

1. PCB 사진과 증상을 입력합니다. KiCad 자료는 선택적으로 첨부할 수 있습니다.
2. PCB 작업공간에서 사진과 소자 탐지 결과를 확인하고 검수합니다.
3. 선택한 소자의 증상 후보, 측정 입력, 고장 진단 결과와 오버레이를 확인합니다.
4. 프로젝트와 검수 기록은 기존 Streamlit 세션 상태 및 앱 데이터 경로를 사용합니다.

실행 진입점은 `src/circuitcare/ui.py` 하나이며 사진 입력 화면에서 `diagnosis_entry`를, 프로젝트가 준비되면 `unified_workspace`를 렌더링합니다. 탐지 구현은 `apps/pcb-component-recognition/app_server.py`를 기존 로컬 가중치와 함께 사용합니다.

## 개발 및 검증

```bash
python -m pip install -e ".[dev,ui,vision,pcb-recognition,vision-ocr,datasheet]"
python -m pytest -q
```

통합·스트레스·크래시 주입 테스트는 CI에서 별도 작업으로 실행됩니다. 탐지 비교와 데이터셋 평가 도구는 `scripts/` 및 `evaluation/`에 있으며 앱 실행 경로와 분리되어 있습니다.

## 구조

- `src/circuitcare/dashboard/views/`: 사진 입력, PCB 작업공간, 검수·측정·진단 화면
- `src/circuitcare/vision/`: 사진 로딩과 소자 탐지 연결
- `src/circuitcare/symptom_diagnosis/`: 증상 및 검수 결과 기반 후보 진단
- `apps/pcb-component-recognition/`: 탐지 서버와 로컬 모델 설정
- `tests/`: 앱 흐름, 탐지 계약 및 진단 기능 테스트
- `docs/`: 평가 한계와 구현 계약 문서

평가 기록과 데이터셋 도구의 상세 설명은 각 자료의 문서에 남아 있으며 런타임 UI 진입점으로 사용되지 않습니다.
