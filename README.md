# CircuitCare-AI 증상 기반 PCB 고장 진단

PCB 연결 정보가 없어도 사용자가 확인한 증상과 근거를 바탕으로 고장 **후보**를 생성하는 진단 패키지입니다. 이 저장소에는 진단 모듈과 기존 앱에 연결할 화면 코드만 포함합니다.

## 구성

- 27개 PCB 부품 라벨을 기능 역할 후보에 연결
- YAML 지식베이스: 증상 분류, FMEA 고장 모드, 19개 규칙, 안전 점검 질문
- 증거 상태·출처·시각을 보존하고 충돌 입력은 자동 확정하지 않음
- 증상 누락을 막는 모드 기반 후보 보완 및 응답별 후보 점수 재계산
- OCR 마킹은 미확정 접두사 힌트로만 사용. 검증된 제조사 데이터시트 카탈로그는 초기에는 비어 있음
- 부품 조합·근접도·OCR 힌트를 사용하는 미확정 기능 블록 추정
- 측정값은 결과에 연결하되 부품별 정상 범위가 없으면 순위에 반영하지 않음
- 진단 및 관측 이력을 SQLite에 추가 기록
- 선택적으로 Ollama 로컬 LLM을 사용해 증상을 구조화하고 결과를 설명

## 설치 및 실행

```bash
pip install -e .
```

로컬 LLM 보조를 쓰려면 Ollama를 실행하고 모델을 준비합니다.

```bash
ollama serve
ollama pull qwen3.5:2b
```

LLM이 없어도 YAML 규칙 엔진과 키워드 파서는 동작합니다.

## Python 사용

```python
from circuitcare.symptom_diagnosis import diagnose

result = diagnose(
    symptom_text="전원 표시등은 켜지고 버튼을 눌러도 버저가 울리지 않음",
    components=[{"component_id": "BZ1", "component_type": "buzzer"}],
)

for candidate in result["hypotheses"][:5]:
    print(candidate["title"], candidate["supporting_evidence"])
```

## 기존 CircuitCare UI에 연결

`src/circuitcare/dashboard/views/symptom_diagnosis_panel.py`는 기존 앱의 Streamlit 화면에서 사용하는 패널입니다. `integration/README.md`와 `integration/inspection_panel.patch` 안내대로 기존 측정 기반 진단을 유지한 채 증상 진단을 추가할 수 있습니다.

## 판단 원칙과 제한

- 부품 검출은 고장 확정이 아니며 소자 개수는 후보 점수를 올리지 않습니다.
- 인식된 부품 종류와 위치는 후보 부품을 좁히는 단서일 뿐, 실제 결선으로 취급하지 않습니다.
- 근거가 충돌하면 관련 증상을 확정하지 않고 충돌을 결과에 표시합니다. 측정값은 관측값보다 우선하지만 충돌 이력도 남깁니다.
- OCR 접두사에서 나온 IC 역할은 데이터시트 확인 전 추정입니다. IC 모델, 제조사, 패키지, 핀 수를 자동 확정하지 않습니다.
- 후보 점수는 고장 확률이 아닙니다. 검사를 거치기 전까지 결과는 미확정입니다.
- IC 핀 기능이나 위험한 통전 측정을 임의로 제안하지 않습니다.

초기 규칙과 가중치는 실제 보드 고장 사례로 검증되지 않았으므로 진단 정확도를 보장하지 않습니다.
