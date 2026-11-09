# CircuitCare-AI 증상 기반 PCB 고장 진단 V1

PCB 연결 정보가 없어도 자연어 증상과 확인된 관측값으로 고장 원인 후보를 생성하는 독립 진단 패키지입니다. 이 저장소에는 이번 고장 진단 세트와 실행에 필요한 코드만 포함합니다.

## 구성

- 18종 소자 분류와 내부 `unknown` 상태
- YAML 기반 증상 코드, FMEA, 16개 규칙, 안전 검사 목록
- 출처와 상태를 보존하는 관측 스키마: `observed`, `measured`, `inferred`, `unknown`, `contradicted`
- 제한된 Ollama JSON 증상 파서와 결과 설명기
- 규칙 근거와 모순을 표시하는 후보 점수화
- 정보가 부족하면 보류하고, 위험 검사는 기본 추천에서 제외
- 로컬 SQLite 진단 세션/관측 저장

## 설치

```bash
pip install -e .
```

로컬 LLM을 사용하려면 Ollama를 실행하고 모델을 준비합니다.

```bash
ollama serve
ollama pull qwen3.5:2b
```

모델이 없어도 YAML 규칙 엔진과 키워드 증상 파서는 동작합니다.

## Python 사용

```python
from circuitcare.symptom_diagnosis import diagnose

result = diagnose(
    symptom_text="전원 표시등은 켜지고 버튼을 눌러도 버저가 울리지 않음",
    components=[{"component_id": "BZ1", "component_type": "buzzer"}],
    connectivity_status="none",
)

for candidate in result["hypotheses"][:5]:
    print(candidate["title"], candidate["supporting_evidence"])
```

Ollama를 통한 자연어 구조화와 설명은 `symptom_parser.parse_user_report`와 `llm_assist.explain_candidates`를 사용할 수 있습니다. LLM이 제안한 미등록 원인은 별도 미검증 목록으로 유지하며 규칙 후보 점수나 순위에 합치지 않습니다.

## 판단 원칙

- 탐지된 소자 라벨은 고장 확정이 아닙니다. 일치하는 소자 탐지는 약한 위치 단서만 제공합니다.
- 연결 정보가 없거나 추정 상태여도 증상 규칙 진단은 계속 동작합니다. 추정 연결은 검증된 근거로 취급하지 않습니다.
- `unknown`, `inferred`, `contradicted`는 확인된 참/거짓 사실로 점수화하지 않습니다.
- 가설 점수는 실제 고장 확률이 아닙니다. 후보는 검사 전까지 미확정입니다.
- IC 핀 기능이나 위험한 통전 측정을 임의로 안내하지 않습니다.

현재 지식베이스와 가중치는 V1 초기값입니다. 실제 고장 사례 평가 없이 진단 정확도를 보장하지 않습니다.
