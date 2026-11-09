# 발표 당일 체크리스트

## Phase 18.1.1 stability gates

- [ ] Fast CI matrix passes on Python 3.11 and 3.12.
- [ ] ngspice integration job passes on Linux; local macOS results are reported separately.
- [ ] Stress suite runs with bounded synthetic inputs and no deliberate OOM.
- [ ] Crash termination classification and log preservation are verified in isolated children.
- [ ] Actual ONNX inference is only marked passed when the project model artifact is present and exercised.
- [ ] Physical repair accuracy remains `NOT_EVALUATED` until independently validated on real boards.

## MVP 설치 및 보고서 확인

- [ ] `python -m pip install -e ".[dev,ui,vision,datasheet]"` 설치 완료
- [ ] `python -m pytest -q` 전체 테스트 통과
- [ ] `python -m compileall -q src/circuitcare` 통과
- [ ] 분석 개요에서 Workspace ID와 현재 회로/고장 Context 확인
- [ ] JSON, Markdown, HTML 및 Evidence Bundle ZIP 생성 확인
- [ ] `scripts/build_phase18_demo_report.py`로 cached synthetic demo 보고서 재생성 확인
- [ ] Bundle manifest 해시 검증 및 로컬 열람 확인
- [x] 기존 MVP Release `v0.1.0-mvp`와 Tag 보존
- [x] Phase 18.1 전체 테스트 및 Python 3.11/3.12 CI 확인
- [x] 0V, 실제 측정/What-if 분리, stale cache, invalid prior, bundle 참조 검증 확인
- [ ] Physical Repair Validation은 실측 검증 데이터가 없으면 `NOT_EVALUATED`

## 사전 설치 및 오프라인 준비

- [ ] Python 3.11 이상 가상환경 준비
- [ ] `python -m pip install -e ".[dev,ui]"` 설치 완료
- [ ] ngspice 설치 및 `ngspice --version` 확인
- [ ] 선택 사항: Ollama 설치, `ollama pull gemma4:12b` 완료, 모델 목록에서 확인
- [ ] 모델은 저장소에 포함되지 않으므로 필요한 경우 발표 전 내려받음
- [ ] 의존성·시뮬레이터·모델이 준비된 상태에서 인터넷 연결을 끊고 오프라인 실행 확인
- [ ] `python scripts/demo_check.py` 실행; Streamlit과 저장된 JSON은 필수, ngspice/Ollama는 선택 기능 상태로 확인
- [ ] `python -m pytest -q` 실행
- [ ] `streamlit run src/circuitcare/ui.py` 실행 및 브라우저 접속 확인

## U1 대표 데모

- [ ] 기본 Regulator + divider 회로, U1 output-rise 8 V 선택
- [ ] 레거시 저장 결과는 `PROVENANCE_UNVERIFIED`로 표시되어 현재 분석에서 차단됨
- [ ] 새 실행은 입력/결과 Fingerprint 검증 후 `SUCCESS · Live Simulation`으로 표시됨
- [ ] 정상 5 V와 고장 8 V 비교 표시
- [ ] 위험 표에서 C1 8 V / 6.3 V = 약 1.2698, `EXCEEDED` 표시
- [ ] RLOAD 16 mW / 125 mW = 0.128, `WITHIN_LIMIT` 표시
- [ ] Bayesian 저장 demo의 `simulated_demo` 출처와 조건부 결과를 설명함
- [ ] 로컬 질문 또는 결정론적 Fallback 응답을 확인함
- [ ] 시연 정격과 관측값이 실측·데이터시트 검증값이 아님을 설명함

## 다른 회로 재현

- [ ] Divider: R1 10 kΩ live 실행 시 VOUT 약 2.5000→0.4545 V
- [ ] RC: RCHARGE 100 kΩ live 실행, 파형과 단위가 표시됨; 저장 CSV는 0–50 ms
- [ ] 회로/고장 조건 변경 시 이전 Live 결과와 LLM 답변이 활성 결과로 남지 않음
- [ ] Bayesian 측정 추가/교체/초기화 시 이전 LLM 답변이 무효화됨
- [ ] 동일 조건 시뮬레이션 재실행 실패는 `FAILED`; 이전 live 출력은 활성 결과가 아님

## 장애 시 대체 절차

- **ngspice 실패/미설치:** 저장 캐시는 manifest가 일치할 때만 재사용합니다. 기존 manifest가 없는 캐시는 `PROVENANCE_UNVERIFIED`로 유지하고 성공한 live 실행이라고 말하지 않습니다. CLI 재현 명령은 `README.md`의 ngspice 예시를 따릅니다.
- **Ollama 실패/미설치:** 화면의 결정론적 안내를 사용합니다. 이전에 저장된 LLM JSON을 보여줄 경우 저장된 응답이라고 알립니다.
- **Streamlit 실패:** `python -m pytest -q`와 `scripts/demo_check.py` 결과를 보여주고, 저장된 JSON/CSV 및 평가 결과를 사용합니다. UI가 실행된 것처럼 말하지 않습니다.
- **인터넷 연결 없음:** 사전에 설치된 패키지, ngspice, 선택 모델을 사용합니다. 시스템은 core CLI/UI 분석에서 클라우드 모델/API를 호출하지 않습니다.

## 최종 점검

- [ ] 포스터 콘텐츠 파일을 확인합니다: `docs/POSTER_CONTENT.md`
- [ ] 5분 대본과 질의응답을 확인합니다: `docs/DEMO_SCRIPT.md`
- [ ] 현재 포스터 수치가 저장된 JSON/CSV와 일치하는지 확인합니다
- [ ] 포스터 제출은 이 저장소에서 수행하지 않았으며 제출 완료라고 표시하지 않습니다
- [ ] 실행 산출물, 임시 결과, 모델 가중치, 대용량 캐시는 커밋 대상에 포함하지 않습니다
