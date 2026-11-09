# Phase 19.3 — Fault, Risk, Measurement Overlay

## 구현 개요

Schematic Viewer에는 검증된 Component Symbol Geometry 위에 표시 전용 SVG Layer를 추가했다. Overlay는 Python 순수 Builder에서 구성하고 `context_id`, SVG `asset_key`, 회로와 Schematic 정합성, Geometry 상태를 먼저 검사한다. 브라우저에서는 검증된 Component Reference와 측정 Anchor만 `textContent` 및 SVG 요소 속성으로 렌더링한다. 원본 배선의 전기적 전파 경로는 그리지 않는다.

### 기존 분석 데이터 재사용

- 연결 후보는 기존 `Circuit`, `direct_candidates()` 및 제한 상태를 제공하는 `find_paths_result()`를 사용한다.
- 전기 위험은 현재 `risk_assessment.json` 구조, 비교 결과 Scenario ID, `AnalysisSnapshot` Manifest 및 Integrity, UI의 현재 Fault Fingerprint를 사용한다.
- 검사 순위는 기존 `inspection_ranking.json`의 Prior 또는 Posterior 정렬 목록을 사용한다.
- 측정 추천은 기존 `diagnosis.json`과 `measurement_recommendations.json`을 사용한다. 새로운 AI 추론, SPICE 실행 또는 결과 생성을 수행하지 않는다.

## 19.3.1 Fault Influence

Fault Source는 사용자 선택 고장 가정으로 표시한다. Direct와 Indirect Candidate는 graph connectivity만 증거로 가지며 공유 Net, Hop Count, 경로 후보 및 `evidence_level=connectivity_only`를 유지한다. 경로 검색이 깊이 또는 리소스 제한으로 종료되면 `PARTIAL_*` 상태를 보존하고 UI에도 부분 결과임을 표시한다. 배선 강조와 방향 화살표는 없다.

부품 선택은 기존 검증된 Viewer 이벤트를 이용한다. R1 고장 가정에서 R2를 선택하면 Inspector가 R2로 바뀌고 Fault Source는 R1로 유지된다.

## 19.3.2 Electrical Risk

Risk Overlay는 Simulation 상태가 `SUCCESS` 또는 `CACHED_SIMULATION`이고 Integrity가 `VERIFIED`일 때만 가능하다. Fault Ref/Type/Parameter, Scenario ID, 비교 결과, Manifest Fingerprint Hash, Snapshot에 포함된 Risk를 대조한다. 유효한 상태는 `EXCEEDED`, `NEAR_LIMIT`, `WITHIN_LIMIT`, `UNKNOWN`뿐이다. 보고서나 Metric 형식이 예상과 다르면 전체 Risk 표시를 차단한다.

Inspector에는 Risk Level, 순위, 점검 점수(확률 아님), 주요 Metric, 평가 상태, 스트레스, 기준값, 사용률, Rating Source 및 재평가 출처를 표시한다. 평가되지 않은 Metric은 Unknown으로 유지한다. 위험 마커는 실제 손상 또는 고장 확률이 아니다.

Divider의 기존 `risk_assessment.json`은 R2=`WITHIN_LIMIT`, J1=`UNKNOWN`을 갖지만 연결된 Simulation Manifest가 없어 현재 Integrity는 `PROVENANCE_UNVERIFIED`다. 따라서 실제 UI는 Divider Risk Overlay를 차단한다. 테스트는 해당 기존 보고서를 synthetic verified snapshot에 연결해 Builder의 표시 필드만 확인하며, 이를 실제 운용 데이터 검증으로 취급하지 않는다. 실제 UI의 Fail-closed 안내도 브라우저에서 확인했다.

## 19.3.3 Measurement & Inspection

추천 Node의 핀 후보는 현재 넷리스트의 `(Reference, Pin, Net)`와 Schematic Embedded Library Pin, Unit, Symbol Transform을 대조해 산출한다. 같은 Node의 모든 연결 Pin을 후보로 표시하며 하나를 실제 프로브 지점으로 선택하지 않는다. Pin 후보는 논리적 Schematic 위치일 뿐 PCB에서 접근 가능한 지점이나 안전한 프로빙 위치를 보증하지 않는다. 위치를 검증하지 못하면 회로도 Marker를 만들지 않고 패널 전용 정보로 제한한다.

Bayesian 근거는 지원되는 U1 Regulator 조건에만 연결된다. Prior-only, 사용자 측정 및 simulated-demo 출처를 별도 표시하고 What-if Preview는 입력 근거로 섞지 않는다. `analysis_context_id`와 실제 측정 Set Hash가 현재 UI Context와 맞지 않으면 이전 진단을 사용하지 않는다. `model_mismatch`에서는 추천 강조를 차단한다. 지원되지 않는 Divider 조합은 명시적으로 안내한다.

기본 Regulator 사례에는 검증된 `.kicad_sch` Fixture가 없다. 따라서 이 사례에서는 기존 Bayesian 정보를 Inspector에서만 제공하고 회로도 위치 Overlay를 사용하지 않는다. Divider의 검증된 Geometry에 Regulator 데이터를 합성하지 않는다. Divider Fixture의 VOUT 핀 후보는 실제 KiCad CLI SVG Pin 끝점과 대조했다.

## 검증 기록

로컬 환경: macOS, Python 3.12, KiCad CLI 10.0.6. 전체 테스트 그룹은 아래 결과를 기록했다.

| 확인 | 결과 |
|---|---|
| `compileall -q src scripts tests` | 통과 |
| 비통합 테스트 그룹 (`-m 'not integration and not stress and not crash_injection'`) | 266 passed |
| Integration (`-m integration -rs`) | 32 passed, KiCad CLI 포함 |
| Stress (`-m stress -rs`) | 11 passed |
| Crash Injection (`-m crash_injection -rs`) | 3 passed |
| 집중 Schematic Overlay/Geometry 그룹 | 31 passed, 실제 Divider SVG 포함 |
| Ruff 변경 파일 | 통과 |
| mypy 변경 핵심 모듈 | 통과 (Workspace, Overlay, Measurement Geometry, Geometry, Viewer) |
| JavaScript Syntax (`node --input-type=module --check`) | 통과 |
| GitHub Actions | [Tests run 41975680483](https://github.com/alfred0105/CircuitCare-AI/actions/runs/41975680483) · Python 3.11/3.12, ngspice Integration, OCR Integration, Stress, Crash Injection 모두 통과 |

전체 `ui.py`에 대해 mypy를 실행할 경우 이 변경 이전부터 이어지는 UI 전반의 기존 타입 문제 40개가 출력된다. 새 Workspace/Overlay 모듈과 View 호출 인접 모듈의 검사 결과는 위와 같이 분리해 기록한다.

브라우저 확인 기록:

- E2E-01: Divider 기본 KiCad SVG와 검증된 선택 부품 표시 — 확인.
- E2E-02: 연결 후보 모드에서 R1 Fault Source와 직접 후보 표시 — 확인.
- E2E-03/08: R2 클릭 시 Inspector가 R2로 변경되고 고장 가정은 R1로 유지 — 확인.
- E2E-04/05: 실제 Divider Risk — `PROVENANCE_UNVERIFIED` 사유와 함께 Overlay 차단 확인. R2/J1의 Risk 마커는 표시하지 않음.
- E2E-11: Bayesian 미지원 Divider 안내 — 확인.
- E2E-06/07/09/10/12/13/14: 이 작업에서는 실제 확대/이동, Stale 주입, Dataset 전환 제거, Regulator 회로도 Marker, Source 변환 및 콘솔 전체 검사를 실행하지 않음.

## Known Limitations 및 다음 단계 조건

- Divider Risk Snapshot에 검증된 Manifest를 연결하는 기존 결과 경로가 없어 Risk Overlay 실증은 제한된다. 유효한 기존 분석 결과를 보존한 채 Manifest를 제공하거나 새 검증 결과를 생성해야 실제 Risk 마커를 데모할 수 있다. 이 작업에서 SPICE를 실행하거나 Manifest를 만들어내지 않았다.
- Regulator Bayesian 회로도 Fixture가 없어 Pin Marker의 회로도 시연은 미완료다.
- Multi-sheet Geometry는 기존처럼 partial 상태이며 Overlay Gate가 닫힌다.
- Browser에서 확대/이동 및 marker 정렬의 별도 시나리오를 수행하지 않았다. FPS/지연 시간은 측정하지 않았다.
- Python 3.11은 로컬이 아닌 위 GitHub Actions에서 검증했다.
- 19.4 전체 UI 정리 전에 Divider Risk provenance와 Regulator Schematic Fixture를 검증하고, 아래 미실행 Browser E2E 항목을 추가 수행한다.

## Phase 판정

- 19.3.1: 기능 구현 및 브라우저 연결 후보 상호작용 확인.
- 19.3.2: provenance gate 및 Inspector 구현. 실제 Divider Risk Overlay는 입력 Manifest가 없어 부분 완료.
- 19.3.3: 논리적 Pin 매핑과 패널 fallback 구현. Regulator Schematic Marker 검증이 없어 부분 완료.
- 전체 Phase 19.3: **PARTIAL**. Phase 19.4에 들어가기 전에 Divider의 Risk Snapshot provenance와 Regulator Schematic Fixture를 해결해야 한다.
