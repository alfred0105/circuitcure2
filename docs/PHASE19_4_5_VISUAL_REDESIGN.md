# Phase 19.4.5 — 회로도 배선 복구 및 한글 EDA 작업공간

## 변경 요약

이전 Regulator 회로도는 13개 Net Label로 연결을 표현했고 실제 `(wire ...)`는 0개였습니다. KiCad 넷리스트에는 연결이 나타났지만 회로도 화면만 보고 배선 경로를 따라갈 수 없었습니다. 기존 UI도 11개 화면을 그룹명이 붙은 선택 목록으로 제공해 회로도보다 탐색 UI가 두드러졌습니다.

Regulator fixture에 실제 직교 배선 19개와 Junction 7개를 추가했습니다. VIN, VOUT, LOAD, GND만 필요한 위치에 보조 라벨로 남겼으며 심벌 Reference와 UUID는 유지했습니다. Divider fixture의 기존 배선은 보존했습니다. RC 데모는 독립된 KiCad 회로도 파일이 없어 기존 Graph Fallback으로 표시되며, 전기 구조를 임의로 만들지 않았습니다.

## KiCad 정합성

KiCad CLI 10.0.6으로 변경 전·후 XML 넷리스트를 내보내 비교했습니다. Reference와 값 6개, 핀-넷 매핑 13개가 모두 일치하여 전기적 동등성은 `VERIFIED_EQUIVALENT`입니다. 변경 후 넷 그룹은 다음과 같습니다.

| Net | 연결된 핀 |
|---|---|
| VIN | J1.1, Q1.1, U1.1 |
| VOUT | C1.1, R1.1, U1.3 |
| LOAD | R1.2, RLOAD.1 |
| GND | C1.2, J1.2, Q1.2, RLOAD.2, U1.2 |

실제 KiCad SVG Export가 성공했고 SVG에서 배선을 확인했습니다. 회로도 geometry 및 VOUT·LOAD 측정 앵커 검증은 기존 통합 테스트가 확인합니다. 부품 선택은 브라우저에서 R1을 눌러 Inspector가 R1의 값과 양쪽 Net을 표시하는 것으로 확인했습니다. 회로도 파일에 전선이 없는 상태를 Viewer에서 가상 배선으로 보정하지 않습니다.

## 작업공간과 한글화

- 상단 탐색을 `회로도`, `분석`, `측정·진단`, `보고서` 네 영역으로 바꾸고 나머지 여섯 화면을 `고급 도구`에 배치했습니다. 기존 라우트는 모두 유지됩니다.
- 어두운 EDA 프레임과 밝은 회로도 캔버스, 컴팩트한 도구 모음, 공간을 넓힌 캔버스와 간결한 Inspector를 적용했습니다. 1280px 이하에서는 Explorer를 감추고 캔버스 및 Inspector를 우선 배치하며 850px 이하에서는 세로 배치합니다.
- 부품 선택 상태와 고장 가정은 분리되어 있습니다. 선택만으로 고장 시나리오나 시뮬레이션 상태를 변경하지 않습니다.
- 사용자 표시용 상태·고장 유형은 한국어로 변환하며 내부 상태 코드와 API 데이터는 유지합니다.
- 위험 Overlay는 `PROVENANCE_UNVERIFIED` 상태에서 비활성화됨을 표시합니다. 측정 추천이 검증된 위치를 제공하지 않는 경우 핀 마커를 만들어내지 않습니다.

## 브라우저 화면 검토

실제 로컬 Streamlit 화면에서 캔버스, Inspector, 위험·측정 모드, 고급 도구 및 좁은 창 레이아웃을 검토하고 다음 브라우저 캡처를 저장했습니다.

- [Regulator 전체 회로도](assets/phase19_4_5/regulator_workbench_browser.png)
- [Divider 전체 회로도](assets/phase19_4_5/divider_workbench_browser.png)
- [R1 선택 및 Inspector](assets/phase19_4_5/component_selection_inspector_browser.png)
- [전기적 위험 모드](assets/phase19_4_5/risk_overlay_browser.png)
- [측정·진단 모드](assets/phase19_4_5/measurement_overlay_browser.png)
- [고급 도구 메뉴](assets/phase19_4_5/advanced_tools_browser.png)
- [1280 × 720 화면](assets/phase19_4_5/workbench_1280x720_browser.png)
- [1440 × 900 화면](assets/phase19_4_5/workbench_1440x900_browser.png)
- [1920 × 1080 화면](assets/phase19_4_5/workbench_1920x1080_browser.png)
- [변경 전 KiCad SVG 렌더](assets/phase19_4_5/regulator_before_kicad.png)
- [변경 후 KiCad SVG 렌더](assets/phase19_4_5/regulator_after_kicad.png)
- [Divider KiCad SVG 렌더](assets/phase19_4_5/divider_after_kicad.png)

위험 화면의 상태는 검증 근거가 없어 fail-closed로 유지됩니다. 측정 화면에는 검증된 추천 위치가 없는 현재 컨텍스트를 그대로 표시합니다. 이 두 화면은 연결이나 측정 근거가 없을 때 결과를 꾸며내지 않는지 확인하는 캡처입니다.

## 최소 검증 및 남은 사항

- 전체 테스트: `319 passed` (Python 3.12, 기존 경고 3건)
- KiCad XML netlist 비교: `VERIFIED_EQUIVALENT` (Component 6개, pin-to-net 13개)
- KiCad SVG export: 성공
- JavaScript 구문 확인, Python compileall, `git diff --check`: 성공
- Ruff와 mypy는 현재 환경 및 CI 워크플로에 설치·구성이 없어 실행하지 않았습니다.
- KiCad export는 변경 전에도 나타났던 주석/annotation 경고를 출력합니다. 넷리스트 및 SVG export 결과는 정상이며 이 작업에서 회로도 문서 전체의 annotation 정리는 수행하지 않았습니다.

Phase 19.4.5 구현과 로컬 검증이 완료되었습니다. 코드 커밋 `ce06e0f33a23bbebec89484b31657277f5ac4af1`은 `main`에 반영했고 GitHub Actions `Tests` 실행 [41993724312](https://github.com/alfred0105/CircuitCare-AI/actions/runs/41993724312)이 Python 3.11·3.12, ngspice 통합, OCR 통합, bounded stress, crash injection을 포함해 모두 통과했습니다.
