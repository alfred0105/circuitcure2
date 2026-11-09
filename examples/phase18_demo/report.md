# CircuitCare AI 진단 보고서

- 회로: Regulator + divider + switch (cached demo)
- Circuit Hash: `9f51ab1d0c3a98200e4d52830c632130c63daeba670723cffe7dcbb6ee308294`
- 생성 시각: 2026-11-04T00:39:59.766682+00:00
- 부품 / Net: 6 / 4

## 고장 및 관측

- 모드: KNOWN_FAULT
- 대상: U1 / regulator_output_rise
- 관측 분류: SYNTHETIC_DEMO

## 파라미터 후보 Grid

- 상태: NOT_EVALUATED
- 해석: 후보 Grid 비교를 수행하지 않았습니다.

## SPICE 검증

- 상태: CACHED
- 출처: ngspice_cached_or_live
- Scenario ID: FAULT_U1_REGULATOR_OUTPUT_RISE

## 정격 스트레스 검토

- 상태: RECORDED
- 손상 확정 여부: 아니요

## 근거

- EV-SPICE: SPICE_SIMULATION · RECORDED · comparison.json
- EV-GRAPH: HEURISTIC_ESTIMATE · CONNECTIVITY_ONLY · fault influence graph
- EV-OBS: SYNTHETIC_DEMO · SIMULATED · 원본 파일 없음

## 한계

- Synthetic or simulated evidence is not a physical repair result.
- A priority score or conditional posterior is not a failure probability.
- Unsupported fault models and unapproved ratings remain unevaluated.
- This report collects the current snapshot and does not rerun analysis.

## 재현 정보

- Git SHA: `357294d286b6e21af63aee0b027fdb715c63fce0`
