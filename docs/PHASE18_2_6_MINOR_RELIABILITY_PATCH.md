# Phase 18.2.6 Minor Reliability Patch

## 변경 내용

- Datasheet 정격 재평가 예외가 발생할 때 현재 `applied_rating_session_key()`로 만든 세션 키만 제거한다. 다른 Context와 분석 상태는 유지한다.
- Live Simulation을 성공으로 확정하기 전에 `comparison.csv`의 존재, 크기(64 MiB 이하), 필수 열과 최소 두 데이터 행을 확인하고 SHA-256을 Manifest의 기존 `result_sha256`에 기록한다.
- Cached 결과에서 CSV가 있으면 Manifest 해시를 확인한다. 해시 변경 또는 Manifest 해시를 가진 CSV의 삭제는 `STALE`, CSV가 있지만 해시가 없는 과거 결과는 `PROVENANCE_UNVERIFIED`로 처리한다.
- 해시가 없는 CSV가 포함된 결과는 기존 결과 신뢰 정책에 따라 분석값도 검증되지 않은 것으로 처리한다. CSV가 아예 없고 Manifest에도 항목이 없는 JSON 전용 결과는 기존 JSON 검증을 유지하며 파형은 제공하지 않는다.
- 파형 뷰와 CSV 다운로드는 CSV 무결성이 `VERIFIED`일 때만 파일을 읽거나 제공한다. 자동 재시뮬레이션은 하지 않는다.
- 파형 읽기와 다운로드 시점에도 Manifest의 해시와 다시 대조하여 검증 이후 파일이 바뀌는 경우를 차단한다.

## 변경 파일

- `src/circuitcare/ui.py`
- `src/circuitcare/services/simulation_service.py`
- `src/circuitcare/dashboard/analysis_loader.py`
- `src/circuitcare/dashboard/views/waveforms.py`
- `src/circuitcare/dashboard/views/downloads.py`
- `tests/test_dashboard_architecture.py`
- `tests/test_phase18_workspace_reporting.py`
- `docs/PHASE18_2_6_MINOR_RELIABILITY_PATCH.md`

## 호환성 및 확인 상태

Manifest 스키마, 기존 JSON 결과 키, Fingerprint 구조는 유지한다. 새 Live 결과는 CSV 해시를 포함한다. 이전 Manifest에 CSV 해시가 없지만 CSV 파일이 존재하는 경우 검증되지 않은 CSV를 표시하거나 다운로드하지 않는다.

- `compileall`: PASS
- 기본 suite (`not integration and not stress and not crash_injection`): 200 passed, 28 deselected
- Integration: 14 passed
- Stress: 11 passed
- 변경 소스 및 테스트 파일 Ruff: PASS
- 전체 `ruff check src/circuitcare`: 이번 변경과 무관한 기존 항목 90개 보고. 이번 변경 소스에서는 Ruff 오류 없음.
- mypy: `NOT_VERIFIED` (로컬 가상환경에 mypy 실행 파일이 없음)
- Crash injection: `NOT_VERIFIED` (저장소 지침대로 격리 CI 환경에서만 실행)
- 브라우저: 별도 포트 8503에서 앱 로드 확인. 라이브 시뮬레이션/CSV 변조/다운로드 브라우저 E2E는 `NOT_VERIFIED`; 이 경계는 회귀 테스트와 ngspice Integration으로 확인.
- GitHub Actions: PASS — run [41837419673](https://github.com/alfred0105/CircuitCare-AI/actions/runs/41837419673), all six jobs passed (Python 3.11, Python 3.12, ngspice integration, OCR, bounded stress, isolated crash injection).
- 구현 커밋: `40bed6f91b710556a9c0333c56b86a6bcc5c8653` (pushed to `main`).
- 이 범위에서는 새 P0/P1 문제가 발견되지 않았다. 이전 Repository Audit의 PDF resource isolation P1과 기존 종료 보고서의 원인 미확정 상태는 이 패치의 범위 밖에 남아 있다.
- Phase 18.2.6 완료. Phase 18 hotfix 작업의 종료는 가능하지만, 저장소 전체의 `RELEASE_STABILITY_VERIFIED`는 기존 P1 및 미검증 브라우저 혼합 부하 경로가 남아 있어 계속 `NOT_VERIFIED`로 둔다.
