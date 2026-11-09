# Phase 20.1 — 다중 사진 PCB 복원 기반

## 1. 기존 Vision 기능 조사 및 재사용

새 작업공간은 기존 Vision 파이프라인 위에 구축했습니다.

- `vision/image_loader.py`: 업로드 크기·픽셀 제한, SHA-256, EXIF 방향 보정, RGB 변환 및 축소를 재사용했습니다. 허용 형식에 WebP를 추가했습니다.
- `vision/board_detector.py`, `vision/perspective.py`: 보드 사각 외곽 후보 검출과 정면 원근 보정을 재사용합니다. 검출 외곽은 사용자 선택 대상이며 그 자체로 보드 경계가 확인된 것은 아닙니다.
- `vision/registration.py`: KiCad 좌표와 사진 좌표 사이의 기존 Homography 및 Reference 확인 절차는 보존했습니다.
- `vision/component_detector.py`, `vision/ocr_backend.py`, PCB Vision 화면: 기존 학습 모델의 부품 검출·Reference 확인과 OCR을 그대로 유지했습니다. 새 화면에서는 선택적으로 로컬 OCR을 실행할 수 있고, 모델 검출 기능은 기존 `PCB 사진 인식` 고급 도구에서 계속 사용할 수 있습니다.

기존 KiCad 기반 컴포넌트 위치와 사진 관측 결과를 하나의 출처로 섞지 않습니다.

## 2. 데이터 계층과 증거 상태

`vision/reconstruction/models.py`에 다음 구조를 추가했습니다.

- `PhotoRecord`: 사진 ID, 원본 SHA-256, 사용자 지정 면·촬영 유형, 이미지 크기·형식, 품질 힌트 및 중복 관계
- `BoardGeometryObservation`: 사진 좌표의 보드 경계와 좌표 프레임, 확인 상태
- `RegisteredView`: 변환 행렬, 정합 방법·상태, inlier 수, 재투영 오차 및 겹침 범위
- `CopperObservation`: 면, 압축 마스크와 크기, 벡터 선분, 끝점·분기점, 폭 추정, 불확실성
- `PadViaCandidate`: 위치·형상·크기 후보와 도금 여부 미확인 상태
- `ConnectivityEvidence`, `ObservedTopologyGraph`: 같은 면에서 보인 마스크 접촉 관계
- `DiagnosticEvidence`: 기존 진단 결과를 사진 출처와 연결하기 위한 자료 구조

`OBSERVED`, `INFERRED`, `USER_CONFIRMED`, `DESIGN_VERIFIED`, `CONTINUITY_VERIFIED`, `UNKNOWN`, `CONFLICT` 상태를 제공합니다. 이 단계에서 만들어지는 구리·패드·접촉 결과는 `INFERRED`이며 전기적 넷리스트나 연속성 검증으로 승격하지 않습니다.

## 3. 사진 입력과 품질

`다중 사진 PCB 복원` 화면에서 PNG/JPEG/WebP 사진을 최대 20장 선택하고, 각 사진의 앞면·뒷면·미지정 및 전체·확대·경사·마킹 촬영 유형을 사용자가 지정할 수 있습니다. 파일당 20 MB, 사진당 2,500만 픽셀 제한과 EXIF 방향 보정을 기존 로더가 적용합니다. 원본은 자동 저장하거나 전송하지 않으며, 앱 세션의 업로드 바이트와 SHA-256으로 처리 결과를 추적합니다. 동일한 SHA-256은 중복 사진으로 표시합니다.

품질 검사에서는 Laplacian 흐림 지표, 해상도, 과다·과소 노출, 반사 후보, 검출된 보드 영역 비율, 외곽 비대칭 및 특징점 매칭 가능성 힌트를 계산합니다. 이 값은 촬영 안내이며 사진을 배제하거나 분석을 중단하는 임계값은 아닙니다. 가림 검출은 아직 자동화하지 않고 `NOT_ASSESSED`로 둡니다.

## 4. 기판 프레임과 다중 사진 정합

자동 외곽 후보를 선택해 기존 원근 보정 함수로 사진을 직사각 보드 후보 프레임에 투영합니다. 외곽 검출에 실패하면 원본 픽셀 프레임을 유지하고, 특징점 기반 정합을 자동 시도하지 않습니다.

동일 면 전체 사진에는 ORB 특징점과 RANSAC Homography를 사용합니다. 최소 12개 inlier, 4 px 이하 inlier 재투영 RMSE, 기준 이미지의 4% 이상 특징 분포 범위를 만족해도 결과는 `REGISTERED_CANDIDATE`/`INFERRED`입니다. 조건을 충족하지 않으면 `REGISTRATION_UNVERIFIED`를 반환합니다. 사용자는 수동 대안에서 두 사진의 네 대응점을 입력할 수 있으며, 대응점이 한 직선에 몰리면 거부합니다. 이 정합은 사진 위치 추정이며 회로 연결을 의미하지 않습니다.

앞면과 뒷면은 ORB로 직접 연결하지 않습니다. 각 보드 외곽을 별도로 정면 보정한 뒤 사용자가 뒷면 회전·좌우 반전 방향을 지정하고 기준 모서리와 커넥터를 비교해 확인할 수 있습니다. 확인 전은 `REGISTRATION_UNVERIFIED`/`UNKNOWN`, 확인 후는 방향 변환에만 `USER_CONFIRMED`입니다. 비아의 관통 도금 여부나 동일 Net은 여기서 확정하지 않습니다.

## 5. 가시 동박 후보, 패드·비아 및 초기 그래프

CPU에서 실행하는 Baseline은 HSV의 밝고 낮은 채도인 영역과 국소 대비를 이용합니다. 사용 가능한 기판 마스크 밖은 제외하며, 선택한 투명도로 사진에 중첩합니다. 실크스크린, 반사광, 솔더마스크 색상은 오검출 또는 누락 원인이 될 수 있습니다. 마스크 미검출 픽셀은 구리 부재로 확정되지 않습니다. 가려진 배선은 계속 `NOT_OBSERVABLE`입니다.

연결요소를 스켈레톤화하고, 경로를 따라 폴리라인·끝점·분기점을 벡터화합니다. 넓고 조밀한 후보 영역은 `COPPER_PLANE_CANDIDATE`, 나머지는 `COPPER_TRACE_CANDIDATE`로 구분합니다. 평균 픽셀 폭은 거친 추정입니다.

Hough 원 후보와 동박 후보의 원형·사각형 윤곽으로 `PAD_OR_VIA_CANDIDATE`를 만듭니다. 결과 수는 제한하며 confidence는 낮게 둡니다. 패드·비아·장착 구멍은 서로 구분하지 못할 수 있고 `through_plated`는 미상입니다.

초기 그래프는 같은 사진과 같은 면에서 후보 마스크와 패드 후보가 실제 픽셀상 닿는지 기록합니다. 서로 다른 면 사이 연결 간선은 생성하지 않습니다. 이 그래프는 `OBSERVED_TOPOLOGY_ONLY`이며 가까운 패턴이거나 같은 Net처럼 보인다는 이유로 연결을 확정하지 않습니다.

## 6. 사용자 화면 및 기존 기능

`다중 사진 PCB 복원`을 고급 도구에 추가했습니다. 다중 업로드, 사진별 면·촬영 유형, 외곽 후보 선택, 품질 안내, 원본 보기, 배선 후보, Pad/Via 후보, 접촉 후보 보기, 투명도 조절, 캔버스 맞춤·휠 확대·버튼 확대축소·드래그 이동을 제공합니다. 부품 마킹 OCR은 사용자가 사진별로 요청할 때만 로컬에서 실행합니다.

기존 `PCB 사진 인식`, KiCad 회로도, SPICE, 위험 평가, Bayesian, 보고서 기능은 기존 경로에서 유지됩니다. 새 다중 사진 파이프라인은 현재 해당 분석 결과를 자동으로 사진 위에 얹거나 수리 추천과 결합하지 않습니다.

## 7. 합성 Fixture와 정량 평가

`tests/fixtures/reconstruction_synthetic.py`에서 만든 **합성 전·후면 이미지**와 PNG를 사용했습니다. fixture에는 직선·직각·분기, 가까우나 분리된 선, 교차 모양, 선 간격, 원형 패드, 사각 패드, 장착 구멍, 뒷면 반전, 원근 왜곡, 밝은 반사 패치 및 노이즈가 있습니다. 실제 PCB 사진이 아니며 모델 학습자료도 아닙니다.

알려진 선 픽셀 마스크와 Baseline 결과를 전체 사진 픽셀에서 비교했습니다.

| 합성 면 | Precision | Recall | IoU | 외곽 후보 | 벡터 후보 | 패드/비아 후보 |
|---|---:|---:|---:|---:|---:|---:|
| 앞면 | 0.685 | 0.949 | 0.661 | 1 | 8 | 4 |
| 뒷면 | 0.651 | 0.949 | 0.629 | 1 | 10 | 4 |

동일 합성 영상에 알려진 7 px / −5 px 이동을 적용한 정합 smoke 검증에서는 674개 inlier, RMSE 0.754 px, 겹침 추정치 0.580으로 `REGISTERED_CANDIDATE`가 나왔습니다. 이 값들은 고정된 합성 입력 결과이며 실제 PCB 성능, 패드 검출 정밀도, 연결 정확도를 나타내지 않습니다. 패드·비아 Ground Truth 라벨과 실제 PCB 검증 세트는 없습니다.

## 8. 브라우저 및 자동 검증

Streamlit 화면에서 두 PNG를 동시에 업로드하고 앞면·뒷면을 지정했습니다. 품질 안내, 보드 외곽 후보 및 정면 보정, 뒷면 변환의 미검증 상태, 배선 중첩과 `OBSERVED_TOPOLOGY_ONLY` 출력을 확인했습니다. 실제 브라우저 캡처:

- [다중 사진 및 면 지정 화면](assets/phase20_1_multiview_synthetic_browser.png)
- [앞면·뒷면 방향 검토 화면](assets/phase20_1_registration_review_browser.png)
- [배선 후보 중첩 화면](assets/phase20_1_copper_candidates_browser.png)
- [확대·이동 가능한 PCB 캔버스](assets/phase20_1_zoom_pan_viewer_browser.png)

실제 브라우저 E2E에서는 확대 버튼으로 125%가 표시되는 것과 캔버스를 드래그했을 때 카메라 변환이 바뀌는 것을 확인했습니다. 페이지 JavaScript 오류는 없었습니다. 로컬 Python 3.12 회귀 테스트는 **328 passed**이며 `tests/test_vision_reconstruction.py`, 기존 `tests/test_vision.py`, 전체 CircuitCare 회귀 및 Streamlit AppTest 내비게이션을 포함합니다. Python compileall, JavaScript syntax check, `git diff --check`도 통과했습니다. 세션 상태 및 내부 상태 코드는 유지됩니다.

GitHub Actions [Tests run 42018132645](https://github.com/alfred0105/CircuitCare-AI/actions/runs/42018132645)도 최종 구현 커밋에서 성공했습니다. Python 3.11·3.12, ngspice 통합, OCR 통합, bounded stress 및 isolated crash-injection 작업이 모두 통과했습니다.

## 9. 제한 사항과 Phase 20.2 착수 조건

- 검증 자료가 합성 이미지뿐입니다. 실제 PCB의 배선 색·솔더마스크·반사 편차에 대한 정확도는 아직 모릅니다.
- 마스킹은 설명 가능한 고전 CV 임계값 기반입니다. 모델 기반 segmentation, 가림 복원 및 plane 경계 정리는 하지 않았습니다.
- 부품 모델 검출은 기존 단일 사진 PCB Vision 경로를 재사용하며 새 화면의 다중 사진 좌표로 검출 박스를 자동 합치지 않습니다. OCR Reference는 후보 문자열일 뿐 자동 승인하지 않습니다.
- 같은 면 특징 정합은 후보입니다. 실제 feature 분포와 사용자 확인을 기록하지만 신뢰 설계 좌표와 일치 검증은 없습니다.
- 앞·뒷면 변환의 사용자 확인은 방향 변환만 승인합니다. 비아 대응, 관통 도금, 전기적 Net 확정은 Phase 20.3 이후입니다.
- RC/보드 결함이나 끊어진 패턴을 실제 PCB 고장으로 진단하지 않습니다.

Phase 20.2로 넘어가기 전 실제 PCB 사진의 촬영 권한과 레이블을 확보하고, trace·pad·via 영역별 Ground Truth와 보드별 train/eval 분리를 마련해야 합니다. 이후 임계값 개선은 그 자료에서 평가하고, 사용자 수정 결과와 출처를 보존해야 합니다.

Phase 20.1 기능 커밋 `17134b316b502f8daaa44ed3ae51514cc34f219f`와 캔버스 Pan/Zoom 커밋 `1cd89ad35512a429d9b6d3e1bf2f49f13659cac5`는 `main`에 반영되었습니다. 실제 PCB 정확도는 아직 검증되지 않았으므로 이 단계의 성공 범위는 합성 fixture와 로컬/CI 테스트까지입니다.
