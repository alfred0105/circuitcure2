# Phase 20.2 — Segmentation 실험 및 착수 판정

## 기존 방법과 관찰

Phase 20.1의 후보 검출은 고정 HSV value/saturation 및 local contrast 조건을 사용한다. PRISM 렌더는 보드별 색상, substrate, silkscreen, 노출 copper 표현이 달라 일부 배선 영역이 빠지거나 비 copper가 후보가 될 수 있다.

## 수행한 변경/실험

- 연결 영역이 넓은데도 `vectorize_copper()`가 픽셀 단위 skeletonization을 오래 반복할 수 있어, 큰 plane/broad region은 중심선 추출 없이 별도 불확실 후보로 보존한다. 작은 trace 후보만 기존 vectorization을 계속한다.
- Benchmark 전용으로 Lab 색상 차와 로컬 board field를 사용하는 adaptive 후보 실험을 추가했다. 제품 기본 UI 경로에는 적용하지 않았다.
- 동일한 PRISM 입력/최대 처리 변 길이 500 px 및 기존 rectification 설정으로 12개 보드 24면을 양 실험에서 처리했다.

## 비교 결과

이미지별 candidate pixel fraction은 `docs/PHASE20_1_1_PRISM_BENCHMARK.md` 표를 참고한다. 전체 평균은 HSV 기준 4.09%, adaptive 후보 실험 14.26%였다. 이 수치는 후보 생성량만 설명하며 Precision/Recall/IoU 또는 개선을 뜻하지 않는다. Adaptive 쪽은 Crossover, Elements, Groundplane, Simple 같은 일부 표본에서 급격히 넓은 영역을 표시했고, 배경/실크스크린 오검출이 섞여 있어 품질 향상으로 채택하지 않았다.

Pad/Via ground-truth 위치 대응은 pixel/mm 정합이 없어 평가하지 않았다. 후보 수와 접촉 그래프 edge 수도 예측 품질 점수로 해석하지 않는다. 샘플별 genuine segmentation metric, false merge/split, via F1 비교는 모두 `NOT_VERIFIED`다.

## 남은 오류와 다음 단계

1. 물리 scale 없이 정답 geometry와 raster를 픽셀 비교할 수 없다.
2. 이미지 기준 board boundary 검출은 일부 렌더에 대해 검증된 외곽 정답이 아니다.
3. adaptive 후보에는 배경/실크스크린 오검출이 있다.
4. via 및 pad의 단일/중복 검출 성능을 reference와 연결할 수 없다.
5. observed pixel contact은 실제 electrical net이 아니다.

## Phase 20.3 준비도

현재 same-side 관측 topology를 생성하고 status를 `NOT_VERIFIED`로 보존한다. 하지만 Pin-to-Net 대응, false short/open, 앞/뒷면 via 대응, plane connectivity는 평가할 수 없다. 따라서 Phase 20.3의 전기적 연결 복원은 착수 준비가 되지 않았다. 먼저 raster↔KiCad 좌표 정합과 샘플 독립 검증 절차가 필요하다.
