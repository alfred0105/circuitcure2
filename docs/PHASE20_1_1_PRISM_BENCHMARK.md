# Phase 20.1.1 — PRISM 벤치마크

## 데이터 출처와 보관

- PRISM: [KamMalinski/PRISM](https://github.com/KamMalinski/PRISM), commit `9189c87bbec6995b745b809db99cef6d3269401f`
- PCB 12개, TOP/BOTTOM PNG 24장, KiCad PCB/schematic 각 12개, sample별 reference JSON 5종을 확인했다.
- `scripts/fetch_prism_dataset.py`는 sparse checkout으로 `Dataset/`와 `evaluation/reference/`만 `data/external/prism/`에 받는다. CircuitCare Git 작업 트리는 별도로 유지하며, 기존 다른 데이터는 삭제하지 않는다.
- 외부 데이터와 대용량 결과는 `.gitignore` 대상이다. 입력 파일 SHA-256과 PRISM commit은 실행 결과에 저장한다.
- PRISM checkout의 `LICENSE.txt`는 AGPL-3.0 전문이지만 README는 프로젝트 라이선스가 아직 선택되지 않았다고 적는다. 데이터 파일별 재배포 허가도 별도 명시되지 않았다. 따라서 원본 데이터/PRISM 코드를 CircuitCare에 복사·재배포하지 않는다. 법적 라이선스 판단은 이 통합에서 하지 않았다.

## 데이터 구조와 누수 방지

`prism_dataset.py`는 이미지/설계 파일 경로와 해시를 가진 입력 Adapter를 제공한다. `load_sample()`은 reference JSON을 파싱하지 않는다. 벤치마크가 TOP/BOTTOM 이미지로 복원 파이프라인을 완료한 뒤 별도 `load_references()`로 평가용 JSON을 연다. 이미지 복원 함수에는 정답 좌표, Net 이름, 부품 배치가 전달되지 않는다. 해당 결과는 `IMAGE_ONLY`이다.

KiCad 정답 좌표는 mm이고 PNG는 pixel이다. PRISM은 pixel/mm 변환, 촬영 DPI, 검증된 공통 좌표 정합값을 제공하지 않는다. 따라서 pixel segmentation, pad/via 위치, geometry Precision/Recall/F1/IoU, false merge/split 및 net partition 지표는 **NOT_VERIFIED**다. 계산 불가를 0점으로 대체하지 않았다. TOP/BOTTOM 결과도 서로 다른 이미지 좌표에 남겨 두었다.

## 실행 방법

```bash
python scripts/fetch_prism_dataset.py
python scripts/benchmark_prism.py --sample Simple
python scripts/benchmark_prism.py --all
python scripts/benchmark_prism.py --all --baseline-only
```

외부 데이터는 `--dataset-root`, 산출 위치는 `--output-dir`로 바꿀 수 있다. 기본 처리 긴 변은 500 px이며 정량 정확도 결과가 아니라 제한된 로컬 실행 설정이다. `data/benchmarks/prism/baseline/`에는 기존 Phase 20.1 HSV 처리 결과가, `improved/`에는 adaptive Lab 색상 후보 실험이 저장된다. JSON에는 입력 hash, Python/OpenCV 버전, 실행 시간, 처리상태 및 이미지 overlay 경로가 기록된다.

## 실제 실행

고정 버전에서 12개 전체를 실행했다. TOP/BOTTOM 24장 모두 기존 파이프라인과 adaptive 후보 실험을 완료했다. 마지막 전체 실행은 약 2.5초였으며 실행 시간은 하드웨어·캐시 영향을 받는다. 파이프라인은 기존 보드 외곽 후보 검출, 원근 보정 후보, Copper 후보 segmentation, vectorization, Pad/Via 후보 탐지, same-side 관측 접촉 그래프를 호출했다. 실행별 요약은 `baseline/summary.json`과 `improved/summary.json`에 분리된다.

| PCB | Baseline 후보 픽셀 비율 TOP/BOTTOM | Adaptive 실험 후보 픽셀 비율 TOP/BOTTOM |
|---|---:|---:|
| Cascade | 0.013 / 0.006 | 0.002 / 0.002 |
| Connector | 0.065 / 0.074 | 0.001 / 0.090 |
| Crossover | 0.046 / 0.047 | 0.313 / 0.320 |
| Divider | 0.032 / 0.033 | 0.035 / 0.000 |
| Elements | 0.061 / 0.084 | 0.341 / 0.367 |
| Filter | 0.017 / 0.015 | 0.044 / 0.000 |
| Floating | 0.083 / 0.083 | 0.339 / 0.353 |
| Groundplane | 0.052 / 0.053 | 0.348 / 0.363 |
| Meander | 0.000 / 0.000 | 0.000 / 0.000 |
| Round | 0.004 / 0.004 | 0.000 / 0.004 |
| Simple | 0.012 / 0.091 | 0.137 / 0.362 |
| Transistors | 0.053 / 0.053 | 0.000 / 0.000 |


후보 픽셀 비율과 검출 개수는 알고리즘 출력량을 나타낼 뿐 정확도 지표가 아니다. Adaptive 출력이 커진 것을 개선으로 해석하지 않는다. 모든 12개에서 reference schema와 SHA-256을 기록했다. 상세 결과와 overlay는 무시된 `data/benchmarks/prism/`에 생성된다.

## 기준선 오류와 Phase 20.2 착수 판정

- HSV 기반 기준선은 PRISM 렌더의 soldermask와 동박 색 차이를 잘 분리하지 못하는 샘플이 있다. 일부 면에서는 후보 픽셀이 거의 없고, 다른 샘플은 밝은 영역이 후보가 된다.
- Adaptive Lab 후보 실험은 일부 보드에서 board 바깥 배경이나 실크스크린까지 크게 포함했다. 후보 비율 증가만 관찰됐고 Ground Truth와의 픽셀 비교는 성립하지 않는다.
- 큰 후보 연결 영역을 중심선 계산 없이 `COPPER_BROAD_REGION_CANDIDATE`로 보존하도록 벡터화 경로를 제한했다. 이는 무거운 골격화로 처리 시간이 멈추는 일을 막는 안전/실행성 수정이며, 연결성이나 분류 정확도 향상으로 주장하지 않는다.
- PRISM 통합, 정답 schema 파싱, Simple/Divider/Crossover 포함 12개 이미지 파이프라인 실행, 입력 분리는 확인됐다. 그러나 좌표 등록과 정답 비교 방법은 검증되지 않았다. 따라서 지시서의 자동 착수 조건 중 Ground Truth 비교 검증을 충족하지 못해 정량 성능 개선 단계는 **보류**한다.

다음에 유효한 pixel 평가를 하려면 정답 좌표계와 각 TOP/BOTTOM raster의 대응을 제공하거나, 독립 검증이 가능한 기준점/보정 절차가 필요하다. KiCad reference를 정합에 쓰면 그 결과는 별도 `DESIGN_ASSISTED` 실험으로 기록해야 한다.
