# PCB 사진 위치 매칭 및 위험 오버레이

PCB 사진 처리는 선택 기능입니다. 회로 분석은 사진 없이 계속 동작하며, 사진만으로 연결 구조나 손상을 추정하지 않습니다.

## 설치 및 실행

```bash
python -m pip install -e ".[ui,vision]"
streamlit run src/circuitcare/ui.py
```

`PCB 사진 인식` 탭에서 JPG/JPEG/PNG를 올리거나 브라우저의 카메라 입력을 사용합니다. 이미지는 최대 20 MB, 최대 2,500만 픽셀까지 허용하고 EXIF 방향을 적용한 RGB 복사본을 최대 한 변 2,400 픽셀로 축소합니다. 화면은 원본 방향 크기와 처리 축소율을 표시합니다. 이 과정에서 인터넷 전송을 하지 않고, 원본 이미지는 저장하지 않습니다.

## 촬영 권장

- 보드를 평평한 면에 놓고 가능한 한 정면에서 촬영합니다.
- 그림자와 조명 반사가 적도록 확산광을 사용합니다.
- 기판 외곽과 식별 가능한 부품 중심이 가려지지 않도록 합니다.
- 앞면과 뒷면은 별도 사진 및 별도 정합으로 처리합니다. 뒷면을 자동 좌우 반전하지 않습니다.
- 너무 비스듬하거나 가려진 사진은 수동 모서리/기준점 입력도 신뢰할 수 없습니다.

## 보드 영역과 원근 보정

OpenCV의 grayscale, blur, edge/contour 후보를 표시합니다. 자동 검출은 `candidate_unverified`일 뿐입니다. 사진에 보드와 비슷한 직사각형이 여럿 있거나 외곽이 잘 보이지 않으면 후보 검출에 실패할 수 있습니다.

후보를 확인하고 네 모서리의 처리 이미지 픽셀 좌표를 직접 조정한 뒤 `기판 영역 확정 및 원근 보정`을 누릅니다. 점 클릭 입력 라이브러리는 필수 의존성으로 추가하지 않았으므로, 이미지에 표시된 좌표 축척을 보고 숫자 입력을 사용합니다. 중복·이미지 밖·너무 작은 영역·비볼록 모서리는 거부됩니다.

실제 보드 폭/높이를 아는 경우에만 mm 치수를 입력합니다. 치수를 모르면 선택한 사변형의 마주 보는 변 길이로 출력 비율을 근사하고 실제 크기를 추정하지 않습니다.

## KiCad Component Position CSV

KiCad CLI에서 다음과 같이 내보낼 수 있습니다. 현재 설치한 KiCad에서 명령 옵션을 확인한 뒤 파일 경로를 지정합니다.

```bash
kicad-cli pcb export pos --format csv --units mm --side both -o components.csv board.kicad_pcb
```

KiCad CLI는 position 파일의 CSV 형식과 mm/in 단위, 앞/뒷면/양면 필터, drill/place origin 옵션을 지원합니다. CSV 헤더와 X/Y/회전/면 열은 [KiCad CLI 문서](https://docs.kicad.org/10.0/en/cli/cli.html#pcb-export-position-file)와 [KiCad position CSV 생성 코드](https://docs.kicad.org/doxygen/place__file__exporter_8cpp_source.html)에서 확인했습니다. CSV 헤더는 `Ref,Val,Package,PosX,PosY,Rot,Side`입니다. 본 개발 환경의 KiCad 10.0.6 CLI가 앱 내장 Raspberry Pi HAT 보드로 생성한 한 행 CSV도 읽어 `J1`, `bottom`, mm 좌표로 파싱했습니다. KiCad가 내보내는 y축 부호와 선택 가능한 뒷면 X 반전 옵션이 있으므로, 사진 측정점과 좌표 방향은 임의로 반전하지 말고 정합 기준점으로 연결하세요. 앞면과 뒷면은 각각 따로 기준점을 설정합니다. 다른 열 이름은 일부 별칭을 지원합니다. 파일에 단위 메타데이터가 없는 경우가 있으므로 UI에서 실제 내보내기 단위를 명시해야 합니다. inch는 mm로 변환합니다. 원점 방식은 내보낼 때 사용한 옵션을 확인해 기록합니다. 같은 좌표 원점과 방향이어야 사진 정합이 의미가 있습니다. KiCad CLI가 PATH에 없다면 앱 안의 실행 파일을 직접 지정할 수 있습니다. 예를 들어 macOS는 `/Applications/KiCad.app/Contents/MacOS/kicad-cli`를 사용합니다.

## 사진과 설계 좌표 정합

1. 앞면 또는 뒷면을 선택하고 해당 면의 CSV를 읽습니다.
2. 원근 보정된 큰 사진에서 중심 위치가 식별되는 Reference를 고릅니다.
3. 같은 부품의 설계 mm 좌표는 CSV에서 가져오고, 사진 x/y 픽셀 좌표는 UI에서 입력합니다.
4. 서로 다른 네 개 이상의 기준점을 기판 넓은 영역에 배치합니다. 거의 한 직선에 놓인 점은 거부됩니다.
5. 선택 사항인 독립 검증점은 변환 계산에 넣지 않고 예상 사진 위치와 오차를 따로 표시합니다.
6. 미확정 미리보기의 중심점이 실제 대응 부품과 일치하는지 살핀 뒤 직접 확인해야 위험도 오버레이가 활성화됩니다.

기준점 네 개만 쓴 경우 그 점들에 대한 재투영 오차가 작아도 정확도 보증이 아닙니다. 검증점이나 다른 독립 확인점을 사용하세요. 기준점 입력의 정합 계산, 처리 상태 `pending_verification`, 사용자 확인 `user_confirmed`는 별도입니다. 앞면·뒷면을 하나의 Homography로 섞지 않습니다.

## 위험도 오버레이와 저장

확정한 정합에서 KiCad footprint의 중심점을 표시합니다. 참조번호가 기존 넷리스트에 없으면 사진 마커를 만들지 않습니다. 위험 색상은 선택된 회로/고장 컨텍스트에서 성공한 현재 SPICE/위험 보고서가 있으며 고장 부품·고장 유형이 일치할 때만 연결합니다. 해당 Reference가 위험 후보 목록에 없으면 위치만 표시합니다.

- 빨강: `EXCEEDED`
- 주황: `NEAR_LIMIT`
- 청회색: `WITHIN_LIMIT` (평가한 설정 정격 항목 한정)
- 회색: `UNKNOWN` 또는 현재 위험 결과 없음
- 보라: 사용자가 현재 시나리오의 고장 부품으로 지정한 Reference (`KNOWN_FAILED` 입력 상태)

선택한 Reference의 스트레스 지표, 정격과 출처, 점검 우선순위, 근거, 손상 확정 여부를 JSON으로 확인합니다. 마커는 부품 중심점이며 footprint 외곽 검출이 아닙니다. 정격은 시연 가정일 수 있고, 우선순위는 고장 확률이 아닙니다.

PNG와 JSON을 내려받을 수 있으며, `Vision 결과를 로컬 data/vision_results에 저장`을 직접 눌렀을 때만 결과 파일을 저장합니다. 해시·CSV 출처·보드 면·기준점·Homography·오차·현재 시나리오 및 위험 보고서 경로를 기록합니다. 원본 사진은 결과 폴더에 복사하지 않습니다. `data/vision_results/`는 Git 추적 대상에서 제외됩니다.

## 확인 상태와 제한

저장소와 이번 작업에서 확인한 프로젝트 작업공간에는 동일한 실물 보드에 대응하는 PCB 사진, `.kicad_pcb`, KiCad Position CSV, XML 넷리스트 자료가 없습니다. 따라서 실제 정합은 **NOT VERIFIED — MISSING REAL PCB DATA**이며, 독립 검증 오차를 계산하거나 해당 회로의 위험도를 사진에 검증 오버레이할 수 없습니다. 자동 외곽 검출, 좌표 변환, 사용자 확인 절차, 위험 매칭 로직은 합성 fixture에서 검증한 **Synthetic Fixture Verified** 상태입니다. KiCad 10.0.6 앱 내장 Raspberry Pi HAT 템플릿에서 내보낸 `Ref,Val,Package,PosX,PosY,Rot,Side` 형식의 한 부품 CSV를 파싱했지만, 이 자료는 CircuitCare 보드와 일치하지 않으므로 CSV 형식 확인에만 사용했습니다. **Real PCB Registered** 또는 **Real PCB Independently Validated**로 표시하지 않습니다.

GitHub Actions run [41569204216](https://github.com/alfred0105/CircuitCare-AI/actions/runs/41569204216)의 Python 3.11 테스트 수집에서 `numpy` 누락이 확인됐습니다. editable package 설치는 성공했지만 workflow가 `.[dev]`만 설치해 `tests/test_vision.py`의 선택 Vision 의존성이 빠졌습니다. CI는 `.[dev,vision]`을 설치하도록 수정했습니다. 해당 실패에서는 코드 테스트가 실행되기 전에 수집이 중단됐으며, 이 문서의 실물 정합 상태와는 별개입니다.

핫픽스 후 로컬 Python 3.12에서 `.[dev,ui,vision]` 설치, pytest 29개 통과, 기본 U1 CLI 실행, Streamlit `AppTest`의 6개 탭 로드를 확인했습니다. 수정 커밋의 [Python 3.11 GitHub Actions run 41570036079](https://github.com/alfred0105/CircuitCare-AI/actions/runs/41570036079)도 Vision 의존성 설치와 pytest 29개 통과로 완료됐습니다.

## 선택 기능: AI 부품 후보와 실크스크린 OCR

아래 추가 설치와 모델 준비를 마치면 보정된 PCB 이미지에서 10개 범주의 부품 후보를 로컬 ONNX 모델로 검출하고, RapidOCR로 실크스크린 텍스트 후보를 읽을 수 있습니다.

```bash
python -m pip install -e ".[ui,vision,vision-ai,vision-ocr]"
python scripts/prepare_pcb_detector.py
streamlit run src/circuitcare/ui.py
```

기본 변환기는 모델 가중치와 ONNX 파일을 저장소 밖 사용자 캐시에 보관합니다. `--output models/pcb_component_detector.onnx`로 저장소 내부 경로를 고르면 `.gitignore`가 파일을 제외합니다. 변환을 위해 Ultralytics 설치가 별도로 필요하며, 이 패키지는 앱 실행 의존성에 포함되지 않습니다. 출처, 해시, 별도 라이선스와 exporter 주의 사항은 [PCB AI 탐지 안내](PCB_AI_DETECTION.md)를 확인하세요.

1. `실제 모델로 부품 탐지`와 `로컬 OCR 실행`은 사용자가 각각 직접 눌러 실행합니다. 모델 파일이 없으면 `MODEL_UNAVAILABLE`로 표시합니다.
2. 모델 상자·종류·confidence는 학습 모델의 후보 결과입니다. 낮은 점수 후보는 `unknown_component`로 표시할 수 있으며, confidence를 정답률이나 손상 확률로 해석하지 않습니다.
3. 오검출을 제외하거나 종류를 수정하고, 누락된 부품 상자를 수동 추가할 수 있습니다. 원래 모델 예측과 사용자 수정은 결과 JSON에서 구분됩니다.
4. OCR은 원문과 상자 좌표를 보존합니다. `U26` 같은 Reference 모양의 문자열도 읽기 제안일 뿐 자동 교정이나 부품 식별 확정이 아닙니다.
5. 탐지 상자와 KiCad footprint 위치는 위치·OCR·종류 호환성의 heuristic 점수로 정렬 제안합니다. 각 Reference는 사용자가 직접 확정해야 하며 두 상자를 하나의 Reference로 확정할 수 없습니다.
6. 위험 색을 탐지 상자에 연결하려면 앞선 보드/회로 동일성 확인과 해당 검출의 Reference 확인 모두가 필요합니다. 그리고 기존 성공한 위험 분석과 고장 시나리오가 일치해야 합니다. 그래도 이는 계산상 스트레스 표시이지 사진이 손상을 판정한 결과가 아닙니다.

공개 데이터셋 평가 결과와 실행 시간은 [PCB AI 평가 기록](PCB_AI_EVALUATION.md)에 있습니다. 사용 가능한 데이터에는 CircuitCare의 PCB 사진과 같은 설계의 `.kicad_pcb`, Position CSV, XML 넷리스트가 없으므로 실제 Reference 매칭과 실물 위험 오버레이는 **NOT VERIFIED** 상태입니다. 현재 평가 split에서의 부품 탐지 성능을 독립 보드 일반화로 간주하지 않습니다.

Phase 13 What-if inspection ranking may appear as a separate `inspect #rank` label on an already user-confirmed KiCad/AI Reference match. It does not change the existing SPICE-based electrical risk color. See [the What-if guide](WHAT_IF_DIAGNOSIS.md).
