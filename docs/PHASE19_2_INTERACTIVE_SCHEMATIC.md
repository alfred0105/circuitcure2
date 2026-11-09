# Phase 19.2 — Interactive Schematic Workspace

## 1. 기존 구현

기존 `identity.py`가 KiCad Symbol Reference, UUID, library ID, unit, sheet path를 읽고 `exporter.py`가 KiCad CLI SVG를 content-addressed cache에 저장한다. 기존 Inspector, 분석 상태, XML-only 회로 연결 보기는 유지했다. 선택된 부품을 고장 부품이나 simulation 입력으로 사용하지 않는다.

## 2. Viewer 기술 선택

로컬 환경의 Streamlit 1.66.1에서 지원하는 `st.components.v2`를 사용했다. HTML/CSS와 바닐라 JavaScript만 패키지 자산으로 제공하며 프론트엔드 번들러나 별도 서버는 추가하지 않았다. v2를 지원하지 않는 환경은 기존 SVG 정적 표시로 폴백한다.

## 3. SVG 및 KiCad 좌표 매핑

Geometry는 내장된 KiCad library 심벌 primitive에서 경계점을 읽고 placed symbol의 `(at x y angle)`로 변환한다. SVG `viewBox`는 반드시 유한한 양수 폭·높이 및 `(0, 0)` 원점이어야 한다. Divider fixture의 실제 KiCad CLI 결과는 `0 0 297.0022 210.0072`이고, 1 KiCad mm가 1 SVG viewBox 단위인 변환은 J1/R1/R2의 실제 SVG 도형 좌표와 맞는 것을 통합 테스트에서 대조했다. 화면 크기·확대·이동은 SVG viewBox를 기준으로 동일하게 렌더링된다.

## 4. Hit Region

`geometry.py`가 Symbol UUID, Reference, library ID, unit, root sheet path가 현재 schematic identity와 한 건으로 일치할 때만 library primitive의 bounds를 회전 변환해 다각형 Hit Region을 생성한다. SVG element 순서나 Reference 텍스트 위치를 추정에 사용하지 않는다. 지원되지 않는 primitive, 중첩 영역, identity 불일치, 잘못된 viewBox는 선택을 거부한다. 다중 sheet의 SVG 페이지와 sheet path 대응은 검증되지 않았으므로 `PARTIAL`로 보고하고 클릭 매핑을 끈다. 다중 unit은 현재 sheet에서 추출 가능한 geometry와 unit identity를 region별로 보존한다.

## 5. 이벤트

Viewer의 SVG 이미지와 투명 Hit Region은 같은 SVG viewBox·camera transform을 사용한다. Pointer wheel, drag, Fit, Zoom buttons, Reset, hover는 JavaScript 안에서 처리되어 Python rerun을 호출하지 않는다. 선택만 Components v2 state로 전달한다. Python은 현재 context key, netlist component Reference, UUID, sheet path, unit을 다시 검증한 뒤 점검 선택을 갱신한다.

## 6. Session State 및 context

점검 선택 key는 회로명, 부품 목록, 연결 정보, schematic와 하위 sheet 파일 바이트를 해시한 context로 scope한다. Viewer 이벤트에도 동일 context 해시가 포함된다. 고장 부품은 기존 sidebar fault 선택에서만 변경된다. Divider 브라우저 흐름에서 R2 검색 선택 및 J1 canvas 선택 후 Inspector가 각각 갱신됐고, 고장 부품 R1과 simulation provenance 표시는 유지됐다. RC 회로로 변경했을 때 선택은 해당 회로의 RCHARGE로 초기화되고 schematic 없는 상태에서 기존 connectivity view가 표시됐다.

## 7. 보안

기존 exporter SVG 검사에 script, foreignObject, iframe, object, embed, 외부 CSS import/URL 및 JavaScript URL 거부를 보강했다. SVG는 검증 후 이미지 Blob URL로 표시되고 SVG markup을 부모 페이지 DOM으로 삽입하지 않는다. Click payload는 Python에서 현재 identity와 비교한다. 업로드 크기 제한(20 MB), 프로젝트 경로 제한, exporter SVG 크기 한계를 유지한다. Blob URL과 event listeners는 component cleanup 시 제거한다.

## 8. 성능

소형 Divider fixture에서 wheel zoom으로 배율이 100%에서 112%로 바뀌고 drag pan이 동작하는 것을 브라우저에서 확인했다. Zoom/drag는 JS 내 local state만 갱신하고 Streamlit component state를 건드리지 않는다. KiCad export cache는 기존 방식으로 재사용한다. FPS/지연 시간은 측정하지 않았다. ResizeObserver 1개와 viewer의 고정 event listeners를 mount 단위로 등록하고 cleanup에서 제거하도록 구현했다.

## 9. 브라우저 E2E

실제 로컬 Streamlit 브라우저에서 Divider SVG 표시, Geometry VERIFIED 표시, Fit/Reset toolbar, wheel zoom, drag pan, J1 canvas selection → Inspector J1, 부품 검색 R2 → Inspector R2, fault R1 불변, RC 전환 후 이전 선택 격리 및 connectivity fallback을 확인했다. Pan/Zoom, Fit, Reset 후 Streamlit 전체 rerun이 발생하지 않았다. Console error/warning은 없었다. 심벌 선택은 노출된 Hit Region 접근성 버튼을 통해 브라우저 이벤트 전달과 Inspector 동기화까지 확인했다. OS 포인터의 좌표 기반 클릭은 별도로 검증하지 않았다.

## 10. 검증 결과

개발 환경은 Python 3.12이며, compileall 성공, 비 integration/stress/crash 테스트 231 passed, integration 22 passed, stress 11 passed, Ruff 성공, 변경 Python 4개 파일 mypy 성공, Node.js module syntax check 성공. 기존 pytest 경고는 zipfile 중복 항목 fixture 경고와 Streamlit chart helper Pandas deprecation warning이며 이번 변경으로 새 오류가 발생하지 않았다. Python 3.11은 로컬에서 실행하지 않았고, push 후 Actions 상태를 따로 확인한다.

## 11. 회귀

Integration 집합에서 KiCad CLI로 생성한 Divider netlist와 기존 fixture를 비교하고 실제 SVG의 R1/R2 rectangle 및 pin path 좌표를 검증했다. SPICE, Fault Injector, Risk/Bayesian, report, PCB vision 로직은 변경하지 않았다. SVG export sanitizer와 schematic workspace 선택 연결만 확장했다.

## 12. 제한

- 다중 sheet의 SVG 파일과 sheet path 자동 매칭은 `PARTIAL`이며 해당 SVG에서는 심벌 선택이 비활성이다.
- 지원하는 Library geometry primitive는 rectangle, polyline, circle, pin이다. arc 등 미지원 도형이 포함되면 fail-closed 처리한다.
- Hit Region은 부품의 library geometry bounds 다각형이며, 겹치는 bounds는 모호성으로 거부한다.
- SVG export가 없는 회로도 및 XML-only 입력은 기존 connectivity fallback을 쓴다.
- Canvas highlight는 점검 중인 Reference 표시이며 전기적 위험이나 손상을 뜻하지 않는다.
- 실제 KiCad multi-unit/multi-sheet 프로젝트의 전체 상호작용과 Browser pointer drag는 별도 검증이 필요하다.

## 13. Phase 19.3 착수 조건

Phase 19.3에서 위험도/분석 overlay를 추가하기 전, multi-sheet export와 sheet path의 검증된 매핑 자료가 필요하다. Overlay는 현재 symbol identity 및 viewBox 좌표 API를 재사용하고, 점검 선택 highlight와 전기적 근거 표시를 서로 다른 시각 계층으로 유지해야 한다. 실제 손상 표현은 지원되는 simulation/rating provenance에서만 제공해야 한다.
