"""Component detection and manual box/class review for the PCB Workbench."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[3] / "apps" / "pcb-component-recognition"
MODULE_NAME = "circuitcare_pcb_component_app_server"
CLASS_OPTIONS = [
    "unknown", "resistor", "inductor", "capacitor", "connector", "jumper", "IC",
    "switch", "relay", "transistor", "potentiometer", "battery", "buzzer", "diode",
    "display", "emi_filter", "ferrite_bead", "fuse", "LED", "clock", "oscillator",
    "voltage_regulator", "MOV", "transformer", "heatsink",
]
STATUS_OPTIONS = ["검수 필요", "확인", "제외"]


def review_scope(image_bytes: bytes, project_id: str, photo_id: str) -> str:
    return f"{project_id}_{photo_id}_{hashlib.sha256(image_bytes).hexdigest()[:16]}"


def review_keys(scope: str) -> tuple[str, str, str]:
    return (f"cc_component_recognition_{scope}", f"cc_component_review_rows_{scope}",
            f"cc_component_recognition_error_{scope}")


@st.cache_resource(show_spinner=False)
def _app_server() -> Any:
    if not APP_DIR.is_dir():
        raise FileNotFoundError(f"PCB recognition app missing: {APP_DIR}")
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    cached = sys.modules.get(MODULE_NAME)
    if cached is not None:
        return cached
    source = APP_DIR / "app_server.py"
    spec = importlib.util.spec_from_file_location(MODULE_NAME, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load PCB recognition module: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _record(label: str, bbox: list | tuple, status: str = "검수 필요", confidence: float = 0.0) -> dict:
    return {"소자 종류": label if label in CLASS_OPTIONS else "unknown",
            "상태": status if status in STATUS_OPTIONS else "검수 필요",
            "신뢰도": float(confidence or 0.0),
            "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}


def _store_predictions(state: Any, records_key: str, result: dict) -> None:
    state[records_key] = [_record(row.get("class_name", "unknown"), row["bbox_xyxy"],
                                  confidence=row.get("confidence", 0.0))
                          for row in result.get("detections", [])]


def ensure_component_detection(st_api: Any, *, image_bytes: bytes, project_id: str,
                               photo_id: str) -> tuple[dict | None, str | None]:
    """Automatically detect a newly selected photo once per Streamlit session."""
    scope = review_scope(image_bytes, project_id, photo_id)
    result_key, records_key, error_key = review_keys(scope)
    if result_key in st_api.session_state:
        return st_api.session_state[result_key], None
    if error_key in st_api.session_state:
        return None, st_api.session_state[error_key]
    try:
        with st_api.spinner("선택한 PCB 사진에서 소자를 자동 탐지하고 있습니다…"):
            engine = _app_server()
            engine.load_component_locator()
            result = engine.run_prediction(image_bytes, {"conf": 0.20, "tile": 1024, "overlap": 0.25})
        st_api.session_state[result_key] = result
        _store_predictions(st_api.session_state, records_key, result)
        return result, None
    except Exception as exc:
        st_api.session_state[error_key] = str(exc)
        return None, str(exc)


def render_selected_component_editor(st_api: Any, *, image_bytes: bytes, scope: str,
                                     component_id: str) -> None:
    """Edit class, review status, and coordinates for the selected overlay box."""
    _result_key, records_key, _error_key = review_keys(scope)
    rows = st_api.session_state.get(records_key, [])
    try:
        index = int(component_id.rsplit("-", 1)[-1]) - 1
    except (TypeError, ValueError):
        return
    if not 0 <= index < len(rows):
        return
    row = rows[index]
    form_key = f"cc_component_selected_edit_{scope}_{index}"
    st_api.markdown("**부품 상태 수정**")
    with st_api.form(form_key):
        label = st_api.selectbox("소자 종류", CLASS_OPTIONS,
                                 index=CLASS_OPTIONS.index(row["소자 종류"]), key=f"{form_key}_class")
        status = st_api.selectbox("검수 상태", STATUS_OPTIONS,
                                  index=STATUS_OPTIONS.index(row.get("상태", "검수 필요")),
                                  key=f"{form_key}_status")
        with st_api.expander("박스 좌표 조정"):
            cols = st_api.columns(2)
            x1 = cols[0].number_input("왼쪽 x", min_value=0, value=int(row["x1"]), step=1, key=f"{form_key}_x1")
            y1 = cols[1].number_input("위쪽 y", min_value=0, value=int(row["y1"]), step=1, key=f"{form_key}_y1")
            x2 = cols[0].number_input("오른쪽 x", min_value=0, value=int(row["x2"]), step=1, key=f"{form_key}_x2")
            y2 = cols[1].number_input("아래쪽 y", min_value=0, value=int(row["y2"]), step=1, key=f"{form_key}_y2")
        apply = st_api.form_submit_button("수정 적용", use_container_width=True)
    if apply:
        from PIL import Image
        import io
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
        x1, x2 = max(0, min(width, int(x1))), max(0, min(width, int(x2)))
        y1, y2 = max(0, min(height, int(y1))), max(0, min(height, int(y2)))
        if x2 - x1 < 4 or y2 - y1 < 4:
            st_api.error("박스는 가로·세로 각각 4px 이상이어야 합니다.")
            return
        rows[index].update({"소자 종류": label, "상태": status,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        st_api.session_state[records_key] = rows
        st_api.rerun()


def render_component_recognition_panel(st_api: Any, *, image_bytes: bytes, project_id: str,
                                       photo_id: str, auto_result: dict | None = None,
                                       auto_error: str | None = None) -> None:
    """Compact controls and export; drawing/editing happens on the shared PCB viewer."""
    scope = review_scope(image_bytes, project_id, photo_id)
    result_key, records_key, error_key = review_keys(scope)
    if result_key not in st_api.session_state:
        if auto_result is not None:
            st_api.session_state[result_key] = auto_result
            _store_predictions(st_api.session_state, records_key, auto_result)
        elif auto_error:
            st_api.warning(f"자동 소자 탐지 실패 · {auto_error}")
            if st_api.button("소자 탐지 다시 시도", key=f"cc_component_retry_{scope}"):
                st_api.session_state.pop(error_key, None)
                st_api.rerun()
            return
        else:
            st_api.info("선택한 사진에서 자동 탐지를 실행하고 있습니다.")
            return
    st_api.caption("이미지 위의 ‘박스 그리기’를 눌러 드래그하세요. 박스를 선택하면 오른쪽에서 종류·상태·위치를 수정할 수 있습니다.")
    if st_api.button("선택 사진 다시 탐지", key=f"cc_component_run_{scope}"):
        st_api.session_state.pop(result_key, None)
        st_api.session_state.pop(records_key, None)
        st_api.session_state.pop(error_key, None)
        st_api.rerun()
    rows = st_api.session_state.get(records_key, [])
    st_api.caption(f"박스 {len(rows)}개 · 확인 {sum(row.get('상태') == '확인' for row in rows)}개 · 제외 {sum(row.get('상태') == '제외' for row in rows)}개")
    result = st_api.session_state[result_key]
    export_rows = [{"project_id": project_id, "photo_id": photo_id,
                    "active": row.get("상태") != "제외", "class_name": row["소자 종류"],
                    "review_status": row.get("상태", "검수 필요"),
                    **{key: row[key] for key in ("x1", "y1", "x2", "y2")}}
                   for row in rows]
    st_api.download_button("검수 결과 CSV 저장", pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{photo_id}_component_review.csv", mime="text/csv",
                           key=f"cc_component_review_download_{scope}")
