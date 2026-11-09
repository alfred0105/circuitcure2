"""Optional local PP-OCR text recognition and conservative KiCad reference extraction."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any


class OCRUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_engine():
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise OCRUnavailable("RapidOCR가 없습니다. `python -m pip install -e '.[vision-ocr]'`를 실행하세요.") from exc
    return RapidOCR()


def extract_reference_candidates(text: str) -> list[str]:
    """Extract literal reference-shaped tokens without OCR character correction."""
    return list(dict.fromkeys(match.group(0).upper() for match in re.finditer(
        r"(?<![A-Z0-9])(?:R|C|U|D|Q|J|P|L|F|T|Y|CN|TP)\s*[-_]?\s*\d{1,4}[A-Z]?(?![A-Z0-9])",
        str(text), flags=re.IGNORECASE,
    )))


def run_local_ocr(image: Any) -> dict[str, Any]:
    """Recognize text locally with RapidOCR and retain raw text plus candidate refs."""
    try:
        import numpy as np
        engine = _load_engine()
        result = engine(np.asarray(image.convert("RGB")))
    except OCRUnavailable:
        raise
    except Exception as exc:
        raise OCRUnavailable(f"로컬 OCR 추론이 실패했습니다: {exc}") from exc
    records = []
    boxes = getattr(result, "boxes", None)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if texts is None:
        texts = ()
    try:
        text_count = len(texts)
    except TypeError as exc:
        raise OCRUnavailable("OCR 엔진의 텍스트 결과 형식이 올바르지 않습니다.") from exc
    score_count = len(scores) if scores is not None else 0
    for index in range(text_count):
        raw_text = texts[index]
        text = str(raw_text).strip()
        if not text:
            continue
        polygon = np.asarray(boxes[index], dtype=float).reshape(-1, 2).tolist() if boxes is not None else []
        if polygon:
            x_values = [point[0] for point in polygon]
            y_values = [point[1] for point in polygon]
            bbox = [float(min(x_values)), float(min(y_values)), float(max(x_values)), float(max(y_values))]
        else:
            bbox = None
        score = float(scores[index]) if scores is not None and index < score_count else 0.0
        records.append({"text": text, "raw_text": text, "confidence": score,
                        "bbox_xyxy": bbox, "polygon_xy": polygon, "source": "local_ocr",
                        "verification_status": "unverified", "reference_candidates": extract_reference_candidates(text)})
    return {"status": "OCR_COMPLETE", "engine": "RapidOCR / PP-OCR ONNX", "runtime": "local-onnxruntime",
            "line_count": len(records), "records": records}
