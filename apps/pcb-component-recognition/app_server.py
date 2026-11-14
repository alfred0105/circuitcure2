"""Local HTML test UI for the upstream PCB YOLO26 detector."""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import time
import webbrowser
import zipfile
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from circuitcare.vision.box_reconciliation import deduplicate_component_boxes
from circuitcare.vision.weight_paths import resolve_pcb_weights_root
from circuitcare.component_taxonomy import normalize_component_class
from hr.infer.predict import agnostic_nms, combine, detect as _detect, draw, draw_elements

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "data" / "samples"


def _default_weights_root() -> Path:
    return resolve_pcb_weights_root(ROOT)


WEIGHTS = _default_weights_root()
SAMPLE_FILES = sorted(p.name for p in SAMPLES.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
DEVICE = "0" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
ANY_MODEL: YOLO | None = None
FINE_MODEL: YOLO | None = None
FINE_NAMES: list[str] = []
EXTERNAL_MODELS: dict[str, YOLO] = {}
EVAL_MODEL_SPECS = {
    "pcb_ai_doctor_v8n": ("PCB AI Doctor YOLOv8n", "pcb-ai-doctor/best_v8n.pt"),
    "pcb_ai_doctor_v12n": ("PCB AI Doctor YOLOv12n", "pcb-ai-doctor/best_v12n.pt"),
    "pcbinspect_yolov12m": ("PCBInspect YOLOv12m", "PCBInspect-AI/YoloV12-Medium-160-FineTuned.pt"),
}
EVAL_ENSEMBLES = {
    "ensemble_parsa_doctor": ("Parsa locator + PCB AI Doctor v12n", ["parsa", "pcb_ai_doctor_v12n"]),
    "ensemble_parsa_pcbinspect": ("Parsa locator + PCBInspect v12m", ["parsa", "pcbinspect_yolov12m"]),
    "ensemble_doctor_pcbinspect": ("PCB AI Doctor v12n + PCBInspect v12m", ["pcb_ai_doctor_v12n", "pcbinspect_yolov12m"]),
    "ensemble_all_three": ("Parsa + PCB AI Doctor v12n + PCBInspect v12m", ["parsa", "pcb_ai_doctor_v12n", "pcbinspect_yolov12m"]),
}
EVAL_WEIGHT_ROOT = WEIGHTS / "external"
COMPONENT_CLASS_MAP = {name: normalize_component_class(name) for name in (
    "battery", "button", "buzzer", "capacitor", "clock", "connector", "diode",
    "display", "fuse", "heatsink", "ic", "inductor", "led", "potentiometer",
    "relay", "resistor", "switch", "transformer", "transistor",
)}


def detect(model, image, conf, imgsz, tile, overlap, iou, device, max_det=1000):
    """Run inference with a safe CPU fallback when Apple's MPS backend fails."""
    global DEVICE
    height, width = image.shape[:2]
    if height * width > 60_000_000:
        raise ValueError("사진이 너무 큽니다 (최대 60MP). 긴 변을 줄여 다시 시도해 주세요.")
    if tile:
        step = max(1, int(tile * (1 - overlap)))
        tile_count = max(1, len(range(0, max(1, width - tile) + 1, step))) * max(1, len(range(0, max(1, height - tile) + 1, step)))
        if tile_count > 256:
            raise ValueError(f"타일 수가 제한을 넘었습니다 ({tile_count}/256). 더 작은 사진 또는 타일 크기를 사용해 주세요.")
    started = time.monotonic()
    try:
        result = _detect(model, image, conf, imgsz, tile, overlap, iou, device, max_det=max_det)
    except (RuntimeError, NotImplementedError) as exc:
        if str(device).lower() not in {"mps", "mps:0"}:
            raise
        DEVICE = "cpu"
        try:
            if hasattr(model, "to"):
                model.to("cpu")
            result = _detect(model, image, conf, imgsz, tile, overlap, iou, "cpu", max_det=max_det)
        except Exception as fallback_exc:
            raise RuntimeError(f"MPS 추론 실패 후 CPU 재시도도 실패했습니다: {fallback_exc}") from exc
    if time.monotonic() - started > 180:
        raise TimeoutError("PCB 소자 탐지 시간이 180초 제한을 넘었습니다.")
    return result


def load_component_locator() -> YOLO:
    """Load the locator used by the component-recognition pipeline on demand."""
    global ANY_MODEL
    if ANY_MODEL is None:
        weight_path = WEIGHTS / "any_best.pt"
        if not weight_path.is_file():
            raise FileNotFoundError(f"YOLO26s locator weight not found: {weight_path}")
        ANY_MODEL = YOLO(str(weight_path))
    return ANY_MODEL


def classify_component_crop(image_bytes: bytes, bbox_xyxy: list[float], confidence: float = 0.10) -> dict:
    """Run the component-type detector inside one user-selected box."""
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("선택한 PCB 사진을 읽을 수 없습니다.")
    height, width = image.shape[:2]
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox_xyxy]
    x0, x1 = max(0, min(width, x0)), max(0, min(width, x1))
    y0, y1 = max(0, min(height, y0)), max(0, min(height, y1))
    if x1 - x0 < 6 or y1 - y0 < 6:
        raise ValueError("박스 크기는 가로·세로 각각 6 px 이상이어야 합니다.")
    crop = image[y0:y1, x0:x1]
    weight_path = EVAL_WEIGHT_ROOT / "pcb-yolov8s-detection" / "best.pt"
    if not weight_path.is_file():
        raise FileNotFoundError(f"소자 종류 모델을 찾을 수 없습니다: {weight_path}")
    model_key = "arshia_pcb_yolov8s"
    if model_key not in EXTERNAL_MODELS:
        EXTERNAL_MODELS[model_key] = YOLO(str(weight_path))
    model = EXTERNAL_MODELS[model_key]
    boxes, scores, classes = detect(
        model, crop, min(0.95, max(0.01, float(confidence))), 640, 0, 0.0, 0.5, DEVICE,
    )
    candidates = []
    for index, class_id in enumerate(classes.tolist()):
        raw = model.names.get(int(class_id), str(class_id)) if isinstance(model.names, dict) else model.names[int(class_id)]
        label = COMPONENT_CLASS_MAP.get(str(raw).strip().lower())
        if label:
            candidates.append((float(scores[index]), label, [float(value) for value in boxes[index]]))
    if not candidates:
        return {"class_name": None, "confidence": None, "bbox_xyxy": None}
    score, label, crop_box = max(candidates, key=lambda item: item[0])
    return {"class_name": label, "confidence": score,
            "bbox_xyxy": [crop_box[0] + x0, crop_box[1] + y0,
                          crop_box[2] + x0, crop_box[3] + y0]}


def jpeg_data_url(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("이미지 결과를 JPEG로 인코딩하지 못했습니다.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def run_prediction(image_bytes: bytes, options: dict) -> dict:
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("업로드 한도는 40 MB입니다.")
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("이미지를 읽지 못했습니다. JPG 또는 PNG 파일을 선택해 주세요.")

    external_id = options.get("eval_model", "parsa")
    if external_id in EVAL_MODEL_SPECS or external_id in EVAL_ENSEMBLES:
        return run_external_prediction(image_bytes, options, external_id)
    # The user-facing detection route always runs the single selected production pipeline.
    mode = "pcb_component_types"
    if mode not in {"any", "fine", "both", "ensemble2", "ensemble3", "ensemble_body", "pcb_component_types"}:
        raise ValueError("mode 값이 올바르지 않습니다.")
    conf = min(0.95, max(0.01, float(options.get("conf", 0.20))))
    tile = int(options.get("tile", 1024))
    if tile not in {0, 640, 1024, 1280}:
        raise ValueError("타일 크기는 0, 640, 1024, 1280 중에서 선택해 주세요.")
    overlap = min(0.50, max(0.0, float(options.get("overlap", 0.25))))
    started = time.perf_counter()
    types = None

    if mode == "any":
        B, S, C = agnostic_nms(*detect(ANY_MODEL, image, conf, 1024, tile, overlap, 0.5, DEVICE), 0.5)
        labels = ["component"] * len(B)
        cls_conf = [float(score) for score in S]
        elements = draw_elements(image, B)
        types = None
    elif mode == "fine":
        B, S, C = detect(FINE_MODEL, image, conf, 1024, tile, overlap, 0.5, DEVICE)
        labels = [FINE_NAMES[int(cls)] for cls in C]
        cls_conf = [float(score) for score in S]
        elements = draw_elements(image, B)
        types, _ = draw(image, B, labels, cls_conf, "auto", 24)
    elif mode == "both":
        da = agnostic_nms(*detect(ANY_MODEL, image, conf, 1024, tile, overlap, 0.5, DEVICE), 0.5)
        df = detect(FINE_MODEL, image, conf, 1024, tile, overlap, 0.5, DEVICE)
        B, S, _ = da
        labels, cls_conf = combine(da, df, FINE_NAMES)
        elements = draw_elements(image, B)
        types, _ = draw(image, B, labels, cls_conf, "auto", 24)
    elif mode == "pcb_component_types":
        result = run_pcb_component_types_prediction(image_bytes, options)
        return attach_pin_pad_result(image, result) if options.get("include_pin_pad") else result
    elif mode == "ensemble_body":
        return run_body_only_prediction(image_bytes, options)
    else:
        locator_model = "ensemble_parsa_doctor" if mode == "ensemble2" else "ensemble_all_three"
        locator = run_external_prediction(image_bytes,
            {"eval_model": locator_model, "conf": conf, "tile": tile, "overlap": overlap},
            locator_model)
        B = torch.tensor([d["bbox_xyxy"] for d in locator["detections"]], dtype=torch.float32).reshape(-1,4)
        S = torch.tensor([d["confidence"] for d in locator["detections"]], dtype=torch.float32)
        C = torch.zeros(len(B), dtype=torch.long)
        labels = ["component"] * len(B)
        cls_conf = [float(score) for score in S]
        elements = draw_elements(image, B)
        if mode == "ensemble3":
            da = (B, S, C)
            df = detect(FINE_MODEL, image, conf, 1024, tile, overlap, 0.5, DEVICE)
            labels, cls_conf = combine(da, df, FINE_NAMES)
            types, _ = draw(image, B, labels, cls_conf, "auto", 24)

    if mode in {"fine", "both", "ensemble3"}:
        keep = [i for i, label in enumerate(labels) if label not in {"Other", "?", "Marker"}]
        B = B[keep]
        S = S[keep]
        labels = [labels[i] for i in keep]
        cls_conf = [cls_conf[i] for i in keep]
        elements = draw_elements(image, B)
        types, _ = draw(image, B, labels, cls_conf, "auto", 24)

    counts = dict(Counter(labels).most_common())
    table = io.StringIO()
    writer = csv.writer(table)
    writer.writerow(["id", "label", "box_confidence", "label_confidence", "x0", "y0", "x1", "y1"])
    for i, box in enumerate(B):
        label_score = float(cls_conf[i]) if mode in {"fine", "both", "ensemble3"} else float(S[i])
        writer.writerow([i + 1, labels[i], f"{float(S[i]):.4f}", f"{label_score:.4f}", *[int(v) for v in box]])

    detections = []
    for i, box in enumerate(B):
        detections.append({"class_name": labels[i], "confidence": float(S[i]),
                           "bbox_xyxy": [float(v) for v in box]})
    return {
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "mode": mode,
        "model_name": ("YOLO26s Locator + PCB AI Doctor v12n (position detection only)" if mode == "ensemble2" else
                       "YOLO26s Locator + AI Doctor (component body only; pins excluded)" if mode == "ensemble_body" else
                       "Three-model position ensemble + Parsa fine classifier" if mode == "ensemble3" else
                       f"Parsa YOLO26 · {mode}"),
        "tile": tile,
        "overlap": overlap,
        "confidence": conf,
        "device": DEVICE,
        "total": len(B),
        "counts": counts,
        "detections": detections,
        "seconds": round(time.perf_counter() - started, 2),
        "elements_image": jpeg_data_url(elements),
        "types_image": jpeg_data_url(types) if types is not None else None,
        "csv": table.getvalue(),
    }


def run_body_only_prediction(image_bytes: bytes, options: dict) -> dict:
    """Use the selected final locator ensemble and keep only component-body detections."""
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("이미지를 읽지 못했습니다. JPG 또는 PNG 파일을 선택해 주세요.")
    conf = min(0.95, max(0.01, float(options.get("conf", 0.20))))
    tile = int(options.get("tile", 1024))
    if tile not in {0, 640, 1024, 1280}:
        raise ValueError("타일 크기는 0, 640, 1024, 1280 중에서 선택해 주세요.")
    overlap = min(0.50, max(0.0, float(options.get("overlap", 0.25))))
    started = time.perf_counter()
    routed = {"component_body": []}
    members = ("parsa", "pcb_ai_doctor_v12n")
    for member in members:
        if member == "parsa":
            model = load_component_locator()
        else:
            _, relative_weight = EVAL_MODEL_SPECS[member]
            weight_path = EVAL_WEIGHT_ROOT / relative_weight
            if not weight_path.is_file():
                raise ValueError(f"평가 가중치 파일을 찾지 못했습니다: {weight_path.name}")
            if member not in EXTERNAL_MODELS:
                EXTERNAL_MODELS[member] = YOLO(str(weight_path))
            model = EXTERNAL_MODELS[member]
        boxes, scores, classes = detect(model, image, conf, 640, tile, overlap, 0.5, DEVICE)
        for i, _class_id in enumerate(classes.tolist()):
            key = "component_body"
            routed[key].append({"box": [float(v) for v in boxes[i]], "score": float(scores[i]),
                                "source": member})

    # Apply one conservative geometry pass to the complete two-model candidate
    # set, including locator-vs-locator and locator-vs-Doctor pairs.
    body, suppressed_duplicates = deduplicate_component_boxes(routed["component_body"])
    suppressed_details = [{"kept_source": row["kept"].get("source"),
        "suppressed_source": row["suppressed"].get("source"),
        "kept_box": row["kept"].get("box"), "suppressed_box": row["suppressed"].get("box"),
        "kept_confidence": row["kept"].get("score"), "suppressed_confidence": row["suppressed"].get("score"),
        "geometry": row["metrics"]} for row in suppressed_duplicates]
    items = [("component_body", item) for item in body]
    items.sort(key=lambda pair: -pair[1]["score"])
    boxes = torch.tensor([item["box"] for _, item in items], dtype=torch.float32).reshape(-1, 4)
    scores = torch.tensor([item["score"] for _, item in items], dtype=torch.float32)
    labels = [label for label, _ in items]
    overlay = draw(image, boxes, labels, [float(x) for x in scores], "auto", 24)[0]
    counts = dict(Counter(labels).most_common())
    table = io.StringIO()
    writer = csv.writer(table)
    writer.writerow(["id", "label", "confidence", "x0", "y0", "x1", "y1"])
    detections = []
    for i, (label, item) in enumerate(items):
        box = [int(v) for v in item["box"]]
        writer.writerow([i + 1, label, f"{item['score']:.4f}", *box])
        detections.append({"class_name": label, "confidence": item["score"],
                           "bbox_xyxy": item["box"]})
    return {"image_width": int(image.shape[1]), "image_height": int(image.shape[0]),
            "mode": "ensemble_body",
            "model_name": "YOLO26s Locator + AI Doctor (component body only; pins excluded)",
            "tile": tile, "overlap": overlap, "confidence": conf, "device": DEVICE,
            "total": len(items), "counts": counts, "detections": detections,
            "duplicate_suppressed_count": len(suppressed_details),
            "duplicate_suppressed": suppressed_details,
            "seconds": round(time.perf_counter() - started, 2),
            "elements_image": jpeg_data_url(overlay), "types_image": jpeg_data_url(overlay),
            "csv": table.getvalue()}



def run_pcb_component_types_prediction(image_bytes: bytes, options: dict) -> dict:
    """Classify component-body proposals and exclude pin detections."""
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("이미지를 읽지 못했습니다. JPG 또는 PNG 파일을 선택해 주세요.")
    conf = min(0.95, max(0.01, float(options.get("conf", 0.20))))
    tile = int(options.get("tile", 1024))
    overlap = min(0.50, max(0.0, float(options.get("overlap", 0.25))))
    started = time.perf_counter()

    # Windows-compatible locator proposal and type-classification sequence.
    base = run_body_only_prediction(image_bytes, options)
    bodies = [d for d in base["detections"] if d["class_name"] == "component_body"]
    weight_path = EVAL_WEIGHT_ROOT / "pcb-yolov8s-detection" / "best.pt"
    if not weight_path.is_file():
        raise ValueError("PCB 분류 가중치 best.pt를 찾지 못했습니다.")
    model_key = "arshia_pcb_yolov8s"
    if model_key not in EXTERNAL_MODELS:
        EXTERNAL_MODELS[model_key] = YOLO(str(weight_path))
    model = EXTERNAL_MODELS[model_key]
    boxes, scores, classes = detect(model, image, conf, 640, tile, overlap, 0.5, DEVICE)

    # pads/pins are deliberately omitted from the component classifier output.
    assignments: dict[int, list[dict]] = {i: [] for i in range(len(bodies))}
    for i, class_id in enumerate(classes.tolist()):
        raw = model.names.get(int(class_id), str(class_id)) if isinstance(model.names, dict) else model.names[int(class_id)]
        label = COMPONENT_CLASS_MAP.get(str(raw).strip().lower())
        if label is None or not bodies:
            continue
        pred = [float(v) for v in boxes[i]]
        best_index, best_match = -1, 0.0
        for j, body in enumerate(bodies):
            ref = body["bbox_xyxy"]
            x1, y1 = max(pred[0], ref[0]), max(pred[1], ref[1])
            x2, y2 = min(pred[2], ref[2]), min(pred[3], ref[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_pred = max(0.0, pred[2] - pred[0]) * max(0.0, pred[3] - pred[1])
            area_ref = max(0.0, ref[2] - ref[0]) * max(0.0, ref[3] - ref[1])
            iou = inter / (area_pred + area_ref - inter + 1e-9)
            smaller_coverage = inter / (min(area_pred, area_ref) + 1e-9)
            match_score = iou + 0.1 * smaller_coverage
            if (iou >= 0.15 or smaller_coverage >= 0.60) and match_score > best_match:
                best_index, best_match = j, match_score
        if best_index >= 0:
            assignments[best_index].append({"label": label, "score": float(scores[i]),
                                            "match_score": best_match})

    typed = []
    for i, body in enumerate(bodies):
        options_for_body = assignments[i]
        best = max(options_for_body, key=lambda p: p["score"]) if options_for_body else None
        typed.append({"class_name": best["label"] if best else "unknown",
                      "confidence": best["score"] if best else 0.0,
                      "location_confidence": float(body["confidence"]),
                      "type_confidence": best["score"] if best else None,
                      "match_score": best["match_score"] if best else None,
                      "bbox_xyxy": body["bbox_xyxy"],
                      "body_confidence": body["confidence"]})
    detections = typed
    all_boxes = torch.tensor([d["bbox_xyxy"] for d in detections], dtype=torch.float32).reshape(-1, 4)
    labels = [d["class_name"] for d in detections]
    confs = [float(d["confidence"]) for d in detections]
    counts = dict(Counter(labels).most_common())
    elements = draw_elements(image, all_boxes)
    overlay, _ = draw(image, all_boxes, labels, confs, "auto", 24)
    table = io.StringIO()
    writer = csv.writer(table)
    writer.writerow(["id", "label", "confidence", "x0", "y0", "x1", "y1"])
    for i, item in enumerate(detections):
        writer.writerow([i + 1, item["class_name"], f"{item['confidence']:.4f}",
                         *[int(v) for v in item["bbox_xyxy"]]])
    return {"image_width": int(image.shape[1]), "image_height": int(image.shape[0]),
            "mode": "pcb_component_types",
            "model_name": "YOLO26s Locator + AI Doctor + Arshia YOLOv8s types (pins excluded)",
            "tile": tile, "overlap": overlap, "confidence": conf, "device": DEVICE,
            "total": len(detections), "counts": counts, "detections": detections,
            "seconds": round(time.perf_counter() - started, 2),
            "elements_image": jpeg_data_url(elements), "types_image": jpeg_data_url(overlay),
            "csv": table.getvalue()}



def attach_pin_pad_result(image: np.ndarray, result: dict) -> dict:
    """Attach only component-conditioned pin/pad candidates to the detector output."""
    started = time.perf_counter()
    try:
        from pin_pad_candidates import run_pin_pad_pipeline
        pin_result = run_pin_pad_pipeline(image, result.get("detections", []))
        result["trace_pads_image"] = jpeg_data_url(pin_result["pad_overlay"])
        result["pin_pad_pads"] = pin_result["pads"]
        result["pin_pad_priors"] = pin_result["pin_priors"]
        result["pin_pad_matches"] = pin_result["matches"]
        result["pin_pad_diagnostics"] = pin_result["diagnostics"]
    except Exception as exc:
        result["pin_pad_error"] = f"핀·패드 후보 계산 실패: {exc}"
        result["trace_pads_image"] = jpeg_data_url(image)
        result["pin_pad_pads"] = []
        result["pin_pad_priors"] = []
        result["pin_pad_matches"] = []
        result["pin_pad_diagnostics"] = {"pad_candidates": 0, "pin_priors": 0,
                                          "pin_pad_matches": 0, "pin_prior_unavailable_components": []}
    result["seconds"] = round(float(result.get("seconds", 0.0)) + time.perf_counter() - started, 2)
    return result

def run_external_prediction(image_bytes: bytes, options: dict, model_id: str) -> dict:
    if model_id in EVAL_ENSEMBLES:
        display_name, members = EVAL_ENSEMBLES[model_id]
    else:
        display_name, _ = EVAL_MODEL_SPECS[model_id]
        members = [model_id]
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("이미지를 읽지 못했습니다. JPG 또는 PNG 파일을 선택해 주세요.")
    conf = min(0.95, max(0.01, float(options.get("conf", 0.20))))
    tile = int(options.get("tile", 1024))
    if tile not in {0, 640, 1024, 1280}:
        raise ValueError("타일 크기는 0, 640, 1024, 1280 중에서 선택해 주세요.")
    overlap = min(0.50, max(0.0, float(options.get("overlap", 0.25))))
    imgsz = 640
    started = time.perf_counter()
    candidates = []
    for member in members:
        if member == "parsa":
            model = ANY_MODEL
            if model is None: raise ValueError("Parsa YOLO26 locator가 로드되지 않았습니다.")
        else:
            _, relative_weight = EVAL_MODEL_SPECS[member]
            weight_path = EVAL_WEIGHT_ROOT / relative_weight
            if not weight_path.is_file(): raise ValueError(f"평가 가중치 파일을 찾지 못했습니다: {weight_path.name}")
            if member not in EXTERNAL_MODELS: EXTERNAL_MODELS[member] = YOLO(str(weight_path))
            model = EXTERNAL_MODELS[member]
        boxes, scores, classes = detect(model, image, conf, imgsz, tile, overlap, 0.5, DEVICE)
        for i, cid in enumerate(classes.tolist()):
            raw = model.names.get(int(cid), str(cid)) if isinstance(model.names, dict) else model.names[int(cid)]
            if member == "pcbinspect_yolov12m" and raw != "ComponentBody": continue
            candidates.append({"box": [float(v) for v in boxes[i]], "score": float(scores[i])})
    candidates.sort(key=lambda p: -p["score"])
    fused = []
    for candidate in candidates:
        duplicate = False
        for existing in fused:
            a, b = candidate["box"], existing["box"]
            x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0.0, x2-x1)*max(0.0, y2-y1)
            aa = max(0.0,a[2]-a[0])*max(0.0,a[3]-a[1]); bb = max(0.0,b[2]-b[0])*max(0.0,b[3]-b[1])
            if inter/(aa+bb-inter+1e-9) >= 0.5: duplicate=True; break
        if not duplicate: fused.append(candidate)
    B = torch.tensor([p["box"] for p in fused],dtype=torch.float32).reshape(-1,4)
    S = torch.tensor([p["score"] for p in fused],dtype=torch.float32)
    labels = ["component"]*len(fused)
    elements = draw_elements(image, B)
    types, _ = draw(image, B, labels, S, "auto", 24)
    counts = dict(Counter(labels).most_common())
    table = io.StringIO(); writer = csv.writer(table)
    writer.writerow(["id", "label", "confidence", "x0", "y0", "x1", "y1"])
    for i, box in enumerate(B):
        writer.writerow([i+1, labels[i], f"{float(S[i]):.4f}", *[int(v) for v in box]])
    detections = [{"class_name": labels[i], "confidence": float(S[i]),
                   "bbox_xyxy": [float(v) for v in B[i]]} for i in range(len(B))]
    return {"image_width": int(image.shape[1]), "image_height": int(image.shape[0]),
            "mode": "any", "model_name": display_name, "tile": tile, "overlap": overlap,
            "confidence": conf, "device": DEVICE, "total": len(B), "counts": counts,
            "seconds": round(time.perf_counter()-started, 2),
            "elements_image": jpeg_data_url(elements), "types_image": jpeg_data_url(types),
            "detections": detections, "csv": table.getvalue()}


def _class_name_list(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split())


def _parse_class_map(text: str) -> dict[str, str]:
    result = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        source, target = line.split("=", 1)
        if source.strip() and target.strip():
            result[_normalize_label(source)] = target.strip()
    return result


def run_evaluation(image_bytes: bytes, options: dict, label_text: str,
                   class_names_text: str, class_map_text: str) -> dict:
    if not label_text.strip():
        raise ValueError("YOLO 정답 라벨 파일(.txt)을 선택해 주세요.")
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("이미지를 읽지 못했습니다.")
    height, width = image.shape[:2]
    class_names = _class_name_list(class_names_text)
    class_map = _parse_class_map(class_map_text)
    ground_truth = []
    for line_no, line in enumerate(label_text.splitlines(), 1):
        row = line.strip().split()
        if not row:
            continue
        if len(row) != 5:
            raise ValueError(f"라벨 {line_no}행 형식 오류: class x_center y_center width height 형식이어야 합니다.")
        try:
            class_id = int(row[0]); cx, cy, bw, bh = map(float, row[1:])
        except ValueError as exc:
            raise ValueError(f"라벨 {line_no}행의 숫자를 읽지 못했습니다.") from exc
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f"라벨 {line_no}행의 좌표는 0~1 정규화 범위여야 합니다.")
        raw = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
        mapped = class_map.get(_normalize_label(raw), raw)
        ground_truth.append({"class_name": mapped, "raw_class": raw,
                             "bbox_xyxy": [(cx-bw/2)*width, (cy-bh/2)*height,
                                           (cx+bw/2)*width, (cy+bh/2)*height]})
    if not ground_truth:
        raise ValueError("라벨 파일에 평가할 박스가 없습니다.")

    eval_options = dict(options)
    eval_options["_skip_pin_pad"] = True
    if eval_options.get("eval_model", "parsa") == "parsa":
        # This screen evaluates the component locator; type prediction is outside the metric.
        eval_options["mode"] = "any"
    result = run_prediction(image_bytes, eval_options)
    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2-x1) * max(0.0, y2-y1)
        aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
        bb = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
        return inter / (aa + bb - inter + 1e-9)
    def norm_class(name):
        if mode == "any":
            return "component"
        choices = {_normalize_label(x): x for x in FINE_NAMES}
        return choices.get(_normalize_label(name), name)
    used = set(); tp = fp = fn = ignored = 0
    per_class = {}
    for gt in ground_truth:
        per_class.setdefault(gt["class_name"], {"TP": 0, "FP": 0, "FN": 0})
    for pred in sorted(result["detections"], key=lambda item: -item["confidence"]):
        pb = pred["bbox_xyxy"]
        # Per request: outside all labeled boxes is unscored and never a false positive.
        if not any(g["bbox_xyxy"][0] <= (pb[0]+pb[2])/2 <= g["bbox_xyxy"][2]
                   and g["bbox_xyxy"][1] <= (pb[1]+pb[3])/2 <= g["bbox_xyxy"][3]
                   for g in ground_truth):
            ignored += 1
            continue
        candidates = [(iou(pb, gt["bbox_xyxy"]), index, gt)
                      for index, gt in enumerate(ground_truth) if index not in used]
        best = max(candidates, default=(0.0, -1, None), key=lambda item: item[0])
        matched = best[2] is not None and best[0] >= 0.5
        if matched:
            used.add(best[1]); tp += 1
            per_class[best[2]["class_name"]]["TP"] += 1
        else:
            fp += 1
            pass
    for index, gt in enumerate(ground_truth):
        if index not in used:
            fn += 1; per_class[gt["class_name"]]["FN"] += 1
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    rows = []
    for cls, counts in sorted(per_class.items()):
        r = counts["TP"]/(counts["TP"]+counts["FN"]) if counts["TP"]+counts["FN"] else 0.0
        rows.append({"class": cls, **counts, "precision": None, "recall": r, "F1": None})
    result["evaluation"] = {"GT": len(ground_truth), "TP": tp, "FP_labeled_region": fp,
                            "FN": fn, "ignored_unlabeled_region": ignored,
                            "precision": precision, "recall": recall, "F1": f1,
                            "iou_threshold": 0.5, "class_aware": False,
                            "per_class": rows}
    return result


def run_dataset_evaluation(zip_bytes: bytes, options: dict, class_names_text: str,
                           class_map_text: str) -> dict:
    if len(zip_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("데이터셋 ZIP은 40 MB 이하로 업로드해 주세요.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError("유효한 ZIP 파일이 아닙니다.") from exc
    with archive:
        members = []
        uncompressed = 0
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            parts = Path(name).parts
            if name.startswith("/") or ".." in parts:
                continue
            if info.file_size > 30 * 1024 * 1024:
                continue
            uncompressed += info.file_size
            if uncompressed > 500 * 1024 * 1024:
                raise ValueError("압축 해제 기준 데이터셋 크기는 500 MB까지 허용합니다.")
            members.append((name, info))
        if len(members) > 8000:
            raise ValueError("ZIP 안의 파일 수가 너무 많습니다. 8,000개 이하로 나눠 주세요.")
        by_name = {name: info for name, info in members}
        labels_by_stem = {}
        for name, info in members:
            if Path(name).suffix.lower() == ".txt":
                labels_by_stem.setdefault(Path(name).stem.lower(), []).append((name, info))
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        images = [(name, info) for name, info in members if Path(name).suffix.lower() in image_exts]
        pairs = []
        for name, info in images:
            p = Path(name); stem = p.stem
            sibling = str(p.parent.parent / "labels" / (stem + ".txt")) if p.parent.name.lower() == "images" else str(p.parent / (stem + ".txt"))
            sibling = sibling.replace("\\", "/")
            label_info = by_name.get(sibling)
            if label_info:
                label_name, label_member = sibling, label_info
            else:
                options_for_stem = labels_by_stem.get(stem.lower(), [])
                if len(options_for_stem) != 1:
                    continue
                label_name, label_member = options_for_stem[0]
            pairs.append((name, info, label_name, label_member))
        if not pairs:
            raise ValueError("이미지와 대응하는 YOLO 라벨(.txt)을 ZIP에서 찾지 못했습니다.")
        if len(pairs) > 1000:
            raise ValueError("한 번의 평가에는 라벨이 있는 이미지 1,000장까지 허용합니다.")
        tp = fp = fn = gt = ignored = 0; total_seconds = 0.0
        class_totals = {}; first_result = None; rows = []
        for index, (image_name, image_info, label_name, label_info) in enumerate(pairs, 1):
            image_bytes = archive.read(image_info)
            label_text = archive.read(label_info).decode("utf-8-sig")
            result = run_evaluation(image_bytes, options, label_text, class_names_text, class_map_text)
            e = result["evaluation"]
            tp += e["TP"]; fp += e["FP_labeled_region"]; fn += e["FN"]
            gt += e["GT"]; ignored += e["ignored_unlabeled_region"]; total_seconds += result["seconds"]
            if first_result is None: first_result = result
            rows.append({"image": image_name, "GT": e["GT"], "TP": e["TP"],
                         "FP": e["FP_labeled_region"], "FN": e["FN"],
                         "ignored": e["ignored_unlabeled_region"], "precision": e["precision"],
                         "recall": e["recall"], "F1": e["F1"]})
            for clsrow in e["per_class"]:
                c = class_totals.setdefault(clsrow["class"], {"TP": 0, "FP": 0, "FN": 0})
                c["TP"] += clsrow["TP"]; c["FP"] += clsrow["FP"]; c["FN"] += clsrow["FN"]
            if index % 10 == 0:
                print(f"Dataset evaluation progress {index}/{len(pairs)}", flush=True)
        precision = tp/(tp+fp) if tp+fp else 0.0; recall = tp/(tp+fn) if tp+fn else 0.0
        overall_f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        per_class = []
        for cls, c in sorted(class_totals.items()):
            p = c["TP"]/(c["TP"]+c["FP"]) if c["TP"]+c["FP"] else 0.0
            r = c["TP"]/(c["TP"]+c["FN"]) if c["TP"]+c["FN"] else 0.0
            per_class.append({"class": cls, **c, "precision": p, "recall": r,
                              "F1": 2*p*r/(p+r) if p+r else 0.0})
        first_result["evaluation"] = {"images": len(pairs), "GT": gt, "TP": tp,
            "FP_labeled_region": fp, "FN": fn, "ignored_unlabeled_region": ignored,
            "precision": precision, "recall": recall, "F1": overall_f1,
            "class_aware": options.get("mode", "both") != "any", "per_class": per_class,
            "per_image": rows}
        first_result["seconds"] = round(total_seconds, 2)
        first_result["dataset_name"] = "Uploaded YOLO dataset ZIP"
        return first_result


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            page = ROOT / "index.html"
            if page.is_file():
                return self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            return self._send(503, b"PCB browser UI assets are not included. Run the CircuitCare integrated UI instead.", "text/plain; charset=utf-8")
        if path in {"/benchmark", "/benchmark.html"}:
            page_path = ROOT / "benchmark.html"
            if not page_path.is_file():
                return self._send(404, b"Benchmark UI is not included in this package.", "text/plain; charset=utf-8")
            page = page_path.read_bytes()
            return self._send(200, page, "text/html; charset=utf-8")
        if path in {"/pin-pad", "/pin-pad.html"}:
            page_path = ROOT / "pin_pad.html"
            if not page_path.is_file():
                return self._send(404, b"Pin/pad UI is not included in this package.", "text/plain; charset=utf-8")
            page = page_path.read_bytes()
            return self._send(200, page, "text/html; charset=utf-8")
        if path == "/api/health":
            payload = {"ready": True, "models_loaded": {"locator": ANY_MODEL is not None,
                       "fine": FINE_MODEL is not None, "external": sorted(EXTERNAL_MODELS)}, "device": DEVICE,
                       "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else DEVICE.upper(),
                       "fine_classes": FINE_NAMES, "sample_count": len(SAMPLE_FILES)}
            return self._send(200, json.dumps(payload).encode(), "application/json; charset=utf-8")
        if path == "/api/samples":
            return self._send(200, json.dumps(SAMPLE_FILES).encode(), "application/json; charset=utf-8")
        if path == "/api/sample":
            name = unquote(parse_qs(urlparse(self.path).query).get("name", [""])[0])
            if name not in SAMPLE_FILES:
                return self._send(404, b"Sample not found", "text/plain; charset=utf-8")
            content_type = {
                ".png": "image/png", ".bmp": "image/bmp",
            }.get(Path(name).suffix.lower(), "image/jpeg")
            return self._send(200, (SAMPLES / name).read_bytes(), content_type)
        return self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        endpoint = urlparse(self.path).path
        if endpoint not in {"/api/predict", "/api/evaluate", "/api/pin-pad", "/api/classify-box"}:
            return self._send(404, b"Not found", "text/plain; charset=utf-8")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_UPLOAD_BYTES * 2:
                raise ValueError("요청 크기가 비어 있거나 40 MB 제한을 넘었습니다.")
            body = json.loads(self.rfile.read(length))
            if endpoint == "/api/classify-box":
                if body.get("sample"):
                    name = unquote(str(body["sample"]))
                    if name not in SAMPLE_FILES:
                        raise ValueError("허용된 샘플 목록에서 선택해 주세요.")
                    image_bytes = (SAMPLES / name).read_bytes()
                elif body.get("image_base64"):
                    image_bytes = base64.b64decode(body["image_base64"], validate=True)
                else:
                    raise ValueError("샘플을 선택하거나 이미지를 업로드해 주세요.")
                result = classify_component_crop(
                    image_bytes, body.get("bbox_xyxy", []), body.get("confidence", 0.05),
                )
                return self._send(200, json.dumps(result).encode(), "application/json; charset=utf-8")
            if endpoint == "/api/pin-pad":
                if body.get("sample"):
                    name = unquote(str(body["sample"]))
                    if name not in SAMPLE_FILES:
                        raise ValueError("허용된 샘플 목록에서 선택해 주세요.")
                    image_bytes = (SAMPLES / name).read_bytes()
                elif body.get("image_base64"):
                    image_bytes = base64.b64decode(body["image_base64"], validate=True)
                else:
                    raise ValueError("샘플을 선택하거나 이미지를 업로드해 주세요.")
                from pin_pad_experiment import run_pin_pad_experiment
                result = run_pin_pad_experiment(image_bytes, body.get("conf", 0.20))
                return self._send(200, json.dumps(result).encode(), "application/json; charset=utf-8")
            is_eval = endpoint == "/api/evaluate"
            if is_eval and body.get("dataset_zip_base64"):
                zip_bytes = base64.b64decode(body["dataset_zip_base64"], validate=True)
                result = run_dataset_evaluation(zip_bytes, body, str(body.get("class_names", "")),
                                                str(body.get("class_map", "")))
                return self._send(200, json.dumps(result).encode(), "application/json; charset=utf-8")
            if body.get("sample"):
                name = unquote(str(body["sample"]))
                if name not in SAMPLE_FILES:
                    raise ValueError("허용된 샘플 목록에서 선택해 주세요.")
                image_bytes = (SAMPLES / name).read_bytes()
            elif body.get("image_base64"):
                image_bytes = base64.b64decode(body["image_base64"], validate=True)
            else:
                raise ValueError("샘플을 선택하거나 이미지를 업로드해 주세요.")
            if endpoint == "/api/evaluate":
                result = run_evaluation(image_bytes, body, str(body.get("labels_yolo", "")),
                                        str(body.get("class_names", "")), str(body.get("class_map", "")))
            else:
                result = run_prediction(image_bytes, body)
            return self._send(200, json.dumps(result).encode(), "application/json; charset=utf-8")
        except Exception as exc:
            return self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json; charset=utf-8")

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Local browser UI for the upstream PCB YOLO26 detector")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(f"Starting PCB service; models will load on first use ({DEVICE})...", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"READY http://127.0.0.1:{args.port}", flush=True)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else DEVICE}", flush=True)
    if os.environ.get("CIRCUITCARE_PCB_OPEN_BROWSER") == "1":
        webbrowser.open(f"http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
