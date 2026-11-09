"""Isolated pin/pad inference using Nuts Vision YOLOv8 and PAINE U-Net."""
from __future__ import annotations

import base64
import math
import os
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
WEIGHTS = Path(os.environ.get("CIRCUITCARE_PCB_WEIGHTS", ROOT / "weights")).expanduser().resolve()
NUTS_WEIGHTS = WEIGHTS / "external" / "nuts_best.pt"
PAINE_WEIGHTS = WEIGHTS / "external" / "paine2026" / "unet_trained.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH = 512
_nuts: YOLO | None = None
_paine_model = None


def _encode(image: np.ndarray, ext: str = ".jpg") -> str:
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError("결과 이미지를 인코딩하지 못했습니다.")
    return "data:image/" + ("png" if ext == ".png" else "jpeg") + ";base64," + base64.b64encode(buf).decode("ascii")


def _load_nuts() -> YOLO:
    global _nuts
    if _nuts is None:
        if not NUTS_WEIGHTS.is_file():
            raise FileNotFoundError(f"Nuts Vision 가중치가 없습니다: {NUTS_WEIGHTS}")
        _nuts = YOLO(str(NUTS_WEIGHTS))
    return _nuts


def _load_paine():
    global _paine_model
    if _paine_model is None:
        if not PAINE_WEIGHTS.is_file():
            raise FileNotFoundError(f"PAINE 가중치가 없습니다: {PAINE_WEIGHTS}")
        from fastai.vision.all import CrossEntropyLossFlat, DataLoaders, Tensor, unet_learner, resnet34

        dummy_x = torch.zeros(1, 3, PATCH, PATCH)
        dummy_y = torch.zeros(1, PATCH, PATCH).long()
        dls = DataLoaders.from_dsets([(dummy_x[0], dummy_y[0])], [(dummy_x[0], dummy_y[0])], bs=1, device=DEVICE)
        learner = unet_learner(dls, resnet34, n_out=2, pretrained=False, loss_func=CrossEntropyLossFlat(axis=1))
        learner.model_dir = str(PAINE_WEIGHTS.parent)
        learner.load("unet_trained")
        _paine_model = learner.model.eval().to(DEVICE)
    return _paine_model


def _paine_mask(image: np.ndarray) -> np.ndarray:
    """Run the official grayscale 512x512 patch inference recipe and stitch class IDs."""
    model = _load_paine()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    pad_h, pad_w = (-h) % PATCH, (-w) % PATCH
    padded = cv2.copyMakeBorder(gray, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    H, W = padded.shape
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    mask = np.zeros((H, W), dtype=np.uint8)
    with torch.inference_mode():
        for y in range(0, H, PATCH):
            for x in range(0, W, PATCH):
                tile = padded[y:y + PATCH, x:x + PATCH]
                rgb = np.repeat(tile[..., None], 3, axis=2).copy()
                t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE).float() / 255.0
                t = (t - mean) / std
                logits = model(t)
                mask[y:y + PATCH, x:x + PATCH] = logits.argmax(dim=1)[0].byte().cpu().numpy()
    return mask[:h, :w]


def run_pin_pad_experiment(image_bytes: bytes, confidence: float = 0.20) -> dict:
    start = time.perf_counter()
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("이미지를 읽지 못했습니다. JPG 또는 PNG 파일을 선택해 주세요.")
    h, w = image.shape[:2]
    conf = min(0.95, max(0.01, float(confidence)))

    det_start = time.perf_counter()
    model = _load_nuts()
    pred = model.predict(source=image, conf=conf, device=0 if DEVICE == "cuda" else "cpu", verbose=False)[0]
    det_image = image.copy()
    detections: list[dict] = []
    names = pred.names
    class_names = [str(names[i]) for i in sorted(names)] if isinstance(names, dict) else [str(n) for n in names]
    has_pin_pad_classes = any(name.lower().rstrip("s") in {"pin", "pad"} for name in class_names)
    counts: Counter[str] = Counter()
    if pred.boxes is not None:
        for box in pred.boxes:
            cls_id = int(box.cls.item())
            label = names[cls_id] if isinstance(names, dict) else names[cls_id]
            score = float(box.conf.item())
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            counts[str(label)] += 1
            color = (255, 170, 40) if str(label).lower().rstrip("s") == "pin" else (40, 210, 255)
            cv2.rectangle(det_image, (x1, y1), (x2, y2), color, max(2, round(min(h, w) / 400)))
            cv2.putText(det_image, f"{label} {score:.2f}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.45, min(h, w) / 1100), color, 1, cv2.LINE_AA)
            detections.append({"class": str(label), "confidence": round(score, 4), "bbox_xyxy": [x1, y1, x2, y2]})
    det_seconds = time.perf_counter() - det_start

    seg_start = time.perf_counter()
    class_mask = _paine_mask(image)
    seg_seconds = time.perf_counter() - seg_start
    foreground = (class_mask == 1).astype(np.uint8) * 255
    # Keep class IDs available as a grayscale PNG; render class 1 in magenta for review.
    color_mask = np.zeros_like(image)
    color_mask[class_mask == 1] = (220, 40, 220)
    overlay = cv2.addWeighted(image, 0.70, color_mask, 0.30, 0)
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 80, 255), 1)

    return {
        "model_names": {"detector": "Nuts Vision YOLOv8", "segmenter": "PAINE 2026 U-Net ResNet34"},
        "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU",
        "image_width": w, "image_height": h,
        "detector_confidence": conf,
        "detections": detections,
        "detector_classes": class_names,
        "detector_has_pin_pad_classes": has_pin_pad_classes,
        "detector_notice": ("현재 GitHub에 공개된 best.pt 체크포인트는 IC 단일 클래스입니다. 저장소 README의 13개 클래스 모델(smd_comp.pt)은 저장소에서 제공되지 않아, 이 체크포인트로 핀·패드를 탐지할 수 없습니다." if not has_pin_pad_classes else "핀·패드 클래스를 포함한 체크포인트입니다."),
        "counts": dict(counts),
        "pins_pads_count": sum(v for k, v in counts.items() if k.lower().rstrip("s") in {"pin", "pad"}),
        "mask_foreground_pixels": int(np.count_nonzero(class_mask == 1)),
        "mask_foreground_percent": round(float(np.mean(class_mask == 1) * 100), 2),
        "class_mask_image": _encode(class_mask, ".png"),
        "segmentation_overlay": _encode(overlay),
        "detector_image": _encode(det_image),
        "detector_seconds": round(det_seconds, 2),
        "segmenter_seconds": round(seg_seconds, 2),
        "seconds": round(time.perf_counter() - start, 2),
        "mask_note": "PAINE 공식 구현의 2개 출력 클래스 중 class 1을 전경 핀 영역으로 표시했습니다. 픽셀 라벨이 없는 업로드 이미지이므로 정밀도 점수는 산출하지 않습니다.",
    }
