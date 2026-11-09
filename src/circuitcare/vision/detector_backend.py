"""Optional ONNX Runtime CPU backend for the published PCB detector."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

from circuitcare.vision.detection_postprocess import (
    PCB_MODEL_CLASSES, PCB_MODEL_ID, PCB_MODEL_IMAGE_SIZE, decode_yolov8_output, letterbox_rgb,
)


class DetectorUnavailable(RuntimeError):
    pass


def model_file_info(path: str | Path) -> dict[str, Any]:
    model_path = Path(path).expanduser()
    if not model_path.is_file():
        return {"status": "MODEL_UNAVAILABLE", "path": str(model_path), "model_id": PCB_MODEL_ID}
    stat = model_path.stat()
    digest = _hash_file(str(model_path.resolve()), stat.st_size, stat.st_mtime_ns,
                        stat.st_ctime_ns, getattr(stat, "st_ino", 0))
    return {"status": "READY", "path": str(model_path.resolve()), "sha256": digest,
            "model_id": PCB_MODEL_ID, "size_bytes": stat.st_size}


@lru_cache(maxsize=8)
def _hash_file(resolved_path: str, size_bytes: int, mtime_ns: int, ctime_ns: int, inode: int) -> str:
    del size_bytes, mtime_ns, ctime_ns, inode  # Metadata participates in replacement detection.
    digest = hashlib.sha256()
    with Path(resolved_path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=3)
def _load_session(resolved_path: str, model_sha256: str):
    del model_sha256  # cache key ensures replacing a model invalidates the old session.
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise DetectorUnavailable("ONNX Runtime가 없습니다. `python -m pip install -e '.[vision-ai]'`를 실행하세요.") from exc
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    try:
        return ort.InferenceSession(resolved_path, sess_options=options, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise DetectorUnavailable(f"PCB 탐지 ONNX 모델을 열 수 없습니다: {exc}") from exc


def run_detector(image: Any, model_path: str | Path, *, confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45) -> dict[str, Any]:
    info = model_file_info(model_path)
    if info["status"] != "READY":
        raise DetectorUnavailable(f"실제 PCB 탐지 모델이 없습니다: {info['path']}")
    if Path(info["path"]).suffix.lower() != ".onnx":
        raise DetectorUnavailable("앱 추론은 ONNX 모델만 실행합니다. 외부 exporter로 체크포인트를 ONNX로 변환하세요.")
    try:
        import numpy as np
    except ImportError as exc:
        raise DetectorUnavailable("NumPy가 없습니다. `python -m pip install -e '.[vision]'`를 실행하세요.") from exc
    session = _load_session(info["path"], info["sha256"])
    input_info = session.get_inputs()[0]
    tensor, ratio, padding = letterbox_rgb(image, PCB_MODEL_IMAGE_SIZE)
    output = session.run(None, {input_info.name: tensor})
    if not output:
        raise DetectorUnavailable("탐지 모델이 출력을 반환하지 않았습니다.")
    detections = decode_yolov8_output(
        np.asarray(output[0]), image_size=image.size, ratio=ratio, padding=padding,
        confidence_threshold=confidence_threshold, iou_threshold=iou_threshold,
    )
    return {"status": "INFERENCE_COMPLETE", "model": info, "runtime": "onnxruntime-cpu",
            "input_size_px": PCB_MODEL_IMAGE_SIZE, "class_names": list(PCB_MODEL_CLASSES),
            "detections": detections}
