#!/usr/bin/env python3
"""Download the pinned checkpoint and export ONNX using a separately installed CLI."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuitcare.vision.model_artifacts import default_model_dir, download_detector_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the optional PCB component detector ONNX model.")
    parser.add_argument("--weights", type=Path, help="trusted local checkpoint; downloads the pinned file if omitted")
    parser.add_argument("--output", type=Path, help="output ONNX path")
    parser.add_argument("--exporter", help="path to a separately installed `yolo` exporter executable")
    args = parser.parse_args()
    model_dir = default_model_dir()
    weights = args.weights or model_dir / "yolov8s_pcb_best.pt"
    output = args.output or ROOT / "models" / "pcb_component_detector.onnx"
    if not weights.exists():
        if args.weights:
            parser.error(f"checkpoint file does not exist: {weights}")
        print("Downloading the pinned 85.4 MB checkpoint and verifying SHA-256...")
        download_detector_checkpoint(weights)
    exporter = args.exporter or shutil.which("yolo")
    if not exporter:
        parser.error("External `yolo` exporter is not installed. See docs/PCB_AI_DETECTION.md for its separate license and setup.")
    print(f"Exporting {weights} to ONNX at 960 px. The application runtime does not load the .pt checkpoint.")
    subprocess.run([exporter, "export", f"model={weights}", "format=onnx", "imgsz=960", "opset=17",
                    "simplify=False", "nms=False", "device=cpu"], check=True)
    generated = weights.with_suffix(".onnx")
    if not generated.is_file():
        parser.error(f"Exporter did not produce the expected file: {generated}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if generated.resolve() != output.resolve():
        shutil.move(str(generated), str(output))
    print(f"ONNX model ready: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
