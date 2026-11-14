#!/usr/bin/env python3
"""Isolated Apple Silicon YOLO training utilities for PCB components."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "component_training"


def probe_mps():
    import torch
    result = {"torch": torch.__version__, "mps_built": torch.backends.mps.is_built(),
              "mps_available": torch.backends.mps.is_available(), "probe_ok": False, "error": None}
    if result["mps_available"]:
        try:
            device = torch.device("mps")
            a = torch.randn((16, 16), device=device, requires_grad=True)
            (a @ a).mean().backward()
            torch.optim.SGD([a], lr=1e-3).step()
            torch.mps.synchronize()
            result["probe_ok"] = True
        except Exception as exc:  # report actual MPS failure, allow CPU fallback
            result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def select_device(probe: dict) -> str:
    """Select MPS only after the runtime operator probe passes."""
    return "mps" if bool(probe.get("probe_ok")) else "cpu"


def doctor():
    import psutil
    import torch
    mps = probe_mps()
    selected = select_device(mps)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "machine": platform.machine(),
              "platform": platform.platform(), "python": sys.version, "processor": platform.processor(),
              "logical_cpus": psutil.cpu_count(), "memory_total_gib": round(psutil.virtual_memory().total / 2**30, 2),
              "memory_available_gib": round(psutil.virtual_memory().available / 2**30, 2), "mps": mps,
              "selected_device": selected, "torchvision": __import__("torchvision").__version__}
    print(json.dumps(report, indent=2))
    return 0 if selected in {"mps", "cpu"} else 1


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_fingerprint(data_yaml: Path) -> tuple[str, list[str]]:
    """Hash the YAML and all referenced training/validation images and labels."""
    import yaml
    cfg = yaml.safe_load(data_yaml.read_text())
    base = Path(cfg.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    content_config = {key: cfg.get(key) for key in ("train", "val", "test", "names", "nc") if key in cfg}
    digest = hashlib.sha256(json.dumps(content_config, sort_keys=True, separators=(",", ":")).encode())
    files: set[Path] = set()
    image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    for split_key in ("train", "val", "test"):
        value = cfg.get(split_key)
        if not value:
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            source = Path(item)
            if not source.is_absolute():
                source = base / source
            if source.is_dir():
                files.update(path for path in source.rglob("*") if path.suffix.lower() in image_suffixes)
            elif source.is_file() and source.suffix.lower() in image_suffixes:
                files.add(source)
            elif source.is_file():  # Ultralytics image-list text file
                for line in source.read_text().splitlines():
                    candidate = Path(line.strip())
                    if candidate and not candidate.is_absolute():
                        candidate = source.parent / candidate
                    if candidate.is_file():
                        files.add(candidate)
    for image in sorted(files):
        label = image.parent.parent.parent / "labels" / image.parent.name / f"{image.stem}.txt" if image.parent.parent.name == "images" else None
        for path in (image, label):
            if path and path.is_file():
                digest.update(str(path.relative_to(base) if path.is_relative_to(base) else path).encode())
                digest.update(bytes.fromhex(_hash(path)))
    return digest.hexdigest(), [str(path) for path in sorted(files)]


def _new_experiment(data: Path, profile: str, resume: Path | None = None, epochs: int | None = None,
                   init_weights: Path | None = None, model_override: str | None = None,
                   seed: int = 21, patience: int = 20, workers: int = 0,
                   batch_override: int | None = None, output_root: Path | None = None):
    import psutil
    import torch
    from ultralytics import __version__ as ultra_version
    from ultralytics import YOLO

    mps = probe_mps()
    device = select_device(mps)
    if not mps["probe_ok"]:
        print(f"MPS probe failed or unavailable; using CPU. Details: {mps['error']}", file=sys.stderr)
    profiles = {
        "baseline": dict(model="yolov8n.pt", imgsz=640, batch=2, epochs=80),
        "expanded": dict(model="yolov8n.pt", imgsz=640, batch=2, epochs=100),
        "highres": dict(model="yolov8n.pt", imgsz=960, batch=1, epochs=80),
    }
    cfg = dict(profiles[profile])
    if model_override:
        cfg["model"] = model_override
    if batch_override is not None:
        if batch_override < 1 or batch_override > 128:
            raise ValueError("batch must be between 1 and 128")
        cfg["batch"] = batch_override
    if workers < 0 or patience < 0:
        raise ValueError("workers and patience must be non-negative")
    if epochs is not None:
        if epochs < 1 or epochs > 1000:
            raise ValueError("epochs must be between 1 and 1000")
        cfg["epochs"] = epochs
    if resume and init_weights:
        raise ValueError("choose either an exact resume checkpoint or warm-start weights, not both")
    if init_weights and not init_weights.is_file():
        raise FileNotFoundError(f"warm-start checkpoint does not exist: {init_weights}")
    if psutil.virtual_memory().available < 4 * 2**30:
        raise RuntimeError("Less than 4 GiB system memory is available; close other workloads before training.")
    dataset_hash, dataset_files = _dataset_fingerprint(data)
    project = (output_root or DATA_ROOT / "runs").resolve()
    if resume and not resume.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume}")
    if resume:
        out = resume.resolve().parent.parent
        if not out.is_relative_to(project.resolve()) or not (out / "experiment.json").is_file():
            raise ValueError("resume checkpoint must belong to a recorded experiment under data/component_training/runs")
        metadata = json.loads((out / "experiment.json").read_text())
        if metadata.get("dataset_sha256") != dataset_hash:
            raise ValueError("resume dataset content hash does not match the recorded experiment")
        run_id = out.name
        cfg["imgsz"] = int(metadata.get("imgsz", cfg["imgsz"]))
        cfg["batch"] = int(metadata.get("batch_effective", metadata.get("batch_initial", cfg["batch"])))
        cfg["epochs"] = epochs or int(metadata.get("epochs", cfg["epochs"]))
        metadata["epochs"] = cfg["epochs"]
        metadata["resume_checkpoint"] = str(resume.resolve())
        metadata["resume_checkpoint_sha256"] = _hash(resume)
    else:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        out = project / run_id
        out.mkdir(parents=True, exist_ok=False)
        metadata = {"experiment_id": run_id, "profile": profile, "model": cfg["model"], "data_yaml": str(data.resolve()),
                    "data_yaml_sha256": _hash(data), "dataset_sha256": dataset_hash,
                    "dataset_image_count": len(dataset_files), "device": device, "mps_probe": mps,
                    "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
                    "torch": torch.__version__, "ultralytics": ultra_version, "seed": seed,
                    "imgsz": cfg["imgsz"], "batch_initial": cfg["batch"], "epochs": cfg["epochs"],
                    "patience": patience, "workers": workers,
                    "started_utc": datetime.now(timezone.utc).isoformat(), "memory_start_gib": psutil.virtual_memory().available / 2**30}
        (out / "experiment.json").write_text(json.dumps(metadata, indent=2))
    model = YOLO(str(resume or init_weights) if (resume or init_weights) else cfg["model"])
    base_checkpoint = Path(getattr(model, "ckpt_path", cfg["model"]))
    metadata["base_checkpoint"] = str(base_checkpoint.resolve()) if base_checkpoint.exists() else str(base_checkpoint)
    metadata["base_checkpoint_sha256"] = _hash(base_checkpoint) if base_checkpoint.is_file() else None
    try:
        import yaml
        data_config = yaml.safe_load(data.read_text()) or {}
        names = data_config.get("names", {})
        metadata["class_mapping"] = [names[key] for key in sorted(names)] if isinstance(names, dict) else names
    except Exception as exc:
        metadata["class_mapping_error"] = str(exc)
    source_repo = DATA_ROOT / "raw" / "source-pinned"
    if (source_repo / ".git").exists():
        try:
            metadata["dataset_source_commit"] = subprocess.run(
                ["git", "-C", str(source_repo), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            metadata["dataset_source_commit"] = None
    if init_weights:
        metadata["warm_start_checkpoint"] = str(init_weights.resolve())
        metadata["warm_start_sha256"] = _hash(init_weights)
    metadata["log_directory"] = str(out.resolve())
    (out / "experiment.json").write_text(json.dumps(metadata, indent=2))
    effective_batch = cfg["batch"]
    while True:
        try:
            model.train(data=str(data.resolve()), imgsz=cfg["imgsz"], batch=effective_batch,
                        epochs=cfg["epochs"], workers=workers, device=device, amp=False, seed=seed,
                        patience=patience, project=str(project), name=run_id, exist_ok=True, resume=bool(resume),
                        save=True, save_period=5, cache=False, plots=True, verbose=True)
            break
        except (RuntimeError, MemoryError) as exc:
            # Preserve run and error. Only retry a clear MPS/OOM failure, at half batch.
            log = out / "training-errors.log"
            with log.open("a") as stream:
                stream.write(f"\n{datetime.now(timezone.utc).isoformat()} batch={effective_batch}\n{type(exc).__name__}: {exc}\n")
            is_oom = any(token in str(exc).lower() for token in ("out of memory", "mps backend out of memory", "memory pressure"))
            if not is_oom or effective_batch <= 1 or profile == "highres":
                metadata["failure"] = f"{type(exc).__name__}: {exc}"
                break
            effective_batch = max(1, effective_batch // 2)
            metadata["batch_retries"] = metadata.get("batch_retries", []) + [effective_batch]
            print(f"Memory error; retrying at batch={effective_batch}. Failure preserved in {log}.")
            if device == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
            last_checkpoint = out / "weights" / "last.pt"
            if not last_checkpoint.is_file():
                metadata["failure"] = "OOM occurred before a resumable checkpoint was saved"
                break
            resume = last_checkpoint
            model = YOLO(str(last_checkpoint))
    metadata["batch_effective"] = effective_batch
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["memory_end_gib"] = psutil.virtual_memory().available / 2**30
    last = out / "weights" / "last.pt"
    metadata["last_checkpoint_sha256"] = _hash(last) if last.exists() else None
    best = out / "weights" / "best.pt"
    metadata["best_checkpoint_sha256"] = _hash(best) if best.exists() else None
    metrics_file = out / "results.csv"
    if metrics_file.is_file():
        with metrics_file.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            best_row = max(rows, key=lambda row: float(row.get("metrics/mAP50-95(B)", 0) or 0))
            metadata["best_validation_epoch"] = int(best_row["epoch"])
            metadata["best_validation_metrics"] = {key: float(best_row[key]) for key in
                ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")}
    (out / "experiment.json").write_text(json.dumps(metadata, indent=2))
    print(f"Experiment record: {out / 'experiment.json'}")
    return 0 if last.exists() else 1


def smoke(data: Path | None):
    import torch
    from ultralytics import YOLO
    report = probe_mps()
    device = select_device(report)
    if data is None:
        from PIL import Image, ImageDraw
        data = DATA_ROOT / "raw" / "smoke" / "data.yaml"
        base = data.parent
        for split, files in {"train": ("board_a", "board_b"), "val": ("board_c",)}.items():
            (base / "images" / split).mkdir(parents=True, exist_ok=True)
            (base / "labels" / split).mkdir(parents=True, exist_ok=True)
            for name in files:
                im = Image.new("RGB", (512, 512), (30, 50, 35)); d = ImageDraw.Draw(im)
                d.rectangle((80, 100, 180, 150), fill=(180, 110, 50)); d.rectangle((260, 260, 300, 300), fill=(60, 60, 60))
                im.save(base / "images" / split / f"{name}.jpg")
                (base / "labels" / split / f"{name}.txt").write_text("0 0.253906 0.244141 0.195312 0.097656\n1 0.546875 0.546875 0.078125 0.078125\n")
        data.write_text("path: " + str(base.resolve()) + "\ntrain: images/train\nval: images/val\nnames:\n  0: resistor\n  1: capacitor\n")
        print(f"Using generated synthetic pipeline fixture only: {data}")
    model = YOLO("yolov8n.pt")
    out = DATA_ROOT / "runs" / ("smoke-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.parent.mkdir(parents=True, exist_ok=True)
    model.train(data=str(data.resolve()), imgsz=512, batch=2, epochs=1, workers=0,
                device=device, amp=False, seed=21, project=str(out.parent), name=out.name,
                exist_ok=False, cache=False, save=True, verbose=True)
    checkpoint = out / "weights" / "last.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"Smoke training did not save checkpoint: {checkpoint}")
    reloaded = YOLO(str(checkpoint))
    reloaded.export(format="onnx", imgsz=512, opset=17, nms=False, device="cpu")
    print(f"Smoke checkpoint/reload/export succeeded: {checkpoint}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("doctor")
    p = subs.add_parser("smoke"); p.add_argument("--data", type=Path)
    p = subs.add_parser("train"); p.add_argument("--data", type=Path, required=True)
    p.add_argument("--profile", choices=("baseline", "expanded", "highres"), default="baseline")
    p.add_argument("--resume", type=Path, help="resume from an existing Ultralytics checkpoint")
    p.add_argument("--init-weights", type=Path, help="start a new run from prior weights with a fresh optimizer")
    p.add_argument("--epochs", type=int, help="override profile epoch count (for controlled short runs)")
    p.add_argument("--model", help="override profile model, for example yolov8s.pt")
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--batch", type=int, help="override initial batch size; OOM fallback is recorded")
    p.add_argument("--output-root", type=Path, help="write isolated candidate runs under this directory")
    p = subs.add_parser("export"); p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=640); p.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "doctor": return doctor()
    if args.command == "smoke": return smoke(args.data)
    if args.command == "train": return _new_experiment(args.data, args.profile, args.resume, args.epochs,
        args.init_weights, args.model, args.seed, args.patience, args.workers, args.batch, args.output_root)
    if args.command == "export":
        from ultralytics import YOLO
        from ultralytics import __version__ as version
        target = args.output or (args.weights.parent.parent / "export" / (args.weights.stem + ".onnx"))
        target.parent.mkdir(parents=True, exist_ok=True)
        generated = Path(YOLO(str(args.weights)).export(format="onnx", imgsz=args.imgsz, opset=17, nms=False, device="cpu"))
        shutil.copy2(generated, target)
        import onnx, onnxruntime as ort
        onnx.checker.check_model(str(target))
        session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
        record = {"weights": str(args.weights.resolve()), "weights_sha256": _hash(args.weights), "onnx": str(target.resolve()),
                  "onnx_sha256": _hash(target), "ultralytics": version,
                  "input": session.get_inputs()[0].name, "input_shape": session.get_inputs()[0].shape,
                  "outputs": [{"name": x.name, "shape": x.shape} for x in session.get_outputs()]}
        target.with_suffix(".metadata.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
