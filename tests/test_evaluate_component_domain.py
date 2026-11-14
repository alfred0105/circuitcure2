import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_component_domain", ROOT / "scripts" / "evaluate_component_domain.py")
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def _review_manifest(tmp_path, types=("BARE_PCB", "BREADBOARD")):
    evaluator.ROOT = tmp_path
    rows = []
    for index, image_type in enumerate(types):
        relative = Path("raw") / f"negative-{index}.jpg"
        path = tmp_path / "data/component_training" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (index, 100, 120)).save(path)
        rows.append({"image_id": f"negative-{index}", "image_path": str(relative),
            "image_type": image_type, "hard_negative_eligible": True,
            "board_group_id": f"board-{index}"})
    manifest = tmp_path / "review.json"
    manifest.write_text(json.dumps({"images": rows}))
    return manifest, rows


def _successful_result(detections=None):
    return {"status": "INFERENCE_COMPLETE", "model": {"status": "READY"},
            "detections": detections or []}


def test_negative_evaluation_all_success(monkeypatch, tmp_path):
    manifest, _ = _review_manifest(tmp_path)
    monkeypatch.setattr(evaluator, "detect_components", lambda *a, **k: _successful_result())
    result = evaluator.evaluate_negatives(Path("model.onnx"), manifest, confidence=0.25, nms_iou=0.45)
    assert result["BARE_PCB"]["reviewed_image_count"] == 1
    assert result["BARE_PCB"]["evaluated_image_count"] == 1
    assert result["BARE_PCB"]["inference_failure_count"] == 0
    assert result["BARE_PCB"]["image_level_false_positive_rate"] == 0
    assert result["BARE_PCB"]["mean_false_positives_per_image"] == 0
    assert result["BREADBOARD"]["status"] == "MEASURED_REVIEWED_EMPTY_IMAGES"


def test_negative_evaluation_partial_failure_is_not_scored_as_zero(monkeypatch, tmp_path):
    manifest, _ = _review_manifest(tmp_path)

    def infer(image, model, **kwargs):
        if image.getpixel((0, 0))[0] == 0:
            raise RuntimeError("inference failed")
        return _successful_result([{"class": "ic"}])

    monkeypatch.setattr(evaluator, "detect_components", infer)
    result = evaluator.evaluate_negatives(Path("model.onnx"), manifest, confidence=0.25, nms_iou=0.45)
    assert result["BARE_PCB"]["evaluated_image_count"] == 0
    assert result["BARE_PCB"]["inference_failure_count"] == 1
    assert result["BARE_PCB"]["image_level_false_positive_rate"] is None
    assert result["BREADBOARD"]["evaluated_image_count"] == 1
    assert result["BREADBOARD"]["image_level_false_positive_rate"] == 1
    assert result["BREADBOARD"]["mean_false_positives_per_image"] == 1


def test_negative_evaluation_all_inference_failed(monkeypatch, tmp_path):
    manifest, _ = _review_manifest(tmp_path)
    monkeypatch.setattr(evaluator, "detect_components", lambda *a, **k: (_ for _ in ()).throw(OSError("no model")))
    result = evaluator.evaluate_negatives(Path("model.onnx"), manifest, confidence=0.25, nms_iou=0.45)
    assert all(value["evaluated_image_count"] == 0 for value in result.values())
    assert all(value["inference_failure_count"] == 1 for value in result.values())
    assert all(value["image_level_false_positive_rate"] is None for value in result.values())
    assert all(value["mean_false_positives_per_image"] is None for value in result.values())
    assert all(value["status"] == "NO_SUCCESSFUL_INFERENCES" for value in result.values())


def test_negative_evaluation_zero_reviewed_images(monkeypatch, tmp_path):
    manifest, _ = _review_manifest(tmp_path, types=())
    monkeypatch.setattr(evaluator, "detect_components", lambda *a, **k: pytest.fail("must not infer"))
    result = evaluator.evaluate_negatives(Path("model.onnx"), manifest, confidence=0.25, nms_iou=0.45)
    assert all(value["reviewed_image_count"] == 0 for value in result.values())
    assert all(value["evaluated_image_count"] == 0 for value in result.values())
    assert all(value["status"] == "NO_REVIEWED_IMAGES" for value in result.values())
    assert all(value["image_level_false_positive_rate"] is None for value in result.values())


def test_cli_main_writes_report_and_prints_existing_negative_count_fields(monkeypatch, tmp_path, capsys):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    review, _ = _review_manifest(tmp_path, types=())
    evaluation = tmp_path / "positive.json"
    evaluation.write_text("{}")
    output = tmp_path / "report.json"

    def run(command, **kwargs):
        positive_path = Path(command[command.index("--output") + 1])
        positive_path.write_text(json.dumps({"evaluation_status": "REVIEWED_VALIDATION",
            "dataset": {"image_count": 4, "input_fingerprints": {}}, "metrics": {},
            "steady_per_image_latency_ms": {}, "peak_process_rss_bytes": 1}))

        class Completed:
            returncode = 0
            stdout = ""
            stderr = ""
        return Completed()

    monkeypatch.setattr(evaluator.subprocess, "run", run)
    monkeypatch.setattr(evaluator, "evaluate_negatives", lambda *a, **k: {
        "BARE_PCB": {"reviewed_image_count": 0, "evaluated_image_count": 0,
            "inference_failure_count": 0, "status": "NO_REVIEWED_IMAGES"},
        "BREADBOARD": {"reviewed_image_count": 0, "evaluated_image_count": 0,
            "inference_failure_count": 0, "status": "NO_REVIEWED_IMAGES"}})
    monkeypatch.setattr(sys, "argv", ["evaluate_component_domain.py", "--model", str(model),
        "--review-manifest", str(review), "--evaluation-manifest", str(evaluation), "--output", str(output)])
    assert evaluator.main() == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["positive_image_count"] == 4
    assert printed["negative_counts"]["BARE_PCB"] == {"reviewed": 0, "evaluated": 0, "failed": 0}
    assert json.loads(output.read_text())["reviewed_empty_domain_metrics"]["BARE_PCB"]["status"] == "NO_REVIEWED_IMAGES"
