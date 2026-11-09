from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CASES = [
    pytest.param("tests/fixtures/fault_demo_netlist.xml", "U1", "regulator_output_rise", "8",
                 "data/circuits/fault_demo.cir", "data/circuits/fault_demo_mapping.json",
                 "data/circuits/fault_demo_probes.json", "data/component_specs/fault_demo_ratings.json", id="regulator"),
    pytest.param("tests/fixtures/divider_netlist.xml", "R1", "resistance_drift", "10k",
                 "data/circuits/divider.cir", "data/circuits/divider_mapping.json",
                 "data/circuits/divider_probes.json", "data/component_specs/divider_ratings.json", id="divider"),
    pytest.param("tests/fixtures/rc_netlist.xml", "RCHARGE", "resistance_drift", "100k",
                 "data/circuits/rc_charge.cir", "data/circuits/rc_charge_mapping.json",
                 "data/circuits/rc_charge_probes.json", "data/component_specs/rc_charge_ratings.json", id="rc"),
]


@pytest.mark.integration
@pytest.mark.parametrize("runtime", ["onnx", "ocr"])
def test_native_vision_inference_isolated_child(runtime):
    model = ROOT / "models/pcb_component_detector.onnx"
    def unavailable(message: str) -> None:
        if os.environ.get("CIRCUITCARE_REQUIRE_NATIVE_INFERENCE") == "1":
            pytest.fail(message)
        pytest.skip(message)

    if runtime == "onnx" and (not model.is_file() or not importlib.util.find_spec("onnxruntime")):
        unavailable("local detector model or ONNX Runtime is unavailable")
    if runtime == "ocr":
        if not importlib.util.find_spec("rapidocr") or not importlib.util.find_spec("onnxruntime"):
            unavailable("RapidOCR or ONNX Runtime is unavailable")
        spec = importlib.util.find_spec("rapidocr")
        model_root = Path(spec.origin).parent / "models" if spec and spec.origin else Path("/nonexistent")
        if not any(model_root.glob("*.onnx")):
            unavailable("local OCR ONNX model files are unavailable")
    script = r'''
import json, sys
from PIL import Image
image = Image.new("RGB", (512, 512), "white")
if sys.argv[1] == "onnx":
    from circuitcare.vision.detector_backend import run_detector
    result = run_detector(image, sys.argv[2])
    payload = {"status": result["status"], "detection_count": len(result["detections"])}
else:
    from circuitcare.vision.ocr_backend import run_local_ocr
    result = run_local_ocr(image)
    payload = {"status": result["status"], "line_count": result["line_count"]}
print("NATIVE_RESULT=" + json.dumps(payload))
'''
    args = [sys.executable, "-c", script, runtime]
    if runtime == "onnx":
        args.append(str(model))
    completed = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=90, check=False)
    if completed.returncode != 0:
        pytest.fail(f"isolated {runtime} child failed, return code {completed.returncode}; stderr={completed.stderr[-2000:]}")
    result_line = next(line for line in completed.stdout.splitlines() if line.startswith("NATIVE_RESULT="))
    assert json.loads(result_line.split("=", 1)[1])["status"] in {"INFERENCE_COMPLETE", "OCR_COMPLETE"}


@pytest.mark.integration
@pytest.mark.parametrize("netlist,fault,fault_type,value,deck,mapping,probes,ratings", CASES)
def test_live_ngspice_representative_scenarios(tmp_path, netlist, fault, fault_type, value,
                                                deck, mapping, probes, ratings):
    if not shutil.which("ngspice"):
        pytest.skip("ngspice executable is unavailable")
    output = tmp_path / "isolated-simulation"
    command = [sys.executable, "-m", "circuitcare.cli", "--netlist", netlist, "--fault", fault,
               "--fault-type", fault_type, "--fault-value", value, "--spice-base", deck,
               "--spice-mapping", mapping, "--output", str(output), "--simulate", "--probes", probes,
               "--ratings", ratings, "--assess-risk"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120, check=False)
    assert completed.returncode == 0, f"stdout={completed.stdout[-2000:]}\nstderr={completed.stderr[-2000:]}"
    payload = json.loads(completed.stdout)
    analysis = payload["fault_scenario"]["simulation_analysis"]
    assert analysis["simulation_status"] == "success"
    assert analysis["simulation_run_status"] == {"baseline": "success", "fault": "success"}
    result_path = output / "simulation_results" / f"fault_{fault.lower()}_{fault_type}" / "comparison.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["simulation_success"] is True
    assert result["time_range_s"]["samples"] >= 2
