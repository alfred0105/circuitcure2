"""Spawn-safe simulation result validation and input fingerprint helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

from circuitcare.dashboard.context import ROOT
from circuitcare.dashboard.downloads import MAX_DOWNLOAD_CSV_BYTES
from circuitcare.workspace.simulation_context import build_simulation_fingerprint


class SimulationRunError(RuntimeError):
    def __init__(self, details: str):
        self.details = details
        super().__init__(details)


def _current_simulation_fingerprint(case: dict, fault: str, fault_type: str,
                                    fault_value: str | None) -> dict:
    if not case.get("base") or not case.get("mapping") or not case.get("probes"):
        return {"fingerprint_sha256": None}
    deck = ROOT / case["base"]
    model_paths = []
    try:
        for line in deck.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^\s*\.(?:include|inc|lib)\s+[\"']?([^\"'\s]+)", line, flags=re.IGNORECASE)
            if match:
                model_paths.append((deck.parent / match.group(1)).resolve())
    except OSError:
        return {"fingerprint_sha256": None}
    settings = {"engine": "ngspice", "analysis": "transient", "runner": "circuitcare-cli"}
    if case.get("fingerprint_ngspice_version"):
        try:
            version_result = subprocess.run(["ngspice", "--version"], capture_output=True,
                                            text=True, timeout=5, check=True, close_fds=False)
        except (OSError, subprocess.SubprocessError):
            return {"fingerprint_sha256": None}
        version_match = re.search(r"\bngspice-[A-Za-z0-9._-]+", version_result.stdout)
        if not version_match:
            return {"fingerprint_sha256": None}
        settings["ngspice_version"] = version_match.group(0)
    return build_simulation_fingerprint(
        netlist=ROOT / case["netlist"], spice_deck=deck,
        component_mapping=ROOT / case["mapping"], probes=ROOT / case["probes"],
        fault={"component_ref": fault, "fault_type": fault_type, "parameter": fault_value},
        settings=settings,
        relevant_models=model_paths,
        ratings=ROOT / case["ratings"] if case.get("ratings") else None,
    )


def _validate_live_result(completed: subprocess.CompletedProcess, output: Path, fault: str, fault_type: str,
                          *, case: dict | None = None, fault_value: str | None = None) -> Path:
    if completed.returncode != 0:
        raise SimulationRunError(_simulation_error_details(completed, output))
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SimulationRunError(f"CLI JSON을 읽을 수 없습니다: {exc}\n{_simulation_error_details(completed, output)}") from exc
    scenario = payload.get("fault_scenario", {})
    scenario_id = f"FAULT_{fault}_{fault_type}".upper()
    report = scenario.get("simulation_analysis", {})
    if (scenario.get("component_ref") != fault or scenario.get("fault_type") != fault_type
            or scenario.get("scenario_id") != scenario_id or scenario.get("simulation_status") != "success"
            or report.get("simulation_status") != "success"
            or report.get("simulation_run_status") != {"baseline": "success", "fault": "success"}):
        raise SimulationRunError(f"CLI 출력이 현재 고장 조건의 성공 결과와 일치하지 않습니다.\n{_simulation_error_details(completed, output)}")
    result_dir = output / "simulation_results" / f"fault_{fault.lower()}_{fault_type}"
    paths = {name: result_dir / f"{name}.json" for name in ("comparison", "component_stress", "risk_assessment")}
    if not all(path.is_file() for path in paths.values()):
        raise SimulationRunError(f"필요한 새 결과 파일이 부족합니다: {', '.join(p.name for p in paths.values() if not p.is_file())}\n{_simulation_error_details(completed, output)}")
    try:
        comparison, stress, risk = (json.loads(paths[key].read_text(encoding="utf-8")) for key in ("comparison", "component_stress", "risk_assessment"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SimulationRunError(f"결과 JSON을 읽을 수 없습니다: {exc}\n{_simulation_error_details(completed, output)}") from exc
    if (comparison.get("simulation_success") is not True or comparison.get("scenario_id") != scenario_id
            or stress.get("simulation_success") is not True or stress.get("fault_type") != fault_type
            or risk.get("simulation_status") != "success" or risk.get("scenario_id") != scenario_id
            or risk.get("failed_component") != fault or risk.get("fault_type") != fault_type):
        raise SimulationRunError(f"결과 파일 상태 또는 고장 조건이 일치하지 않습니다.\n{_simulation_error_details(completed, output)}")
    csv_path = result_dir / "comparison.csv"
    try:
        if not csv_path.is_file() or csv_path.stat().st_size == 0:
            raise ValueError("comparison.csv가 없거나 비어 있습니다.")
        if csv_path.stat().st_size > MAX_DOWNLOAD_CSV_BYTES:
            raise ValueError("comparison.csv가 허용된 크기를 초과했습니다.")
        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            fields = reader.fieldnames or []
            node_columns = {f"{prefix}_V({node})_V" for node in comparison.get("voltage_nodes", {})
                            for prefix in ("baseline", "fault")}
            if ("time_s" not in fields or not node_columns.issubset(fields)
                    or not next(reader, None) or not next(reader, None)):
                raise ValueError("comparison.csv의 시간 열 또는 데이터 행이 유효하지 않습니다.")
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        raise SimulationRunError(f"Live Simulation 파형 CSV를 검증할 수 없습니다: {exc}\n{_simulation_error_details(completed, output)}") from exc
    if case is None:
        raise SimulationRunError("성공한 시뮬레이션에 입력 Fingerprint를 기록할 회로 설정이 없습니다.")
    fingerprint = _current_simulation_fingerprint(case, fault, fault_type, fault_value)
    result_paths = [*paths.values(), csv_path]
    manifest = {"status": "LIVE_SUCCESS", "fingerprint": fingerprint,
                "result_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in result_paths}}
    (result_dir / "simulation_context.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return result_dir


def _simulation_error_details(completed: subprocess.CompletedProcess, output: Path) -> str:
    stderr = (completed.stderr or "").strip()[-3000:] or "(stderr 없음)"
    logs = sorted(output.rglob("*.log")) if output.exists() else []
    log_text = ", ".join(str(path) for path in logs) if logs else f"{output}/simulation_results/의 ngspice 로그를 확인하세요."
    return f"종료 코드: {completed.returncode}\nstderr:\n{stderr}\nngspice 로그: {log_text}"


def _valid_parameter(value: str) -> bool:
    match = re.fullmatch(r"\s*((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([pnumkKMG]?)\s*", value)
    if not match:
        return False
    multiplier = {"": 1.0, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9}[match.group(2)]
    try:
        return float(match.group(1)) * multiplier > 0
    except ValueError:
        return False


def execute_simulation(cmd: list[str], output: Path, fault: str, fault_type: str,
                       *, case: dict, fault_value: str | None = None) -> Path:
    """Run the existing CLI through macOS spawn-safe subprocess options, then validate outputs."""
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                               check=False, close_fds=False)
    return _validate_live_result(completed, output, fault, fault_type,
                                 case=case, fault_value=fault_value)

