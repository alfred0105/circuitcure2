from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from circuitcare.graph import find_paths_result
from circuitcare.models import Circuit, Component, Connection
from circuitcare.vision.board_detector import detect_board_candidates, detection_cache_stats
from circuitcare.vision.image_loader import clear_image_cache, image_cache_stats, load_board_image
from circuitcare.workspace.atomic_json import write_json_atomic


def _dense_circuit(count: int) -> Circuit:
    components = {f"U{i}": Component(f"U{i}", "IC") for i in range(count)}
    connections = [Connection(ref, str(pin), f"N{pin}") for ref in components for pin in range(8)]
    return Circuit(components=components, connections=connections)


def test_graph_small_result_complete_and_depth_limit_is_explicit():
    circuit = Circuit(
        components={f"U{i}": Component(f"U{i}", "IC") for i in range(10)},
        connections=[Connection(f"U{i}", "1", f"N{i}") for i in range(9)]
        + [Connection(f"U{i+1}", "1", f"N{i}") for i in range(9)],
    )
    result = find_paths_result(circuit, "U0", max_hops=10)
    assert result.status == "COMPLETE"
    assert result.peak_queue_size <= 5000
    assert any(row.target_ref == "U9" and row.hops == 9 for row in result.paths)
    partial = find_paths_result(circuit, "U0", max_hops=2)
    assert partial.status == "PARTIAL_DEPTH_LIMIT"


@pytest.mark.stress
@pytest.mark.parametrize("count", [100, 300, 1000])
def test_dense_graph_obeys_global_queue_and_expansion_budgets(count):
    result = find_paths_result(_dense_circuit(count), "U0", max_hops=20,
                               max_expanded_states=120, max_queue_size=80,
                               max_total_paths=200, max_paths_per_candidate=2)
    assert result.status == "PARTIAL_RESOURCE_LIMIT"
    assert result.expanded_states <= 120
    assert result.peak_queue_size <= 80
    assert result.total_paths <= 200


@pytest.mark.stress
def test_pdf_parse_results_are_reused_and_cache_is_bounded():
    pypdf = pytest.importorskip("pypdf")
    from circuitcare.datasheet.pdf_loader import cache_stats, clear_pdf_cache, load_pdf

    clear_pdf_cache()
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=300, height=300)
    output = BytesIO()
    writer.write(output)
    data = output.getvalue()
    first = load_pdf(data, "first.pdf")
    second = load_pdf(data, "renamed.pdf")
    assert first["sha256"] == second["sha256"]
    assert second["filename"] == "renamed.pdf"
    assert cache_stats()["document_hits"] == 1


@pytest.mark.stress
def test_pdf_medium_repeated_rerun_and_malformed_input_are_safe():
    pypdf = pytest.importorskip("pypdf")
    from circuitcare.datasheet.pdf_loader import DatasheetError, cache_stats, clear_pdf_cache, load_pdf

    writer = pypdf.PdfWriter()
    for _ in range(50):
        writer.add_blank_page(width=300, height=300)
    buffer = BytesIO()
    writer.write(buffer)
    raw = buffer.getvalue()
    clear_pdf_cache()
    cache_stats(reset=True)
    for _ in range(20):
        result = load_pdf(raw, "medium.pdf")
    assert result["page_count"] == 50
    assert cache_stats()["document_hits"] == 19
    with pytest.raises(DatasheetError):
        load_pdf(b"%PDF-1.7 damaged internal structure", "damaged.pdf")


@pytest.mark.stress
def test_image_decode_and_board_detection_are_cached():
    clear_image_cache()
    image = Image.new("RGB", (512, 512), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    raw = buffer.getvalue()
    first = load_board_image(raw)
    second = load_board_image(raw)
    assert first.sha256 == second.sha256
    assert image_cache_stats()["hits"] == 1

    detection_cache_stats(reset=True)
    first_result = detect_board_candidates(first.image)
    second_result = detect_board_candidates(second.image)
    assert first_result == second_result
    assert detection_cache_stats()["hits"] == 1


@pytest.mark.stress
def test_image_2000_square_and_sequential_cache_eviction():
    clear_image_cache()
    for index in range(20):
        image = Image.new("RGB", (2000, 2000), (index, 20, 30))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=80)
        loaded = load_board_image(buffer.getvalue())
        assert loaded.processed_size == (2000, 2000)
        if index == 0:
            assert detect_board_candidates(loaded.image)["image_size_px"] == [2000, 2000]
    assert image_cache_stats()["entries"] <= 4


def test_atomic_json_keeps_existing_file_when_serialization_fails(tmp_path: Path):
    target = tmp_path / "registry.json"
    target.write_text('{"records": []}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        write_json_atomic(target, {"bad": float("nan")})
    assert json.loads(target.read_text(encoding="utf-8")) == {"records": []}
    write_json_atomic(target, {"records": [{"ok": True}]})
    assert json.loads(target.read_text(encoding="utf-8")) == {"records": [{"ok": True}]}


@pytest.mark.stress
def test_wrdata_parser_streams_rows_and_rejects_sample_budget(tmp_path: Path, monkeypatch):
    from circuitcare import spice_runner

    output = tmp_path / "sample.wrdata"
    output.write_text("time v(out)\n0 1\n1 2\n2 3\n", encoding="utf-8")
    parsed = spice_runner._read_wrdata(output, {"v(out)": "voltage:out"})
    assert parsed["voltage:out"] == [1.0, 2.0, 3.0]
    monkeypatch.setattr(spice_runner, "MAX_WRDATA_SAMPLES", 2)
    with pytest.raises(spice_runner.SimulationError, match="sample count"):
        spice_runner._read_wrdata(output, {"v(out)": "voltage:out"})


@pytest.mark.stress
def test_large_wrdata_stream_preserves_all_exact_samples(tmp_path: Path):
    from circuitcare.spice_runner import _read_wrdata

    output = tmp_path / "large.wrdata"
    count = 50_000
    with output.open("w", encoding="ascii") as stream:
        stream.write("time v(out)\n")
        for index in range(count):
            stream.write(f"{index * 1e-9:.12g} {index / 1000:.6f}\n")
    columns = _read_wrdata(output, {"v(out)": "voltage:out"})
    assert len(columns["time_s"]) == count
    assert columns["voltage:out"][-1] == pytest.approx((count - 1) / 1000)


def test_ngspice_single_sigsegv_retry_preserves_failed_attempt_log(tmp_path: Path, monkeypatch):
    import subprocess
    from circuitcare import spice_runner

    deck = tmp_path / "input.cir"
    deck.write_text("Demo\nV1 out 0 5\n.tran 1n 2n\n.end\n", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        log = Path(command[command.index("-o") + 1])
        assert kwargs["close_fds"] is False
        if len(calls) == 1:
            log.write_text("first attempt crashed\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, -11)
        log.write_text("retry succeeded\n", encoding="utf-8")
        (tmp_path / "sim.wrdata").write_text("time v(out)\n0 4.9\n1 5.0\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(spice_runner.shutil, "which", lambda _program: "/fake/ngspice")
    monkeypatch.setattr(spice_runner.subprocess, "run", fake_run)
    result = spice_runner.run_ngspice(deck, {"voltage_nodes": ["out"]}, {}, tmp_path, "sim")
    assert result.columns["voltage:out"] == [4.9, 5.0]
    assert len(calls) == 2
    assert "first attempt crashed" in Path(result.log_path).read_text(encoding="utf-8")


def test_ngspice_two_sigsegv_attempts_fail_and_preserve_both_logs(tmp_path: Path, monkeypatch):
    import subprocess
    from circuitcare import spice_runner

    deck = tmp_path / "input.cir"
    deck.write_text("Demo\nV1 out 0 5\n.tran 1n 2n\n.end\n", encoding="utf-8")

    def fake_run(command, **kwargs):
        assert kwargs["close_fds"] is False
        Path(command[command.index("-o") + 1]).write_text("segfault log\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, -11)

    monkeypatch.setattr(spice_runner.shutil, "which", lambda _program: "/fake/ngspice")
    monkeypatch.setattr(spice_runner.subprocess, "run", fake_run)
    with pytest.raises(spice_runner.SimulationError, match="SIGSEGV") as error:
        spice_runner.run_ngspice(deck, {"voltage_nodes": ["out"]}, {}, tmp_path, "sim")
    text = Path(tmp_path, "sim.log").read_text(encoding="utf-8")
    assert text.count("segfault log") == 2
    assert error.value.diagnostic["retry_count"] == 1
    assert error.value.diagnostic["status"] == "FAILED_NATIVE_CRASH"


def test_ngspice_timeout_is_failure_and_preserves_attempt_logs(tmp_path: Path, monkeypatch):
    import subprocess
    from circuitcare import spice_runner

    deck = tmp_path / "input.cir"
    deck.write_text("Demo\nV1 out 0 5\n.tran 1n 2n\n.end\n", encoding="utf-8")

    def fake_run(command, **kwargs):
        assert kwargs["close_fds"] is False
        Path(command[command.index("-o") + 1]).write_text("started\n", encoding="utf-8")
        raise subprocess.TimeoutExpired(command, timeout=0.01)

    monkeypatch.setattr(spice_runner.shutil, "which", lambda _program: "/fake/ngspice")
    monkeypatch.setattr(spice_runner.subprocess, "run", fake_run)
    with pytest.raises(spice_runner.SimulationError, match="timed out") as error:
        spice_runner.run_ngspice(deck, {"voltage_nodes": ["out"]}, {}, tmp_path, "sim", timeout_seconds=1)
    assert error.value.diagnostic["status"] == "TIMEOUT"
    assert "started" in Path(tmp_path, "sim.log").read_text(encoding="utf-8")


def test_ocr_numpy_scores_have_explicit_array_length(monkeypatch):
    from circuitcare.vision import ocr_backend

    class FakeResult:
        boxes = np.array([[[0, 0], [2, 0], [2, 2], [0, 2]]])
        txts = ["R1"]
        scores = np.array([0.9])

    class FakeEngine:
        def __call__(self, _image):
            return FakeResult()

    monkeypatch.setattr(ocr_backend, "_load_engine", lambda: FakeEngine())
    result = ocr_backend.run_local_ocr(Image.new("RGB", (8, 8)))
    assert result["records"][0]["confidence"] == pytest.approx(0.9)
