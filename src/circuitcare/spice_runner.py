"""Run transient SPICE decks with ngspice and collect numeric probe data."""

import csv
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SimulationError(RuntimeError):
    """Raised when ngspice cannot produce a valid transient result."""

    def __init__(self, message: str, *, diagnostic: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostic = diagnostic or {"engine": "ngspice", "status": "FAILED"}


MAX_WRDATA_BYTES = 256 * 1024 * 1024
MAX_WRDATA_SAMPLES = 1_000_000
MAX_WRDATA_VECTORS = 128
MAX_WRDATA_VALUES = 2_000_000
MAX_PROCESS_OUTPUT_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class SimulationRun:
    run_name: str
    circuit_path: str
    log_path: str
    csv_path: str
    columns: dict[str, list[float]]
    unsupported_currents: dict[str, str]


def ngspice_available(executable: str = "ngspice") -> bool:
    return shutil.which(executable) is not None


def run_ngspice(
    circuit_path: str | Path,
    probes: dict[str, Any],
    component_mapping: dict[str, dict[str, Any]],
    output_dir: str | Path,
    run_name: str,
    *,
    executable: str = "ngspice",
    timeout_seconds: int = 120,
) -> SimulationRun:
    """Run one transient deck and save its log and normalized CSV samples."""
    program = shutil.which(executable)
    if program is None:
        raise SimulationError(f"ngspice executable {executable!r} was not found")
    source_path = Path(circuit_path).resolve()
    source = source_path.read_text(encoding="utf-8")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_name(run_name)
    runner_path = output / f"{safe_name}.runner.cir"
    raw_path = output / f"{safe_name}.wrdata"
    log_path = output / f"{safe_name}.log"
    csv_path = output / f"{safe_name}.csv"

    voltage_nodes, current_elements, unsupported = _select_vectors(source, probes, component_mapping)
    expressions = [f"v({node})" for node in voltage_nodes]
    expressions.extend(f"i({element})" for _, element in current_elements)
    if len(expressions) > MAX_WRDATA_VECTORS:
        raise SimulationError(f"Probe count exceeds the {MAX_WRDATA_VECTORS}-vector limit")
    if not expressions:
        raise SimulationError("At least one voltage node or supported source current must be probed")
    vector_columns = {f"v({node})".casefold(): f"voltage:{node}" for node in voltage_nodes}
    vector_columns.update({f"i({element})".casefold(): f"current:{reference}" for reference, element in current_elements})

    runner_text = _instrument_deck(source, str(raw_path), expressions)
    runner_path.write_text(runner_text, encoding="utf-8")
    attempts: list[tuple[int, Path, Path]] = []
    max_attempts = 2
    completed = None
    for attempt in range(max_attempts):
        attempt_log = output / f"{safe_name}.attempt{attempt + 1}.log"
        stdout_path = output / f"{safe_name}.attempt{attempt + 1}.stdout"
        stderr_path = output / f"{safe_name}.attempt{attempt + 1}.stderr"
        try:
            with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
                # Absolute paths and close_fds=False select posix_spawn on macOS;
                # forking from native-library threads was observed to SIGSEGV.
                completed = subprocess.run([program, "-b", "-o", str(attempt_log), str(runner_path)],
                    stdout=stdout_file, stderr=stderr_file, timeout=timeout_seconds, check=False,
                    close_fds=False)
        except subprocess.TimeoutExpired as exc:
            _preserve_attempt_logs(log_path, attempts + [(attempt + 1, attempt_log, None),
                                                           (attempt + 1, stdout_path, None),
                                                           (attempt + 1, stderr_path, None)])
            raise SimulationError(f"ngspice timed out for {source_path}; log: {log_path}",
                                  diagnostic={"engine": "ngspice", "status": "TIMEOUT",
                                              "retry_count": attempt, "logs_preserved": True}) from exc
        attempts.append((attempt + 1, attempt_log, stdout_path))
        attempts.append((attempt + 1, stderr_path, stderr_path))
        if completed.returncode != -11 or attempt + 1 >= max_attempts:
            break
        raw_path.unlink(missing_ok=True)
    _preserve_attempt_logs(log_path, attempts)
    retried_after_segfault = len(attempts) > 2 and completed.returncode == 0

    if completed.returncode != 0:
        if completed.returncode < 0:
            import signal
            try:
                signal_name = signal.Signals(-completed.returncode).name
            except ValueError:
                signal_name = "UNKNOWN_SIGNAL"
            retry_count = min(max(len(attempts) // 2 - 1, 0), 1)
            diagnostic = {"engine": "ngspice", "status": "FAILED_NATIVE_CRASH",
                          "return_code": completed.returncode, "signal": signal_name,
                          "retry_count": retry_count, "logs_preserved": True}
            raise SimulationError(f"ngspice child terminated by {signal_name} (return code {completed.returncode}); diagnostic={diagnostic}; logs preserved at {log_path}", diagnostic=diagnostic)
        raise SimulationError(f"ngspice exited with code {completed.returncode}; logs preserved at {log_path}",
                              diagnostic={"engine": "ngspice", "status": "FAILED",
                                          "return_code": completed.returncode,
                                          "retry_count": max(len(attempts) // 2 - 1, 0), "logs_preserved": True})
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    if re.search(r"\b(error|fatal)\s*:", log_text, flags=re.IGNORECASE) or re.search(
        r"singular matrix|timestep too small|no convergence", log_text, flags=re.IGNORECASE
    ):
        raise SimulationError(f"ngspice reported an error; log: {log_path}")
    retry_note = "[warning] ngspice exited with SIGSEGV once; the single retry succeeded.\n" if retried_after_segfault else ""
    log_path.write_text(log_text.rstrip() + "\n" + retry_note, encoding="utf-8")
    if not raw_path.is_file() or raw_path.stat().st_size == 0:
        raise SimulationError(f"ngspice produced no probe data; log: {log_path}")
    if raw_path.stat().st_size > MAX_WRDATA_BYTES:
        raise SimulationError(f"ngspice output exceeds the {MAX_WRDATA_BYTES}-byte limit; log: {log_path}")

    columns = _read_wrdata(raw_path, vector_columns)
    if len(columns.get("time_s", [])) < 2:
        raise SimulationError(f"ngspice produced fewer than two time samples; log: {log_path}")
    _write_csv(csv_path, columns)
    runner_path.unlink(missing_ok=True)
    raw_path.unlink(missing_ok=True)
    return SimulationRun(
        run_name=safe_name,
        circuit_path=str(source_path),
        log_path=str(log_path),
        csv_path=str(csv_path),
        columns=columns,
        unsupported_currents=unsupported,
    )


def _instrument_deck(source: str, data_filename: str, expressions: list[str]) -> str:
    transient: list[str] | None = None
    retained: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("*"):
            retained.append(line)
            continue
        tokens = stripped.split()
        directive = tokens[0].casefold()
        if directive == ".control":
            raise SimulationError("Baseline decks with an existing .control block are not supported")
        if directive == ".tran":
            if transient is not None:
                raise SimulationError("Only one .tran analysis is supported per input deck")
            if len(tokens) < 3:
                raise SimulationError("Malformed .tran directive; expected step and stop time")
            transient = tokens[1:]
            continue
        if directive == ".end":
            continue
        retained.append(line)
    if transient is None:
        raise SimulationError("Baseline deck must contain one .tran directive")

    control = [
        ".control",
        "set wr_singlescale",
        "set wr_vecnames",
        "tran " + " ".join(transient),
        f"wrdata {data_filename} {' '.join(expressions)}",
        "quit",
        ".endc",
        ".end",
    ]
    return "\n".join([*retained, *control, ""])


def _select_vectors(
    source: str,
    probes: dict[str, Any],
    component_mapping: dict[str, dict[str, Any]],
) -> tuple[list[str], list[tuple[str, str]], dict[str, str]]:
    nodes = []
    for node in probes.get("voltage_nodes", []):
        _validate_spice_identifier(str(node), "voltage node")
        if str(node) != "0" and str(node).casefold() not in {item.casefold() for item in nodes}:
            nodes.append(str(node))

    currents: list[tuple[str, str]] = []
    unsupported: dict[str, str] = {}
    for reference in probes.get("components", []):
        entry = component_mapping.get(reference)
        if not isinstance(entry, dict):
            unsupported[reference] = "No explicit KiCad-to-SPICE mapping is available."
            continue
        element = str(entry.get("spice_element", "")).strip()
        if not element:
            unsupported[reference] = "The mapping has no SPICE element name."
            continue
        _validate_spice_identifier(element, "SPICE element")
        tokens = _find_element_tokens(source, element)
        if tokens is None:
            unsupported[reference] = f"Mapped SPICE element {element} is absent from this circuit."
            continue
        model_type = entry.get("model_type")
        if model_type == "resistor" and tokens[0][0].upper() == "R" and len(tokens) >= 4:
            terminals = tokens[1:3]
        elif model_type == "capacitor" and tokens[0][0].upper() == "C" and len(tokens) >= 4:
            terminals = tokens[1:3]
        elif model_type == "regulator_output" and tokens[0][0].upper() == "V" and len(tokens) >= 5:
            terminals = tokens[1:3]
            currents.append((reference, element))
        elif model_type == "switch" and tokens[0][0].upper() == "S" and len(tokens) >= 6:
            terminals = tokens[1:3]
            unsupported[reference] = "Switch current and power are not reported by this probe backend."
        else:
            unsupported[reference] = f"Mapped model type {model_type!r} does not have a supported probe model."
            continue
        for node in terminals:
            _validate_spice_identifier(node, "SPICE terminal")
            if node != "0" and node.casefold() not in {item.casefold() for item in nodes}:
                nodes.append(node)

    return nodes, currents, unsupported


def _find_element_tokens(source: str, element: str) -> list[str] | None:
    matches = []
    for line in source.splitlines():
        tokens = line.strip().split()
        if tokens and not tokens[0].startswith("*") and tokens[0].casefold() == element.casefold():
            matches.append(tokens)
    if len(matches) > 1:
        raise SimulationError(f"SPICE element {element} appears more than once")
    return matches[0] if matches else None


def _validate_spice_identifier(value: str, label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.$+-]+", value):
        raise SimulationError(f"Unsupported {label} name {value!r}; use a simple SPICE identifier")


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    if not name:
        raise SimulationError("run_name must contain a letter or number")
    return name


def _preserve_attempt_logs(destination: Path, attempts: list[tuple[int, Path, Path | None]]) -> None:
    """Combine bounded child logs without discarding a failed retry attempt."""
    with destination.open("w", encoding="utf-8", errors="replace") as output:
        for attempt, path, _ in attempts:
            output.write(f"\n--- attempt {attempt}: {path.name} ---\n")
            if path.is_file():
                if path.stat().st_size > MAX_PROCESS_OUTPUT_BYTES:
                    output.write("[truncated: child output exceeded configured log limit]\n")
                    with path.open("rb") as source:
                        output.write(source.read(MAX_PROCESS_OUTPUT_BYTES).decode("utf-8", errors="replace"))
                else:
                    output.write(path.read_text(encoding="utf-8", errors="replace"))


def _read_wrdata(path: Path, vector_columns: dict[str, str]) -> dict[str, list[float]]:
    if len(vector_columns) > MAX_WRDATA_VECTORS:
        raise SimulationError(f"Requested vector count exceeds limit in {path}")
    iterator = (line.split() for line in path.open(encoding="utf-8", errors="replace"))
    header_row = next((row for row in iterator if row), None)
    if not header_row:
        raise SimulationError(f"Empty ngspice wrdata file: {path}")
    headers = [item.casefold() for item in header_row]
    if len(headers) > MAX_WRDATA_VECTORS + 1:
        raise SimulationError(f"ngspice header exceeds the {MAX_WRDATA_VECTORS}-vector limit in {path}")
    if not headers or headers[0] != "time":
        raise SimulationError(f"Unexpected ngspice wrdata header in {path}: {header_row}")
    requested_indices = {}
    for header, key in vector_columns.items():
        if header.casefold() not in headers:
            raise SimulationError(f"ngspice result is missing requested vector {header}")
        requested_indices[key] = headers.index(header.casefold())
    columns: dict[str, list[float]] = {"time_s": []}
    columns.update({key: [] for key in requested_indices})
    sample_count = 0
    for row in iterator:
        if not row:
            continue
        sample_count += 1
        if sample_count > MAX_WRDATA_SAMPLES:
            raise SimulationError(f"ngspice sample count exceeds {MAX_WRDATA_SAMPLES} in {path}")
        if sample_count * len(headers) > MAX_WRDATA_VALUES:
            raise SimulationError(f"ngspice numeric value count exceeds {MAX_WRDATA_VALUES} in {path}")
        if len(row) != len(headers):
            raise SimulationError(f"Malformed ngspice data row in {path}")
        try:
            values = [float(item) for item in row]
        except ValueError as exc:
            raise SimulationError(f"Non-numeric ngspice data row in {path}") from exc
        if not all(math.isfinite(item) for item in values):
            raise SimulationError(f"Non-finite ngspice sample in {path}")
        columns["time_s"].append(values[0])
        for key, index in requested_indices.items():
            columns[key].append(values[index])
    if any(right <= left for left, right in zip(columns["time_s"], columns["time_s"][1:])):
        raise SimulationError(f"ngspice time samples are not strictly increasing in {path}")
    return columns


def _write_csv(path: Path, columns: dict[str, list[float]]) -> None:
    keys = list(columns)
    length = len(columns["time_s"])
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for index in range(length):
            writer.writerow({key: columns[key][index] for key in keys})
