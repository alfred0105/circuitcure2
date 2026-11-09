"""Application-level validation and execution orchestration for a simulation request."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from circuitcare.services.simulation_service import execute_simulation


@dataclass(frozen=True)
class SimulationRequest:
    command: tuple[str, ...]
    output_dir: Path
    component_ref: str
    fault_type: str
    fault_value: str | None
    case: dict


@dataclass(frozen=True)
class SimulationOutcome:
    status: str
    result_dir: Path | None = None
    error: str | None = None


def build_simulation_command(*, python_executable: str, project_root: Path,
                             netlist: Path, case: dict,
                             component_ref: str, fault_type: str, output_dir: Path,
                             fault_value: str | None) -> tuple[str, ...]:
    """Build the existing absolute-path CLI invocation without Streamlit dependencies."""
    command = [python_executable, "-X", "faulthandler", "-m", "circuitcare.cli",
               "--netlist", str(netlist.resolve()), "--fault", component_ref,
               "--fault-type", fault_type,
               "--spice-base", str((project_root / case["base"]).resolve()),
               "--spice-mapping", str((project_root / case["mapping"]).resolve()),
               "--output", str(output_dir.resolve()), "--simulate",
               "--probes", str((project_root / case["probes"]).resolve()),
               "--ratings", str((project_root / case["ratings"]).resolve()),
               "--assess-risk"]
    if fault_value is not None:
        command.extend(["--fault-value", fault_value])
    return tuple(command)


def execute_simulation_request(request: SimulationRequest) -> SimulationOutcome:
    """Run one validated command and normalize its application outcome."""
    try:
        result_dir = execute_simulation(
            list(request.command), request.output_dir, request.component_ref,
            request.fault_type, case=request.case, fault_value=request.fault_value,
        )
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        details = getattr(exc, "details", str(exc))
        return SimulationOutcome("FAILED", error=details)
    return SimulationOutcome("SUCCESS", result_dir=result_dir)
