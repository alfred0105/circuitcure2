"""Check local prerequisites and committed demo inputs without rerunning analyses."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_JSON = (
    "data/fault_scenarios/overvoltage_demo/simulation_results/fault_u1_regulator_output_rise/comparison.json",
    "data/fault_scenarios/overvoltage_demo/simulation_results/fault_u1_regulator_output_rise/component_stress.json",
    "data/fault_scenarios/overvoltage_demo/simulation_results/fault_u1_regulator_output_rise/risk_assessment.json",
    "data/diagnosis/prior_results/diagnosis.json",
    "data/diagnosis/posterior_results/diagnosis.json",
    "data/repair_advice/demo_c1/llm_explanation.json",
    "data/repair_advice/fallback_no_llm/repair_advice.json",
)


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    try:
        import circuitcare  # noqa: F401
        print(f"Python: OK ({sys.version.split()[0]})")
    except ImportError as exc:
        print(f"Python: MISSING ({exc})")
        return 1

    if importlib.util.find_spec("streamlit"):
        print(f"Streamlit: OK ({_module_version('streamlit')})")
        streamlit_ok = True
    else:
        print("Streamlit: MISSING (install with python -m pip install -e \".[dev,ui]\")")
        streamlit_ok = False

    ngspice = shutil.which("ngspice")
    print(f"ngspice: {'OK (' + ngspice + ')' if ngspice else 'Unavailable · cached results only'}")

    missing: list[str] = []
    invalid: list[str] = []
    for relative in REQUIRED_JSON:
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid.append(relative)
    data_ok = not missing and not invalid
    if data_ok:
        print("Cached Demo Data: OK (required JSON present and parseable)")
    else:
        print(f"Cached Demo Data: MISSING/INVALID · missing={missing} invalid={invalid}")

    models = _ollama_models()
    if models is None:
        print("Ollama: Unavailable")
    elif models:
        names = ", ".join(item.get("name", "unknown") for item in models)
        print(f"Ollama: Available · local model(s): {names}")
    else:
        print("Ollama: Available · no local models installed")
    print("Fallback: Available (deterministic repair advisor; no Ollama required)")
    return 0 if streamlit_ok and data_ok else 1


def _module_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "installed"


def _ollama_models() -> list[dict] | None:
    try:
        request = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
        with urllib.request.urlopen(request, timeout=2) as response:
            body = json.loads(response.read().decode("utf-8"))
        models = body.get("models")
        return models if isinstance(models, list) else None
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
