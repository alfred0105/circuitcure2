"""Apply a small set of explicitly mapped faults to SPICE text netlists."""

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .fault_models import FaultScenario

_EXPECTED_MODELS = {
    "resistor_open": "resistor",
    "resistor_short": "resistor",
    "resistance_drift": "resistor",
    "regulator_output_rise": "regulator_output",
    "regulator_output_drop": "regulator_output",
    "regulator_output_loss": "regulator_output",
    "stuck_open": "switch",
    "stuck_on": "switch",
}
_ELEMENT_PREFIXES = {"resistor": "R", "regulator_output": "V", "switch": "S"}
_SPICE_SUFFIXES = {
    "": 1.0,
    "T": 1e12,
    "G": 1e9,
    "MEG": 1e6,
    "K": 1e3,
    "M": 1e-3,
    "U": 1e-6,
    "N": 1e-9,
    "P": 1e-12,
    "F": 1e-15,
}


@dataclass(frozen=True)
class FaultInjectionChange:
    component_ref: str
    spice_element: str
    before: str
    after: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class FaultInjectionResult:
    baseline_netlist: str
    faulted_netlist: str
    changes: tuple[FaultInjectionChange, ...]


def inject_fault(
    base_netlist: str | Path,
    scenario: FaultScenario,
    component_mapping: dict[str, dict[str, Any]] | None = None,
) -> FaultInjectionResult:
    """Return a faulted copy; never writes to or mutates the baseline file."""
    if component_mapping is None:
        raise ValueError("An explicit KiCad-to-SPICE component mapping is required")
    mapping = component_mapping.get(scenario.component_ref)
    if not isinstance(mapping, dict):
        raise ValueError(f"No explicit SPICE mapping for {scenario.component_ref}")
    expected_model = _EXPECTED_MODELS[scenario.fault_type]
    if mapping.get("model_type") != expected_model:
        raise ValueError(
            f"Fault {scenario.fault_type} requires a {expected_model!r} mapping for "
            f"{scenario.component_ref}"
        )
    spice_element = str(mapping.get("spice_element", "")).strip()
    if not spice_element:
        raise ValueError(f"Mapping for {scenario.component_ref} is missing spice_element")

    baseline = _read_netlist(base_netlist)
    lines = baseline.splitlines(keepends=True)
    target_index, target_tokens = _find_element(lines, spice_element)
    if target_tokens[0][0].upper() != _ELEMENT_PREFIXES[expected_model]:
        raise ValueError(
            f"Mapped SPICE element {spice_element} is not a {expected_model} element"
        )

    if expected_model == "resistor":
        changes = [_inject_resistor(lines, target_index, target_tokens, scenario, spice_element)]
    elif expected_model == "regulator_output":
        changes = [_inject_regulator(lines, target_index, target_tokens, scenario, spice_element)]
    else:
        changes = _inject_switch(lines, target_index, target_tokens, scenario, spice_element, mapping)
    return FaultInjectionResult(baseline, "".join(lines), tuple(changes))


def _read_netlist(base_netlist: str | Path) -> str:
    if isinstance(base_netlist, Path):
        return base_netlist.read_text(encoding="utf-8")
    if "\n" in base_netlist or "\r" in base_netlist:
        return base_netlist
    return Path(base_netlist).read_text(encoding="utf-8")


def _find_element(lines: list[str], spice_element: str) -> tuple[int, list[str]]:
    matches = []
    for index, line in enumerate(lines):
        tokens = line.strip().split()
        if tokens and not tokens[0].startswith("*") and tokens[0].casefold() == spice_element.casefold():
            matches.append((index, tokens))
    if len(matches) != 1:
        reason = "not found" if not matches else "ambiguous"
        raise ValueError(f"Mapped SPICE element {spice_element!r} is {reason} in the baseline")
    return matches[0]


def _replace_line(lines: list[str], index: int, tokens: list[str]) -> tuple[str, str]:
    before = lines[index]
    ending = "\r\n" if before.endswith("\r\n") else ("\n" if before.endswith("\n") else "")
    after = " ".join(tokens) + ending
    lines[index] = after
    return before.rstrip("\r\n"), after.rstrip("\r\n")


def _positive_value(value: Any, parameter: str) -> str:
    result = str(value).strip()
    number = _spice_number(result)
    if number <= 0:
        raise ValueError(f"{parameter} must be a positive SPICE numeric value")
    return result


def _spice_number(value: str) -> float:
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([a-zA-Z]*)", value.strip())
    if not match:
        raise ValueError(f"Unsupported non-numeric SPICE value: {value!r}")
    suffix = match.group(2).upper()
    if suffix not in _SPICE_SUFFIXES:
        raise ValueError(f"Unsupported SPICE suffix in value: {value!r}")
    return float(match.group(1)) * _SPICE_SUFFIXES[suffix]


def _inject_resistor(
    lines: list[str], index: int, tokens: list[str], scenario: FaultScenario, spice_element: str
) -> FaultInjectionChange:
    if len(tokens) < 4:
        raise ValueError(f"Malformed resistor element {spice_element}")
    old_value = tokens[3]
    if _spice_number(old_value) <= 0:
        raise ValueError(f"Baseline resistance for {spice_element} must be positive")

    if scenario.fault_type == "resistor_open":
        new_value = _positive_value(scenario.parameters.get("fault_resistance", "1T"), "fault_resistance")
        description = "Open-circuit approximation using a high resistance"
    elif scenario.fault_type == "resistor_short":
        new_value = _positive_value(scenario.parameters.get("fault_resistance", "1u"), "fault_resistance")
        description = "Short-circuit approximation using a small non-zero resistance"
    else:
        if "fault_resistance" not in scenario.parameters:
            raise ValueError("resistance_drift requires a fault_resistance parameter")
        new_value = _positive_value(scenario.parameters["fault_resistance"], "fault_resistance")
        description = "Set resistor to the explicitly requested drift value"

    changed = list(tokens)
    changed[3] = new_value
    before, after = _replace_line(lines, index, changed)
    return FaultInjectionChange(scenario.component_ref, spice_element, before, after, description)


def _dc_source_value(tokens: list[str], spice_element: str) -> float:
    if len(tokens) < 5 or tokens[3].casefold() != "dc":
        raise ValueError(f"{spice_element} must be an independent voltage source with an explicit DC value")
    return _spice_number(tokens[4])


def _inject_regulator(
    lines: list[str], index: int, tokens: list[str], scenario: FaultScenario, spice_element: str
) -> FaultInjectionChange:
    if len(tokens) < 3:
        raise ValueError(f"Malformed regulator output source {spice_element}")
    if scenario.fault_type == "regulator_output_loss":
        if len(tokens) < 5 or tokens[3].casefold() != "dc":
            raise ValueError(f"{spice_element} must be an independent DC voltage source")
        resistor_name = f"RFAULT_{spice_element}"
        existing_names = {
            line.strip().split()[0].casefold()
            for line in lines if line.strip() and not line.lstrip().startswith("*")
        }
        if resistor_name.casefold() in existing_names:
            raise ValueError(f"Generated SPICE element name {resistor_name} already exists")
        open_value = _positive_value(scenario.parameters.get("open_resistance", "1T"), "open_resistance")
        replacement = [resistor_name, tokens[1], tokens[2], open_value]
        before, after = _replace_line(lines, index, replacement)
        return FaultInjectionChange(
            scenario.component_ref, spice_element, before, after,
            "Replace ideal regulator output source with a high-resistance output-loss approximation",
        )

    original_voltage = _dc_source_value(tokens, spice_element)
    if "output_voltage" not in scenario.parameters:
        raise ValueError(f"{scenario.fault_type} requires an output_voltage parameter")
    new_value = str(scenario.parameters["output_voltage"]).strip()
    new_voltage = _spice_number(new_value)
    if scenario.fault_type == "regulator_output_rise" and new_voltage <= original_voltage:
        raise ValueError("regulator_output_rise requires a value above the baseline output voltage")
    if scenario.fault_type == "regulator_output_drop" and new_voltage >= original_voltage:
        raise ValueError("regulator_output_drop requires a value below the baseline output voltage")
    changed = list(tokens)
    changed[4] = new_value
    before, after = _replace_line(lines, index, changed)
    return FaultInjectionChange(
        scenario.component_ref, spice_element, before, after,
        "Change the ideal regulator output source setpoint",
    )


def _inject_switch(
    lines: list[str],
    index: int,
    tokens: list[str],
    scenario: FaultScenario,
    spice_element: str,
    mapping: dict[str, Any],
) -> list[FaultInjectionChange]:
    if len(tokens) < 6:
        raise ValueError(f"{spice_element} is not a supported voltage-controlled SPICE switch")
    model_name = tokens[5]
    model_line = None
    for line in lines:
        model_tokens = line.strip().split()
        if len(model_tokens) >= 2 and model_tokens[0].casefold() == ".model" and model_tokens[1].casefold() == model_name.casefold():
            model_line = (line.strip(), model_tokens)
            break
    if model_line is None or len(model_line[1]) < 3 or not model_line[1][2].casefold().startswith("sw"):
        raise ValueError(f"No supported .model {model_name} SW(...) definition found")
    threshold_match = re.search(r"\bVt\s*=\s*([^\s)]+)", model_line[0], flags=re.IGNORECASE)
    if threshold_match is None:
        raise ValueError(f"Switch model {model_name} must define an explicit Vt threshold")

    required = ("control_source", "switch_threshold", "open_control_voltage", "closed_control_voltage")
    missing = [key for key in required if key not in mapping]
    if missing:
        raise ValueError(f"Switch mapping for {scenario.component_ref} requires: {', '.join(missing)}")
    threshold = _spice_number(str(mapping["switch_threshold"]))
    model_threshold = _spice_number(threshold_match.group(1))
    if abs(threshold - model_threshold) > 1e-12:
        raise ValueError("Mapped switch_threshold does not match the SPICE .model Vt")

    control_source = str(mapping["control_source"]).strip()
    control_index, control_tokens = _find_element(lines, control_source)
    if not control_tokens[0].upper().startswith("V") or len(control_tokens) < 4:
        raise ValueError(f"Mapped switch control source {control_source} must be a voltage source")
    control_voltage_key = "open_control_voltage" if scenario.fault_type == "stuck_open" else "closed_control_voltage"
    new_voltage = str(mapping[control_voltage_key]).strip()
    voltage_value = _spice_number(new_voltage)
    if scenario.fault_type == "stuck_open" and voltage_value >= threshold:
        raise ValueError("open_control_voltage must be below the switch threshold")
    if scenario.fault_type == "stuck_on" and voltage_value <= threshold:
        raise ValueError("closed_control_voltage must be above the switch threshold")

    changed_control = [control_tokens[0], control_tokens[1], control_tokens[2], "DC", new_voltage]
    before, after = _replace_line(lines, control_index, changed_control)
    return [FaultInjectionChange(
        scenario.component_ref, control_source, before, after,
        f"Set {spice_element} control to {scenario.fault_type.removeprefix('stuck_')} "
        f"at {new_voltage} V relative to mapped threshold {threshold:g} V",
    )]
