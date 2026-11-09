"""Small, explicit fault scenario records and supported fault types."""

from dataclasses import asdict, dataclass, field
from typing import Any

SUPPORTED_FAULT_TYPES = {
    "resistor_open",
    "resistor_short",
    "resistance_drift",
    "regulator_output_rise",
    "regulator_output_drop",
    "regulator_output_loss",
    "stuck_open",
    "stuck_on",
}

_ASSUMPTIONS = {
    "resistor_open": ("The resistor is approximated as an ideal 1 TΩ resistance.",),
    "resistor_short": ("The resistor is approximated as an ideal 1 µΩ resistance.",),
    "resistance_drift": ("The resistor is modeled as an ideal linear element at the specified value.",),
    "regulator_output_rise": (
        "The regulator is an ideal DC output source whose setpoint is raised.",
        "This abstracts the output symptom and does not model an internal regulator failure mechanism.",
    ),
    "regulator_output_drop": (
        "The regulator is an ideal DC output source whose setpoint is lowered.",
        "This abstracts the output symptom and does not model an internal regulator failure mechanism.",
    ),
    "regulator_output_loss": (
        "The ideal output source is replaced by a 1 TΩ resistance across the same terminals.",
        "This abstracts output loss and does not model an internal regulator failure mechanism.",
    ),
    "stuck_open": (
        "The mapped voltage-controlled switch is held below its explicit switching threshold.",
        "MOSFET parasitics, body diode, and thermal behavior are not modeled.",
    ),
    "stuck_on": (
        "The mapped voltage-controlled switch is held above its explicit switching threshold.",
        "MOSFET parasitics, body diode, and thermal behavior are not modeled.",
    ),
}


@dataclass(frozen=True)
class FaultScenario:
    scenario_id: str
    component_ref: str
    fault_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("scenario_id must not be empty")
        if not self.component_ref.strip():
            raise ValueError("component_ref must not be empty")
        if self.fault_type not in SUPPORTED_FAULT_TYPES:
            raise ValueError(f"Unsupported fault type: {self.fault_type}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["assumptions"] = list(self.assumptions)
        return data


def create_scenario(
    component_ref: str,
    fault_type: str,
    parameters: dict[str, Any] | None = None,
    *,
    scenario_id: str | None = None,
    description: str | None = None,
) -> FaultScenario:
    """Create a reproducible scenario with a stable default ID and assumptions."""
    normalized_type = fault_type.strip().lower()
    if normalized_type not in SUPPORTED_FAULT_TYPES:
        raise ValueError(f"Unsupported fault type: {fault_type}")
    normalized_ref = component_ref.strip()
    if not normalized_ref:
        raise ValueError("component_ref must not be empty")
    scenario_parameters = dict(parameters or {})
    defaults = {
        "resistor_open": {"fault_resistance": "1T"},
        "resistor_short": {"fault_resistance": "1u"},
        "regulator_output_loss": {"open_resistance": "1T"},
    }
    for key, value in defaults.get(normalized_type, {}).items():
        scenario_parameters.setdefault(key, value)
    stable_id = scenario_id or f"FAULT_{normalized_ref}_{normalized_type.upper()}"
    if description is None:
        description = f"Apply {normalized_type.replace('_', ' ')} to {normalized_ref}"
    return FaultScenario(
        scenario_id=stable_id,
        component_ref=normalized_ref,
        fault_type=normalized_type,
        parameters=scenario_parameters,
        description=description,
        assumptions=_ASSUMPTIONS[normalized_type],
    )
