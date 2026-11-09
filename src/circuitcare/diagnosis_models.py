"""Small typed records for offline fault hypotheses and user measurements."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DiagnosisHypothesis:
    hypothesis_id: str
    component_ref: str
    fault_type: str
    parameters: dict[str, Any]
    prior_weight: float
    predicted_measurements: dict[str, float]
    simulation_source: str
    risk_assessment_source: str
    scenario_id: str
    source_base: str


@dataclass(frozen=True)
class Measurement:
    node: str
    value: float
    unit: str
    uncertainty: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "value": self.value,
            "unit": self.unit,
            "uncertainty": self.uncertainty,
            "source": self.source,
        }
