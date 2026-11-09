"""Small JSON-compatible records used by the fault hypothesis pipeline."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FaultHypothesis:
    hypothesis_id: str
    component_ref: str
    fault_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    supporting_observations: tuple[str, ...] = ()
    generation_method: str = "catalog_rule"
    status: str = "proposed"
    simulation_status: str = "SIMULATION_NOT_RUN"
    validation_status: str = "INSUFFICIENT_EVIDENCE"
    validation_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["supporting_observations"] = list(self.supporting_observations)
        value["validation_reasons"] = list(self.validation_reasons)
        return value
