"""Translation boundary between frontend view models and backend outputs."""
from __future__ import annotations
from typing import Any, Protocol
from circuitcare.frontend.models import DiagnosisSummaryViewModel

class DiagnosisAdapter(Protocol):
    def diagnosis_summary(self, *, project_id: str) -> DiagnosisSummaryViewModel: ...
    def update_measurements(self, *, project_id: str, measurements: list[dict[str, Any]]) -> DiagnosisSummaryViewModel: ...

class MockDiagnosisAdapter:
    """Explicit demo-only adapter; never represents a real diagnosis."""
    def diagnosis_summary(self, *, project_id: str) -> DiagnosisSummaryViewModel:
        return DiagnosisSummaryViewModel(
            overall_status="DEMO_ONLY", verification_status="DEMO_ONLY",
            provenance={"source": "MOCK", "verification_status": "DEMO_ONLY", "project_id": project_id},
        )

    def update_measurements(self, *, project_id: str, measurements: list[dict[str, Any]]) -> DiagnosisSummaryViewModel:
        result = self.diagnosis_summary(project_id=project_id)
        return DiagnosisSummaryViewModel(
            overall_status=result.overall_status, verification_status=result.verification_status,
            provenance={**result.provenance, "measurement_count": len(measurements)},
        )
