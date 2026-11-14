"""Versioned, photo-scoped state for PCB fault analysis."""

from .models import AnalysisFingerprints, AnalysisScope, AnalysisStatus
from .state_manager import (
    begin_analysis,
    commit_analysis_result,
    fail_analysis,
    get_current_result,
    get_current_state,
    invalidate_analysis,
    migrate_legacy_result,
)

__all__ = [
    "AnalysisFingerprints", "AnalysisScope", "AnalysisStatus",
    "begin_analysis", "commit_analysis_result", "fail_analysis",
    "get_current_result", "get_current_state", "invalidate_analysis",
    "migrate_legacy_result",
]
