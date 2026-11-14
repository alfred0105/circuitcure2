"""Serializable contracts for current PCB diagnosis state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import hashlib
import json


class AnalysisStatus(StrEnum):
    NOT_RUN = "NOT_RUN"
    RUNNING = "RUNNING"
    CURRENT = "CURRENT"
    STALE = "STALE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class AnalysisScope:
    project_id: str
    board_id: str
    target_id: str

    def to_dict(self) -> dict[str, str]:
        return {"project_id": self.project_id, "board_id": self.board_id,
                "target_id": self.target_id}

    @property
    def key(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
        return "analysis:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class AnalysisFingerprints:
    input_sha256: str
    measurement_sha256: str
    rules_sha256: str
    final_sha256: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)
