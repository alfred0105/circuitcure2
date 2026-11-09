"""Artifact provenance helpers with workspace-root containment checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from circuitcare.workspace.context import file_sha256


def artifact_record(path: Path | str, *, workspace_root: Path | str,
                    evidence_type: str, verification_status: str) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    candidate = Path(path).resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Evidence artifact must be inside the workspace root.") from exc
    if not candidate.is_file():
        raise ValueError(f"Evidence artifact does not exist: {relative}")
    return {"path": relative.as_posix(), "sha256": file_sha256(candidate),
            "evidence_type": evidence_type, "verification_status": verification_status}
