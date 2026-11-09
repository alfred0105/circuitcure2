"""Stable identifiers and content hashes for the active circuit context."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from circuitcare.workspace.models import WorkspaceContext


def file_sha256(path: Path | str | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def build_workspace_context(*, circuit_id: str, netlist_path: Path | str,
                            input_paths: dict[str, Path | str | None],
                            fault: dict[str, Any], generated_at: str | None = None) -> WorkspaceContext:
    netlist = Path(netlist_path)
    hashes = {name: file_sha256(path) for name, path in sorted(input_paths.items())}
    circuit_hash = file_sha256(netlist) or "MISSING"
    identity = {"circuit_id": circuit_id, "circuit_hash": circuit_hash,
                "input_hashes": hashes, "fault": fault}
    workspace_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
    version = hashlib.sha256(json.dumps({"circuit_hash": circuit_hash, "inputs": hashes},
                                        sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    return WorkspaceContext(
        workspace_id=workspace_id,
        circuit_id=circuit_id,
        circuit_hash=circuit_hash,
        circuit_version=version,
        # Keep the exported context portable and avoid exposing the operator's home path.
        netlist_path=netlist.name,
        input_hashes=hashes,
        active_fault_scenario=dict(fault),
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
    )


def store_uploaded_netlist(content: bytes, *, workspace_root: Path | str) -> Path:
    """Store a content-addressed XML upload in the ignored local-private folder."""
    if not content or len(content) > 25 * 1024 * 1024:
        raise ValueError("KiCad XML upload must contain 1 byte to 25 MiB.")
    if b"<!DOCTYPE" in content.upper() or b"<!ENTITY" in content.upper():
        raise ValueError("XML documents with DTD or entity declarations are not accepted.")
    digest = hashlib.sha256(content).hexdigest()
    target = Path(workspace_root).resolve() / "data/private/workspaces/netlists" / f"{digest}.xml"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(content)
    return target
