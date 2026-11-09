"""Bounded immutable download snapshots for dashboard exports."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

MAX_DOWNLOAD_CSV_BYTES = 64 * 1024 * 1024

@lru_cache(maxsize=4)
def _cached_download_bytes(path: str, digest: str, context_key: str) -> bytes:
    """Cache a bounded, immutable CSV snapshot for one analysis context and content hash."""
    del context_key  # Part of the cache key to prevent cross-context reuse.
    csv_path = Path(path)
    before = csv_path.stat()
    if before.st_size > MAX_DOWNLOAD_CSV_BYTES:
        raise ValueError("CSV exceeds the 64 MiB download limit")
    with csv_path.open("rb") as source:
        data = source.read(MAX_DOWNLOAD_CSV_BYTES + 1)
    if len(data) > MAX_DOWNLOAD_CSV_BYTES:
        raise ValueError("CSV exceeds the 64 MiB download limit")
    after = csv_path.stat()
    stable = (before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
              and getattr(before, "st_ino", 0) == getattr(after, "st_ino", 0))
    if not stable or len(data) != after.st_size or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("CSV changed while its download snapshot was being prepared")
    return data


def _download_file_digest(path: Path) -> str:
    if path.stat().st_size > MAX_DOWNLOAD_CSV_BYTES:
        raise ValueError("CSV exceeds the 64 MiB download limit")
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            total += len(chunk)
            if total > MAX_DOWNLOAD_CSV_BYTES:
                raise ValueError("CSV exceeds the 64 MiB download limit")
            digest.update(chunk)
    return digest.hexdigest()


