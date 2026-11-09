"""Private subprocess entry point for CPU- and memory-bounded local PDF parsing."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from .pdf_loader import (
    MAX_PDF_BYTES,
    MAX_PDF_PAGES,
    _extract_pdf_tables_local,
    _load_pdf_local,
)

MAX_RESULT_BYTES = 16 * 1024 * 1024


def _atomic_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_RESULT_BYTES:
        raise ValueError("PDF worker result exceeds its output size limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("document", "tables"))
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--filename", default="datasheet.pdf")
    parser.add_argument("--max-bytes", type=int, default=MAX_PDF_BYTES)
    parser.add_argument("--max-pages", type=int, default=MAX_PDF_PAGES)
    args = parser.parse_args()
    try:
        data = args.input_path.read_bytes()
        if len(data) > args.max_bytes:
            raise ValueError("PDF exceeds the worker input size limit")
        if args.operation == "document":
            result = _load_pdf_local(data, args.filename, max_bytes=args.max_bytes, max_pages=args.max_pages)
        else:
            result = _extract_pdf_tables_local(data, max_pages=args.max_pages)
        _atomic_json(args.output_path, {"status": "OK", "result": result})
        return 0
    except Exception as exc:  # noqa: BLE001 - serialize any parser failure across the worker boundary.
        try:
            _atomic_json(args.output_path, {"status": "ERROR", "error_type": type(exc).__name__,
                                            "error": str(exc)[:1000]})
        except Exception:  # noqa: BLE001,S110 - the parent classifies a missing result envelope.
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
