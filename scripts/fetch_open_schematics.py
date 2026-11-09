#!/usr/bin/env python3
"""Fetch a bounded KiCad-source subset from the Hugging Face rows API.

Only the schematic text and source project name are saved. Image and PCB
payloads are not downloaded. Use --limit to keep the first acquisition bounded.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sys
import time
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
API = "https://datasets-server.huggingface.co/rows"
DATASET = "bshada/open-schematics"
LICENSE = "CC-BY-4.0"


def fetch_page(offset: int, length: int) -> dict:
    query = urlencode({"dataset": DATASET, "config": "default", "split": "train",
                       "offset": offset, "length": length})
    request = Request(f"{API}?{query}", headers={"User-Agent": "CircuitCare-AI dataset preparation"})
    for attempt in range(6):
        try:
            with urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 5:
                raise
            retry_after = exc.headers.get("Retry-After", "")
            try:
                delay = max(1.0, min(float(retry_after), 120.0)) if retry_after else min(60.0, 2.0 ** (attempt + 1))
            except ValueError:
                delay = min(60.0, 2.0 ** (attempt + 1))
            print(f"HTTP {exc.code}; retrying request at row {offset} after {delay:.0f}s", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"retry loop ended unexpectedly at row {offset}")


def safe_project_parts(name: str) -> tuple[str, ...]:
    path = PurePosixPath(name.strip().strip("/"))
    parts = tuple(part for part in path.parts if part not in {"", ".", ".."})
    return parts if parts else ("unknown-project",)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/external/open-schematics")
    parser.add_argument("--offset", type=int, default=0, help="First source row (0-based)")
    parser.add_argument("--limit", type=int, default=100, help="Rows to fetch; use 0 for all remaining rows")
    parser.add_argument("--page-size", type=int, default=25, help="Rows per API request (1-100)")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent page requests (1-8)")
    parser.add_argument("--pause", type=float, default=0.2, help="Seconds between API pages")
    args = parser.parse_args()
    if args.offset < 0 or args.limit < 0 or not 1 <= args.page_size <= 100 or not 1 <= args.workers <= 8:
        parser.error("offset/limit must be non-negative, page-size 1-100, and workers 1-8")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "source_manifest.jsonl"
    remaining = args.limit
    offset = args.offset
    fetched = saved = reused = 0
    manifest_mode = "a" if manifest_path.exists() else "w"
    existing_rows = sum(1 for _ in manifest_path.open("r", encoding="utf-8")) if manifest_path.exists() else 0
    try:
        with manifest_path.open(manifest_mode, encoding="utf-8") as manifest:
            while args.limit == 0 or remaining > 0:
                batch = []
                slots = args.workers
                if args.limit:
                    slots = min(slots, (remaining + args.page_size - 1) // args.page_size)
                for _ in range(slots):
                    length = args.page_size
                    batch.append((offset, length))
                    offset += length
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = [(page_offset, length, pool.submit(fetch_page, page_offset, length))
                               for page_offset, length in batch]
                    pages = [(page_offset, length, future.result())
                             for page_offset, length, future in futures]
                for page_offset, length, page in pages:
                    rows = page.get("rows", [])
                    if not rows:
                        offset = page_offset
                        break
                    for row_index, wrapper in enumerate(rows):
                        row = wrapper.get("row", {})
                        schematic = row.get("schematic")
                        name = str(row.get("name") or "unknown-project")
                        if not isinstance(schematic, str) or not schematic.lstrip().startswith("("):
                            continue
                        digest = hashlib.sha256(schematic.encode("utf-8")).hexdigest()
                        project_parts = safe_project_parts(name)
                        relative = Path(*project_parts) / f"{digest[:16]}.kicad_sch"
                        target = output / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if target.exists():
                            reused += 1
                        else:
                            target.write_text(schematic, encoding="utf-8", newline="")
                            saved += 1
                        record = {
                            "source_row_offset": page_offset + row_index,
                            "project_group": "/".join(project_parts),
                            "source_name": name,
                            "schematic_path": relative.as_posix(),
                            "sha256": digest,
                            "source_dataset": DATASET,
                            "source_url": f"https://huggingface.co/datasets/{DATASET}",
                            "license": LICENSE,
                        }
                        manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fetched += 1
                        if args.limit:
                            remaining -= 1
                            if remaining == 0:
                                break
                    if args.limit and remaining == 0:
                        break
                manifest.flush()
                print(f"Fetched {fetched} rows in this run; saved {saved}; next row {offset}", flush=True)
                if args.limit and remaining == 0:
                    break
                if any(len(page.get("rows", [])) < length for _, length, page in pages):
                    break
                if args.pause > 0:
                    time.sleep(args.pause)
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        print(f"Fetch stopped at source row {offset}: {exc}", file=sys.stderr)
        return 1
    summary = {"dataset": DATASET, "start_offset": args.offset, "rows_read_this_run": fetched,
               "rows_in_manifest": existing_rows + fetched, "files_saved_this_run": saved,
               "files_reused_this_run": reused, "manifest": str(manifest_path)}
    (output / "fetch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
