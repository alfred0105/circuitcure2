#!/usr/bin/env python3
"""Fetch the individually verified photo + CAD pairs from a supplied CSV."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/external/pcb-photo-cad-2026-11-09"
UA = "CircuitCare-PCB-Photo-CAD-Acquisition/1.0"
MAX_BYTES = 100 * 1024 * 1024


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "asset"


def raw_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.lower() == "github.com" and "/blob/" in parsed.path:
        before, after = parsed.path.split("/blob/", 1)
        return urllib.parse.urlunparse(("https", "raw.githubusercontent.com", before + "/" + after,
                                       "", parsed.query, ""))
    return url


def image_size(data: bytes, mime: str):
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data.startswith(b"\xff\xd8"):
        i = 2
        while i + 4 < len(data):
            if data[i] != 0xFF:
                i += 1; continue
            while i < len(data) and data[i] == 0xFF: i += 1
            if i >= len(data): break
            marker = data[i]; i += 1
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7: continue
            if i + 2 > len(data): break
            size = int.from_bytes(data[i:i+2], "big")
            if size < 2: break
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                          0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF) and i + 7 <= len(data):
                return int.from_bytes(data[i+5:i+7], "big"), int.from_bytes(data[i+3:i+5], "big")
            i += size
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP" and len(data) >= 30:
        kind = data[12:16]
        if kind == b"VP8X":
            return 1 + int.from_bytes(data[24:27], "little"), 1 + int.from_bytes(data[27:30], "little")
        if kind == b"VP8 " and len(data) >= 30:
            return int.from_bytes(data[26:28], "little") & 0x3FFF, int.from_bytes(data[28:30], "little") & 0x3FFF
    return None, None


def download(url: str, dest: Path, expected: str):
    url = raw_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as response:
        mime = response.headers.get_content_type()
        final_url = response.geturl()
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > MAX_BYTES:
            raise ValueError(f"asset exceeds {MAX_BYTES} byte limit")
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f"asset exceeds {MAX_BYTES} byte limit")
    if expected == "photo":
        if not (data.startswith(b"\xff\xd8") or data.startswith(b"\x89PNG\r\n\x1a\n")
                or (data.startswith(b"RIFF") and data[8:12] == b"WEBP")):
            raise ValueError(f"photo URL returned non-image bytes ({mime})")
        width, height = image_size(data, mime)
        if not width or not height:
            raise ValueError("could not parse photo dimensions")
    else:
        if len(data) < 100 or b"<html" in data[:2048].lower() or b"<!doctype html" in data[:2048].lower():
            raise ValueError("CAD URL returned an empty or HTML document")
        width = height = None
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".part")
    temp.write_bytes(data)
    temp.replace(dest)
    return {"status": "downloaded", "source_url": url, "final_url": final_url,
        "path": str(dest.relative_to(ROOT)).replace("\\", "/"), "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(), "content_type": mime,
        "width": width, "height": height}


def fetch_row(row):
    repo_key = safe_name(row["repository"].replace("/", "__"))
    cad_name = safe_name(Path(row["cad_path"].replace("\\", "/")).name)
    photo_ext = Path(urllib.parse.urlparse(row["photo_url"]).path).suffix.lower()
    if photo_ext not in (".jpg", ".jpeg", ".png", ".webp"): photo_ext = ".img"
    record = {"repository": row["repository"], "tier": row["tier"],
        "cad_path": row["cad_path"], "repo_url": row["repo_url"], "product_page": row["product_page"],
        "revision_match": row["revision_match"], "notes": row["notes"], "assets": {}}
    for kind, url, filename in (
        ("cad", row["cad_url"], cad_name),
        ("photo", row["photo_url"], "board_photo" + photo_ext),
    ):
        dest = DEST / "assets" / repo_key / filename
        try:
            record["assets"][kind] = download(url, dest, kind)
        except Exception as exc:
            record["assets"][kind] = {"status": "failed", "source_url": url,
                                       "error": f"{type(exc).__name__}: {exc}"}
    record["pair_status"] = "verified_downloads" if all(
        record["assets"][k]["status"] == "downloaded" for k in ("cad", "photo")) else "incomplete"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8-sig", newline="")))
    verified = [r for r in rows if r.get("cad_file_verified", "").strip().lower() == "yes"
                and r.get("photo_in_readme_verified", "").strip().lower() == "yes"
                and r.get("cad_url", "").strip() and r.get("photo_url", "").strip()]
    DEST.mkdir(parents=True, exist_ok=True)
    manifest_path = DEST / "manifest.json"
    manifest = {"source_csv": str(args.csv), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selection": "Only rows marked yes for CAD and README photo verification, with both direct URLs.",
        "resolution_note": "Image dimensions are read from file headers; high resolution is not assumed from the source listing.",
        "source_rows": len(rows), "eligible_rows": len(verified), "assets": []}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(fetch_row, row) for row in verified]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record = future.result()
            manifest["assets"].append(record)
            manifest["assets"].sort(key=lambda r: r["repository"])
            manifest["downloaded_pairs"] = sum(x["pair_status"] == "verified_downloads" for x in manifest["assets"])
            manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{done}/{len(futures)}] {record['repository']}: {record['pair_status']}", flush=True)
    photos = [r["assets"]["photo"] for r in manifest["assets"] if r["assets"]["photo"]["status"] == "downloaded"]
    manifest["photo_resolution_summary"] = {
        "downloaded": len(photos),
        "at_least_1280px_wide": sum((p.get("width") or 0) >= 1280 for p in photos),
        "at_least_1920px_wide": sum((p.get("width") or 0) >= 1920 for p in photos),
        "max_width": max((p.get("width") or 0 for p in photos), default=0),
        "max_height": max((p.get("height") or 0 for p in photos), default=0),
    }
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("source_rows", "eligible_rows", "downloaded_pairs", "photo_resolution_summary")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
