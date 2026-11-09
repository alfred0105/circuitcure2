#!/usr/bin/env python3
"""Download selected CAD/photo pairs discovered in public source repositories."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from acquire_pcb_photo_cad_sources import DEST, image_size, safe_name

ROOT = Path(__file__).resolve().parents[1]
UA = "CircuitCare-PCB-Photo-CAD-Acquisition/1.0"
MAX_BYTES = 100 * 1024 * 1024
BRANCHES = ("main", "master", "develop")


def fetch(url: str, dest: Path, kind: str):
    errors = []
    data = None
    final_url = None
    mime = None
    for attempt in range(3):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                mime = response.headers.get_content_type()
                final_url = response.geturl()
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > MAX_BYTES:
                    raise ValueError(f"asset exceeds {MAX_BYTES} byte limit")
                data = response.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise ValueError(f"asset exceeds {MAX_BYTES} byte limit")
            break
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    if data is None:
        raise RuntimeError("; ".join(errors))
    width = height = None
    if kind == "photo":
        if not (data.startswith(b"\xff\xd8") or data.startswith(b"\x89PNG\r\n\x1a\n")
                or (data.startswith(b"RIFF") and data[8:12] == b"WEBP")):
            raise ValueError(f"photo URL returned non-image bytes ({mime})")
        width, height = image_size(data, mime)
        if not width or not height:
            raise ValueError("could not parse photo dimensions")
    elif len(data) < 100 or b"<html" in data[:2048].lower():
        raise ValueError("CAD URL returned an empty or HTML document")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".part")
    temp.write_bytes(data)
    temp.replace(dest)
    return {"status": "downloaded", "source_url": url, "final_url": final_url,
            "path": str(dest.relative_to(ROOT)).replace("\\", "/"), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "content_type": mime,
            "width": width, "height": height}


def download_asset(repo: str, source: dict, kind: str, dest_dir: Path):
    path = source["path"]
    encoded_path = urllib.parse.quote(path, safe="/")
    last_error = None
    for branch in BRANCHES:
        url = f"https://raw.githubusercontent.com/{repo}/{branch}/{encoded_path}"
        try:
            result = fetch(url, dest_dir / Path(path).name, kind)
            result["git_blob_sha"] = source.get("sha")
            result["repository_path"] = path
            return result
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (404, 400):
                raise
        except Exception as exc:
            last_error = exc
            if "HTTP Error 404" not in str(exc):
                # A failed photo may be a transient error; retry branches only for not found.
                if branch == BRANCHES[-1]:
                    raise
    raise RuntimeError(f"No supported default branch found for {repo}/{path}: {last_error}")


def fetch_record(item: dict):
    repo_dir = DEST / "assets" / "repository_discovered" / safe_name(item["repo"].replace("/", "__"))
    record = {"repository": item["repo"], "source_kind": item["source_kind"],
              "revision": item.get("revision"), "revision_match": item["source_kind"] == "version_matched_soldered_output",
              "repo_url": f"https://github.com/{item['repo']}", "assets": {}}
    for name in ("board", "photo", "schematic"):
        source = item.get(name)
        if not source:
            continue
        try:
            record["assets"][name] = download_asset(item["repo"], source, "photo" if name == "photo" else "cad", repo_dir / name)
        except Exception as exc:
            record["assets"][name] = {"status": "failed", "repository_path": source.get("path"),
                                       "error": f"{type(exc).__name__}: {exc}"}
    record["pair_status"] = "downloaded" if all(record["assets"].get(k, {}).get("status") == "downloaded"
                                                  for k in ("board", "photo")) else "incomplete"
    return record


def main():
    candidate_path = DEST / "repository_candidates.json"
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    manifest_path = DEST / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"assets": []}
    manifest["repository_discovery"] = {
        "selection": "One largest non-panel board file and one largest candidate product image per Adafruit repository; latest shared CAD/OUTPUTS version per Soldered repository.",
        "candidate_file": str(candidate_path.relative_to(ROOT)).replace("\\", "/"),
        "revision_policy": "Soldered CAD and OUTPUTS use the same version directory; Adafruit revision matching is not verified.",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "assets": []}
    items = candidates["soldered"] + candidates["adafruit"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(fetch_record, item) for item in items]
        for n, done in enumerate(concurrent.futures.as_completed(futures), 1):
            record = done.result()
            manifest["repository_discovery"]["assets"].append(record)
            manifest["repository_discovery"]["assets"].sort(key=lambda x: x["repository"])
            manifest["repository_discovery"]["downloaded_pairs"] = sum(
                a["pair_status"] == "downloaded" for a in manifest["repository_discovery"]["assets"])
            manifest["repository_discovery"]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{n}/{len(items)}] {record['repository']}: {record['pair_status']}", flush=True)
    photos = [a["assets"]["photo"] for a in manifest["repository_discovery"]["assets"]
              if a["assets"].get("photo", {}).get("status") == "downloaded"]
    manifest["repository_discovery"]["photo_resolution_summary"] = {
        "downloaded": len(photos), "at_least_1280px_wide": sum((p.get("width") or 0) >= 1280 for p in photos),
        "at_least_1920px_wide": sum((p.get("width") or 0) >= 1920 for p in photos),
        "max_width": max((p.get("width") or 0 for p in photos), default=0),
        "max_height": max((p.get("height") or 0 for p in photos), default=0)}
    manifest["repository_discovery"]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_pairs": len(items),
                      "downloaded_pairs": manifest["repository_discovery"]["downloaded_pairs"],
                      "photo_resolution_summary": manifest["repository_discovery"]["photo_resolution_summary"]}, indent=2))


if __name__ == "__main__":
    main()
