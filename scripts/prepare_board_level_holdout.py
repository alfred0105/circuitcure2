"""Fetch a board-level PCB DSLR holdout without downloading full archive parts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import struct
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import zipfile
import zlib
import math

ZENODO_RECORD = "https://zenodo.org/api/records/3886553"
DATASET_PAGE = "https://cvl.tuwien.ac.at/research/cvl-databases/pcb-dslr-dataset/"


class _RangeZip:
    """Small random-access reader backed by HTTP Range requests."""

    def __init__(self, url: str, size: int, block_size: int = 4 * 1024 * 1024):
        self.url, self.size, self.block_size, self.pos = url, size, block_size, 0
        self.cache: dict[int, bytes] = {}

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        self.pos = offset if whence == os.SEEK_SET else self.pos + offset if whence == os.SEEK_CUR else self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read_at(self, offset: int, count: int) -> bytes:
        if count <= 0:
            return b""
        end = min(self.size, offset + count) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={offset}-{end}"})
        with urllib.request.urlopen(request, timeout=90) as response:
            if response.status != 206:
                raise OSError(f"Zenodo did not honor byte range: HTTP {response.status}")
            return response.read()

    def read(self, count: int = -1) -> bytes:
        if count < 0:
            count = self.size - self.pos
        count = min(count, self.size - self.pos)
        output = bytearray()
        while count > 0:
            block_index = self.pos // self.block_size
            block_start = block_index * self.block_size
            if block_index not in self.cache:
                end = min(self.size, block_start + self.block_size) - 1
                request = urllib.request.Request(self.url, headers={"Range": f"bytes={block_start}-{end}"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206:
                        raise OSError(f"Zenodo did not honor byte range: HTTP {response.status}")
                    self.cache[block_index] = response.read()
            block = self.cache[block_index]
            offset = self.pos - block_start
            part = block[offset:offset + count]
            if not part:
                break
            output.extend(part)
            self.pos += len(part)
            count -= len(part)
        return bytes(output)


def _read_member(reader: _RangeZip, info: zipfile.ZipInfo) -> bytes:
    local = reader.read_at(info.header_offset, 30)
    if len(local) != 30:
        raise OSError(f"Truncated ZIP header: {info.filename}")
    fields = struct.unpack("<IHHHHHIIIHH", local)
    if fields[0] != 0x04034B50:
        raise OSError(f"Invalid ZIP entry: {info.filename}")
    name_length, extra_length = fields[-2:]
    compressed = reader.read_at(info.header_offset + 30 + name_length + extra_length, info.compress_size)
    if len(compressed) != info.compress_size:
        raise OSError(f"Truncated ZIP data: {info.filename}")
    if info.compress_type == zipfile.ZIP_STORED:
        raw = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        raw = zlib.decompress(compressed, -15)
    else:
        raise OSError(f"Unsupported ZIP compression {info.compress_type}: {info.filename}")
    if len(raw) != info.file_size or (zlib.crc32(raw) & 0xFFFFFFFF) != info.CRC:
        raise OSError(f"ZIP integrity check failed: {info.filename}")
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare 50 complete PCB-board photos with IC box labels.")
    parser.add_argument("--destination", type=Path,
                        default=Path.home() / ".local/share/circuitcare-ai/external_datasets/pcb-dslr")
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.count <= 165:
        parser.error("--count must be 1..165 unique PCB boards")

    record = json.load(urllib.request.urlopen(ZENODO_RECORD, timeout=45))
    metadata = record["metadata"]
    files = {item["key"]: item for item in record["files"]}
    parts = sorted((name for name in files if name.startswith("cvl_pcb_dslr_") and name.endswith(".zip")),
                   key=lambda name: int(name.split("_")[-1].split(".")[0]))
    selected: list[tuple[str, zipfile.ZipInfo, zipfile.ZipInfo, str]] = []
    for part in parts:
        item = files[part]
        remote = _RangeZip(item["links"]["self"], item["size"])
        with zipfile.ZipFile(remote) as archive:
            by_name = {entry.filename: entry for entry in archive.infolist()}
            board_ids = sorted({name.split("/")[0] for name in by_name
                                if name.startswith("pcb") and name.endswith("-annot.txt")},
                               key=lambda value: int(value[3:]))
            for board_id in board_ids:
                if any(row[3] == board_id for row in selected):
                    continue
                prefix = f"{board_id}/"
                annotations = sorted((name for name in by_name if name.startswith(prefix) and name.endswith("-annot.txt")),
                                     key=lambda value: int(Path(value).stem[3:].split("-")[0]))
                chosen = None
                for annotation_name in annotations:
                    label_info = by_name[annotation_name]
                    labels = _read_member(remote, label_info).decode("utf-8", "replace").splitlines()
                    if not any(line.strip() for line in labels):
                        continue
                    recording = Path(annotation_name).stem.removesuffix("-annot")
                    image_name = f"{prefix}{recording}.jpg"
                    if image_name in by_name:
                        chosen = (by_name[image_name], label_info)
                        break
                if chosen:
                    selected.append((part, chosen[0], chosen[1], board_id))
                    if len(selected) >= args.count:
                        break
        print(f"선별 진행: {len(selected)}/{args.count}개 서로 다른 기판", flush=True)
        if len(selected) >= args.count:
            break

    if len(selected) != args.count:
        raise RuntimeError(f"요청한 {args.count}개 중 라벨 있는 전체 기판 사진 {len(selected)}개만 확보 가능")

    args.destination.mkdir(parents=True, exist_ok=True)
    image_dir = args.destination / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    by_part = {item["key"]: item for item in record["files"]}
    def fetch(row):
        part, image_info, label_info, board_id = row
        image_path = image_dir / f"{board_id}-{Path(image_info.filename).name}"
        label_path = image_dir / f"{board_id}-{Path(label_info.filename).name}"
        if image_path.is_file() and label_path.is_file():
            image_bytes, label_bytes = image_path.read_bytes(), label_path.read_bytes()
        else:
            item = by_part[part]
            reader = _RangeZip(item["links"]["self"], item["size"])
            image_bytes = _read_member(reader, image_info)
            label_bytes = _read_member(reader, label_info)
            image_path.write_bytes(image_bytes)
            label_path.write_bytes(label_bytes)
        image_sha = hashlib.sha256(image_bytes).hexdigest()
        from PIL import Image
        with Image.open(io.BytesIO(image_bytes)) as source_image:
            rgb = source_image.convert("RGB")
            pixel_fingerprint = hashlib.sha256(
                f"{rgb.width}x{rgb.height}:RGB:".encode() + rgb.tobytes()
            ).hexdigest()
        components = []
        # Source annotations are rotated rectangles: cx cy width height angle (pixels).
        for line in label_bytes.decode("utf-8", "replace").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            cx, cy, width, height, angle = (float(value) for value in fields[:5])
            radians = math.radians(angle)
            box_width = abs(width * math.cos(radians)) + abs(height * math.sin(radians))
            box_height = abs(width * math.sin(radians)) + abs(height * math.cos(radians))
            components.append({"component_type": "ic", "bbox_xyxy": [cx - box_width / 2, cy - box_height / 2,
                                                                           cx + box_width / 2, cy + box_height / 2],
                               "source_label": "IC bounding box; axis-aligned bounds of source rectangle"})
        return {
            "case_id": f"pcb-dslr-{board_id}", "board_id": f"tuwien-{board_id}",
            "case_kind": "REAL_UNKNOWN", "photo_scope": "BOARD_LEVEL",
            "dataset_name": "PCB DSLR Dataset", "dataset_url": "https://doi.org/10.5281/zenodo.3886553",
            "photo_source": "TU Wien Computer Vision Lab, PCB DSLR Dataset; one full-board DSLR photo per unique PCB; not a component crop/tile",
            "label_source": "Original per-PCB IC annotation text; bounding boxes only; labels are not independent adjudication",
            "license": "CC BY 4.0; cite Pramerdorfer & Kampel, MVA 2015",
            "inference_input": {"photo_path": str(image_path), "skip_qwen_review": True},
            "reference": {"adjudication_status": "UNKNOWN", "gold_components_status": "UNKNOWN",
                          "source_label_proxy_components": components,
                          "evidence_sources": ["PCB DSLR Dataset original annotation"]},
            "source_image_sha256": image_sha, "training_exclusion_pixel_sha256": pixel_fingerprint,
            "source_board_id": board_id,
            "annotation_path": str(label_path), "source_archive": part,
        }

    with ThreadPoolExecutor(max_workers=6, thread_name_prefix="pcb-holdout-download") as pool:
        for index, case in enumerate(pool.map(fetch, selected), 1):
            cases.append(case)
            if index % 10 == 0 or index == args.count:
                print(f"다운로드 진행: {index}/{args.count}장 전체 기판", flush=True)
    manifest = {"schema_version": 1, "dataset_name": "PCB DSLR Dataset", "dataset_url": "https://doi.org/10.5281/zenodo.3886553",
                "dataset_page": DATASET_PAGE, "license": record["metadata"].get("license", {}).get("id", "unknown"),
                "citation": "C. Pramerdorfer and M. Kampel, A Dataset for Computer-Vision-Based PCB Analysis, MVA 2015, doi:10.1109/MVA.2015.7153209",
                "selection": "one original full-board photo with non-empty IC labels per unique PCB board ID; no isolated components or tile crops",
                "training_use": "HOLDOUT_ONLY; excluded from training by source image hash", "cases": cases}
    manifest_path = args.destination / f"holdout-{args.count}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "count": len(cases), "distinct_boards": len({row['board_id'] for row in cases}),
                      "source": manifest["dataset_url"], "license": manifest["license"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
