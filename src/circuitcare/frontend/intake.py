"""Safe, session-local intake validation and project draft construction."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import uuid
from pathlib import PurePath
from typing import Any, Iterable, Mapping
from xml.etree import ElementTree

from circuitcare.frontend.models import ImageRecord, ProjectContext
from circuitcare.frontend.photo_adapter import normalize_photo_metadata
from circuitcare.vision.image_loader import ImageLoadError, load_board_image
from circuitcare.schematic.identity import parse_sexpr

MAX_PHOTOS = 8
MAX_TOTAL_IMAGE_BYTES = 80 * 1024 * 1024
MAX_CIRCUIT_FILE_BYTES = 20 * 1024 * 1024
MAX_SYMPTOM_CHARS = 1000

SYMPTOM_CATEGORIES = {
    "power_failure": "전원 불량",
    "output_anomaly": "출력 이상",
    "overheating": "과열",
    "intermittent_operation": "간헐적 작동",
    "unknown": "증상 불명",
}

PHOTO_SIDES = {"front": "앞면", "back": "뒷면", "unknown": "미지정"}
CAPTURE_TYPES = {
    "full_view": "전체 사진", "close_up": "확대 사진", "oblique": "비스듬한 사진",
    "marking_detail": "부품 표기", "unknown": "미지정",
}
# Compatibility alias for the F1 widget/tests; new code must use PHOTO_SIDES/CAPTURE_TYPES.
IMAGE_ROLES = PHOTO_SIDES


def safe_filename(filename: str) -> str:
    return PurePath(filename.replace("\\", "/")).name[:255]


def normalize_symptom_description(value: str | None) -> str | None:
    normalized = re.sub(r"\s+", " ", value or "").strip()
    if len(normalized) > MAX_SYMPTOM_CHARS:
        raise ValueError(f"상세 증상은 {MAX_SYMPTOM_CHARS}자 이하로 입력해 주세요.")
    return normalized or None


def validate_symptom_category(category: str) -> str:
    if category not in SYMPTOM_CATEGORIES:
        raise ValueError("고장 증상을 선택해 주세요.")
    return category


def validate_image_upload(filename: str, data: bytes) -> tuple[str, int, int]:
    """Return digest and source dimensions after the existing bounded decoder validates it."""
    suffix = PurePath(filename).suffix.lower()
    allowed = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}
    if suffix not in allowed:
        raise ValueError("PNG, JPG, JPEG 또는 WebP 사진만 지원합니다.")
    loaded = load_board_image(data)
    if loaded.format != allowed[suffix]:
        raise ValueError("파일 확장자와 실제 이미지 형식이 다릅니다.")
    return loaded.sha256, loaded.original_size[0], loaded.original_size[1]


def add_image_batch(
    current: list[dict[str, Any]],
    uploads: Iterable[tuple[str, bytes]],
    *,
    max_photos: int = MAX_PHOTOS,
    max_total_bytes: int = MAX_TOTAL_IMAGE_BYTES,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Add valid non-duplicate images; per-file failures do not discard valid uploads."""
    images = [dict(item) for item in current]
    messages: list[str] = []
    existing_hashes = {item["sha256"] for item in images}
    total_bytes = sum(len(item["data"]) for item in images)
    for filename, data in uploads:
        try:
            digest, width, height = validate_image_upload(filename, data)
        except (ImageLoadError, ValueError) as exc:
            messages.append(f"{filename}: {exc}")
            continue
        if digest in existing_hashes:
            messages.append(f"{filename}: 중복 사진이라 건너뛰었습니다.")
            continue
        if len(images) >= max_photos:
            messages.append(f"{filename}: 사진은 최대 {max_photos}장까지 추가할 수 있습니다.")
            continue
        if total_bytes + len(data) > max_total_bytes:
            messages.append("전체 사진 용량 한도를 초과해 이후 사진을 추가하지 않았습니다.")
            break
        images.append({
            "image_id": str(uuid.uuid4()), "sha256": digest, "filename": safe_filename(filename),
            "data": bytes(data), "side": "unknown", "capture_type": "unknown",
            "media_type": f"image/{'jpeg' if suffix_is_jpeg(filename) else image_subtype(filename)}",
            "width": width, "height": height,
            "image_dimensions": (width, height), "image_format": "JPEG" if suffix_is_jpeg(filename) else image_subtype(filename).upper(),
        })
        existing_hashes.add(digest)
        total_bytes += len(data)
    return images, messages


def suffix_is_jpeg(filename: str) -> bool:
    return PurePath(filename).suffix.lower() in {".jpg", ".jpeg"}


def image_subtype(filename: str) -> str:
    suffix = PurePath(filename).suffix.lower().lstrip(".")
    return "png" if suffix == "png" else "webp"


def validate_circuit_upload(filename: str, data: bytes) -> dict[str, str]:
    """Validate only formats currently supported by the local codebase."""
    if not data or len(data) > MAX_CIRCUIT_FILE_BYTES:
        raise ValueError("설계 자료는 비어 있지 않은 20 MB 이하 파일이어야 합니다.")
    suffix = PurePath(filename).suffix.lower()
    if suffix == ".xml":
        from circuitcare.netlist import load_netlist

        try:
            with tempfile.NamedTemporaryFile(suffix=".xml") as temporary:
                temporary.write(data)
                temporary.flush()
                load_netlist(temporary.name)
        except (ElementTree.ParseError, ValueError, OSError) as exc:
            raise ValueError("유효한 KiCad XML Netlist가 아닙니다.") from exc
        kind = "kicad_xml_netlist"
    elif suffix == ".kicad_sch":
        try:
            root = parse_sexpr(data.decode("utf-8-sig"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("KiCad 회로도 파일을 읽을 수 없습니다.") from exc
        if not root or root[0] != "kicad_sch":
            raise ValueError("KiCad S-expression 회로도만 지원합니다.")
        kind = "kicad_schematic"
    else:
        raise ValueError("현재 지원하는 설계 자료는 KiCad XML Netlist(.xml)와 회로도(.kicad_sch)입니다.")
    return {
        "filename": safe_filename(filename),
        "content_sha256": hashlib.sha256(data).hexdigest(),
        "source_type": kind,
        "analysis_status": "NOT_RUN",
    }


def create_project_context(
    *,
    images: Iterable[Mapping[str, Any]],
    symptom_category: str,
    symptom_description: str | None,
    circuit_source: Mapping[str, Any] | None = None,
    project_id: str | None = None,
    board_id: str | None = None,
) -> ProjectContext:
    image_items = list(images)
    if not image_items:
        raise ValueError("PCB 사진을 한 장 이상 추가해 주세요.")
    category = validate_symptom_category(symptom_category)
    description = normalize_symptom_description(symptom_description)
    records: list[ImageRecord] = []
    fingerprint_images = []
    for item in image_items:
        normalized, _ = normalize_photo_metadata(item)
        if not normalized.image_id or not normalized.content_sha256:
            raise ValueError("사진 식별자와 콘텐츠 SHA-256이 필요합니다.")
        record = normalized
        records.append(record)
        fingerprint_images.append({
            "sha256": record.content_sha256,
            "side": record.side,
            "capture_type": record.capture_type,
        })
    source = dict(circuit_source) if circuit_source else None
    source_fingerprint = (
        {"content_sha256": source.get("content_sha256"), "source_type": source.get("source_type")}
        if source else None
    )
    fingerprint_payload = {
        "images": fingerprint_images,
        "symptom_category": category,
        "symptom_description": description,
        "circuit_source": source_fingerprint,
    }
    fingerprint = hashlib.sha256(json.dumps(
        fingerprint_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return ProjectContext(
        project_id=project_id or str(uuid.uuid4()), board_id=board_id or str(uuid.uuid4()),
        image_records=tuple(records), symptom_category=category,
        symptom_description=description, circuit_source=source,
        input_fingerprint=fingerprint,
    )


def create_preview_context(
    *, symptom_category: str, symptom_description: str | None,
    circuit_source: Mapping[str, Any] | None = None,
    project_id: str | None = None, board_id: str | None = None,
) -> ProjectContext:
    """Create an explicitly photo-free UI preview, never a diagnostic project."""
    category = validate_symptom_category(symptom_category)
    description = normalize_symptom_description(symptom_description)
    source = dict(circuit_source) if circuit_source else None
    payload = {
        "mode": "PREVIEW_ONLY", "images": [], "symptom_category": category,
        "symptom_description": description,
        "circuit_source": ({
            "content_sha256": source.get("content_sha256"),
            "source_type": source.get("source_type"),
        } if source else None),
    }
    fingerprint = hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return ProjectContext(
        project_id=project_id or str(uuid.uuid4()), board_id=board_id or str(uuid.uuid4()),
        image_records=(), symptom_category=category, symptom_description=description,
        circuit_source=source, input_fingerprint=fingerprint,
    )
