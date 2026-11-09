"""Optional, validated boundary to the Phase 21.2 multi-image component-map service."""

from __future__ import annotations

import base64
import binascii
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib
import io
import json
import logging
import math
import os
from pathlib import Path
from collections.abc import Mapping, MutableMapping
from typing import Any, Protocol

from PIL import Image

from circuitcare.frontend.photo_adapter import ComponentMapPhotoInput

SCHEMA_VERSION = "circuitcare.pcb-component-map.v1"
COMPONENT_MAP_STATE_KEY = "circuitcare_v3_component_map_entry"
LEGACY_COMPONENT_MAP_KEYS = (
    "circuitcare_v3_component_map_result", "circuitcare_v3_component_map_cache_key",
    "circuitcare_v3_photo_map", "circuitcare_v3_selected_canonical_image",
    "circuitcare_v3_registration_status", "circuitcare_v3_processing_error",
    "circuitcare_v3_active_side", "circuitcare_v3_view_mode", "circuitcare_v3_failed_artifacts",
    "circuitcare_v2_view_transform", "circuitcare_v2_active_photo_id", "circuitcare_v2_view_zoom",
    "circuitcare_v4_selection_map_key", "circuitcare_v4_current_components",
    "circuitcare_v4_detection_status",
)
LOGGER = logging.getLogger(__name__)
MAX_CANONICAL_BYTES = 24 * 1024 * 1024
MAX_CANONICAL_PIXELS = 8_000_000
MAX_CANONICAL_EDGE = 4000
REGISTRATION_LABELS = {
    "REFERENCE": "기준 사진",
    "REGISTERED_CANDIDATE": "정합 후보 · 미검증",
    "UNREGISTERED": "정합 실패",
    "DUPLICATE_SKIPPED": "중복 제외",
}


@lru_cache(maxsize=8)
def _model_sha256(path: str, size: int, modified_ns: int) -> str:
    """Hash a model once per file version; Streamlit reruns must stay cheap."""
    del size, modified_ns  # Metadata intentionally participates in the cache key.
    digest = hashlib.sha256()
    with Path(path).open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ComponentMapProvider(Protocol):
    def build_component_map(self, *, board_id: str, photos: list[dict[str, Any]], previous_map: dict | None = None, options: dict | None = None) -> dict:
        """Build the backend's component map for one board."""


@dataclass(frozen=True)
class PhotoMapEntry:
    project_id: str
    board_id: str
    frontend_photo_id: str
    backend_photo_id: str | None
    content_sha256: str
    board_side: str
    capture_role: str
    registration_status: str
    mapping_status: str
    reason: str | None = None


@dataclass(frozen=True)
class CanonicalImage:
    image: Image.Image
    frame_id: str
    width: int
    height: int
    reference_backend_photo_id: str | None
    registration_status: str


def optional_backend_provider() -> ComponentMapProvider | None:
    """Load the backend lazily when frontend and backend are installed together."""
    try:
        module = importlib.import_module("circuitcare.vision.reconstruction.multi_image_component_map")
        service_type = getattr(module, "MultiImageComponentMapService")
        photo_type = getattr(module, "PhotoInput")
    except (ImportError, AttributeError):
        return None

    class LocalServiceProvider:
        def __init__(self) -> None:
            configured = os.environ.get("CIRCUITCARE_PCB_MODEL_PATH")
            default_path = Path(__file__).resolve().parents[3] / "models" / "pcb_component_detector.onnx"
            self.model_path = Path(configured).expanduser() if configured else default_path if default_path.is_file() else None
            model_digest = None
            if self.model_path and self.model_path.is_file():
                metadata = self.model_path.stat()
                model_digest = _model_sha256(str(self.model_path.resolve()), metadata.st_size, metadata.st_mtime_ns)
            self.cache_options = {
                "model_path": str(self.model_path) if self.model_path else None,
                "model_sha256": model_digest,
            }

        def build_component_map(self, *, board_id: str, photos: list[dict[str, Any]], previous_map: dict | None = None, options: dict | None = None) -> dict:
            service = service_type()
            inputs = [photo_type(data=item["data"], side=item["side"], role=item["role"], name=item["name"])
                      for item in photos]
            session = service.ingest(board_id, inputs)
            actual_options = options or self.cache_options
            model_path = actual_options.get("model_path")
            return service.build_map(session, model_path=model_path, previous_component_map=previous_map)

    return LocalServiceProvider()


def validate_component_map(value: Any, *, board_id: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Component Map schema 불일치")
    if value.get("board_id") != board_id or not isinstance(value.get("photos"), list) or not isinstance(value.get("sides"), dict):
        raise ValueError("Component Map 보드 또는 필수 필드 불일치")
    for side, entry in value["sides"].items():
        if side not in {"TOP", "BOTTOM"} or not isinstance(entry, dict):
            raise ValueError("Component Map 촬영면 형식 불일치")
        canonical = entry.get("canonical_image")
        if canonical is not None and (not isinstance(canonical, dict) or not isinstance(canonical.get("image_id"), str)
                                      or not canonical.get("image_id")):
            raise ValueError("Canonical Image 메타데이터 불일치")
        for registration in entry.get("registrations", []):
            if not isinstance(registration, dict) or registration.get("status") not in {
                "REFERENCE", "REGISTERED_CANDIDATE", "UNREGISTERED", "DUPLICATE_SKIPPED",
            }:
                raise ValueError("사진 정합 상태 형식 불일치")
            matrix = registration.get("transform_moving_to_canonical")
            if matrix is not None and (not isinstance(matrix, list) or len(matrix) != 3
                    or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
                    or any(not isinstance(number, (int, float)) or not math.isfinite(number)
                           for row in matrix for number in row)):
                raise ValueError("정합 변환 행렬이 유효하지 않습니다.")
    return value


def map_backend_photo_ids(*, project_id: str, board_id: str, inputs: list[ComponentMapPhotoInput], result: dict[str, Any]) -> list[PhotoMapEntry]:
    """Map only unique (side, role, digest) tuples; names and hashes alone are insufficient."""
    backend = result.get("photos", [])
    mapped: list[PhotoMapEntry] = []
    tuples = Counter((item.side, item.role, item.content_sha256) for item in inputs)
    for item in inputs:
        key = (item.side, item.role, item.content_sha256)
        matches = [row for row in backend if isinstance(row, dict) and row.get("side") == item.side
                   and row.get("role") == item.role and row.get("sha256") == item.content_sha256]
        if tuples[key] != 1 or len(matches) != 1:
            mapped.append(PhotoMapEntry(project_id, board_id, item.frontend_photo_id, None, item.content_sha256,
                                        item.side, item.role, "UNKNOWN", "UNCONFIRMED", "사진 식별이 모호합니다."))
            continue
        row = matches[0]
        if not isinstance(row.get("photo_id"), str) or not row["photo_id"]:
            mapped.append(PhotoMapEntry(project_id, board_id, item.frontend_photo_id, None, item.content_sha256,
                                        item.side, item.role, "UNKNOWN", "UNCONFIRMED", "Backend 사진 ID가 없습니다."))
            continue
        reg = next((registration for side_map in result.get("sides", {}).values()
                    for registration in side_map.get("registrations", [])
                    if registration.get("photo_id") == row.get("photo_id")), {})
        status = str(reg.get("status", "UNKNOWN"))
        mapped.append(PhotoMapEntry(project_id, board_id, item.frontend_photo_id, row.get("photo_id"),
                                    item.content_sha256, item.side, item.role, status, "MAPPED"))
    return mapped


def decode_canonical_image(canonical: dict[str, Any]) -> CanonicalImage:
    if (not isinstance(canonical, dict) or canonical.get("coordinate_frame") != "MOSAIC_PIXEL"
            or not isinstance(canonical.get("image_id"), str) or not canonical["image_id"]):
        raise ValueError("Canonical Image 프레임 정보가 유효하지 않습니다.")
    encoded = canonical.get("image_png_base64")
    if not isinstance(encoded, str) or len(encoded) > (MAX_CANONICAL_BYTES * 4 // 3 + 16):
        raise ValueError("통합 이미지 크기 제한 초과 또는 데이터 누락")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("통합 이미지 데이터가 손상되었습니다.") from exc
    if len(raw) > MAX_CANONICAL_BYTES:
        raise ValueError("통합 이미지 크기 제한 초과")
    with Image.open(io.BytesIO(raw)) as probe:
        if probe.format != "PNG" or probe.width < 1 or probe.height < 1 or probe.width > MAX_CANONICAL_EDGE or probe.height > MAX_CANONICAL_EDGE or probe.width * probe.height > MAX_CANONICAL_PIXELS:
            raise ValueError("통합 이미지 해상도 제한 또는 포맷 불일치")
        probe.verify()
    with Image.open(io.BytesIO(raw)) as loaded:
        image = loaded.convert("RGB")
    width, height = image.size
    declared = canonical.get("image_size_px")
    if (not isinstance(declared, list) or len(declared) != 2
            or any(not isinstance(value, int) or value < 1 for value in declared)
            or declared != [width, height]):
        raise ValueError("통합 이미지 크기 메타데이터 불일치")
    return CanonicalImage(image, canonical["image_id"], width, height,
                          canonical.get("reference_photo_id"), "CANDIDATE_GEOMETRY_ONLY")


def photo_input_fingerprint(photos: list[Any]) -> str:
    """Fingerprint every photo record, independent of symptom-only changes."""
    manifest = []
    for photo in photos:
        get = photo.get if isinstance(photo, Mapping) else lambda name, default=None: getattr(photo, name, default)
        manifest.append({
            "photo_id": get("image_id", get("frontend_photo_id", "")),
            "sha256": get("content_sha256", get("sha256", "")),
            "side": get("side", "unknown"),
            "capture_type": get("capture_type", get("role", "unknown")),
        })
    raw = json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def component_map_cache_key(
    *, project_id: str, board_id: str, inputs: list[ComponentMapPhotoInput],
    options: dict | None = None, photo_manifest: list[Any] | None = None,
) -> str:
    payload = {
        "schema": SCHEMA_VERSION, "project": project_id, "board": board_id,
        "photos": [(item.frontend_photo_id, item.content_sha256, item.side, item.role) for item in inputs],
        "photo_manifest": photo_input_fingerprint(photo_manifest or []),
        "options": options or {},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def clear_component_map_state(state: MutableMapping[str, Any], *, clear_viewer_selection: bool = True) -> None:
    """Clear F3 results and viewer coordinates without touching intake inputs."""
    state.pop(COMPONENT_MAP_STATE_KEY, None)
    for key in tuple(state):
        if str(key).startswith(("cc_v4_pcb_viewer_", "circuitcare_v4_pcb_")):
            state.pop(key, None)
    for key in LEGACY_COMPONENT_MAP_KEYS:
        if not clear_viewer_selection and key in {
            "circuitcare_v3_active_side", "circuitcare_v2_active_photo_id",
        }:
            continue
        state.pop(key, None)
    if clear_viewer_selection:
        current = state.get("circuitcare_v2_selection")
        if current is not None and hasattr(current, "selected_component"):
            try:
                from dataclasses import replace
                state["circuitcare_v2_selection"] = replace(current, selected_component=None, selected_probe=None)
            except (TypeError, ValueError):
                pass


def component_map_entry_is_current(
    entry: Any, *, project_id: str, board_id: str | None, photo_fingerprint: str, cache_key: str,
) -> bool:
    return bool(
        isinstance(entry, dict)
        and entry.get("project_id") == project_id
        and entry.get("board_id") == board_id
        and entry.get("input_fingerprint") == photo_fingerprint
        and entry.get("component_map_cache_key") == cache_key
        and isinstance(entry.get("component_map"), dict)
        and isinstance(entry.get("photo_id_map"), list)
    )


def available_photo_sides(photo_records: list[Any]) -> list[str]:
    """Return only represented sides in stable PCB order, including unknown metadata."""
    present = {
        photo.get("side", "unknown") if isinstance(photo, Mapping) else getattr(photo, "side", "unknown")
        for photo in photo_records
    }
    return [side for side in ("front", "back", "unknown") if side in present]
