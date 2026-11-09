"""Bounded same-board photo intake, registration, mosaic, and component map.

Image-to-image registration in this module is separate from KiCad design
registration. Inferred detections and registrations remain candidates until
reviewed; no electrical connectivity is created here.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import numbers
import os
import time
import base64
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps

from circuitcare.vision.board_detector import detect_board_candidates
from circuitcare.vision.component_detector import detect_components, detect_components_tiled
from circuitcare.vision.detector_backend import DetectorUnavailable
from circuitcare.vision.detection_postprocess import PCB_MODEL_CLASSES
from circuitcare.vision.image_loader import ImageLoadError, load_board_image
from circuitcare.vision.perspective import rectify_board
from circuitcare.vision.reconstruction.multiview_registration import register_same_side


class BoardSide(StrEnum):
    TOP = "TOP"
    BOTTOM = "BOTTOM"


class CaptureRole(StrEnum):
    REFERENCE = "REFERENCE"
    OVERVIEW = "OVERVIEW"
    CLOSEUP = "CLOSEUP"
    ANGLED = "ANGLED"


@dataclass(frozen=True)
class ImageBudget:
    max_images: int = 20
    max_file_bytes: int = 20 * 1024 * 1024
    max_total_file_bytes: int = 80 * 1024 * 1024
    max_image_pixels: int = 12_000_000
    max_total_pixels: int = 40_000_000
    max_decoded_edge: int = 1600
    max_canvas_edge: int = 4000
    max_canvas_pixels: int = 8_000_000
    max_runtime_seconds: float = 60.0

    def __post_init__(self) -> None:
        ints = (self.max_images, self.max_file_bytes, self.max_total_file_bytes, self.max_image_pixels,
                self.max_total_pixels, self.max_decoded_edge, self.max_canvas_edge,
                self.max_canvas_pixels)
        if any(value < 1 for value in ints) or not math.isfinite(self.max_runtime_seconds) or self.max_runtime_seconds <= 0:
            raise ValueError("image budgets must be positive")
        if (self.max_canvas_edge < self.max_decoded_edge
                or self.max_canvas_pixels < self.max_decoded_edge * self.max_decoded_edge):
            raise ValueError("canvas limits must accommodate a single maximum-size decoded image")


@dataclass(frozen=True)
class PhotoInput:
    data: bytes | bytearray | str | Path
    side: BoardSide | str
    role: CaptureRole | str = CaptureRole.OVERVIEW
    name: str = "photo"
    rectification_corners_px: tuple[tuple[float, float], ...] | None = None


@dataclass
class PhotoAsset:
    photo_id: str
    sha256: str
    side: str
    role: str
    name: str
    legacy_photo_id: str
    original_size_px: tuple[int, int]
    processed_size_px: tuple[int, int]
    decoded_size_px: tuple[int, int]
    image_format: str
    image: Image.Image = field(repr=False)
    original_bytes: bytes = field(repr=False)
    duplicate_of: str | None = None
    rectification: dict[str, Any] | None = None
    coordinate_transforms: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"photo_id": self.photo_id, "sha256": self.sha256, "side": self.side,
                "role": self.role, "name": self.name, "original_size_px": list(self.original_size_px),
                "legacy_photo_id": self.legacy_photo_id,
                "processed_size_px": list(self.processed_size_px), "decoded_size_px": list(self.decoded_size_px),
                "image_format": self.image_format,
                "duplicate_of": self.duplicate_of, "original_preserved": True,
                "rectification": self.rectification, "coordinate_transforms": self.coordinate_transforms}


@dataclass
class PCBPhotoSession:
    board_id: str
    photos: list[PhotoAsset]
    created_at_epoch: float
    budget: ImageBudget


@dataclass(frozen=True)
class RegistrationRecord:
    photo_id: str
    side: str
    reference_photo_id: str
    status: str
    method: str
    transform_moving_to_canonical: tuple[tuple[float, ...], ...] | None
    candidate_matches: int = 0
    inlier_count: int = 0
    inlier_ratio: float | None = None
    rmse_px: float | None = None
    overlap_ratio: float = 0.0
    failure_reason: str | None = None
    p95_reprojection_error_px: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "transform_moving_to_canonical":
                [list(row) for row in self.transform_moving_to_canonical]
                if self.transform_moving_to_canonical else None}


class MultiImageError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class MultiImageComponentMapService:
    """Create a partial but provenance-preserving map for one board session."""

    SCHEMA_VERSION = "circuitcare.pcb-component-map.v1"

    def __init__(self, *, budget: ImageBudget = ImageBudget()):
        self.budget = budget

    def ingest(self, board_id: str, inputs: Iterable[PhotoInput]) -> PCBPhotoSession:
        if not isinstance(board_id, str) or not board_id.strip():
            raise MultiImageError("INVALID_BOARD_ID", "board_id is required")
        values = []
        for item in inputs:
            if len(values) >= self.budget.max_images:
                raise MultiImageError("IMAGE_COUNT_LIMIT", f"Provide 1 to {self.budget.max_images} photos")
            values.append(item)
        if not values or len(values) > self.budget.max_images:
            raise MultiImageError("IMAGE_COUNT_LIMIT", f"Provide 1 to {self.budget.max_images} photos")
        seen: dict[str, str] = {}
        photo_key_counts: dict[str, int] = {}
        assets: list[PhotoAsset] = []
        total_pixels = 0
        total_bytes = 0
        intake_started = time.perf_counter()
        for index, item in enumerate(values):
            if time.perf_counter() - intake_started > self.budget.max_runtime_seconds:
                raise MultiImageError("TIMEOUT", "Photo intake exceeded its time budget")
            try:
                side = BoardSide(item.side).value
                role = CaptureRole(item.role).value
            except ValueError as exc:
                raise MultiImageError("INVALID_SIDE_OR_ROLE", str(exc)) from exc
            raw = _read_bytes(item.data, self.budget.max_file_bytes)
            total_bytes += len(raw)
            if total_bytes > self.budget.max_total_file_bytes:
                raise MultiImageError("TOTAL_FILE_SIZE_LIMIT", "Session exceeds total input byte budget")
            try:
                loaded = load_board_image(raw, max_file_bytes=self.budget.max_file_bytes,
                                          max_pixels=self.budget.max_image_pixels,
                                          max_edge=self.budget.max_decoded_edge)
            except ImageLoadError as exc:
                raise MultiImageError("INVALID_IMAGE", str(exc)) from exc
            encoded_size, exif_orientation = _encoded_image_info(raw)
            total_pixels += loaded.original_size[0] * loaded.original_size[1]
            if total_pixels > self.budget.max_total_pixels:
                raise MultiImageError("TOTAL_PIXEL_LIMIT", "Session exceeds total original image pixel budget")
            stable_name = Path(str(item.name)).name
            photo_key = hashlib.sha256(
                f"{board_id}|{side}|{role}|{stable_name}|{loaded.sha256}".encode()).hexdigest()[:16]
            occurrence = photo_key_counts.get(photo_key, 0)
            photo_key_counts[photo_key] = occurrence + 1
            photo_id = f"photo_{photo_key}" + (f"_dup{occurrence}" if occurrence else "")
            legacy_photo_id = "photo_" + hashlib.sha256(
                f"{board_id}|{side}|{role}|{item.name}|{index}|{loaded.sha256}".encode()).hexdigest()[:16]
            asset = PhotoAsset(photo_id, loaded.sha256, side, role, str(item.name), legacy_photo_id,
                               loaded.original_size, loaded.processed_size, loaded.processed_size, loaded.format,
                               loaded.image.copy(), raw, seen.get(loaded.sha256))
            asset.coordinate_transforms = {
                "encoded_image_frame": "IMAGE_FILE_PIXEL",
                "encoded_size_px": list(encoded_size),
                "exif_orientation": exif_orientation,
                "encoded_to_exif_corrected": _exif_orientation_matrix(exif_orientation, encoded_size).tolist(),
                "exif_corrected_frame": "EXIF_CORRECTED_IMAGE_PIXEL",
                "exif_corrected_size_px": list(loaded.original_size),
                "exif_corrected_to_decoded": [[loaded.scale_xy[0], 0.0, 0.0],
                                                [0.0, loaded.scale_xy[1], 0.0], [0.0, 0.0, 1.0]],
                "decoded_frame": "DOWNSCALED_IMAGE_PIXEL",
                "decoded_size_px": list(loaded.processed_size),
                "decoded_to_rectified": np.eye(3).tolist(),
            }
            if item.rectification_corners_px is not None:
                asset = self._rectify_asset(asset, item.rectification_corners_px, method="USER_SUPPLIED_BOARD_CORNERS")
            else:
                asset = self._try_auto_rectification(asset)
            seen.setdefault(loaded.sha256, photo_id)
            assets.append(asset)
        return PCBPhotoSession(board_id.strip(), assets, time.time(), self.budget)

    def build_map(self, session: PCBPhotoSession, *, detector: Callable[[Any], dict[str, Any]] | None = None,
                  model_path: str | Path | None = None,
                  inference_mode: str = "single",
                  tile_size_px: int = 1536,
                  tile_overlap_px: int = 320,
                  previous_component_map: dict[str, Any] | str | Path | None = None,
                  confirmed_identity_by_observation_id: dict[str, str] | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        if not session.photos:
            raise MultiImageError("EMPTY_SESSION", "No photos are available")
        if inference_mode not in {"single", "tiled"}:
            raise MultiImageError("INVALID_INFERENCE_MODE", "inference_mode must be 'single' or 'tiled'")
        previous = _load_previous_map(previous_component_map)
        if previous and previous.get("board_id") != session.board_id:
            raise MultiImageError("PREVIOUS_MAP_BOARD_MISMATCH", "Previous component map belongs to another board")
        sides: dict[str, list[PhotoAsset]] = {side.value: [p for p in session.photos if p.side == side.value]
                                              for side in BoardSide}
        side_maps = {}
        for side, photos in sides.items():
            if photos:
                if time.perf_counter() - started > session.budget.max_runtime_seconds:
                    raise MultiImageError("TIMEOUT", "Multi-image processing exceeded its time budget")
                previous_side = (previous or {}).get("sides", {}).get(side, {})
                if previous_side:
                    previous_side = {**previous_side, "_previous_photos": (previous or {}).get("photos", [])}
                side_maps[side] = self._build_side(session.board_id, side, photos, detector, model_path,
                    started, session.budget, inference_mode=inference_mode, tile_size_px=tile_size_px,
                    tile_overlap_px=tile_overlap_px, previous_side=previous_side,
                    confirmed_identity_by_observation_id=confirmed_identity_by_observation_id or {})
        registration_by_photo = {item["photo_id"]: item for side_map in side_maps.values()
                                 for item in side_map["registrations"]}
        photo_records = []
        for photo in session.photos:
            record = photo.to_dict()
            registration = registration_by_photo.get(photo.photo_id)
            transform = registration.get("transform_moving_to_canonical") if registration else None
            if transform:
                image_to_canonical = np.asarray(transform, dtype=np.float64)
                decoded_to_rectified = np.asarray(photo.coordinate_transforms["decoded_to_rectified"], dtype=np.float64)
                exif_to_decoded = np.asarray(photo.coordinate_transforms["exif_corrected_to_decoded"], dtype=np.float64)
                encoded_to_exif = np.asarray(photo.coordinate_transforms["encoded_to_exif_corrected"], dtype=np.float64)
                chain = record["coordinate_transforms"]
                chain["processed_to_canonical"] = image_to_canonical.tolist()
                chain["decoded_to_canonical"] = (image_to_canonical @ decoded_to_rectified).tolist()
                chain["exif_corrected_to_canonical"] = (image_to_canonical @ decoded_to_rectified @ exif_to_decoded).tolist()
                chain["encoded_to_canonical"] = (
                    image_to_canonical @ decoded_to_rectified @ exif_to_decoded @ encoded_to_exif).tolist()
            else:
                record["coordinate_transforms"]["processed_to_canonical"] = None
                record["coordinate_transforms"]["encoded_to_canonical"] = None
            photo_records.append(record)
        return {"schema_version": self.SCHEMA_VERSION, "board_id": session.board_id,
                "created_at_epoch": session.created_at_epoch, "processing_status": "PARTIAL_MAP_READY",
                "design_registration": {"status": "NOT_RUN_SEPARATE_FROM_PHOTO_REGISTRATION"},
                "sides": side_maps,
                "detection_rejections": [record for value in side_maps.values()
                                         for record in value.get("detection_rejections", [])],
                "detection_warnings": [record for value in side_maps.values()
                                       for record in value.get("detection_warnings", [])],
                "identity_manifests": {side: value["identity_manifest"] for side, value in side_maps.items()},
                "photos": photo_records,
                "resource_limits": session.budget.__dict__.copy(),
                "runtime_seconds": time.perf_counter() - started,
                "electrical_connectivity": "NOT_INFERRED_FROM_PHOTOS"}

    def _try_auto_rectification(self, photo: PhotoAsset) -> PhotoAsset:
        try:
            result = detect_board_candidates(photo.image, max_candidates=3)
            candidates = result.get("candidates", [])
            if not candidates:
                photo.rectification = {"status": "RECTIFICATION_UNVERIFIED", "reason": "NO_BOARD_BOUNDARY_CANDIDATE",
                                       "source_to_rectified": None, "confidence_diagnostics": result}
                return photo
            top = candidates[0]
            second_area = float(candidates[1]["area_px"]) if len(candidates) > 1 else 0.0
            clear_margin = second_area <= 0 or float(top["area_px"]) >= second_area * 1.35
            corners = np.asarray(top.get("corners_px", []), dtype=np.float64).reshape(-1, 2)
            width, height = photo.image.size
            edge_margin = min(width, height) * 0.015
            touches_frame = (corners.shape != (4, 2) or np.any(corners[:, 0] <= edge_margin)
                             or np.any(corners[:, 0] >= width - 1 - edge_margin)
                             or np.any(corners[:, 1] <= edge_margin)
                             or np.any(corners[:, 1] >= height - 1 - edge_margin))
            if touches_frame or float(top.get("area_ratio", 0.0)) < 0.15 or not clear_margin:
                photo.rectification = {"status": "RECTIFICATION_UNVERIFIED",
                    "reason": "BOUNDARY_TOUCHES_IMAGE_FRAME" if touches_frame else "AMBIGUOUS_OR_SMALL_BOARD_BOUNDARY",
                    "source_to_rectified": None,
                    "confidence_diagnostics": {"candidate_count": len(candidates),
                                                "top_area_ratio": top.get("area_ratio"),
                                                "second_area_px": second_area,
                                                "image_size_px": [width, height]}}
                return photo
            return self._rectify_asset(photo, top["corners_px"],
                method="AUTO_UNIQUE_LARGE_BOUNDARY_CANDIDATE",
                diagnostics={"candidate_count": len(candidates), "area_ratio": top.get("area_ratio"),
                             "runner_up_area_px": second_area})
        except (ValueError, cv2.error) as exc:
            photo.rectification = {"status": "RECTIFICATION_UNVERIFIED", "reason": str(exc),
                                   "source_to_rectified": None}
            return photo

    def _rectify_asset(self, photo: PhotoAsset, corners: Any, *, method: str,
                       diagnostics: dict[str, Any] | None = None) -> PhotoAsset:
        try:
            output_edge = min(self.budget.max_decoded_edge, max(photo.image.size))
            result = rectify_board(photo.image, corners, max_output_edge=max(64, output_edge))
            photo.rectification = {"status": "RECTIFIED_CANDIDATE_NOT_PHYSICALLY_VALIDATED",
                                   "source_to_rectified": result["transform_matrix"],
                                   "output_size_px": result["output_size_px"],
                                   "confidence_diagnostics": {"method": method,
                                                               "boundary": diagnostics,
                                                               "physical_dimensions_known": False}}
            photo.image = result["image"]
            photo.processed_size_px = tuple(result["output_size_px"])
            photo.coordinate_transforms["decoded_to_rectified"] = result["transform_matrix"]
            photo.coordinate_transforms["processed_frame"] = "RECTIFIED_IMAGE_PIXEL"
            photo.coordinate_transforms["rectified_size_px"] = result["output_size_px"]
        except (ValueError, cv2.error) as exc:
            photo.rectification = {"status": "RECTIFICATION_FAILED_KEEP_ORIGINAL", "reason": str(exc),
                                   "source_to_rectified": None}
        return photo

    def _build_side(self, board_id: str, side: str, photos: list[PhotoAsset],
                    detector: Callable[[Any], dict[str, Any]] | None, model_path: str | Path | None,
                    started: float, budget: ImageBudget, *, inference_mode: str,
                    tile_size_px: int, tile_overlap_px: int, previous_side: dict[str, Any],
                    confirmed_identity_by_observation_id: dict[str, str]) -> dict[str, Any]:
        unique = []
        duplicate_records = {}
        known_sha: dict[str, str] = {}
        for photo in photos:
            if photo.sha256 in known_sha:
                duplicate_records[photo.photo_id] = known_sha[photo.sha256]
            else:
                unique.append(photo)
                known_sha[photo.sha256] = photo.photo_id
        prior_reference_id = _previous_reference_id(previous_side)
        reference = next((photo for photo in unique
                          if prior_reference_id in {photo.photo_id, photo.legacy_photo_id}), None)
        reference = reference or min(unique, key=lambda p: (0 if p.role == CaptureRole.REFERENCE.value else
                                               1 if p.role == CaptureRole.OVERVIEW.value else 2,
                                               -p.image.width * p.image.height, p.sha256))
        previous_ref_photo = next((photo for photo in (previous_side.get("_previous_photos") or [])
                                   if photo.get("photo_id") == prior_reference_id), None)
        previous_rectification = (previous_side.get("canonical_image", {}).get("reference_rectification_matrix")
                                  or ((previous_ref_photo or {}).get("rectification") or {}).get("source_to_rectified"))
        current_rectification = (reference.rectification or {}).get("source_to_rectified")
        rectification_compatible = _same_matrix(previous_rectification, current_rectification)
        registrations: dict[str, RegistrationRecord] = {
            reference.photo_id: RegistrationRecord(reference.photo_id, side, reference.photo_id,
                "REFERENCE", "IDENTITY", tuple(tuple(float(x) for x in row) for row in np.eye(3)))
        }
        for photo in unique:
            if photo.photo_id == reference.photo_id:
                continue
            if time.perf_counter() - started > budget.max_runtime_seconds:
                registrations[photo.photo_id] = RegistrationRecord(photo.photo_id, side, reference.photo_id,
                    "UNREGISTERED", "ORB_RANSAC_HOMOGRAPHY", None, failure_reason="TIMEOUT")
                continue
            try:
                result = register_same_side(reference.image, photo.image, source_photo_id=photo.photo_id,
                                            reference_photo_id=reference.photo_id,
                                            board_side="front" if side == "TOP" else "back")
            except (ValueError, cv2.error) as exc:
                registrations[photo.photo_id] = RegistrationRecord(photo.photo_id, side, reference.photo_id,
                    "UNREGISTERED", "ORB_RANSAC_HOMOGRAPHY", None, failure_reason=str(exc))
                continue
            record = _validate_registration(photo, reference, result)
            registrations[photo.photo_id] = record
        for duplicate_id, original_id in duplicate_records.items():
            registrations[duplicate_id] = RegistrationRecord(duplicate_id, side, reference.photo_id,
                "DUPLICATE_SKIPPED", "SHA256_DUPLICATE", registrations[original_id].transform_moving_to_canonical)

        mosaic, source_map, canvas_origin, included_ids, mosaic_metrics = _make_mosaic(
            reference, unique, registrations, budget)
        for photo_id, record in list(registrations.items()):
            if photo_id not in included_ids and record.status in {"REFERENCE", "REGISTERED_CANDIDATE"}:
                registrations[photo_id] = RegistrationRecord(photo_id, side, reference.photo_id,
                    "UNREGISTERED", record.method, None, record.candidate_matches, record.inlier_count,
                    record.inlier_ratio, record.rmse_px, record.overlap_ratio,
                    "CANVAS_RESOURCE_LIMIT")
        shift = np.array([[1, 0, -canvas_origin[0]], [0, 1, -canvas_origin[1]], [0, 0, 1]], dtype=np.float64)
        for photo_id, record in list(registrations.items()):
            if record.transform_moving_to_canonical is not None:
                canonical_transform = shift @ np.asarray(record.transform_moving_to_canonical)
                registrations[photo_id] = RegistrationRecord(
                    record.photo_id, record.side, record.reference_photo_id, record.status, record.method,
                    tuple(tuple(float(x) for x in row) for row in canonical_transform),
                    record.candidate_matches, record.inlier_count, record.inlier_ratio, record.rmse_px,
                    record.overlap_ratio, record.failure_reason, record.p95_reprojection_error_px)
        observations = []
        detection_rejections: list[dict[str, Any]] = []
        detection_warnings: list[dict[str, Any]] = []
        detector_status = "DETECTOR_NOT_CONFIGURED"
        for photo in unique:
            if time.perf_counter() - started > budget.max_runtime_seconds:
                detector_status = "PARTIAL_TIMEOUT"
                break
            if detector is None and model_path is None:
                continue
            try:
                if detector:
                    detection_result = detector(photo.image)
                elif inference_mode == "tiled":
                    native_image, native_to_processed = _native_rectified_image(photo)
                    remaining = max(0.001, budget.max_runtime_seconds - (time.perf_counter() - started))
                    detection_result = detect_components_tiled(native_image, model_path,
                        tile_size_px=tile_size_px, tile_overlap_px=tile_overlap_px,
                        timeout_seconds=remaining)
                else:
                    detection_result = detect_components(photo.image, model_path)
                if not isinstance(detection_result, dict) or not isinstance(detection_result.get("detections"), list):
                    detection_rejections.append({"photo_id": photo.photo_id,
                        "error_code": "MALFORMED_DETECTOR_RESPONSE", "detection_index": None,
                        "details": "response must be an object with a detections list"})
                    detector_status = "DETECTOR_ERROR"
                    continue
                result_status = detection_result.get("status", "INFERENCE_COMPLETE")
                detector_status = result_status if isinstance(result_status, str) else "DETECTOR_ERROR"
                if detector_status not in {"INFERENCE_COMPLETE", "PARTIAL_INFERENCE"}:
                    detection_rejections.append({"photo_id": photo.photo_id,
                        "error_code": "MALFORMED_DETECTOR_RESPONSE", "detection_index": None,
                        "details": f"unsupported detector status: {detector_status}"})
                    detector_status = "DETECTOR_ERROR"
                    continue
                detection_frame_size = native_image.size if inference_mode == "tiled" and detector is None and model_path is not None else photo.image.size
                valid_detections = []
                seen_source_ids: dict[str, int] = {}
                for detection_index, raw_detection in enumerate(detection_result["detections"]):
                    normalized, error_code, details = _normalize_detection(raw_detection, detection_frame_size)
                    if error_code:
                        detection_rejections.append({"photo_id": photo.photo_id,
                            "detection_id": raw_detection.get("detection_id") if isinstance(raw_detection, dict) else None,
                            "detection_index": detection_index, "error_code": error_code, "details": details})
                        continue
                    source_id = normalized.get("detection_id")
                    if source_id:
                        occurrence = seen_source_ids.get(source_id, 0)
                        seen_source_ids[source_id] = occurrence + 1
                        if occurrence:
                            detection_warnings.append({"photo_id": photo.photo_id, "detection_id": source_id,
                                "detection_index": detection_index, "error_code": "DUPLICATE_DETECTION_ID",
                                "disposition": "RETAINED_WITH_DISAMBIGUATED_ID"})
                    valid_detections.append(normalized)
                transform = registrations[photo.photo_id].transform_moving_to_canonical
                if transform is None or registrations[photo.photo_id].status not in {"REFERENCE", "REGISTERED_CANDIDATE"}:
                    continue
                raw_detections = sorted(valid_detections, key=_detection_sort_key)
                seen_detection_ids: set[str] = set()
                seen_observation_ids: dict[str, int] = {}
                for detection in raw_detections:
                    if inference_mode == "tiled" and detector is None and model_path is not None:
                        detection = _detection_to_processed_frame(detection, native_to_processed)
                    observation = _map_detection(detection, photo, np.asarray(transform), (0, 0),
                                                 registrations[photo.photo_id])
                    if observation.get("status"):
                        detection_rejections.append({"photo_id": photo.photo_id,
                            "detection_id": detection.get("detection_id"), "detection_index": None,
                            "error_code": observation["status"],
                            "details": observation.get("reason", "invalid normalized detection")})
                        continue
                    raw_id = observation["detection_id"]
                    if raw_id in seen_detection_ids:
                        original_id = raw_id
                        raw_id = f"{original_id}__{hashlib.sha256(observation['observation_id'].encode()).hexdigest()[:8]}"
                        observation["source_detection_id"] = observation.get("source_detection_id") or original_id
                        observation["detection_id"] = raw_id
                        observation["observation_id"] = _stable_observation_id(photo.photo_id, raw_id,
                            observation["class_name"], observation["processed_bbox_xyxy"], observation.get("ocr_marking"))
                        if raw_id in seen_detection_ids:
                            duplicate_count = 2
                            candidate = raw_id
                            while candidate in seen_detection_ids:
                                candidate = f"{raw_id}_{duplicate_count}"
                                duplicate_count += 1
                            raw_id = candidate
                            observation["detection_id"] = raw_id
                            observation["observation_id"] = _stable_observation_id(photo.photo_id, raw_id,
                                observation["class_name"], observation["processed_bbox_xyxy"], observation.get("ocr_marking"))
                    seen_detection_ids.add(raw_id)
                    observation_count = seen_observation_ids.get(observation["observation_id"], 0)
                    seen_observation_ids[observation["observation_id"]] = observation_count + 1
                    if observation_count:
                        observation["observation_id"] = hashlib.sha256(
                            f"{observation['observation_id']}|duplicate-instance:{observation_count + 1}".encode()
                        ).hexdigest()[:24]
                        observation["identity_review_status"] = "IDENTITY_AMBIGUOUS"
                        observation["identity_review_reason"] = "DUPLICATE_DETECTION_SIGNATURE"
                    observations.append(observation)
            except TimeoutError as exc:
                detector_status = "PARTIAL_TIMEOUT"
                observations.append({"photo_id": photo.photo_id, "error": str(exc), "status": detector_status})
            except DetectorUnavailable as exc:
                detector_status = "BLOCKED_MODEL_UNAVAILABLE" if model_path else "DETECTOR_ERROR"
                observations.append({"photo_id": photo.photo_id, "error": str(exc), "status": detector_status})
            except (ValueError, OSError, RuntimeError, cv2.error) as exc:
                detector_status = "DETECTOR_ERROR"
                observations.append({"photo_id": photo.photo_id, "error": str(exc), "status": "DETECTION_FAILED"})
        components, identity_mappings = _fuse_observations(
            board_id, side, observations, previous_side=previous_side,
            confirmed_identity_by_observation_id=confirmed_identity_by_observation_id,
            previous_canvas_origin=previous_side.get("canonical_image", {}).get("canvas_origin_in_reference_px"),
            current_canvas_origin=canvas_origin,
            previous_reference_id=prior_reference_id,
            current_reference_id=reference.photo_id,
            previous_frame_compatible=(prior_reference_id in {reference.photo_id, reference.legacy_photo_id}
                                       and rectification_compatible))
        canonical_id = f"mosaic:{board_id}:{side}"
        registration_records = []
        photo_by_id = {photo.photo_id: photo for photo in photos}
        for record in registrations.values():
            serialized = record.to_dict()
            source_photo = photo_by_id.get(record.photo_id)
            serialized["source_coordinate_frame"] = (
                (source_photo.coordinate_transforms.get("processed_frame")
                 or source_photo.coordinate_transforms.get("decoded_frame"))
                if source_photo else "IMAGE_PIXEL")
            serialized["target_coordinate_frame"] = "MOSAIC_PIXEL"
            serialized["matrix_direction"] = "SOURCE_TO_CANONICAL"
            serialized["matrix_units"] = {"source": "px", "target": "px"}
            registration_records.append(serialized)
        return {"board_id": board_id, "board_side": side,
                "canonical_image": {"image_id": canonical_id, "coordinate_frame": "MOSAIC_PIXEL",
                    "reference_photo_id": reference.photo_id,
                    "reference_rectification_matrix": current_rectification,
                    "image_size_px": [int(mosaic.shape[1]), int(mosaic.shape[0])],
                    "canvas_origin_in_reference_px": list(canvas_origin),
                    "image_png_base64": _png_base64(mosaic),
                    "pixel_source_map_png_base64": _png_base64(source_map),
                    "pixel_source_map_values": {str(index + 1): photo.photo_id for index, photo in enumerate(unique)},
                    "rectification_validation": "CANDIDATE_GEOMETRY_ONLY",
                    "coverage_metrics": mosaic_metrics},
                "registrations": registration_records,
                "components": components, "component_observations": observations,
                "detection_rejections": detection_rejections,
                "detection_warnings": detection_warnings,
                "historical_component_id_mappings": identity_mappings,
                "identity_manifest": {"schema_version": "circuitcare.component-identities.v1",
                    "board_id": board_id, "board_side": side, "reference_photo_id": reference.photo_id,
                    "canonical_coordinate_frame": "MOSAIC_PIXEL",
                    "canonical_image": {"reference_photo_id": reference.photo_id,
                        "canvas_origin_in_reference_px": list(canvas_origin),
                        "reference_rectification_matrix": current_rectification}, "components": components},
                "detection_status": detector_status, "inference_mode": inference_mode,
                "tile_configuration": {"tile_size_px": tile_size_px, "tile_overlap_px": tile_overlap_px}
                    if inference_mode == "tiled" else None,
                "board_boundary_candidates": _board_boundary_records(unique, started, budget),
                "functional_blocks": {"status": "NOT_EVALUATED", "candidates": []},
                "warnings": ["Photo registration does not establish KiCad coordinate alignment or electrical connectivity."]}

    @staticmethod
    def fault_overlay(component_map: dict[str, Any], side: str,
                      candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
        try:
            side_value = BoardSide(side).value
        except ValueError as exc:
            raise MultiImageError("INVALID_SIDE_OR_ROLE", "side must be TOP or BOTTOM") from exc
        side_record = component_map.get("sides", {}).get(side_value)
        if side_record is None:
            raise MultiImageError("BOARD_SIDE_NOT_FOUND", "No component map exists for the requested side")
        components = {item["component_id"]: item for item in side_record["components"]}
        result = []
        for candidate in candidates:
            component_id = str(candidate.get("component_id", ""))
            state = str(candidate.get("status", "UNDETERMINED"))
            if state not in {"CONFIRMED_FAULT", "SUSPECTED_FAULT", "RELATED_COMPONENT", "UNDETERMINED"}:
                raise MultiImageError("INVALID_FAULT_STATUS", f"Unsupported fault candidate state: {state}")
            if state == "CONFIRMED_FAULT" and not candidate.get("confirmed_by_measurement_or_inspection"):
                state = "UNDETERMINED"
            component = components.get(component_id)
            result.append({"board_id": component_map["board_id"], "board_side": side,
                "canonical_image_reference": side_record["canonical_image"]["image_id"],
                "component_id": component_id if component else None,
                "component_position": component.get("bbox_xyxy") if component else None,
                "mapping_status": "MAPPED" if component else "UNMAPPED_COMPONENT_ID",
                "fault_candidate_status": state, "candidate_ranking": candidate.get("ranking"),
                "diagnostic_evidence": candidate.get("diagnostic_evidence", []),
                "suggested_measurement": candidate.get("suggested_measurement"),
                "measurement_result": candidate.get("measurement_result"),
                "confidence": candidate.get("confidence"),
                "uncertainty": candidate.get("uncertainty", "NOT_PROVIDED"),
                "registration_validation_status": "IMAGE_TO_IMAGE_CANDIDATE_ONLY"})
        return {"schema_version": "circuitcare.pcb-fault-overlay.v1", "board_id": component_map["board_id"],
                "board_side": side, "candidates": result}


def _read_bytes(value: Any, limit: int) -> bytes:
    if isinstance(value, (str, Path)):
        path = Path(value)
        try:
            if path.stat().st_size > limit:
                raise MultiImageError("FILE_SIZE_LIMIT", f"Image file exceeds {limit} bytes")
            data = path.read_bytes()
        except OSError as exc:
            raise MultiImageError("INVALID_IMAGE", str(exc)) from exc
    else:
        data = bytes(value)
    if not data or len(data) > limit:
        raise MultiImageError("FILE_SIZE_LIMIT", f"Image bytes must be between 1 and {limit}")
    return data


def _encoded_image_info(raw: bytes) -> tuple[tuple[int, int], int]:
    try:
        with Image.open(io.BytesIO(raw)) as source:
            width, height = source.size
            orientation = int(source.getexif().get(274, 1))
    except (OSError, ValueError) as exc:
        raise MultiImageError("INVALID_IMAGE", f"Could not inspect EXIF image frame: {exc}") from exc
    if orientation not in range(1, 9):
        orientation = 1
    return (int(width), int(height)), orientation


def _exif_orientation_matrix(orientation: int, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    matrices = {
        1: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        2: [[-1, 0, width - 1], [0, 1, 0], [0, 0, 1]],
        3: [[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]],
        4: [[1, 0, 0], [0, -1, height - 1], [0, 0, 1]],
        5: [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        6: [[0, -1, height - 1], [1, 0, 0], [0, 0, 1]],
        7: [[0, -1, height - 1], [-1, 0, width - 1], [0, 0, 1]],
        8: [[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]],
    }
    return np.asarray(matrices[orientation], dtype=np.float64)


def _validate_registration(photo: PhotoAsset, reference: PhotoAsset, result: Any) -> RegistrationRecord:
    matrix = np.asarray(result.transformation, dtype=np.float64) if result.transformation else None
    if matrix is None or matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return RegistrationRecord(photo.photo_id, photo.side, reference.photo_id, "UNREGISTERED",
                                  result.method, None, None,
                                  result.inlier_count, result.inlier_ratio or 0.0, result.reprojection_error_px,
                                  failure_reason=result.registration_status)
    determinant = float(np.linalg.det(matrix))
    condition = float(np.linalg.cond(matrix))
    corners = np.array([[0, 0], [photo.image.width - 1, 0],
                        [photo.image.width - 1, photo.image.height - 1], [0, photo.image.height - 1]], dtype=np.float32)
    try:
        projected = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        area = abs(float(cv2.contourArea(projected.astype(np.float32))))
    except cv2.error:
        projected, area = np.zeros((4, 2)), 0.0
    ref_area = float(reference.image.width * reference.image.height)
    intersection_area = 0.0
    if area > 0 and ref_area > 0:
        ref_polygon = np.array([[0, 0], [reference.image.width - 1, 0],
                                [reference.image.width - 1, reference.image.height - 1],
                                [0, reference.image.height - 1]], dtype=np.float32)
        try:
            intersection_area, _ = cv2.intersectConvexConvex(
                cv2.convexHull(projected.astype(np.float32)), ref_polygon)
        except cv2.error:
            intersection_area = 0.0
    overlap = min(1.0, float(intersection_area) / min(area, ref_area)) if area and ref_area else 0.0
    valid_geometry = (abs(determinant) > 1e-12 and condition < 1e10 and area > 0
                      and area <= ref_area * 8
                      and np.isfinite(projected).all()
                      and np.max(np.abs(projected)) < max(reference.image.size) * 8)
    status = result.registration_status
    if status != "REGISTERED_CANDIDATE" or not valid_geometry:
        status = "UNREGISTERED"
    if overlap < 0.02:
        status = "UNREGISTERED"
    return RegistrationRecord(photo.photo_id, photo.side, reference.photo_id, status,
        result.method, tuple(tuple(float(x) for x in row) for row in matrix) if status != "UNREGISTERED" else None,
        result.candidate_match_count, result.inlier_count, result.inlier_ratio or 0.0,
        result.reprojection_error_px, overlap,
        None if status != "UNREGISTERED" else "INVALID_TRANSFORM_OR_INSUFFICIENT_OVERLAP",
        result.p95_reprojection_error_px)


def _make_mosaic(reference: PhotoAsset, photos: list[PhotoAsset],
                 records: dict[str, RegistrationRecord], budget: ImageBudget):
    placements = []
    all_points = []
    for photo in photos:
        record = records[photo.photo_id]
        if record.status not in {"REFERENCE", "REGISTERED_CANDIDATE"} or record.transform_moving_to_canonical is None:
            continue
        matrix = np.asarray(record.transform_moving_to_canonical, dtype=np.float64)
        corners = np.array([[0, 0], [photo.image.width - 1, 0],
                            [photo.image.width - 1, photo.image.height - 1], [0, photo.image.height - 1]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        placements.append((photo, matrix, transformed))
        all_points.extend(transformed.tolist())
    points = np.asarray(all_points)
    min_x, min_y = np.floor(points.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(points.max(axis=0)).astype(int)
    width, height = max_x - min_x + 1, max_y - min_y + 1
    if width <= 0 or height <= 0 or width > budget.max_canvas_edge or height > budget.max_canvas_edge or width * height > budget.max_canvas_pixels:
        reference_corners = np.array([[0, 0], [reference.image.width-1, 0],
                                      [reference.image.width-1, reference.image.height-1],
                                      [0, reference.image.height-1]], dtype=np.float32)
        placements = [(reference, np.eye(3), reference_corners)]
        all_points = reference_corners.tolist()
        min_x = min_y = 0
        width, height = reference.image.size
        # The reference itself is already capped by the intake pixel and edge budgets.
    shift = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
    mosaic = np.zeros((height, width, 3), dtype=np.uint8)
    sources = np.zeros((height, width), dtype=np.uint8)
    quality = np.full((height, width), -1.0, dtype=np.float32)
    coverage_count = np.zeros((height, width), dtype=np.uint8)
    overlap_pixel_count = 0
    overlap_residuals = []
    for photo, matrix, _ in sorted(placements, key=lambda item: item[0].photo_id):
        transform = shift @ matrix
        array = np.asarray(photo.image.convert("RGB"))
        warped = cv2.warpPerspective(array, transform, (width, height))
        mask = cv2.warpPerspective(np.ones(array.shape[:2], dtype=np.uint8), transform,
                                   (width, height), flags=cv2.INTER_NEAREST).astype(bool)
        overlap_mask = mask & (sources > 0)
        if np.any(overlap_mask):
            current_pixels = mosaic[overlap_mask].astype(np.int16)
            new_pixels = warped[overlap_mask].astype(np.int16)
            overlap_pixel_count += int(np.count_nonzero(overlap_mask))
            overlap_residuals.extend(np.mean(np.abs(current_pixels - new_pixels), axis=1)[::max(1, int(np.count_nonzero(overlap_mask)) // 2000)].tolist())
            channel_delta = np.median(current_pixels - new_pixels, axis=0)
            warped = np.clip(warped.astype(np.int16) + np.clip(channel_delta, -32, 32), 0, 255).astype(np.uint8)
        coverage_count += mask.astype(np.uint8)
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        score = max(0.0, float(cv2.Laplacian(gray, cv2.CV_32F).var()))
        replace_mask = mask & (score > quality)
        mosaic[replace_mask] = warped[replace_mask]
        photo_index = next(index for index, candidate in enumerate(photos) if candidate.photo_id == photo.photo_id)
        sources[replace_mask] = photo_index + 1
        quality[replace_mask] = score
    valid_pixels = int(np.count_nonzero(coverage_count))
    canvas_pixels = int(width * height)
    footprint_hull = cv2.convexHull(np.asarray(all_points, dtype=np.float32)) if all_points else None
    footprint_pixels = abs(float(cv2.contourArea(footprint_hull))) if footprint_hull is not None else 0.0
    overlap_union = int(np.count_nonzero(coverage_count > 1))
    assigned = {photo.photo_id: int(np.count_nonzero(sources == index + 1))
                for index, photo in enumerate(photos)}
    metrics = {"valid_pixel_count": valid_pixels, "canvas_pixel_count": canvas_pixels,
               "valid_pixel_coverage_ratio": valid_pixels / canvas_pixels if canvas_pixels else 0.0,
               "canvas_coverage_ratio": min(1.0, footprint_pixels / canvas_pixels) if canvas_pixels else 0.0,
               "unobserved_pixel_count": canvas_pixels - valid_pixels,
               "unobserved_area_ratio": (canvas_pixels - valid_pixels) / canvas_pixels if canvas_pixels else 0.0,
               "overlap_union_pixel_count": overlap_union,
               "overlap_union_ratio_to_canvas": overlap_union / canvas_pixels if canvas_pixels else 0.0,
               "overlap_pixel_count_accumulated": overlap_pixel_count,
               "source_contribution_pixels": assigned,
               "source_contribution_ratio": {photo_id: count / valid_pixels if valid_pixels else 0.0
                                              for photo_id, count in assigned.items()},
               "overlap_photometric_residual_mean_abs_rgb": float(np.mean(overlap_residuals)) if overlap_residuals else None,
               "ghosting_diagnostics": "NOT_EVALUATED_PARALLAX_AND_EDGE_ARTIFACT_GROUND_TRUTH_UNAVAILABLE"}
    return mosaic, sources, (int(min_x), int(min_y)), {photo.photo_id for photo, _, _ in placements}, metrics


def _normalize_detection(detection: Any, frame_size: tuple[int, int]
                         ) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if not isinstance(detection, dict):
        return None, "MALFORMED_DETECTOR_RESPONSE", "detection record must be an object"
    bbox = detection.get("bbox_xyxy")
    if bbox is None:
        bbox = detection.get("bbox")
    if isinstance(bbox, (str, bytes)) or bbox is None:
        return None, "INVALID_DETECTION_BOX", "bounding box is missing or not a numeric sequence"
    try:
        values = list(bbox)
        if len(values) != 4:
            return None, "INVALID_DETECTION_BOX", "bounding box must contain exactly four coordinates"
        x1, y1, x2, y2 = map(float, values)
    except (TypeError, ValueError, OverflowError):
        return None, "INVALID_DETECTION_BOX", "bounding box coordinates must be numeric"
    width, height = frame_size
    if (not all(math.isfinite(value) for value in (x1, y1, x2, y2))
            or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)):
        return None, "INVALID_DETECTION_BOX", "bounding box must be finite, positive, and inside its source image"
    raw_class = detection.get("class_name")
    if not isinstance(raw_class, str) or not raw_class.strip():
        raw_class = detection.get("predicted_class")
    if not isinstance(raw_class, str) or not raw_class.strip():
        return None, "INVALID_DETECTION_CLASS", "component class is missing"
    class_name = raw_class.strip().lower()
    if class_name not in set(PCB_MODEL_CLASSES) | {"unknown_component"}:
        return None, "INVALID_DETECTION_CLASS", f"unsupported component class: {class_name}"
    confidence = detection.get("confidence")
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, numbers.Real):
            return None, "INVALID_CONFIDENCE", "confidence must be a finite number in [0, 1] or null"
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            return None, "INVALID_CONFIDENCE", "confidence must be a finite number in [0, 1] or null"
    polygon = detection.get("source_polygon_xy")
    if polygon is None:
        polygon = detection.get("polygon_xy")
    normalized = dict(detection)
    normalized["bbox_xyxy"] = [x1, y1, x2, y2]
    normalized["class_name"] = class_name
    normalized["confidence"] = confidence
    detection_id = detection.get("detection_id")
    if detection_id is not None:
        normalized["detection_id"] = (str(detection_id).strip()
            if isinstance(detection_id, (str, int, float)) and not isinstance(detection_id, bool) else None)
    if polygon is not None:
        try:
            points = np.asarray(polygon, dtype=np.float64)
        except (TypeError, ValueError):
            return None, "INVALID_DETECTION_POLYGON", "polygon must be a finite Nx2 coordinate sequence"
        if (points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3
                or not np.isfinite(points).all()
                or np.any(points[:, 0] < 0) or np.any(points[:, 0] > width)
                or np.any(points[:, 1] < 0) or np.any(points[:, 1] > height)
                or abs(float(cv2.contourArea(points.astype(np.float32)))) < 1e-6):
            return None, "INVALID_DETECTION_POLYGON", "polygon needs at least three distinct in-bounds points with nonzero area"
        normalized["source_polygon_xy"] = points.tolist()
    return normalized, None, None


def _map_detection(detection: dict[str, Any], photo: PhotoAsset,
                   matrix: np.ndarray, canvas_origin: tuple[int, int],
                   registration: RegistrationRecord) -> dict[str, Any]:
    bbox = detection.get("bbox_xyxy")
    if bbox is None:
        bbox = detection.get("bbox")
    if bbox is None or len(bbox) != 4:
        return {"photo_id": photo.photo_id, "detection_id": detection.get("detection_id"),
                "status": "INVALID_DETECTION_BOX"}
    x1, y1, x2, y2 = map(float, bbox)
    if (not all(math.isfinite(value) for value in (x1, y1, x2, y2))
            or not (x1 < x2 and y1 < y2)):
        return {"photo_id": photo.photo_id, "detection_id": detection.get("detection_id"),
                "status": "INVALID_DETECTION_BOX"}
    raw_polygon = detection.get("source_polygon_xy")
    corners = np.asarray(raw_polygon if raw_polygon is not None else
                         [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32).reshape(-1, 2)
    if (corners.shape[0] < 3 or not np.isfinite(corners).all()
            or np.any(corners[:, 0] < 0) or np.any(corners[:, 0] > photo.image.width)
            or np.any(corners[:, 1] < 0) or np.any(corners[:, 1] > photo.image.height)):
        return {"photo_id": photo.photo_id, "detection_id": detection.get("detection_id"),
                "status": "INVALID_DETECTION_BOX"}
    mapped = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    mapped[:, 0] -= canvas_origin[0]
    mapped[:, 1] -= canvas_origin[1]
    min_xy, max_xy = mapped.min(axis=0), mapped.max(axis=0)
    transforms = photo.coordinate_transforms
    decoded_to_rectified = np.asarray(transforms["decoded_to_rectified"], dtype=np.float64)
    exif_corrected_to_decoded = np.asarray(transforms["exif_corrected_to_decoded"], dtype=np.float64)
    encoded_to_exif_corrected = np.asarray(transforms["encoded_to_exif_corrected"], dtype=np.float64)
    processed_to_exif_corrected = np.linalg.inv(decoded_to_rectified @ exif_corrected_to_decoded)
    processed_to_encoded = np.linalg.inv(
        decoded_to_rectified @ exif_corrected_to_decoded @ encoded_to_exif_corrected)
    exif_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2),
                                            processed_to_exif_corrected).reshape(-1, 2)
    encoded_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), processed_to_encoded).reshape(-1, 2)
    exif_min, exif_max = exif_corners.min(axis=0), exif_corners.max(axis=0)
    encoded_min, encoded_max = encoded_corners.min(axis=0), encoded_corners.max(axis=0)
    confidence = detection.get("confidence")
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = None
        if confidence is not None and (not math.isfinite(confidence) or not 0 <= confidence <= 1):
            confidence = None
    class_name = str(detection.get("class_name", detection.get("predicted_class", "unknown_component")))
    source_detection_id = str(detection.get("detection_id") or "").strip() or None
    detection_id = source_detection_id or _generated_detection_id(photo.photo_id, class_name, [x1, y1, x2, y2],
        detection.get("model_id"), detection.get("ocr_text"))
    observation_id = _stable_observation_id(photo.photo_id, detection_id, class_name,
        [x1, y1, x2, y2], detection.get("ocr_text"))
    return {"observation_id": observation_id, "photo_id": photo.photo_id, "detection_id": detection_id,
            "source_detection_id": source_detection_id,
            "class_name": class_name,
            "bbox_xyxy": [float(min_xy[0]), float(min_xy[1]), float(max_xy[0]), float(max_xy[1])],
            "polygon_xy": [[float(x), float(y)] for x, y in mapped],
            "processed_bbox_xyxy": [float(corners[:, 0].min()), float(corners[:, 1].min()),
                                    float(corners[:, 0].max()), float(corners[:, 1].max())],
            "original_bbox_xyxy": [float(exif_min[0]), float(exif_min[1]),
                                   float(exif_max[0]), float(exif_max[1])],
            "exif_corrected_bbox_xyxy": [float(exif_min[0]), float(exif_min[1]),
                                         float(exif_max[0]), float(exif_max[1])],
            "encoded_bbox_xyxy": [float(encoded_min[0]), float(encoded_min[1]),
                                  float(encoded_max[0]), float(encoded_max[1])],
            "encoded_image_size_px": list(transforms["encoded_size_px"]),
            "original_coordinate_frame": "EXIF_CORRECTED_IMAGE_PIXEL",
            "source_image_size_px": list(photo.processed_size_px),
            "confidence": confidence, "ocr_marking": detection.get("ocr_text"),
            "recognition_status": "LOW_CONFIDENCE" if confidence is None or confidence < 0.35 else "DETECTED_CANDIDATE",
            "possible_part_number": None, "identification_status": "SUGGESTED_CLASS_ONLY",
            "evidence": "MODEL_DETECTION_CANDIDATE", "coordinate_transform": "PHOTO_TO_CANONICAL_CANDIDATE",
            "registration_status": registration.status, "registration_rmse_px": registration.rmse_px,
            "registration_p95_px": registration.p95_reprojection_error_px,
            "registration_inlier_ratio": registration.inlier_ratio,
            "registration_uncertainty": "INFERRED_NOT_INDEPENDENTLY_VALIDATED",
            "is_reference_photo": registration.status == "REFERENCE",
            "inference_mode": detection.get("inference_mode", "single_image"),
            "source_tile": detection.get("source_tile")}


def _native_rectified_image(photo: PhotoAsset) -> tuple[Image.Image, np.ndarray]:
    """Decode a bounded native-resolution frame and return native-to-processed mapping.

    The normal path deliberately uses a reduced image for registration and maps.
    Tiled inference can recover detail from the original bytes without changing
    the canonical coordinate frame used by the rest of the component map.
    """
    with Image.open(io.BytesIO(photo.original_bytes)) as source:
        native = ImageOps.exif_transpose(source).convert("RGB")
    scale_matrix = photo.coordinate_transforms["exif_corrected_to_decoded"]
    sx, sy = float(scale_matrix[0][0]), float(scale_matrix[1][1])
    if sx <= 0 or sy <= 0 or not math.isfinite(sx + sy):
        raise ValueError("Invalid native-to-decoded scale")
    source_scale = np.diag([sx, sy, 1.0])
    target_scale = np.diag([1.0 / sx, 1.0 / sy, 1.0])
    decoded_to_processed = np.asarray(photo.coordinate_transforms["decoded_to_rectified"], dtype=np.float64)
    native_to_high = target_scale @ decoded_to_processed @ source_scale
    output_size = (max(1, int(round(photo.image.width / sx))),
                   max(1, int(round(photo.image.height / sy))))
    if output_size != native.size or not np.allclose(native_to_high, np.eye(3), atol=1e-6):
        warped = cv2.warpPerspective(np.asarray(native), native_to_high, output_size,
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        native = Image.fromarray(warped, mode="RGB")
    return native, np.linalg.inv(target_scale)


def _detection_to_processed_frame(detection: dict[str, Any], native_to_processed: np.ndarray) -> dict[str, Any]:
    """Attach a processed-frame polygon to a native-resolution tile detection."""
    bbox = detection.get("bbox_xyxy") or detection.get("bbox")
    if bbox is None or len(bbox) != 4:
        return detection
    x1, y1, x2, y2 = map(float, bbox)
    corners = np.asarray(detection.get("source_polygon_xy") or
                         [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32).reshape(-1, 2)
    polygon = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), native_to_processed).reshape(-1, 2)
    row = dict(detection)
    row["source_polygon_xy"] = polygon.tolist()
    row["bbox_xyxy"] = [float(polygon[:, 0].min()), float(polygon[:, 1].min()),
                        float(polygon[:, 0].max()), float(polygon[:, 1].max())]
    row["source_coordinate_frame"] = "NATIVE_RECTIFIED_IMAGE_PIXEL"
    return row


def _detection_sort_key(detection: dict[str, Any]) -> tuple[Any, ...]:
    box = detection.get("bbox_xyxy") or detection.get("bbox") or []
    try:
        normalized_box = tuple(round(float(value), 3) for value in box)
    except (TypeError, ValueError):
        normalized_box = ()
    try:
        confidence_key = -float(detection.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence_key = 0.0
    return (str(detection.get("class_name", detection.get("predicted_class", "unknown_component"))),
            normalized_box, confidence_key,
            str(detection.get("detection_id") or ""), str(detection.get("ocr_text") or ""))


def _generated_detection_id(photo_id: str, class_name: str, box: list[float], model_id: Any,
                            ocr_text: Any) -> str:
    payload = json.dumps([photo_id, class_name, [round(float(value), 2) for value in box],
                          str(model_id or ""), str(ocr_text or "").strip().upper()], separators=(",", ":"))
    return "det_auto_" + hashlib.sha256(payload.encode()).hexdigest()[:20]


def _stable_observation_id(photo_id: str, detection_id: str, class_name: str,
                           box: list[float], ocr_text: Any) -> str:
    payload = json.dumps([photo_id, detection_id, class_name,
                          [round(float(value), 2) for value in box],
                          str(ocr_text or "").strip().upper()], separators=(",", ":"))
    return "observation_" + hashlib.sha256(payload.encode()).hexdigest()[:20]


def _fuse_observations(board_id: str, side: str, observations: list[dict[str, Any]], *,
                       previous_side: dict[str, Any] | None = None,
                       confirmed_identity_by_observation_id: dict[str, str] | None = None,
                       previous_canvas_origin: list[int] | tuple[int, int] | None = None,
                       current_canvas_origin: tuple[int, int] = (0, 0),
                       previous_reference_id: str | None = None,
                       current_reference_id: str | None = None,
                       previous_frame_compatible: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid = [item for item in observations if item.get("bbox_xyxy")]
    groups: list[list[dict[str, Any]]] = []
    for item in sorted(valid, key=lambda row: (row["class_name"], row["bbox_xyxy"], row["photo_id"])):
        # Complete-link fusion prevents a chain A~B and A~C from merging B/C
        # if those observations do not independently satisfy the same gates.
        targets = [group for group in groups if all(_same_component(item, member) for member in group)]
        if len(targets) == 1:
            targets[0].append(item)
        else:
            groups.append([item])
    previous_components = list((previous_side or {}).get("components", []))
    confirmation_map = confirmed_identity_by_observation_id or {}
    adjusted_previous = []
    if previous_frame_compatible and previous_reference_id and current_reference_id:
        old_origin = previous_canvas_origin or (0, 0)
        dx = float(old_origin[0]) - float(current_canvas_origin[0])
        dy = float(old_origin[1]) - float(current_canvas_origin[1])
        for prior in previous_components:
            item = dict(prior)
            if item.get("bbox_xyxy"):
                item["bbox_xyxy"] = [item["bbox_xyxy"][0] + dx, item["bbox_xyxy"][1] + dy,
                                     item["bbox_xyxy"][2] + dx, item["bbox_xyxy"][3] + dy]
                item["canonical_position_xy"] = [item["canonical_position_xy"][0] + dx,
                                                   item["canonical_position_xy"][1] + dy]
            adjusted_previous.append(item)
    prior_by_id = {str(item.get("component_id")): item for item in previous_components if item.get("component_id")}
    used_previous_ids: set[str] = set()
    components = []
    for group in groups:
        boxes = np.asarray([item["bbox_xyxy"] for item in group], dtype=np.float64)
        box = np.median(boxes, axis=0).tolist()
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        observation_ids = sorted({str(item.get("observation_id") or _observation_id(item)) for item in group})
        explicit_ids = {confirmation_map[item_id] for item_id in observation_ids if item_id in confirmation_map}
        identity_status = "NEW_COMPONENT"
        identity_confidence = None
        identity_evidence: list[dict[str, Any]] = []
        matched_prior = None
        if len(explicit_ids) > 1:
            identity_status = "IDENTITY_CONFLICT"
        elif explicit_ids:
            requested_id = next(iter(explicit_ids))
            requested = prior_by_id.get(requested_id)
            if requested and str(requested.get("component_type", "")).lower() == group[0]["class_name"].lower() and requested_id not in used_previous_ids:
                matched_prior = requested
                identity_status, identity_confidence = "USER_CONFIRMED", 1.0
                identity_evidence.append({"method": "EXPLICIT_USER_IDENTITY_MAPPING", "component_id": requested_id})
            else:
                identity_status = "IDENTITY_CONFLICT"
                identity_evidence.append({"method": "USER_MAPPING_REJECTED", "requested_component_id": requested_id,
                                          "reason": "unknown_id_duplicate_assignment_or_class_conflict"})
        elif adjusted_previous:
            ranked = sorted(((score, prior) for prior in adjusted_previous
                             if str(prior.get("component_id")) not in used_previous_ids
                             for score in [_identity_match_score(group, box, prior)] if score is not None),
                            key=lambda row: row[0], reverse=True)
            if ranked:
                best_score, best_prior = ranked[0]
                next_score = ranked[1][0] if len(ranked) > 1 else 0.0
                if best_score >= 0.78 and best_score - next_score >= 0.15:
                    matched_prior = best_prior
                    identity_status, identity_confidence = "IDENTITY_MATCHED", float(best_score)
                    if matched_prior.get("user_confirmed_identity"):
                        identity_status, identity_confidence = "USER_CONFIRMED", 1.0
                    identity_evidence.append({"method": "ONE_TO_ONE_GEOMETRY_AND_CLASS_MATCH",
                                              "score": float(best_score), "runner_up_score": float(next_score),
                                              "coordinate_frame_compatible": True})
                elif best_score >= 0.60:
                    identity_status = "IDENTITY_AMBIGUOUS"
                    identity_evidence.append({"method": "AMBIGUOUS_PRIOR_COMPONENT_MATCH",
                                              "best_score": float(best_score), "runner_up_score": float(next_score)})
        if matched_prior:
            component_id = str(matched_prior["component_id"])
            used_previous_ids.add(component_id)
        else:
            reference_observations = [item for item in group if item.get("is_reference_photo")]
            primary_observation = min(reference_observations or group,
                                      key=lambda item: str(item.get("observation_id") or _observation_id(item)))
            stable_seed = str(primary_observation.get("observation_id") or _observation_id(primary_observation))
            component_id = "component_" + hashlib.sha256(
                f"{board_id}|{side}|{group[0]['class_name']}|{stable_seed}".encode()).hexdigest()[:16]
        markings = {str(item["ocr_marking"]).strip().upper() for item in group if item.get("ocr_marking")}
        if len(markings) > 1:
            identity_status = "IDENTITY_CONFLICT"
            identity_evidence.append({"method": "OCR_MARKING_CONFLICT", "markings": sorted(markings)})
        for other in valid:
            if other in group:
                continue
            proximity_by_member = [(member, _observation_pair_metrics(member, other)) for member in group]
            nearby = [(member, metrics) for member, metrics in proximity_by_member
                      if metrics["iou"] >= 0.65 and metrics["normalized_center_distance"] <= 0.25
                      and metrics["bbox_scale_ratio"] >= 0.65]
            if not nearby:
                continue
            proximity = max((metrics for _, metrics in nearby),
                            key=lambda metrics: (metrics["iou"], -metrics["normalized_center_distance"]))
            if other.get("photo_id") in {item.get("photo_id") for item in group}:
                if identity_status != "IDENTITY_CONFLICT":
                    identity_status = "IDENTITY_AMBIGUOUS"
                identity_evidence.append({"method": "SAME_PHOTO_OVERLAPPING_DETECTIONS",
                    "observation_id": other.get("observation_id"), **proximity})
                continue
            if str(other.get("class_name", "")).lower() != str(group[0].get("class_name", "")).lower():
                identity_status = "IDENTITY_CONFLICT"
                identity_evidence.append({"method": "OVERLAPPING_CLASS_CONFLICT",
                    "observation_id": other.get("observation_id"), "other_class": other.get("class_name"),
                    **proximity})
            elif any(_same_component(member, other) for member in group) and not all(
                    _same_component(other, member) for member in group):
                if identity_status != "IDENTITY_CONFLICT":
                    identity_status = "IDENTITY_AMBIGUOUS"
                identity_evidence.append({"method": "PAIRWISE_FUSION_DISAGREEMENT",
                    "observation_id": other.get("observation_id"),
                    "pairwise_matches": [item.get("observation_id") for item, _ in nearby
                                         if _same_component(item, other)],
                    "group_observation_ids": [item.get("observation_id") for item in group], **proximity})
            elif not all(item.get("registration_status") in {None, "REFERENCE", "REGISTERED_CANDIDATE"}
                         and (item.get("registration_rmse_px") is None or item["registration_rmse_px"] <= 4.0)
                         and (item.get("registration_inlier_ratio") is None or item["registration_inlier_ratio"] >= 0.2)
                         for item in (*group, other)):
                if identity_status != "IDENTITY_CONFLICT":
                    identity_status = "IDENTITY_AMBIGUOUS"
                identity_evidence.append({"method": "POTENTIAL_MATCH_REJECTED_UNCERTAIN_REGISTRATION",
                                          "observation_id": other.get("observation_id"), **proximity})
        if len(group) > 1:
            identity_evidence.extend({"method": "SAME_COMPONENT_GEOMETRY_GATE",
                "observation_ids": [str(first.get("observation_id") or _observation_id(first)),
                                    str(second.get("observation_id") or _observation_id(second))],
                **_observation_pair_metrics(first, second)}
                for index, first in enumerate(group) for second in group[index + 1:])
        if any(item.get("registration_rmse_px") is not None and item["registration_rmse_px"] > 4.0 for item in group):
            identity_status = "IDENTITY_AMBIGUOUS"
            identity_evidence.append({"method": "REGISTRATION_UNCERTAINTY", "reason": "RMSE_ABOVE_4PX"})
        historical_refs = list((matched_prior or {}).get("measurement_history_refs", []))
        if (matched_prior or {}).get("measurement_history_ref"):
            historical_refs.append(matched_prior["measurement_history_ref"])
        components.append({"component_id": component_id, "component_type": group[0]["class_name"],
            "canonical_position_xy": [float(center[0]), float(center[1])], "bbox_xyxy": [float(x) for x in box],
            "polygon_xy": [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]],
            "source_image_ids": sorted({item["photo_id"] for item in group}),
            "component_observation_ids": observation_ids,
            "original_image_positions": [{"photo_id": item["photo_id"], "bbox_xyxy": item["original_bbox_xyxy"]} for item in group],
            "detection_confidence": max((float(item["confidence"]) for item in group if item.get("confidence") is not None), default=None),
            "ocr_marking": next((item["ocr_marking"] for item in group if item.get("ocr_marking")), None),
            "possible_part_number": None, "identification_status": "SUGGESTED_UNCONFIRMED",
            "identity_fusion_status": "MULTI_OBSERVATION_FUSED" if len(group) > 1 else "SINGLE_OBSERVATION",
            "identity_match_status": identity_status,
            "identity_confidence": identity_confidence,
            "identity_evidence": identity_evidence,
            "user_confirmed_identity": identity_status == "USER_CONFIRMED" or bool((matched_prior or {}).get("user_confirmed_identity")),
            "measurement_history_refs": sorted(set(map(str, historical_refs))),
            "historical_component_ids": sorted(set((matched_prior or {}).get("historical_component_ids", []))),
            "evidence": [{"photo_id": item["photo_id"], "detection_id": item["detection_id"],
                          "observation_id": item.get("observation_id"), "confidence": item["confidence"],
                          "registration_status": item.get("registration_status"),
                          "registration_uncertainty": item.get("registration_uncertainty"),
                          "source": item["evidence"]} for item in group]})
    mappings = []
    for prior in previous_components:
        prior_id = str(prior.get("component_id", ""))
        if not prior_id:
            continue
        current = next((item for item in components if item["component_id"] == prior_id), None)
        mappings.append({"previous_component_id": prior_id,
                         "current_component_id": prior_id if current else None,
                         "status": "PRESERVED" if current else "NOT_OBSERVED_NO_ID_REASSIGNMENT",
                         "measurement_history_refs_preserved": list(prior.get("measurement_history_refs", [])),
                         "identity_review_required": not bool(current)})
    return components, mappings


def _same_component(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if first["photo_id"] == second["photo_id"]:
        return False
    if first["class_name"].lower() != second["class_name"].lower():
        return False
    first_marking = str(first.get("ocr_marking") or "").strip().upper()
    second_marking = str(second.get("ocr_marking") or "").strip().upper()
    if first_marking and second_marking and first_marking != second_marking:
        return False
    for observation in (first, second):
        if (observation.get("registration_status") not in {None, "REFERENCE", "REGISTERED_CANDIDATE"}
                or (observation.get("registration_rmse_px") is not None
                    and observation["registration_rmse_px"] > 4.0)
                or (observation.get("registration_inlier_ratio") is not None
                    and observation["registration_inlier_ratio"] < 0.2)):
            return False
    metrics = _observation_pair_metrics(first, second)
    return metrics["iou"] >= 0.65 and metrics["normalized_center_distance"] <= 0.25 and metrics["bbox_scale_ratio"] >= 0.65


def _observation_pair_metrics(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    a, b = first["bbox_xyxy"], second["bbox_xyxy"]
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(1e-9, (a[2]-a[0]) * (a[3]-a[1]))
    area_b = max(1e-9, (b[2]-b[0]) * (b[3]-b[1]))
    iou = intersection / (area_a + area_b - intersection)
    center_a = np.array([(a[0]+a[2])/2, (a[1]+a[3])/2])
    center_b = np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])
    diagonal_a = max(1.0, math.hypot(a[2]-a[0], a[3]-a[1]))
    diagonal_b = max(1.0, math.hypot(b[2]-b[0], b[3]-b[1]))
    distance = float(np.linalg.norm(center_a-center_b)) / max(1.0, (diagonal_a + diagonal_b) / 2)
    return {"iou": float(iou), "normalized_center_distance": float(distance),
            "bbox_scale_ratio": float(min(area_a, area_b) / max(area_a, area_b))}


def _observation_id(item: dict[str, Any]) -> str:
    return "observation_" + hashlib.sha256(
        f"{item.get('photo_id')}|{item.get('detection_id')}|{item.get('class_name')}".encode()).hexdigest()[:16]


def _identity_match_score(group: list[dict[str, Any]], new_box: list[float], prior: dict[str, Any]) -> float | None:
    if str(prior.get("component_type", "")).lower() != str(group[0].get("class_name", "")).lower():
        return None
    old = prior.get("bbox_xyxy")
    if not old or len(old) != 4:
        return None
    metrics = _observation_pair_metrics({"bbox_xyxy": new_box}, {"bbox_xyxy": old})
    if metrics["normalized_center_distance"] > 0.8 or metrics["bbox_scale_ratio"] < 0.3:
        return None
    return 0.55 * metrics["iou"] + 0.30 * max(0.0, 1.0 - metrics["normalized_center_distance"]) + 0.15 * metrics["bbox_scale_ratio"]


def _previous_reference_id(previous_side: dict[str, Any]) -> str | None:
    canonical = previous_side.get("canonical_image", {})
    if canonical.get("reference_photo_id"):
        return canonical["reference_photo_id"]
    return next((item.get("photo_id") for item in previous_side.get("registrations", [])
                 if item.get("status") == "REFERENCE"), None)


def _same_matrix(first: Any, second: Any) -> bool:
    if first is None or second is None:
        return first is None and second is None
    try:
        a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
        return a.shape == (3, 3) and b.shape == (3, 3) and np.allclose(a, b, rtol=0.0, atol=1e-7)
    except (TypeError, ValueError):
        return False


def _load_previous_map(value: dict[str, Any] | str | Path | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        try:
            value = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MultiImageError("INVALID_PREVIOUS_MAP", str(exc)) from exc
    if not isinstance(value, dict):
        raise MultiImageError("INVALID_PREVIOUS_MAP", "Previous identity data must be a JSON object")
    if isinstance(value.get("sides"), dict):
        return value
    if value.get("schema_version") == "circuitcare.component-identity-bundle.v1":
        sides = {}
        for side, manifest in value.get("sides", {}).items():
            sides[side] = {"components": manifest.get("components", []),
                           "canonical_image": manifest.get("canonical_image", {})}
        return {"board_id": value.get("board_id"), "sides": sides}
    if value.get("schema_version") == "circuitcare.component-identities.v1":
        side = value.get("board_side")
        if side not in {"TOP", "BOTTOM"}:
            raise MultiImageError("INVALID_PREVIOUS_MAP", "Identity manifest has an invalid side")
        return {"board_id": value.get("board_id"), "sides": {side: {
            "components": value.get("components", []), "canonical_image": value.get("canonical_image", {})}}}
    raise MultiImageError("INVALID_PREVIOUS_MAP", "Unsupported previous component map or identity manifest schema")


def save_identity_manifest(path: str | Path, component_map: dict[str, Any]) -> None:
    """Atomically persist stable IDs and provenance without serializing image blobs."""
    destination = Path(path)
    bundle = {"schema_version": "circuitcare.component-identity-bundle.v1",
              "board_id": component_map.get("board_id"),
              "sides": {side: record["identity_manifest"]
                        for side, record in component_map.get("sides", {}).items()}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _board_boundary_records(photos: list[PhotoAsset], started: float,
                            budget: ImageBudget) -> list[dict[str, Any]]:
    results = []
    for photo in photos:
        if time.perf_counter() - started > budget.max_runtime_seconds:
            results.append({"photo_id": photo.photo_id, "status": "NOT_EVALUATED_TIMEOUT"})
            continue
        try:
            detected = detect_board_candidates(photo.image, max_candidates=3)
            results.append({"photo_id": photo.photo_id, "status": detected["status"],
                            "candidates": detected["candidates"], "verification": "UNCONFIRMED_CANDIDATE"})
        except (ValueError, cv2.error) as exc:
            results.append({"photo_id": photo.photo_id, "status": "NOT_DETECTED", "reason": str(exc)})
    return results


def to_json(component_map: dict[str, Any]) -> str:
    """Stable JSON serialization for API/UI adapters; NumPy values are excluded."""
    return json.dumps(component_map, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _png_base64(array: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", array)
    if not ok:
        raise MultiImageError("ARTIFACT_ENCODING_FAILED", "Could not encode canonical image artifact")
    return base64.b64encode(encoded.tobytes()).decode("ascii")
