"""Diagnostic overlays for automatic registration; all generated views stay marked unverified."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from circuitcare.vision.reconstruction.anchor_detection import ImageAnchor
from circuitcare.vision.reconstruction.anchor_matching import AutomaticMatchResult, DesignAnchor


def save_registration_overlay(image_path: str | Path, output_path: str | Path, *,
                              image_anchors: list[ImageAnchor],
                              design_anchors: list[DesignAnchor],
                              result: AutomaticMatchResult) -> None:
    """Compare detected image centers with transformed design anchors; never imply validation."""
    with Image.open(image_path) as loaded:
        canvas = loaded.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    matrix = (np.asarray(result.transform.matrix, dtype=np.float64)
              if result.transform is not None else None)
    design_by_id = {anchor.anchor_id: anchor for anchor in design_anchors}
    image_by_id = {anchor.anchor_id: anchor for anchor in image_anchors}
    for index, anchor in enumerate(image_anchors, start=1):
        x, y = anchor.center_px
        radius = 6
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=(0, 210, 90), width=2)
        draw.text((x+7, y-9), f"I{index}", fill=(0, 120, 50))
    if matrix is not None:
        for index, anchor in enumerate(design_anchors, start=1):
            point = np.asarray([[[anchor.x_mm, anchor.y_mm]]], dtype=np.float64)
            u, v = cv2.perspectiveTransform(point, matrix)[0, 0]
            if not (0 <= u < width and 0 <= v < height):
                continue
            x, y = int(round(u)), int(round(v))
            draw.line((x-6, y, x+6, y), fill=(235, 40, 180), width=2)
            draw.line((x, y-6, x, y+6), fill=(235, 40, 180), width=2)
            draw.text((x+7, y+2), f"D{index}", fill=(150, 0, 110))
    for match in result.matches:
        design = design_by_id.get(match.design_anchor_id)
        image_anchor = image_by_id.get(match.image_anchor_id)
        if design is None or image_anchor is None or matrix is None:
            continue
        u, v = cv2.perspectiveTransform(
            np.asarray([[[design.x_mm, design.y_mm]]], dtype=np.float64), matrix,
        )[0, 0]
        x, y = image_anchor.center_px
        draw.line((x, y, int(round(u)), int(round(v))), fill=(255, 150, 0), width=1)
    # Opaque title panel and warning are embedded so exports retain their status.
    panel_height = 54
    draw.rectangle((0, 0, width, panel_height), fill=(255, 255, 255))
    draw.text((8, 7), f"DESIGN-ASSISTED REGISTRATION · {result.status}", fill=(130, 0, 0))
    draw.text((8, 28), "Green circles: image candidates · Magenta crosses: transformed design anchors · Orange: match residual",
              fill=(25, 25, 25))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_top_bottom_comparison(top_path: str | Path, bottom_path: str | Path,
                               output_path: str | Path) -> None:
    """Create a side-by-side view only; it does not claim physical TOP/BOTTOM alignment."""
    with Image.open(top_path) as top_source, Image.open(bottom_path) as bottom_source:
        top, bottom = top_source.convert("RGB"), bottom_source.convert("RGB")
    target_height = max(top.height, bottom.height)
    scale_top = target_height / top.height
    scale_bottom = target_height / bottom.height
    top = top.resize((round(top.width * scale_top), target_height), Image.Resampling.LANCZOS)
    bottom = bottom.resize((round(bottom.width * scale_bottom), target_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (top.width + bottom.width, target_height + 42), "white")
    canvas.paste(top, (0, 42))
    canvas.paste(bottom, (top.width, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), "TOP and BOTTOM source views · displayed side-by-side only · not co-registered",
              fill=(130, 0, 0))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
