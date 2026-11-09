"""Generated, unlicensed PCB-like images for algorithm/UI smoke checks only."""

from __future__ import annotations

import cv2
import numpy as np


def synthetic_front_back(*, include_ground_truth: bool = False):
    """Make front/back pictures with pads, routes, fiducials, glare, and perspective."""
    front = np.full((360, 480, 3), (28, 85, 58), dtype=np.uint8)
    cv2.rectangle(front, (35, 30), (445, 330), (32, 102, 67), -1)
    trace_truth = np.zeros(front.shape[:2], dtype=np.uint8)
    copper = (215, 216, 205)
    # Straight, right-angle, branched, near-parallel, and crossing candidates.
    routes = [((80, 95), (390, 95), 5), ((110, 150), (280, 150), 5),
              ((280, 150), (280, 260), 5), ((210, 150), (210, 205), 5),
              ((105, 220), (350, 220), 4), ((190, 180), (310, 250), 3)]
    for start, end, width in routes:
        cv2.line(front, start, end, copper, width)
        cv2.line(trace_truth, start, end, 255, width)
    # Deliberate gap models an occluded route: pixels in the gap remain absent.
    for start, end in (((80, 280), (210, 280)), ((250, 280), (390, 280))):
        cv2.line(front, start, end, copper, 5)
        cv2.line(trace_truth, start, end, 255, 5)
    for point in ((80, 95), (390, 95), (110, 150), (280, 260), (105, 220), (350, 220)):
        cv2.circle(front, point, 10, (225, 224, 214), -1)
        cv2.circle(front, point, 4, (25, 28, 24), -1)
    # Rectangular SMD-like pads and mounting holes.
    cv2.rectangle(front, (370, 145), (398, 164), (220, 221, 210), -1)
    for point in ((55, 50), (425, 50), (55, 310), (425, 310)):
        cv2.circle(front, point, 7, (15, 17, 15), -1)
    cv2.putText(front, "SYNTHETIC PCB", (115, 55), cv2.FONT_HERSHEY_SIMPLEX, .55,
                (220, 225, 220), 1, cv2.LINE_AA)
    # A glare patch and deterministic sensor noise create quality-test cases.
    cv2.ellipse(front, (355, 70), (42, 13), -25, 0, 360, (250, 250, 248), -1)
    rng = np.random.default_rng(2041)
    noise = rng.normal(0, 2.0, front.shape).astype(np.int16)
    front = np.clip(front.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Perspective photograph: a trapezoid maps to the image canvas.
    src = np.float32([[0, 0], [479, 0], [479, 359], [0, 359]])
    dst = np.float32([[10, 7], [468, 0], [479, 350], [0, 359]])
    perspective = cv2.getPerspectiveTransform(src, dst)
    front = cv2.warpPerspective(front, perspective, (480, 360))
    trace_truth = cv2.warpPerspective(trace_truth, perspective, (480, 360), flags=cv2.INTER_NEAREST)
    cv2.polylines(front, [dst.astype(np.int32)], True, (11, 20, 15), 5, cv2.LINE_AA)
    back = cv2.flip(front, 1).copy()
    back_truth = cv2.flip(trace_truth, 1)
    cv2.putText(back, "BACK", (205, 180), cv2.FONT_HERSHEY_SIMPLEX, .8,
                (205, 210, 205), 2, cv2.LINE_AA)
    return (front, back, trace_truth > 0, back_truth > 0) if include_ground_truth else (front, back)
