"""Small, explicit metric helpers for future aligned PRISM evaluations."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def binary_mask_metrics(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Compute binary segmentation metrics only after callers verify alignment."""
    pred = np.asarray(predicted).astype(bool)
    truth = np.asarray(reference).astype(bool)
    if pred.shape != truth.shape:
        raise ValueError("Prediction and reference masks must have the same shape")
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    union = tp + fp + fn
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "iou": tp / union if union else 1.0,
            "false_positive_pixels": fp, "false_negative_pixels": fn}


def greedy_point_match(predicted: Iterable[tuple[float, float]],
                       reference: Iterable[tuple[float, float]],
                       tolerance_px: float) -> dict[str, float | int]:
    """One-to-one nearest matching; neither reference nor prediction is reused."""
    if tolerance_px <= 0:
        raise ValueError("tolerance_px must be positive")
    pred, truth = list(predicted), list(reference)
    pairs = sorted((float(np.hypot(px-rx, py-ry)), i, j)
                   for i, (px, py) in enumerate(pred)
                   for j, (rx, ry) in enumerate(truth))
    used_pred: set[int] = set()
    used_truth: set[int] = set()
    errors = []
    for distance, i, j in pairs:
        if distance > tolerance_px:
            break
        if i not in used_pred and j not in used_truth:
            used_pred.add(i)
            used_truth.add(j)
            errors.append(distance)
    tp = len(errors)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    return {"true_positive": tp, "false_positive": len(pred)-tp, "false_negative": len(truth)-tp,
            "precision": precision, "recall": recall,
            "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
            "mean_error_px": float(np.mean(errors)) if errors else 0.0,
            "matched_errors_px": errors}
