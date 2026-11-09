"""Render concise Markdown summaries from immutable benchmark result JSON."""

from __future__ import annotations

from typing import Any


def render_summary_markdown(record: dict[str, Any]) -> str:
    metrics = record.get("methods", {})
    lines = ["# CircuitCare Benchmark Summary", "", f"- Run ID: `{record.get('run_id', 'unknown')}`",
             f"- Status: `{record.get('status', 'NOT_EVALUATED')}`",
             f"- Dataset: `{record.get('dataset_id', 'unknown')}`",
             f"- Split: `{record.get('split', 'unknown')}`", "", "| Method | Component Recall | Fault Mode Recall | Exact Recall | Top-1 | Runtime |",
             "|---|---:|---:|---:|---:|---:|"]
    for method, value in metrics.items():
        levels = value.get("levels", {})
        candidate = value.get("candidate_seconds")
        lines.append("| {method} | {component} | {fault} | {exact} | {top1} | {runtime} |".format(
            method=method,
            component=_cell(levels.get("component", {}).get("recall")),
            fault=_cell(levels.get("fault_mode", {}).get("recall")),
            exact=_cell(levels.get("exact_hypothesis", {}).get("recall")),
            top1=_cell(value.get("ranking", {}).get("top_k_accuracy", {}).get("1")),
            runtime=_cell(candidate)))
    lines.extend(["", "Metrics not present in the saved record are not evaluated. Synthetic benchmark scores are not real-repair accuracy."])
    return "\n".join(lines) + "\n"


def _cell(value: Any) -> str:
    return "NOT_EVALUATED" if value is None else f"{value:.4g}" if isinstance(value, (int, float)) else str(value)
