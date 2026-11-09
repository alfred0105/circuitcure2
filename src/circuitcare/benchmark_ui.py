"""Read-only Streamlit view of saved benchmark runs; never starts evaluations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from circuitcare.benchmark.runner import load_result_records


def render_benchmark_tab(st: Any, *, root: Path) -> None:
    st.subheader("AI 성능 평가")
    st.caption("이 화면은 저장된 평가 결과만 표시합니다. 벤치마크를 자동 실행하지 않습니다.")
    records = []
    for relative in ("evaluation/phase17/results", "evaluation/phase17_1/results"):
        records.extend(load_result_records(root / relative))
    records.sort(key=lambda item: item.get("metadata", {}).get("timestamp_utc", ""))
    if not records:
        st.info("저장된 Phase 17 평가 결과가 없습니다. 결과가 없으면 정확도를 0%나 100%로 표시하지 않습니다.")
        st.write({"Holdout": "NOT_EVALUATED", "Real repair accuracy": "NOT_EVALUATED",
                  "Phase 16 learned prior": "NOT_EVALUATED"})
        return
    record = records[-1]
    st.caption(f"최근 실행: {record.get('run_id')} · {record.get('status')} · {record.get('split')}")
    overview = st.tabs(["Overview", "Method Comparison", "Accuracy", "Efficiency", "Evaluation Integrity"])
    with overview[0]:
        cols = st.columns(4)
        cols[0].metric("Cases", record.get("case_count", 0))
        cols[1].metric("Topology Groups", record.get("topology_group_count", 0))
        cols[2].metric("Supported Truth", record.get("supported_case_count", 0))
        cols[3].metric("Holdout", _display(record.get("holdout_status")))
        st.write({"dataset": record.get("dataset_id"), "split": record.get("split"),
                  "independent_holdout": record.get("independent_holdout"),
                  "metric_schema_version": record.get("metric_schema_version", "NOT_EVALUATED"),
                  "candidate_grid_analysis": record.get("candidate_grid_analysis", "NOT_EVALUATED"),
                  "Phase 16 prior": record.get("phase16_prior_status")})
    with overview[1]:
        table = []
        for name, result in record.get("methods", {}).items():
            generation = result.get("generation_metrics", {})
            recall = generation.get("recall", {})
            ranking = result.get("ranking", {})
            top_k = ranking.get("top_k_accuracy", {})
            table.append({"Method": name, "Status": result.get("status"),
                "Component Recall": recall.get("component", {}).get("recall"),
                "Unique Component Precision": recall.get("component", {}).get("precision"),
                "Unique Fault Mode Precision": recall.get("fault_mode", {}).get("precision"),
                "Fault Mode Recall": recall.get("fault_mode", {}).get("recall"),
                "Exact Precision": recall.get("exact_hypothesis", {}).get("precision"),
                "Exact Recall": recall.get("exact_hypothesis", {}).get("recall"),
                "Strict Top-1": ranking.get("strict_top1", top_k.get("1")),
                "Tie-Aware Top-1": ranking.get("tie_aware_top1"),
                "Top-3": top_k.get("3")})
        if table:
            st.dataframe(table, use_container_width=True, hide_index=True)
            chart_rows = [{key: row[key] for key in ("Method", "Component Recall", "Fault Mode Recall", "Exact Recall")}
                          for row in table if any(isinstance(row[key], (int, float))
                              for key in ("Component Recall", "Fault Mode Recall", "Exact Recall"))]
            if chart_rows:
                st.bar_chart(chart_rows, x="Method", y=["Component Recall", "Fault Mode Recall", "Exact Recall"])
        else:
            st.info("Method comparison is NOT_EVALUATED.")
    with overview[2]:
        for name, result in record.get("methods", {}).items():
            st.markdown(f"**{name}** · {result.get('status', 'NOT_EVALUATED')}")
            generation = result.get("generation_metrics", {})
            if generation.get("status") == "MEASURED":
                st.dataframe([{"Level": key, **value} for key, value in generation.get("recall", {}).items()],
                             use_container_width=True, hide_index=True)
                st.json({"candidate_counts": generation.get("candidate_counts", "NOT_EVALUATED"),
                         "raw_candidate_precision": generation.get("raw_candidate_precision", "NOT_EVALUATED")})
            else:
                st.caption("Candidate accuracy: NOT_EVALUATED")
            ranking = result.get("ranking", {})
            st.write({"SPICE ranking": ranking if ranking else "NOT_EVALUATED"})
    with overview[3]:
        for name, result in record.get("methods", {}).items():
            st.write({"method": name, "candidate runtime": result.get("candidate_seconds", "NOT_EVALUATED"),
                      "SPICE runtime seconds": [run.get("seconds") for run in result.get("runs", [])],
                      "LLM wall seconds": [row.get("llm_wall_seconds") for row in record.get("case_runs", [])
                                           if row.get("llm_attempted")],
                      "selected parameter error": [run.get("selected_parameter_error", "NOT_EVALUATED")
                                                   for run in result.get("runs", [])],
                      "SPICE budget": record.get("spice_run_budget_per_method_case", "NOT_EVALUATED")})
    with overview[4]:
        st.json({"input_audit": record.get("input_audit", "NOT_EVALUATED"),
                 "metadata": record.get("metadata", {}),
                 "dataset_id": record.get("dataset_id"), "dataset_hash": record.get("metadata", {}).get("dataset_hash"),
                 "ground_truth_hash": record.get("metadata", {}).get("ground_truth_hash"),
                 "holdout_status": record.get("holdout_status"),
                 "known_limitations": record.get("limitations", [])})


def _display(value: Any) -> str:
    return "NOT_EVALUATED" if value is None else str(value)
