"""Streamlit interaction for constrained fault proposals and explicit SPICE runs."""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Any

from .fault_ai.fault_catalog import generate_rule_candidates
from .fault_ai.hypothesis_generator import merge_candidates
from .fault_ai.hypothesis_validator import validate_hypotheses
from .fault_ai.hypothesis_ranking import conditional_bayesian_update, rank_hypotheses
from .fault_ai.llm_hypothesis import propose_hypotheses
from .fault_ai.simulation_bridge import preflight_candidates, simulate_selected
from .fault_ai.observation_matcher import compare_observations
from .repair_feedback.case_repository import RepairCaseRepository
from .repair_feedback.prior_registry import active_prior
from .workspace.measurement_provenance import measurement_set_sha256


def render_fault_ai_tab(st: Any, *, root: Path, circuit: Any, case_name: str, case: dict[str, Any],
                        client: Any, models: list[dict[str, Any]], context_key: str) -> None:
    st.subheader("AI 고장 가설 분석")
    st.caption("규칙과 로컬 LLM은 후보를 제안하고, 기존 Fault Injector와 ngspice가 제한된 모델을 검증합니다. 일치 결과는 실제 고장 확정이 아닙니다.")
    mapping = _read_json(root / case["mapping"])
    probes = _read_json(root / case["probes"])
    base_path = root / case["base"]
    spice_text = base_path.read_text(encoding="utf-8")
    supported_nodes = [str(node) for node in probes.get("voltage_nodes", [])]
    llm_names = [item.get("name") for item in models if item.get("name")]
    chosen_model = st.selectbox("가설 생성 로컬 모델", llm_names, key=f"fault_ai_model_{context_key}") if client and llm_names else None
    st.caption(f"회로: {case_name} · 부품 {len(circuit.components)}개 · 명시 매핑 {len(mapping)}개 · Probe 노드 {len(supported_nodes)}개")
    with st.form(f"fault_ai_form_{context_key}"):
        symptom = st.text_area("관측 증상", placeholder="예: VOUT가 기대값보다 높게 측정됨", max_chars=2000)
        cols = st.columns(4)
        measurement_recorded = cols[0].checkbox("측정값을 기록함", value=False, key=f"fault_ai_recorded_{context_key}")
        node = cols[1].selectbox("측정 노드", supported_nodes, key=f"fault_ai_node_{context_key}") if supported_nodes else ""
        value = cols[2].number_input("측정값 (V)", value=0.0, format="%.6g", key=f"fault_ai_value_{context_key}")
        uncertainty = cols[3].number_input("불확실도 (V, 알면 입력)", min_value=0.0, value=0.0, format="%.6g", key=f"fault_ai_sigma_{context_key}")
        source_label = st.selectbox("측정 출처", ["실제 측정", "시뮬레이션", "가상 데모"], key=f"fault_ai_source_{context_key}")
        source = {"실제 측정": "user_measurement", "시뮬레이션": "simulation", "가상 데모": "simulated_demo"}[source_label]
        if measurement_recorded:
            st.caption("기록 체크 후 입력한 0.0 V도 유효한 측정값으로 저장됩니다. 실제 측정은 사용자가 신고한 값이며 독립 검증을 뜻하지 않습니다.")
        generate = st.form_submit_button("규칙 기반 후보 생성")
        generate_llm = st.form_submit_button("로컬 AI 가설 제안", disabled=not (client and chosen_model))
    state_key = f"fault_ai_state_{context_key}"
    if generate or generate_llm:
        observations = _collect_observation(measurement_recorded, node, value, uncertainty, source)
        candidates = generate_rule_candidates(circuit, mapping, spice_text, observations)
        candidates = preflight_candidates(candidates, component_refs=set(circuit.components),
                                          component_mapping=mapping, spice_text=spice_text)
        state = {"observations": observations, "symptom": symptom, "rule_candidates": candidates,
                 "hypotheses": candidates, "llm_metadata": None, "context_key": context_key}
        state["measurement_set_sha256"] = measurement_set_sha256(observations)
        state["hypothesis_set_sha256"] = hashlib.sha256(
            json.dumps(candidates, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        if generate_llm:
            components = [{"ref": c.ref, "value": c.value, "library_id": c.library_id,
                           "nets": list(circuit.component_nets.get(c.ref, ())),
                           "spice_mapping": {"model_type": mapping[c.ref].get("model_type"), "available": True}
                           if c.ref in mapping else {"available": False}}
                          for c in circuit.components.values()]
            try:
                proposed = propose_hypotheses(client, chosen_model, circuit_id=case_name,
                    components=components, observations=observations, symptoms=symptom,
                    executable_catalog=candidates)
                llm_rows = validate_hypotheses(proposed["hypotheses"], component_refs=set(circuit.components),
                    component_mapping=mapping, spice_text=spice_text, allowed_catalog=candidates)
                state["hypotheses"] = merge_candidates(candidates, llm_rows)
                state["llm_metadata"] = {k: proposed.get(k) for k in ("model", "model_info", "prompt_version", "latency_seconds", "json_mode")}
            except (ValueError, RuntimeError, OSError) as exc:
                state["llm_metadata"] = {"status": "failed", "error": str(exc)}
        state["hypothesis_set_sha256"] = hashlib.sha256(
            json.dumps(state["hypotheses"], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        st.session_state[state_key] = state
        # The overview/report runs earlier in the page; rerun so it can include
        # the newly generated candidate snapshot in the same active context.
        st.rerun()
    state = st.session_state.get(state_key)
    if not state or state.get("context_key") != context_key:
        st.info("증상과 측정값을 입력하고 후보 생성 버튼을 누르세요. SPICE는 별도 실행 버튼에서만 작동합니다.")
        _render_eval_section(st, root)
        return
    if state.get("llm_metadata"):
        st.write("**로컬 LLM:**", state["llm_metadata"])
    rows = state["hypotheses"]
    summary = {"규칙 후보": len(state["rule_candidates"]), "전체 후보": len(rows),
               "사전검증 통과": sum(row.get("validation_status") == "VALIDATED_EXECUTABLE" for row in rows),
               "실행 실패/미지원": sum(row.get("validation_status") != "VALIDATED_EXECUTABLE" for row in rows)}
    st.json(summary)
    if rows:
        st.dataframe([{"가설 ID": r.get("hypothesis_id"), "부품": r.get("component_ref"),
                       "고장 유형": r.get("fault_type"), "출처": r.get("generation_method"),
                       "사전검증": r.get("validation_status"), "설명": r.get("reason", ""),
                       "미지원 사유": "; ".join(r.get("validation_reasons", []))}
                      for r in rows], use_container_width=True, hide_index=True)
    executable = [row for row in rows if row.get("validation_status") == "VALIDATED_EXECUTABLE"]
    run_key = f"fault_ai_run_{context_key}"
    ids = [row["hypothesis_id"] for row in executable]
    selected = st.multiselect("ngspice 실행 가설 (최대 5개)", ids, default=ids[:min(3, len(ids))], key=f"fault_ai_select_{context_key}")
    if st.button("선택 가설을 ngspice로 검증", key=f"fault_ai_simulate_{context_key}", disabled=not selected):
        with st.spinner("기존 Fault Injector와 ngspice로 선택된 가설을 실행 중…"):
            results = simulate_selected(rows, selected_ids=selected, max_runs=5, base_netlist=base_path,
                component_mapping=mapping, probes=probes, observations=state["observations"], timeout_seconds=60)
        state["hypotheses"] = rank_hypotheses(results)
        state["bayesian_result"] = conditional_bayesian_update(state["hypotheses"], state["observations"])
        st.session_state[state_key] = state
        st.session_state[run_key] = time.time()
        st.rerun()
    for row in rank_hypotheses(state["hypotheses"]):
        with st.expander(f"{row.get('evidence_rank') or '—'} · {row['component_ref']} · {row['fault_type']} · {row.get('simulation_status', 'SIMULATION_NOT_RUN')}"):
            st.json(row)
            try:
                from circuitcare.visualization.fault_map import build_fault_map, render_fault_map_dot
                fault_map = build_fault_map(circuit, row["component_ref"], selected_component=row["component_ref"])
                st.graphviz_chart(render_fault_map_dot(fault_map))
            except (KeyError, ValueError):
                pass
    bayes = state.get("bayesian_result")
    if bayes:
        learned_model = active_prior(RepairCaseRepository(root / "data/private/repair_feedback/repair_cases.sqlite3"), case_name)
        selected_prior_mode = st.radio("Bayesian Prior", ["Default Prior"] + (["Approved Learned Prior"] if learned_model else []),
            horizontal=True, key=f"repair_prior_mode_{context_key}")
        if selected_prior_mode == "Approved Learned Prior" and learned_model:
            bayes = conditional_bayesian_update(state["hypotheses"], state["observations"],
                prior_weights=learned_model["prior_weights"], prior_model_version=learned_model["version"])
        with st.expander("기존 Bayesian 엔진 · 조건부 비교"):
            if bayes.get("status") == "ACTIVE_CONDITIONAL":
                st.caption(f"Prior source: {bayes.get('prior_source', 'DEFAULT_EQUAL')} · Version: {bayes.get('prior_model_version') or 'default'}")
                st.write("**조건부 가설 순위:**")
                st.dataframe(bayes.get("posterior", []), use_container_width=True, hide_index=True)
                st.caption(bayes.get("probability_interpretation", ""))
            else:
                st.info(bayes.get("reason", "Bayesian 업데이트를 적용할 수 없습니다."))
    _render_eval_section(st, root)


def _render_eval_section(st: Any, root: Path) -> None:
    with st.expander("개발자용 평가 상태"):
        for label, relative in (("Fault Hypothesis Evaluation · leakage-free development", "evaluation/fault_hypothesis/results/leakage_free_development_metrics.json"),
                                ("Datasheet Target Rating Pilot · partial", "evaluation/datasheet_extraction/results/corrected_scope_development_metrics.json")):
            result = _read_json(root / relative)
            st.markdown(f"**{label}**")
            st.json(result or {"status": "NOT_EVALUATED", "reason": "No evaluation result has been recorded."})


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _collect_observation(recorded: bool, node: str, value: float, uncertainty: float,
                         source: str) -> list[dict[str, Any]]:
    """A checked zero is a valid reading; an unchecked widget default is missing."""
    if not recorded or not node:
        return []
    return [{"node": node, "value": float(value), "unit": "V",
             "uncertainty": float(uncertainty) if uncertainty > 0 else None,
             "source": source, "measurement_mode": "final_sample"}]
