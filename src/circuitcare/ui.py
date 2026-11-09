"""Local Streamlit dashboard joining the existing CircuitCare analysis stages."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from circuitcare.conversation import answer_question  # noqa: F401 - compatibility export
from circuitcare.dashboard.analysis_loader import (
    json_object_if_exists as _json_if_exists,
)
from circuitcare.dashboard.analysis_loader import (
    load_analysis_snapshot,
)
from circuitcare.dashboard.capabilities import command_exists as _command_exists
from circuitcare.dashboard.case_loader import DEMO_CASES as CASES
from circuitcare.dashboard.case_loader import UPLOAD_CHOICE, fault_options_for
from circuitcare.dashboard.context import (
    _advisor_matches,  # noqa: F401 - compatibility export
    _analysis_context_key,
    _file_sha256,  # noqa: F401 - compatibility export
    _is_cached_case,
    _measurement_set_id,  # noqa: F401 - compatibility export
    _simulation_result_id,  # noqa: F401 - compatibility export
    _simulation_status,  # noqa: F401 - compatibility export
    _supports_bayesian_demo,
    _upsert_measurement,  # noqa: F401 - compatibility export
)
from circuitcare.dashboard.downloads import (
    MAX_DOWNLOAD_CSV_BYTES,  # noqa: F401 - compatibility export
    _cached_download_bytes,  # noqa: F401 - compatibility export
    _download_file_digest,  # noqa: F401 - compatibility export
)
from circuitcare.dashboard.formatters import (
    csv_rows as _csv_rows,  # noqa: F401 - compatibility export
)
from circuitcare.dashboard.formatters import (
    fallback_text as _fallback_text,  # noqa: F401 - compatibility export
)
from circuitcare.dashboard.formatters import (
    risk_table as _risk_table,  # noqa: F401 - compatibility export
)
from circuitcare.dashboard.formatters import (
    stress_table as _stress_table,  # noqa: F401 - compatibility export
)
from circuitcare.dashboard.session_state import (
    reconcile_analysis_context,
    simulation_context_keys_to_clear,
)
from circuitcare.dashboard.simulation_controller import (
    SimulationRequest,
    build_simulation_command,
    execute_simulation_request,
)
from circuitcare.dashboard.views.downloads import render_downloads as _downloads
from circuitcare.dashboard.workspace_adapter import build_analysis_records
from circuitcare.datasheet.datasheet_ui import render_datasheet_tab
from circuitcare.datasheet.rating_adapter import (
    apply_approved_rating,
    reassess_existing_risk,
)
from circuitcare.datasheet.rating_review import (
    applied_rating_session_key,
    find_applications,
)
from circuitcare.evidence_builder import build_evidence  # noqa: F401 - compatibility export
from circuitcare.graph import build_graph, direct_candidates, find_paths_result, to_dot  # noqa: F401 - compatibility exports
from circuitcare.local_llm import LocalLLMError  # noqa: F401 - compatibility export
from circuitcare.netlist import load_netlist
from circuitcare.repair_advisor import build_repair_advice  # noqa: F401 - compatibility export
from circuitcare.services.diagnosis_service import (
    advisor_inputs as _advisor_inputs,  # noqa: F401 - compatibility export
)
from circuitcare.services.diagnosis_service import (
    load_saved_diagnosis as _load_diagnosis,
)
from circuitcare.services.diagnosis_service import (
    run_diagnosis_for_measurements as _run_diagnosis,  # noqa: F401 - compatibility export
)
from circuitcare.services.local_llm_service import list_local_models as _ollama_models
from circuitcare.services.simulation_service import (
    SimulationRunError,
    _current_simulation_fingerprint,
    _valid_parameter,
    _validate_live_result,  # noqa: F401 - compatibility export for existing callers
)
from circuitcare.workspace import (
    build_workspace_context,
    reconcile_session_workspace,
    store_uploaded_netlist,
)

ROOT = Path(__file__).resolve().parents[2]
DEMO_HYPOTHESES = ROOT / "data/diagnosis/demo_hypotheses.json"
MEASUREMENTS = ROOT / "data/diagnosis/demo_measurements.json"


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="CircuitCare AI", page_icon="🔧", layout="wide")
    if not st.session_state.get("circuitcare_v2_legacy_demo"):
        if st.session_state.get("circuitcare_v2_screen") == "workbench":
            from circuitcare.dashboard.views.unified_workspace import render_workbench_shell
            from circuitcare.frontend.state import INTAKE_PROJECT_KEY

            project = st.session_state.get(INTAKE_PROJECT_KEY)
            if project is not None:
                render_workbench_shell(st, project)
            else:
                st.session_state["circuitcare_v2_screen"] = "entry"
                from circuitcare.dashboard.views.diagnosis_entry import render_diagnosis_entry
                render_diagnosis_entry(st)
        else:
            from circuitcare.dashboard.views.diagnosis_entry import render_diagnosis_entry
            if render_diagnosis_entry(st) == "demo":
                st.rerun()
        st.stop()
    if st.sidebar.button("새 PCB 진단으로 돌아가기", key="cc_v2_return_to_entry"):
        st.session_state.pop("circuitcare_v2_legacy_demo", None)
        st.rerun()
    from circuitcare.dashboard.navigation import (
        ADVANCED_VIEWS, ANALYSIS_VIEWS, AREAS, CASE_LABELS, WORKBENCH_VIEW,
        WORKSPACE_VIEWS, korean_status,
    )
    navigation_target = st.session_state.pop("workbench_navigation_target", None)
    if navigation_target in WORKSPACE_VIEWS:
        st.session_state["active_analysis_view"] = navigation_target
        target_area = (
            "회로도" if navigation_target == WORKBENCH_VIEW else
            "분석" if navigation_target in ANALYSIS_VIEWS else
            "측정·진단" if navigation_target == "Bayesian 진단" else None
        )
        if target_area is not None:
            st.session_state["main_area"] = target_area
            st.session_state["workbench_area"] = target_area
            if navigation_target in ANALYSIS_VIEWS:
                st.session_state["analysis_view_choice"] = navigation_target
        else:
            st.session_state["workbench_area"] = "고급 도구"
    from circuitcare.dashboard.workbench_styles import (
        apply_workbench_theme,
        render_workbench_header,
    )
    apply_workbench_theme(st)

    st.sidebar.subheader("프로젝트")
    case_name = st.sidebar.selectbox("회로 데이터셋", [*CASES, UPLOAD_CHOICE], key="case_name",
                                     format_func=lambda name: CASE_LABELS.get(name, name))
    if case_name == UPLOAD_CHOICE:
        uploaded = st.sidebar.file_uploader("KiCad XML 넷리스트", type=["xml"], key="uploaded_netlist")
        if uploaded is None:
            st.info("KiCad XML을 올리면 부품·Net 연결 그래프를 확인할 수 있습니다. 업로드 회로에는 사전 구성된 SPICE 모델/매핑이 없어 시뮬레이션은 지원되지 않습니다.")
            st.stop()
        try:
            uploaded_path = store_uploaded_netlist(uploaded.getvalue(), workspace_root=ROOT)
            circuit = load_netlist(uploaded_path)
        except (OSError, ValueError) as exc:
            st.error(f"회로 검증에 실패했습니다: {exc} · KiCad XML 넷리스트를 다시 내보내 주세요.")
            st.stop()
        case_name = f"Uploaded KiCad netlist · {uploaded_path.stem[:12]}"
        case = {"netlist": str(uploaded_path), "fault": next(iter(circuit.components)),
                "fault_type": "unspecified", "value": "", "fault_types": ["unspecified"],
                "base": None, "mapping": None, "probes": None, "ratings": None,
                "cached": None, "prior": None, "uploaded": True}
    else:
        case = CASES[case_name]
    netlist_path = ROOT / case["netlist"]
    if case_name != UPLOAD_CHOICE and not case.get("uploaded"):
        circuit = load_netlist(netlist_path)
    st.sidebar.subheader("고장 조건")
    fault_options = fault_options_for(case_name, case, list(circuit.components))
    selected_fault = st.sidebar.selectbox("고장 부품", fault_options, index=fault_options.index(case["fault"]) if case["fault"] in fault_options else 0)
    fault_type = st.sidebar.selectbox(
        "고장 유형", case["fault_types"], index=case["fault_types"].index(case["fault_type"]),
        key=f"fault_type_{case_name}", format_func=korean_status)
    needs_value = fault_type in {"resistance_drift", "regulator_output_rise", "regulator_output_drop"}
    fault_value = st.sidebar.text_input("고장 파라미터 (SPICE 숫자 표기)", value=case["value"] if fault_type == case["fault_type"] else ("10k" if fault_type == "resistance_drift" else "8"), disabled=not needs_value, key=f"fault_value_{case_name}_{fault_type}")
    context_key = _analysis_context_key(case_name, case, selected_fault, fault_type, fault_value if needs_value else None)
    workspace_context = build_workspace_context(
        circuit_id=case_name,
        netlist_path=netlist_path,
        input_paths={key: ROOT / case[source] if case.get(source) else None for key, source in
                     (("spice_model", "base"), ("mapping", "mapping"), ("probes", "probes"),
                      ("ratings", "ratings"), ("prior", "prior"))},
        fault={"component_ref": selected_fault, "fault_type": fault_type,
               "parameter": fault_value if needs_value else None},
    ).to_dict()
    # Include content hashes in all downstream session keys so unchanged size/mtime
    # metadata cannot accidentally reuse results for a different circuit input.
    context_key = (context_key, workspace_context["workspace_id"])
    workspace_context["component_refs"] = sorted(circuit.components)
    bayesian_supported = _supports_bayesian_demo(case_name, selected_fault, fault_type, fault_value if needs_value else None)
    reconcile_analysis_context(st.session_state, context_key)
    active_workspace = reconcile_session_workspace(st.session_state, workspace_context)
    st.sidebar.caption(f"고장 모델 · {korean_status(fault_type)} · 값 {fault_value if needs_value else '해당 없음'}")
    st.sidebar.caption("저장 결과는 현재 회로 조건과 검증된 경우에만 사용합니다. 새 계산은 실행 버튼을 눌렀을 때 시작됩니다.")

    run_col, state_col = st.sidebar.columns(2)
    state_display = state_col.empty()
    run_live = run_col.button("ngspice 실행", disabled=not _command_exists("ngspice") or bool(case.get("uploaded")), use_container_width=True)
    live_matches = bool(st.session_state.get("live_result_dir")) and st.session_state.get("live_context_key") == context_key
    if not live_matches:
        st.session_state.pop("live_result_dir", None)
        live_used = False
        if case.get("cached") and selected_fault == case["fault"] and fault_type == case["fault_type"] and fault_value == case["value"]:
            result_dir = ROOT / case["cached"]
        else:
            result_dir = ROOT / "data/no_cached_results" / case_name.replace(" ", "_") / selected_fault
    else:
        result_dir = Path(st.session_state["live_result_dir"])
        live_used = True
    if run_live:
        if needs_value and not _valid_parameter(fault_value):
            st.sidebar.error("파라미터는 양수 숫자 또는 10k 같은 SPICE 숫자 표기만 허용합니다.")
            st.stop()
        state_display.markdown("🟡 RUNNING")
        st.session_state.simulation_status = "RUNNING"
        for key in simulation_context_keys_to_clear(list(st.session_state)):
            st.session_state.pop(key, None)
        live_used = False
        result_dir = ROOT / case["cached"] if case.get("cached") and _is_cached_case(case, selected_fault, fault_type, fault_value, needs_value) else ROOT / "data/no_cached_results" / case_name.replace(" ", "_") / selected_fault
        slug = {"U1": "regulator_8v", "R1": "divider_r1_drift", "RCHARGE": "rc_rcharge_drift"}.get(selected_fault, "scenario")
        output = ROOT / "data/ui_runs" / f"{slug}-{time.time_ns()}"
        output.mkdir(parents=True, exist_ok=True)
        active_marker = output / ".circuitcare-active"
        active_marker.write_text(str(__import__("os").getpid()), encoding="ascii")
        command = build_simulation_command(
            python_executable=sys.executable, project_root=ROOT, netlist=netlist_path,
            case=case, component_ref=selected_fault, fault_type=fault_type,
            output_dir=output, fault_value=fault_value if needs_value else None,
        )
        with st.spinner("Fault Injection → ngspice → stress/rating assessment 실행 중…"):
            try:
                outcome = execute_simulation_request(SimulationRequest(
                    command=command, output_dir=output, component_ref=selected_fault,
                    fault_type=fault_type, fault_value=fault_value if needs_value else None,
                    case=case,
                ))
                if outcome.status != "SUCCESS" or outcome.result_dir is None:
                    raise SimulationRunError(outcome.error or "시뮬레이션을 완료하지 못했습니다.")
                result_dir = outcome.result_dir
                st.session_state.live_result_dir = str(result_dir)
                st.session_state.live_context_key = context_key
                st.session_state.pop("live_error", None)
                st.session_state.pop("last_advisor", None)
                st.session_state.simulation_status = "SUCCESS"
                st.success("새 시뮬레이션 완료")
                st.rerun()
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                details = str(exc)
                if isinstance(exc, SimulationRunError):
                    details = exc.details
                st.session_state.live_error = details
                st.session_state.simulation_status = "FAILED"
                live_used = False
                st.error(f"시뮬레이션 실패 · 성공 결과로 등록하지 않았습니다.\n\n{details}")
            finally:
                active_marker.unlink(missing_ok=True)
    comparison_path = result_dir / "comparison.json"
    risk_path = result_dir / "risk_assessment.json"
    stress_path = result_dir / "component_stress.json"
    current_simulation_fingerprint = _current_simulation_fingerprint(case, selected_fault, fault_type, fault_value)
    cached_selection_for_integrity = bool(case.get("cached")) and _is_cached_case(
        case, selected_fault, fault_type, fault_value, needs_value)
    analysis_snapshot = load_analysis_snapshot(
        result_dir=result_dir, current_fingerprint=current_simulation_fingerprint,
        live_used=live_used, live_error=st.session_state.get("live_error"),
        cached_selection=cached_selection_for_integrity,
    )
    comparison, risk, stress = (analysis_snapshot.comparison, analysis_snapshot.risk,
                                analysis_snapshot.stress)
    cache_manifest = analysis_snapshot.manifest
    cache_integrity = analysis_snapshot.integrity
    base_risk = risk
    component_mapping = _json_if_exists(ROOT / case["mapping"]) if case.get("mapping") else {}
    stress_signature = hashlib.sha256(stress_path.read_bytes()).hexdigest() if stress_path.is_file() else None
    datasheet_context_token = hashlib.sha256(repr(context_key).encode()).hexdigest()[:16]
    applied_key = applied_rating_session_key(datasheet_context_token, stress_signature)
    applied_ratings = st.session_state.get(applied_key)
    if applied_ratings is None and stress_signature and risk.get("scenario_id"):
        applied_ratings = find_applications(ROOT / "data/private/datasheet_ai/applied_ratings.json",
            circuit_path=case["netlist"], scenario_id=risk["scenario_id"], stress_sha256=stress_signature)
        if applied_ratings:
            st.session_state[applied_key] = applied_ratings
    if applied_ratings and risk and stress:
        try:
            applied_specs = json.loads((ROOT / case["ratings"]).read_text(encoding="utf-8"))
            for application in applied_ratings:
                record = application["record"]
                applied_specs = apply_approved_rating(applied_specs, record,
                    component_ref=application["component_ref"], selected_reference=application["component_ref"],
                    mpn_match_status=record["mpn_match_status"], conditions_compatible=application["conditions_compatible"])
            risk = reassess_existing_risk(risk, stress, applied_specs, component_mapping)
        except (OSError, ValueError, KeyError, TypeError):
            _clear_failed_datasheet_cache(st.session_state, applied_key)
    is_cached_selection = bool(case.get("cached")) and _is_cached_case(case, selected_fault, fault_type, fault_value, needs_value)
    status = analysis_snapshot.status
    if status != "RUNNING":
        st.session_state.simulation_status = status
    state_icon = {"SUCCESS": "🟢", "FAILED": "🔴", "CACHED_SIMULATION": "🔵",
                  "STALE": "🟠", "PROVENANCE_UNVERIFIED": "🟠", "NOT_RUN": "⚪"}.get(status, "🟡")
    state_note = " · 저장 결과 출처 미검증, 사용 차단" if status == "FAILED" and is_cached_selection else ""
    state_display.markdown(f"{state_icon} {korean_status(status)}{state_note}")
    render_workbench_header(
        st,
        circuit_name=case_name,
        fault_ref=selected_fault,
        fault_type=str(fault_type or case["fault_type"]),
        fault_value=fault_value if needs_value else None,
        simulation_status=status,
        verification_status=analysis_snapshot.integrity,
    )
    st.caption("연결 후보는 손상 증거가 아닙니다. 위험 등급은 평가된 정격 스트레스이며 실제 고장 확률이 아닙니다.")
    if status == "PROVENANCE_UNVERIFIED":
        st.info("저장된 결과의 출처를 확인하지 못해 분석 자료로 사용하지 않았습니다. 조건을 확인한 뒤 회로를 새로 실행하거나 검증된 결과를 선택하세요.")
    elif status == "NOT_RUN":
        st.info("회로를 선택하고 고장 부품·유형을 확인한 뒤 SPICE 실행을 누르세요. 실행 전에는 검증되지 않은 저장 결과를 켜지 않습니다.")
    active_workspace["analysis_records"] = build_analysis_records(
        workspace_context=workspace_context, active_workspace=active_workspace,
        comparison=comparison, risk=risk, status=status, uploaded=bool(case.get("uploaded")),
    )
    st.session_state["circuitcare_workspace"] = active_workspace
    st.sidebar.markdown("🟢 ngspice 사용 가능" if _command_exists("ngspice") else "🟠 ngspice 없음 · 저장 결과 조회 가능")
    views = WORKSPACE_VIEWS
    if not st.session_state.get("phase194_navigation_migrated"):
        st.session_state["active_analysis_view"] = views[0]
        st.session_state["phase194_navigation_migrated"] = True
    if st.session_state.get("main_area") not in AREAS:
        st.session_state["main_area"] = "회로도"
    area_col, tool_col = st.columns([5, 1])
    area = area_col.radio("작업 영역", AREAS, horizontal=True, key="main_area",
                          label_visibility="collapsed")
    active_view = WORKBENCH_VIEW
    if area == "분석":
        active_view = st.radio("분석 화면", ANALYSIS_VIEWS, horizontal=True,
                               key="analysis_view_choice", label_visibility="collapsed")
    elif area == "측정·진단":
        active_view = "Bayesian 진단"
    elif area == "보고서":
        active_view = "분석 개요"
    with tool_col:
        with st.popover("고급 도구", use_container_width=True):
            st.caption("추가 분석 기능")
            for tool_view in ADVANCED_VIEWS:
                st.button(
                    tool_view, key=f"advanced_{tool_view}", use_container_width=True,
                    on_click=lambda view=tool_view: st.session_state.update(
                        active_analysis_view=view, workbench_area="고급 도구"),
                )
    if st.session_state.get("workbench_area") == "고급 도구" and area == st.session_state.get("last_main_area", "회로도"):
        active_view = st.session_state.get("active_analysis_view")
        if active_view not in ADVANCED_VIEWS:
            active_view = ADVANCED_VIEWS[0]
    elif area != st.session_state.get("last_main_area"):
        st.session_state["workbench_area"] = area
    st.session_state["last_main_area"] = area
    st.session_state["active_analysis_view"] = active_view
    client, models = _ollama_models() if active_view in {"로컬 LLM 수리 지원", "데이터시트 AI", "AI 고장 가설 분석"} else (None, [])
    st.sidebar.markdown("🟢 로컬 AI 연결됨" if client else "🟡 로컬 AI는 고급 기능을 열 때 확인합니다")
    what_if_result = None
    if active_view == views[0]:
        from circuitcare.dashboard.views.schematic_workspace import (
            render_schematic_workspace,
        )
        render_schematic_workspace(st, circuit=circuit, case_name=case_name, case=case,
                                   selected_fault=selected_fault, fault_type=str(fault_type or case["fault_type"]),
                                   fault_value=fault_value if needs_value else None,
                                   root=ROOT, comparison_path=comparison_path,
                                   simulation_status=status,
                                   simulation_integrity=analysis_snapshot.integrity,
                                   risk_assessment=risk, comparison=comparison,
                                   analysis_snapshot=analysis_snapshot,
                                   current_fingerprint=current_simulation_fingerprint,
                                   bayesian_supported=bayesian_supported,
                                   analysis_context_key=context_key)
    elif active_view == "분석 개요":
        what_if_result = _render_analysis_overview(
            st, circuit=circuit, case_name=case_name, case=case, context_key=context_key,
            failed_component=selected_fault, fault_type=fault_type, fault_value=fault_value,
            risk=risk, comparison=comparison, status=status, bayesian_supported=bayesian_supported,
            workspace=active_workspace,
            simulation_context={"status": "VERIFIED" if cache_integrity == "VERIFIED" else cache_integrity,
                                "fingerprint": cache_manifest.get("fingerprint"),
                                "result_sha256": cache_manifest.get("result_sha256")},
        )
    elif active_view == views[2]:
        from circuitcare.dashboard.views.waveforms import render_waveform_view
        render_waveform_view(st, comparison=comparison, result_dir=result_dir, stress=stress,
                             csv_verified=analysis_snapshot.csv_integrity == "VERIFIED",
                             csv_sha256=analysis_snapshot.csv_sha256)
    elif active_view == views[3]:
        from circuitcare.dashboard.views.risk import render_risk_view
        render_risk_view(st, risk=risk)
    elif active_view == views[4]:
        from circuitcare.dashboard.views.bayesian import render_bayesian_view
        render_bayesian_view(st, case=case, supported=bayesian_supported,
                             measurement_example=MEASUREMENTS, context_key=context_key)
    elif active_view == views[5]:
        from circuitcare.dashboard.views.repair_advisor import (
            render_repair_advisor_view,
        )
        render_repair_advisor_view(st, case=case, case_name=case_name, client=client, models=models,
            supported=bayesian_supported, context_key=context_key, risk=risk, stress=stress,
            risk_path=risk_path, stress_path=stress_path, result_dir=result_dir)
    elif active_view == views[6]:
        from circuitcare.vision_ui import render_vision_tab

        render_vision_tab(
            st,
            netlist_component_refs=sorted(circuit.components),
            risk_assessment=risk,
            risk_path=str(risk_path.relative_to(ROOT)) if risk_path.exists() else "unavailable",
            failed_component=selected_fault,
            fault_type=fault_type,
            context_key=hashlib.sha256(repr(context_key).encode()).hexdigest()[:16],
            current_result_status=status,
            inspection_ranking=(what_if_result or {}).get("inspection_ranking"),
            selected_component=st.session_state.get(f"fault_map_component_{hashlib.sha256(repr(context_key).encode()).hexdigest()[:10]}"),
            result_integrity=cache_integrity,
        )
    elif active_view == views[7]:
        if case.get("uploaded"):
            st.info("업로드 회로에는 부품 Mapping과 확인된 SPICE 정격 분석이 없습니다. 데이터시트 정격 적용은 해당 회로 Context에서 지원되지 않습니다.")
        else:
            render_datasheet_tab(st, root=ROOT, circuit=circuit,
                context_key=datasheet_context_token, case=case,
                risk=base_risk, stress=stress, client=client, models=models,
                stress_signature=stress_signature,
                applied_records=applied_ratings,
                current_risk=risk if applied_ratings and risk.get("reassessment_source") else None)
    elif active_view == views[8]:
        if case.get("uploaded"):
            st.info("업로드 회로에 대해 실행 가능한 SPICE Mapping이 없어 고장 가설 실행을 지원하지 않습니다. 기존 3개 데모 회로에서만 사용할 수 있습니다.")
        else:
            from circuitcare.fault_ai_ui import render_fault_ai_tab
            render_fault_ai_tab(st, root=ROOT, circuit=circuit, case_name=case_name, case=case,
                                client=client, models=models,
                                context_key=hashlib.sha256(repr(context_key).encode()).hexdigest()[:16])
    elif active_view == views[9]:
        from circuitcare.repair_feedback_ui import render_repair_feedback_tab
        state_id = hashlib.sha256(repr(context_key).encode()).hexdigest()[:16]
        fault_state = st.session_state.get(f"fault_ai_state_{state_id}")
        render_repair_feedback_tab(st, root=ROOT,
            circuit_context={"circuit_id": case_name, "circuit_family": case_name,
                "circuit_hash": hashlib.sha256(netlist_path.read_bytes()).hexdigest(),
                "component_refs": sorted(circuit.components)},
            fault_state=fault_state)
    elif active_view == views[10]:
        from circuitcare.benchmark_ui import render_benchmark_tab
        render_benchmark_tab(st, root=ROOT)
    elif active_view == "다중 사진 PCB 복원":
        from circuitcare.vision.reconstruction.ui import render_multiview_reconstruction
        render_multiview_reconstruction(st)

    _downloads(st, case, result_dir, comparison, risk, st.session_state.get("diagnosis_result"), selected_fault, fault_type, fault_value if needs_value else None,
                simulation_verified=(cache_integrity == "VERIFIED" and status in {"SUCCESS", "CACHED_SIMULATION"}),
                csv_verified=analysis_snapshot.csv_integrity == "VERIFIED",
                csv_sha256=analysis_snapshot.csv_sha256)


def _clear_failed_datasheet_cache(session_state, applied_key: str) -> None:
    """Drop only the scoped approved-rating cache that failed during reassessment."""
    session_state.pop(applied_key, None)


def _render_analysis_overview(st, *, circuit, case_name, case, context_key, failed_component,
                              fault_type, fault_value, risk, comparison, status, bayesian_supported,
                              workspace, simulation_context):
    """Unified read-only overview plus an isolated simulated-demo What-if preview."""
    st.subheader("고장 영향 후보 지도 및 진단 개요")
    st.caption("지도는 Component–Net 연결 후보를 나타내며 실제 회로도 배치나 방향성 있는 손상 전파도가 아닙니다. 정격 스트레스와 Bayesian 점검 순위는 별도 근거입니다.")
    risk_rows = (risk or {}).get("secondary_candidates", [])
    exceeded_count = sum(row.get("risk_level") == "EXCEEDED" for row in risk_rows) if risk else None
    baseline_inspection = None
    if bayesian_supported:
        base_result = _load_diagnosis(case)
        baseline_inspection = next(iter(base_result.get("inspection_ranking", {}).get("prior_weighted_inspection_order", [])), None)
    cards = st.columns(4)
    cards[0].metric("선택 고장 부품", failed_component)
    cards[1].metric("시뮬레이션 상태", status)
    cards[2].metric("정격 초과 후보", str(exceeded_count) if exceeded_count is not None else "UNAVAILABLE")
    cards[3].metric("Prior 점검 후보", baseline_inspection.get("component_ref") if baseline_inspection else "UNAVAILABLE")
    st.caption(f"회로: {case_name} · 고장 조건: {fault_type} / {fault_value or '—'} · 손상 확정: 아니요")
    from circuitcare.dashboard.views.workflow_status import render_workflow_status
    render_workflow_status(st, workspace=workspace, case=case, status=status)

    from circuitcare.dashboard.views.what_if import render_what_if
    what_if_preview = render_what_if(
        st, case=case, context_key=context_key, bayesian_supported=bayesian_supported,
    )
    what_if_result = what_if_preview.result if what_if_preview else None
    from circuitcare.dashboard.views.fault_map import render_fault_map
    payload, what_if_export = render_fault_map(
        st, circuit=circuit, case=case, context_key=context_key,
        failed_component=failed_component, fault_type=fault_type, fault_value=fault_value,
        risk=risk, status=status, risk_rows=risk_rows, what_if_preview=what_if_preview,
    )

    include_preview = st.checkbox("별도 What-if Preview를 보고서에 포함", value=False,
                                  help="기본 진단과 분리된 PREVIEW_ONLY 절로만 추가됩니다.") if what_if_result else False
    from circuitcare.dashboard.views.report_panel import render_report_panel
    from circuitcare.services.report_service import (
        ReportSnapshotContext,
        build_report_snapshot_for_dashboard,
    )
    report_snapshot = build_report_snapshot_for_dashboard(
        ReportSnapshotContext(
            root=ROOT,
            workspace=workspace,
            circuit=circuit,
            failed_component=failed_component,
            fault_type=fault_type,
            fault_value=fault_value,
            simulation_status=status,
            comparison=comparison,
            risk_assessment=risk,
            simulation_context=simulation_context,
            bayesian_supported=bayesian_supported,
            analysis_context_key=context_key,
            include_preview=include_preview,
            what_if_result=what_if_result,
            what_if_measurements=st.session_state.get("what_if_measurements", []),
            connection_paths=payload.get("paths", []),
        ),
        st.session_state,
    )
    render_report_panel(
        st, snapshot=report_snapshot, root=ROOT, workspace_id=workspace["workspace_id"],
        expected_circuit_hash=workspace.get("circuit_hash"),
    )
    from circuitcare.dashboard.views.fault_map import render_fault_map_save
    render_fault_map_save(
        st, payload=payload, what_if_export=what_if_export, context_key=context_key,
    )
    return what_if_result


if __name__ == "__main__":
    main()
