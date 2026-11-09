"""Schematic-first workspace with the existing connectivity view as fallback."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from circuitcare.graph import direct_candidates, find_paths_result
from circuitcare.dashboard.navigation import korean_status
from circuitcare.schematic.electrical_equivalence import verify_schematic_netlist
from circuitcare.schematic.exporter import export_schematic
from circuitcare.schematic.geometry import (
    GeometryResult,
    build_symbol_geometry,
    read_svg_view_box,
)
from circuitcare.schematic.identity import (
    discover_schematic_files,
    map_component_refs,
    read_symbol_identities,
)
from circuitcare.schematic.interaction import validate_selection_event
from circuitcare.schematic.measurement_geometry import build_measurement_anchors
from circuitcare.schematic.overlay import (
    build_fault_overlay,
    build_measurement_overlay,
    build_risk_overlay,
)
from circuitcare.schematic.viewer import render_interactive_viewer


def spice_verification_label(status: str, integrity: str) -> str:
    """Describe trusted simulation status only when both provenance signals agree."""
    if status == "SUCCESS" and integrity == "VERIFIED":
        return "VERIFIED · Live Simulation"
    if status == "CACHED_SIMULATION" and integrity == "VERIFIED":
        return "VERIFIED · Cached Simulation"
    return f"NOT VERIFIED · {status} / {integrity}"


def _store_schematic(raw: bytes, root: Path) -> Path:
    if not raw or len(raw) > 20_000_000:
        raise ValueError("회로도 파일은 20 MB 이하이어야 합니다.")
    if not raw.lstrip().startswith(b"(kicad_sch"):
        raise ValueError("KiCad S-expression 회로도 형식이 아닙니다.")
    folder = root / "data" / "schematic_uploads"
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(raw).hexdigest()
    target = folder / f"{digest}.kicad_sch"
    if not target.exists():
        target.write_bytes(raw)
    return target


def _json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _korean_overlay_reason(reason: object) -> str:
    text = str(reason)
    if text.startswith("Measurement nodes are logical schematic candidates"):
        return "측정 Node는 논리 회로도 후보입니다. PCB의 안전하거나 접근 가능한 프로브 지점을 뜻하지 않습니다."
    if text.startswith("No verified recommendation or inspection markers"):
        return "현재 Context에 검증된 측정 추천이나 점검 표시가 없습니다."
    for code in ("PROVENANCE_UNVERIFIED", "UNVERIFIED_GEOMETRY", "NOT_VERIFIED",
                 "UNVERIFIED_MEASUREMENT_LOCATION", "UNVERIFIED_MAPPING", "BLOCKED", "UNKNOWN"):
        text = text.replace(code, korean_status(code))
    return text


def _content_view_box(schematic_path: Path, geometry: GeometryResult) -> tuple[float, float, float, float] | None:
    """Find real symbol and wire bounds for camera fitting without altering SVG geometry."""
    if geometry.view_box is None or not geometry.regions:
        return None
    points = [point for region in geometry.regions for point in region.polygon]
    try:
        source = schematic_path.read_text(encoding="utf-8")
    except OSError:
        return None
    number = r"-?\d+(?:\.\d+)?"
    wire_pattern = rf"\(wire\s+\(pts\s+\(xy\s+({number})\s+({number})\)\s+\(xy\s+({number})\s+({number})\)"
    for match in re.finditer(wire_pattern, source):
        x1, y1, x2, y2 = (float(value) for value in match.groups())
        points.extend(((x1, y1), (x2, y2)))
    for match in re.finditer(rf"\(junction\s+\(at\s+({number})\s+({number})", source):
        points.append((float(match.group(1)), float(match.group(2))))
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    page_x, page_y, page_w, page_h = geometry.view_box
    padding = 18.0
    x1, y1 = max(page_x, min(xs) - padding), max(page_y, min(ys) - padding)
    x2, y2 = min(page_x + page_w, max(xs) + padding), min(page_y + page_h, max(ys) + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    view_box = (x1, y1, x2 - x1, y2 - y1)
    return view_box


def render_schematic_workspace(st: Any, *, circuit: Any, case_name: str, case: dict,
                               selected_fault: str, fault_type: str, fault_value: str | None,
                               root: Path, comparison_path: Path, simulation_status: str,
                               simulation_integrity: str, risk_assessment: dict,
                               comparison: dict, analysis_snapshot: Any,
                               current_fingerprint: dict, bayesian_supported: bool,
                               analysis_context_key: Any) -> None:
    fingerprint = {
        "case": case_name,
        "components": [(ref, item.value, item.library_id) for ref, item in sorted(circuit.components.items())],
        "connections": sorted((conn.ref, conn.pin, conn.net) for conn in circuit.connections),
    }
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode())
    schematic_path = None
    upload_token = hashlib.sha256(case_name.encode()).hexdigest()[:12]
    uploaded = st.sidebar.file_uploader(
        "KiCad 회로도 (.kicad_sch)", type=["kicad_sch"],
        key=f"schematic_source_{upload_token}", max_upload_size=20,
    )
    # Include the actual schematic bytes in the selection scope, including child sheets.
    if uploaded is not None:
        try:
            schematic_path = _store_schematic(uploaded.getvalue(), root)
        except (OSError, ValueError) as exc:
            st.error(str(exc))
    elif case.get("schematic"):
        candidate = (root / case["schematic"]).resolve()
        if candidate.is_relative_to(root.resolve()) and candidate.is_file():
            schematic_path = candidate

    if schematic_path is not None:
        try:
            source_files, _ = discover_schematic_files(schematic_path, project_root=root)
            for source_file in source_files:
                digest.update(source_file.relative_to(root.resolve()).as_posix().encode())
                digest.update(source_file.read_bytes())
        except (OSError, ValueError):
            digest.update(schematic_path.name.encode())
    else:
        digest.update(b"no-kicad-schematic")
    netlist_path = Path(case["netlist"])
    if not netlist_path.is_absolute():
        netlist_path = root / netlist_path
    try:
        digest.update(netlist_path.resolve(strict=True).read_bytes())
    except OSError:
        digest.update(b"unavailable-netlist")
    context_key = digest.hexdigest()
    context_token = context_key[:16]
    state_key = f"schematic_inspected_{context_token}"
    navigation_key = f"schematic_navigation_{context_token}"
    initial_ref = st.session_state.get(state_key)
    if initial_ref not in circuit.components:
        initial_ref = selected_fault if selected_fault in circuit.components else min(circuit.components)

    nav, canvas, inspector = st.columns([0.75, 4.8, 1.15], gap="small")
    refs = sorted(circuit.components)
    query = ""
    component_ref = initial_ref
    with nav:
        st.markdown("**회로 탐색기**")
        from circuitcare.dashboard.navigation import korean_status
        st.caption(f"고장 가정 · {selected_fault} · {korean_status(fault_type)}")
        query = st.text_input("부품 검색", key=f"schematic_search_{state_key}").strip().casefold()
        filtered = [ref for ref in refs if query in ref.casefold()]
        if not filtered:
            st.info("검색한 Reference가 현재 회로에 없습니다.")
        else:
            widget_key = f"schematic_component_{context_token}"
            if widget_key not in st.session_state or st.session_state.get(widget_key) not in filtered:
                st.session_state[widget_key] = initial_ref if initial_ref in filtered else filtered[0]
            component_ref = st.selectbox("부품 검색 결과", filtered, key=widget_key)
            if component_ref != st.session_state.get(state_key):
                st.session_state[state_key] = component_ref
                st.session_state[navigation_key] = int(st.session_state.get(navigation_key, 0)) + 1

    status = "NOT_EVALUATED"
    svg = None
    svg_path = None
    view_box = None
    geometry = GeometryResult("UNVERIFIED_GEOMETRY", "회로도 geometry를 사용할 수 없습니다.")
    equivalence = None
    mapping: dict[str, dict] = {}
    overlay_mode = "기본"
    overlay_payload: dict | None = None
    measurement_recommendation: dict | None = None
    measurement_anchors = None
    measurement_source = "NONE"
    measurement_set_id = None
    measurement_bundle: dict = {}
    diagnosis = {}
    inspection_ranking = {}
    if schematic_path is not None:
        try:
            equivalence = verify_schematic_netlist(
                schematic_path, netlist_path, project_root=root,
                cache_root=root / "data" / "schematic_equivalence_cache",
            )
            identities, missing_sheets = read_symbol_identities(schematic_path, project_root=root)
            mapping = map_component_refs(refs, identities)
            if missing_sheets:
                mapping = {ref: {"component_ref": ref, "status": "INCOMPLETE_PROJECT"}
                           for ref in refs}
            result = export_schematic(schematic_path, project_root=root,
                                      cache_root=root / "data" / "schematic_svg_cache")
            if result.status == "EXPORTED" and result.svg_paths:
                svg_paths = result.svg_paths
                page_names = [item.name for item in svg_paths]
                if len(page_names) > 1:
                    with nav:
                        page_key = f"schematic_page_{context_token}"
                        if st.session_state.get(page_key) not in page_names:
                            st.session_state[page_key] = page_names[0]
                        page_name = st.selectbox("회로도 페이지", page_names, key=page_key)
                else:
                    page_name = page_names[0]
                svg_path = next(item for item in svg_paths if item.name == page_name)
                svg = svg_path.read_text(encoding="utf-8")
                status = "SVG_EXPORTED"
                geometry = build_symbol_geometry(schematic_path, result.svg_paths, project_root=root)
                view_box = geometry.view_box or read_svg_view_box(svg_path)
                fit_view_box = _content_view_box(schematic_path, geometry)
                allowed_refs = set(refs)
                regions = [region for region in geometry.regions if region.component_ref in allowed_refs]
                if len(regions) != len(geometry.regions):
                    geometry = GeometryResult("UNVERIFIED_GEOMETRY", "Schematic Reference does not match the current netlist", view_box)
                    regions = []
                interaction_geometry = geometry
                if equivalence is None or equivalence.status != "VERIFIED_EQUIVALENT":
                    reason = equivalence.message if equivalence else "Equivalence check unavailable"
                    interaction_geometry = GeometryResult(
                        "UNVERIFIED_GEOMETRY",
                        f"Electrical equivalence {equivalence.status if equivalence else 'NOT_VERIFIED'}: {reason}",
                        view_box,
                    )
                    regions = []
                asset_key = f"{result.cache_key}:{svg_path.name}"
                with canvas:
                    mode_key = f"schematic_overlay_mode_{context_token}"
                    overlay_mode = st.radio(
                        "회로도 보기",
                        ["기본", "연결 후보", "전기적 위험", "측정·점검"],
                        horizontal=True,
                        key=mode_key,
                    )
                    if overlay_mode == "전기적 위험":
                        st.caption("범례 · ▲ 정격 초과 · ◆ 정격 근접 · ○ 평가 항목 정격 이내 · ? 정보 부족 — 실제 손상 확정을 뜻하지 않습니다.")
                    elif overlay_mode == "측정·점검":
                        st.caption("측정 표시는 논리적 후보 위치입니다. PCB의 안전하거나 접근 가능한 프로브 위치를 의미하지 않습니다.")
                    if overlay_mode == "연결 후보":
                        overlay_payload = build_fault_overlay(
                            circuit=circuit, failed_component=selected_fault,
                            geometry=interaction_geometry,
                            electrical_equivalence=equivalence.status if equivalence else "NOT_VERIFIED",
                            context_id=context_key, asset_key=asset_key,
                        )
                    elif overlay_mode == "전기적 위험":
                        overlay_payload = build_risk_overlay(
                            circuit=circuit, failed_component=selected_fault, fault_type=fault_type,
                            fault_value=fault_value, risk=risk_assessment, comparison=comparison,
                            simulation_status=simulation_status,
                            simulation_integrity=simulation_integrity,
                            manifest=getattr(analysis_snapshot, "manifest", {}),
                            current_fingerprint=current_fingerprint, geometry=interaction_geometry,
                            electrical_equivalence=equivalence.status if equivalence else "NOT_VERIFIED",
                            context_id=context_key, asset_key=asset_key,
                        )
                    elif overlay_mode == "측정·점검":
                        diagnosis_state = st.session_state.get("diagnosis_result") or {}
                        diagnosis_state_id = hashlib.sha256(repr(analysis_context_key).encode()).hexdigest()[:16]
                        state_measurements = st.session_state.get("measurements", [])
                        from circuitcare.workspace.measurement_provenance import (
                            measurement_set_sha256,
                        )
                        current_measurement_set = measurement_set_sha256(state_measurements)
                        state_is_current = (
                            diagnosis_state.get("analysis_context_id") == diagnosis_state_id
                            and diagnosis_state.get("measurement_set_sha256") == current_measurement_set
                        )
                        if state_is_current:
                            measurement_bundle = diagnosis_state
                        elif bayesian_supported and not state_measurements and case.get("prior"):
                            prior_root = root / case["prior"]
                            measurement_bundle = {
                                "diagnosis": _json_object(prior_root / "diagnosis.json"),
                                "inspection_ranking": _json_object(prior_root / "inspection_ranking.json"),
                                "recommendations": _json_object(prior_root / "measurement_recommendations.json"),
                            }
                        if bayesian_supported and measurement_bundle:
                            diagnosis = measurement_bundle.get("diagnosis", {})
                            inspection_ranking = measurement_bundle.get("inspection_ranking", {})
                            used = diagnosis.get("used_measurements", [])
                            sources = {str(item.get("source", "")) for item in used if isinstance(item, dict)}
                            measurement_source = next(iter(sources)) if len(sources) == 1 else (
                                "prior_only" if not used else "mixed_sources"
                            )
                            measurement_set_id = current_measurement_set if state_is_current else "prior_only"
                            recommendation = measurement_bundle.get("recommendations", {}).get("recommended_measurement")
                            if recommendation:
                                measurement_recommendation = dict(recommendation)
                                if geometry.status == "VERIFIED_GEOMETRY":
                                    measurement_anchors = build_measurement_anchors(
                                        node=str(recommendation.get("node", "")), circuit=circuit,
                                        schematic_path=schematic_path, project_root=root,
                                        geometry=geometry,
                                    )
                        overlay_payload = build_measurement_overlay(
                            circuit=circuit, failed_component=selected_fault,
                            geometry=interaction_geometry,
                            electrical_equivalence=equivalence.status if equivalence else "NOT_VERIFIED",
                            context_id=context_key, asset_key=asset_key,
                            support_status="VERIFIED_CURRENT_CONTEXT" if bayesian_supported and measurement_bundle else "NOT_SUPPORTED",
                            source_class=measurement_source,
                            measurement_set_id=measurement_set_id,
                            recommendation=measurement_recommendation,
                            anchors=measurement_anchors,
                            inspection_ranking=inspection_ranking,
                            model_mismatch=bool(diagnosis.get("model_mismatch")),
                        )
                    if overlay_payload:
                        if overlay_payload.get("status") == "BLOCKED":
                            st.info(f"이 보기의 Overlay가 비활성화되었습니다. {_korean_overlay_reason(overlay_payload.get('reason', ''))}")
                        elif overlay_payload.get("status") == "UNVERIFIED_MEASUREMENT_LOCATION":
                            st.info(f"측정 추천은 정보 패널에서만 확인할 수 있습니다. 위치 검증: {_korean_overlay_reason(overlay_payload.get('reason', ''))}")
                        elif overlay_payload.get("reason"):
                            st.caption(_korean_overlay_reason(overlay_payload["reason"]))
                        if overlay_mode == "연결 후보" and overlay_payload.get("status") == "READY":
                            path_status = overlay_payload.get("path_search", {}).get("status", "UNKNOWN")
                            st.caption(f"보라: 사용자 고장 가정 · 파랑: 연결 후보 (손상 증거 아님) · 배선은 표시하지 않음 · 경로 검색: {korean_status(path_status)}")
                        elif overlay_mode == "전기적 위험" and overlay_payload.get("status") == "READY":
                            st.caption("정격 초과·근접 결과는 실제 손상 확정이 아닙니다. 정격 이내 판정은 평가된 항목에만 적용됩니다.")
                        elif overlay_mode == "측정·점검" and overlay_payload.get("status") == "READY":
                            st.caption("측정 핀은 논리적 회로도 후보이며 PCB에서 안전하거나 접근 가능한 프로브 위치를 보증하지 않습니다. 점검 점수는 손상 확률이 아닙니다.")
                    viewer_key = f"schematic_viewer_{context_token}_{hashlib.sha256(svg_path.name.encode()).hexdigest()[:8]}"
                    viewer_result = None
                    if view_box is not None:
                        def handle_viewer_selection() -> None:
                            current = st.session_state.get(viewer_key)
                            event = getattr(current, "selection", None)
                            if not isinstance(event, dict) or event.get("event_id") == st.session_state.get(f"last_event_{viewer_key}"):
                                return
                            valid, reason = validate_selection_event(event, component_refs=set(refs),
                                                                    geometry=interaction_geometry,
                                                                    context_key=context_key,
                                                                    asset_key=asset_key)
                            prior_sequence = int(st.session_state.get(f"last_seq_{viewer_key}", 0))
                            event_sequence = event.get("event_seq")
                            if not valid or type(event_sequence) is not int or event_sequence <= prior_sequence:
                                st.session_state[f"selection_error_{viewer_key}"] = reason
                                return
                            st.session_state[f"last_seq_{viewer_key}"] = event_sequence
                            st.session_state[f"last_event_{viewer_key}"] = event.get("event_id")
                            st.session_state[state_key] = event["component_ref"]
                            st.session_state[f"schematic_component_{context_token}"] = event["component_ref"]
                            st.session_state[navigation_key] = int(st.session_state.get(navigation_key, 0)) + 1

                        viewer_result = render_interactive_viewer(
                            st, svg=svg, asset_key=asset_key, view_box=view_box,
                            fit_view_box=fit_view_box,
                            geometry_status=interaction_geometry.status,
                            overlay=overlay_payload,
                            regions=[{"component_ref": item.component_ref, "symbol_uuid": item.symbol_uuid,
                                      "sheet_path": item.sheet_path, "unit": item.unit,
                                      "polygon": [list(point) for point in item.polygon]} for item in regions],
                            selected_ref=component_ref,
                            navigation_id=int(st.session_state.get(navigation_key, 0)),
                            context_key=context_key,
                            event_sequence_seed=int(st.session_state.get(f"last_seq_{viewer_key}", 0)),
                            key=viewer_key,
                            on_selection_change=handle_viewer_selection,
                        )
                    if viewer_result is None:
                        st.image(str(svg_path), output_format="SVG", use_container_width=True)
                        st.caption("Pan/Zoom Viewer를 사용할 수 없어 SVG 정적 보기로 표시합니다.")
                    else:
                        selection_error = st.session_state.pop(f"selection_error_{viewer_key}", None)
                        if selection_error:
                            st.warning(f"회로도 선택 이벤트를 거부했습니다: {selection_error}")
                        if equivalence is None or equivalence.status != "VERIFIED_EQUIVALENT":
                            st.warning("회로도와 넷리스트의 전기적 정합성을 확인하지 못했습니다. 선택 및 Overlay가 제한됩니다.")
            else:
                with canvas:
                    st.warning(f"회로도 Export를 사용할 수 없습니다 ({korean_status(result.status)}). {result.message}")
            if missing_sheets:
                st.warning("누락되거나 허용 경로 밖인 하위 시트: " + ", ".join(missing_sheets))
        except (OSError, ValueError) as exc:
            with canvas:
                st.error(f"회로도 확인 실패: {exc}")
    else:
        with canvas:
            st.info("실제 KiCad 회로도 파일이 선택되지 않았습니다. 아래 연결 화면은 XML 넷리스트에서 만든 논리 연결 그래프이며 실제 회로도 배치를 나타내지 않습니다.")
            mode_key = f"schematic_overlay_mode_{context_token}"
            overlay_mode = st.radio(
                "분석 보기",
                ["기본", "연결 후보", "전기적 위험", "측정·점검"],
                horizontal=True,
                key=mode_key,
            )
            if overlay_mode == "전기적 위험":
                st.caption("범례 · ▲ 정격 초과 · ◆ 정격 근접 · ○ 평가 항목 정격 이내 · ? 정보 부족 — 실제 손상 확정을 뜻하지 않습니다.")
            elif overlay_mode == "측정·점검":
                st.caption("측정 표시는 회로도의 논리적 후보 위치입니다. PCB의 안전하거나 접근 가능한 프로브 위치를 의미하지 않습니다.")
            if overlay_mode == "측정·점검":
                # Bayesian evidence can be useful in the inspector without a
                # schematic, but no physical or page coordinate is inferred.
                from circuitcare.workspace.measurement_provenance import (
                    measurement_set_sha256,
                )

                diagnosis_state = st.session_state.get("diagnosis_result") or {}
                diagnosis_state_id = hashlib.sha256(repr(analysis_context_key).encode()).hexdigest()[:16]
                state_measurements = st.session_state.get("measurements", [])
                state_is_current = (
                    diagnosis_state.get("analysis_context_id") == diagnosis_state_id
                    and diagnosis_state.get("measurement_set_sha256") == measurement_set_sha256(state_measurements)
                )
                if state_is_current:
                    measurement_bundle = diagnosis_state
                elif bayesian_supported and not state_measurements and case.get("prior"):
                    prior_root = root / case["prior"]
                    measurement_bundle = {
                        "diagnosis": _json_object(prior_root / "diagnosis.json"),
                        "inspection_ranking": _json_object(prior_root / "inspection_ranking.json"),
                        "recommendations": _json_object(prior_root / "measurement_recommendations.json"),
                    }
                if bayesian_supported and measurement_bundle:
                    diagnosis = measurement_bundle.get("diagnosis", {})
                    inspection_ranking = measurement_bundle.get("inspection_ranking", {})
                    used = diagnosis.get("used_measurements", [])
                    sources = {str(item.get("source", "")) for item in used if isinstance(item, dict)}
                    measurement_source = next(iter(sources)) if len(sources) == 1 else (
                        "prior_only" if not used else "mixed_sources"
                    )
                    measurement_set_id = "current_measurements" if state_is_current else "prior_only"
                    measurement_recommendation = measurement_bundle.get("recommendations", {}).get(
                        "recommended_measurement"
                    )
                st.info("검증된 KiCad 회로도가 없어 측정 위치는 회로도에 표시하지 않습니다. 추천은 정보 패널의 Node 이름으로만 제공합니다.")

    if svg is None:
        from circuitcare.dashboard.views.circuit import render_circuit_view
        with canvas:
            render_circuit_view(st, circuit=circuit, case_name=case_name, case=case,
                                selected_fault=selected_fault, fault_type=fault_type,
                                fault_value=fault_value, root=root, comparison_path=comparison_path)

    with inspector:
        item = circuit.components[component_ref]
        candidates = direct_candidates(circuit, component_ref)
        path_search = find_paths_result(circuit, component_ref, max_hops=2)
        from circuitcare.dashboard.navigation import korean_status
        st.markdown("#### 부품 상세")
        st.markdown(f"### {item.ref}")
        st.caption(item.value or "값 없음")
        st.caption(f"선택 부품 · {component_ref}  /  고장 가정 부품 · {selected_fault}")
        st.markdown("**연결**")
        component_connections = [conn for conn in circuit.connections if conn.ref == component_ref]
        if component_connections:
            for conn in component_connections:
                st.caption(f"{conn.pin}번 핀 → {conn.net}")
        else:
            st.caption("연결 Net 정보 없음")
        st.divider()
        st.markdown("**진단 상태**")
        marker = next((row for row in (overlay_payload or {}).get("markers", [])
                       if row.get("component_ref") == component_ref), None)
        if marker and marker.get("kind") == "RISK":
            st.metric("정격 스트레스", korean_status(marker.get("risk_level", "UNKNOWN")))
            st.caption(f"우선순위 {marker.get('priority_rank', '—')} · {marker.get('dominant_metric') or '근거 미확인'}")
            st.caption("정격 스트레스 평가이며 실제 손상 확률이나 손상 확정이 아닙니다.")
        elif overlay_mode == "전기적 위험":
            st.caption("위험 상태 · 검증된 평가 없음")
        else:
            st.caption("위험 상태 · 미평가")
        if measurement_recommendation:
            st.markdown("**다음 측정 후보**")
            st.write(f"{measurement_recommendation.get('node', '—')} · {measurement_recommendation.get('measurement_type', '측정')}")
            if measurement_anchors and measurement_anchors.anchors:
                st.caption("논리적 핀 · " + ", ".join(
                    f"{row['component_ref']}.{row['pin']}" for row in measurement_anchors.anchors))
        else:
            st.caption("다음 측정 · 추천 자료 없음")
        if candidates:
            st.markdown("**연결 후보**")
            st.caption(" · ".join(f"{candidate.target_ref} ({', '.join(candidate.shared_nets)})" for candidate in candidates[:4]))
        identity = mapping.get(component_ref, {"status": "UNVERIFIED_MAPPING"})
        with st.expander("근거 및 검증 정보", expanded=False):
            st.caption(f"시뮬레이션 · {korean_status(simulation_status)} · {spice_verification_label(simulation_status, simulation_integrity)}")
            st.caption(f"내부 상태 코드 · 시뮬레이션 `{simulation_status}` · 출처 `{simulation_integrity}`")
            st.caption(f"회로도 Export · {korean_status(status)}")
            st.caption(f"전기적 정합성 · {equivalence.status if equivalence else 'NOT_VERIFIED'} · {equivalence.message if equivalence else '회로도 없음'}")
            st.caption(f"부품 매칭 · {identity.get('status', 'UNVERIFIED_MAPPING')}")
            st.caption(f"경로 검색 · {korean_status(path_search.status)} · {len(path_search.paths)}개 후보 경로")
            for instance in identity.get("symbol_instances", []):
                st.caption(f"UUID · {instance['symbol_uuid']} · Unit {instance['unit']} · {instance['sheet_path']}")
            st.caption("연결 후보는 손상 또는 전기적 위험의 증거가 아닙니다.")
        if overlay_mode == "측정·점검" and diagnosis.get("model_mismatch"):
            st.warning("모델 불일치로 점검 순위와 위치 표시가 제한됩니다.")
        action_col, report_col = st.columns(2, gap="small")
        if action_col.button("파형 상세", key=f"workbench_waveform_{context_token}", use_container_width=True):
            st.session_state["workbench_navigation_target"] = "SPICE 파형"
            st.rerun()
        if report_col.button("보고서", key=f"workbench_report_{context_token}", use_container_width=True):
            st.session_state["workbench_navigation_target"] = "분석 개요"
            st.rerun()
