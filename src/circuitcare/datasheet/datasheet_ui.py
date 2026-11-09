"""Streamlit review workflow for local datasheet rating candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .evidence_validator import validate_candidate
from .pdf_loader import DatasheetError, extract_pdf_tables, load_pdf
from .rating_extractor import find_rating_candidates
from .rating_review import load_reviews
from .ui_helpers import _attach_document


def render_datasheet_tab(st: Any, *, root: Path, circuit: Any, context_key: str, case: dict[str, Any],
                         risk: dict[str, Any], stress: dict[str, Any], client: Any,
                         models: list[dict[str, Any]], stress_signature: str | None = None,
                         applied_records: list[dict[str, Any]] | None = None,
                         current_risk: dict[str, Any] | None = None) -> None:
    st.subheader("데이터시트 AI · 근거 확인 후 정격 검토")
    st.caption("로컬 PDF와 loopback Ollama만 사용합니다. AI 제안은 사실이 아니며, PDF 원문·부품 모델·조건을 확인하고 승인하기 전에는 Rating Engine에 적용되지 않습니다.")
    references = list(circuit.components)
    component_ref = st.selectbox("KiCad Reference", references, key=f"ds_ref_{context_key}")
    component = circuit.components[component_ref]
    st.write(f"**선택 부품:** {component_ref} · KiCad Value `{component.value}` · Library `{component.library_id}` · Footprint `{component.footprint or '미기록'}` · KiCad datasheet `{component.datasheet or '미기록'}`")
    left, right = st.columns(2)
    with left:
        manufacturer = st.text_input("제조사", key=f"ds_mfr_{context_key}")
        mpn = st.text_input("Manufacturer Part Number (MPN)", key=f"ds_mpn_{context_key}")
        revision = st.text_input("문서 Revision (선택)", key=f"ds_rev_{context_key}")
        uploaded = st.file_uploader("데이터시트 PDF", type=["pdf"], max_upload_size=25, key=f"ds_pdf_{context_key}")
    with right:
        st.write("현재 활성 SPICE 스트레스와 저장된 위험 결과")
        st.write({"simulation_success": stress.get("simulation_success"),
                  "scenario_id": risk.get("scenario_id"), "risk_available": bool(risk),
                  "active_rating_source": "현재 시나리오 JSON" if risk else "없음"})
        current_rating = next((row for row in risk.get("rating_checks", []) if row.get("component_ref") == component_ref), None)
        if current_rating:
            st.write("현재 설정 정격")
            st.json(current_rating)
        active_for_selected = next((row for row in (applied_records or [])
                                     if row.get("component_ref") == component_ref), None)
        if active_for_selected:
            st.success(f"이 Reference에는 승인 정격이 활성화되어 있습니다 · {active_for_selected['record'].get('manufacturer')} {active_for_selected['record'].get('mpn')} · SHA-256 {active_for_selected['record'].get('source_document_sha256')}")
        st.caption("시뮬레이션 스트레스/파형은 고정된 입력입니다. 승인 정격을 비교할 때 SPICE를 다시 실행하지 않습니다.")
    if uploaded is None:
        st.info('PDF 처리에는 선택 의존성이 필요합니다: `python -m pip install -e ".[datasheet]"`')
        return
    raw_pdf = uploaded.getvalue()
    try:
        document = load_pdf(raw_pdf, uploaded.name)
        document["tables"] = extract_pdf_tables(raw_pdf)
    except DatasheetError as exc:
        st.error(str(exc))
        return
    from .ui_document import render_document_summary
    render_document_summary(st, document, manufacturer=manufacturer, mpn=mpn, revision=revision)
    if document["text_status"] == "OCR_REQUIRED":
        st.warning("이미지 기반 또는 텍스트가 없는 PDF입니다. OCR을 자동 실행하지 않았습니다. 이 환경의 선택 OCR 기능은 데이터시트 텍스트 추출용으로 구성되지 않아 수동 텍스트 PDF가 필요합니다.")
        return
    model_names = [item.get("name") or item.get("model") for item in models if item.get("name") or item.get("model")]
    model_name = None
    if client and model_names:
        model_name = st.selectbox("로컬 Ollama 모델", model_names, key=f"ds_model_{context_key}_{component_ref}")
    extraction_model = model_name
    model_meta = next((item for item in models if (item.get("name") or item.get("model")) == model_name), {}) if model_name else {}
    model_digest = model_meta.get("digest")
    state_id = ":".join((context_key, component_ref, document["sha256"], manufacturer.strip(), mpn.strip(), revision.strip(), model_name or "rule-only", model_digest or "no-digest", "datasheet-rating-json-v1"))
    state_key = f"datasheet_state_{state_id}"
    if st.session_state.get("datasheet_active_state") != state_id:
        previous_state = st.session_state.get("datasheet_active_state")
        if previous_state:
            st.session_state.pop(f"datasheet_state_{previous_state}", None)
        st.session_state["datasheet_active_state"] = state_id
        initial_candidates = find_rating_candidates(document["pages"], document["tables"])
        saved_reviews = load_reviews(root / "data/private/datasheet_ai/review_history.json",
            sha256=document["sha256"], mpn=mpn.strip(), reference=component_ref, revision=revision.strip(),
            extraction_model=extraction_model, model_digest=model_digest)
        reviewed_by_source = {}
        for record in saved_reviews:
            original = record.get("original_candidate") or record
            reviewed_by_source[(original.get("rating_key"), original.get("pdf_page_index"), original.get("source_text"))] = record
        initial_review_state = {}
        for index, candidate in enumerate(initial_candidates):
            match = reviewed_by_source.get((candidate.get("rating_key"), candidate.get("pdf_page_index"), candidate.get("source_text")))
            if match:
                initial_review_state[str(index)] = match
        st.session_state[state_key] = {"document": document, "candidates": initial_candidates,
                                       "llm": None, "reviews": initial_review_state}
    ds_state = st.session_state[state_key]
    candidates = ds_state["candidates"]
    from .ui_document import render_candidate_list
    render_candidate_list(st, candidates)
    if client and models and model_name:
        if st.button("로컬 LLM으로 정격 구조화", disabled=not candidates, key=f"ds_extract_{context_key}"):
            try:
                from .llm_extractor import extract_with_local_llm
                with st.spinner("PDF에서 후보가 발견된 페이지만 로컬 LLM으로 분석 중…"):
                    ds_state["llm"] = extract_with_local_llm(client, model_name, manufacturer=manufacturer, mpn=mpn,
                                                             candidates=candidates, pages=document["pages"],
                                                             tables=document["tables"])
                llm_rows = ds_state["llm"]["ratings"]
                ds_state["candidates"] = [_attach_document(validate_candidate({**row, "source_document_sha256": document["sha256"]}, document["pages"],
                                                                                manufacturer=manufacturer, mpn=mpn), document)
                                           for row in llm_rows]
                for row in ds_state["candidates"]:
                    row["llm_model"] = ds_state["llm"].get("model")
                    row["llm_model_info"] = ds_state["llm"].get("model_info")
                    row["llm_model_digest"] = (ds_state["llm"].get("model_info") or {}).get("digest")
                    row["llm_total_duration_ns"] = ds_state["llm"].get("total_duration_ns")
                    row["extraction_prompt_version"] = "datasheet-rating-json-v1"
                saved_reviews = load_reviews(root / "data/private/datasheet_ai/review_history.json",
                    sha256=document["sha256"], mpn=mpn.strip(), reference=component_ref,
                    revision=revision.strip(), extraction_model=ds_state["llm"].get("model"),
                    model_digest=(ds_state["llm"].get("model_info") or {}).get("digest"))
                reviewed_by_source = {}
                for record in saved_reviews:
                    original = record.get("original_candidate") or record
                    reviewed_by_source[(original.get("rating_key"), original.get("pdf_page_index"), original.get("source_text"))] = record
                ds_state["reviews"] = {str(index): reviewed_by_source[(row.get("rating_key"), row.get("pdf_page_index"), row.get("source_text"))]
                    for index, row in enumerate(ds_state["candidates"])
                    if (row.get("rating_key"), row.get("pdf_page_index"), row.get("source_text")) in reviewed_by_source}
                st.rerun()
            except Exception as exc:  # noqa: BLE001 - preserve fallback behavior for optional LLM extraction
                st.error(f"로컬 LLM 추출 실패 · 기존 규칙 후보는 유지됩니다: {exc}")
        if ds_state.get("llm"):
            st.caption(f"로컬 Ollama 모델: {ds_state['llm'].get('model')} · 처리시간 ns: {ds_state['llm'].get('total_duration_ns')}")
    else:
        st.info("Ollama가 연결되지 않았거나 모델이 없습니다. 규칙 후보와 수동 검토 기능은 계속 사용할 수 있습니다.")
    candidates = ds_state["candidates"]
    if not candidates:
        return
    labels = [f"{row['rating_type']} · {row['raw_value']} · p.{row['pdf_page_index'] + 1} · {row.get('extraction_method')}" for row in candidates]
    selected_index = st.selectbox("검토할 정격 후보", range(len(candidates)), format_func=lambda index: labels[index], key=f"ds_candidate_{state_id}")
    selected = candidates[selected_index]
    from .ui_document import render_candidate_source
    selected, source_status = render_candidate_source(
        st, document=document, selected=selected, manufacturer=manufacturer, mpn=mpn,
        revision=revision, candidate_index=selected_index, candidates=candidates,
    )
    from .ui_candidate_review import render_candidate_review
    conditions_fit = render_candidate_review(
        st, root=root, selected=selected, document=document, manufacturer=manufacturer, mpn=mpn,
        revision=revision, component_ref=component_ref, state_id=state_id,
        selected_index=selected_index, ds_state=ds_state, source_status=source_status,
    )
    from .ui_application import render_rating_application
    render_rating_application(
        st, root=root, case=case, risk=risk, stress=stress, component_ref=component_ref,
        selected_index=selected_index, ds_state=ds_state, conditions_fit=conditions_fit,
        context_key=context_key, state_id=state_id, stress_signature=stress_signature,
        current_risk=current_risk,
    )
