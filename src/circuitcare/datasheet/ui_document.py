"""PDF document and extracted candidate evidence presentation."""

from __future__ import annotations

from typing import Any

from .evidence_validator import validate_candidate
from .ui_helpers import _attach_document


def render_document_summary(st: Any, document: dict, *, manufacturer: str,
                            mpn: str, revision: str) -> None:
    st.json({"filename": document["filename"], "sha256": document["sha256"],
             "size_bytes": document["size_bytes"], "page_count": document["page_count"],
             "metadata": document["metadata"], "text_status": document["text_status"],
             "pages_with_text": document["pages_with_text"], "tables_found": len(document["tables"]),
             "manufacturer": manufacturer, "mpn": mpn, "revision": revision})


def render_candidate_list(st: Any, candidates: list[dict]) -> None:
    if not candidates:
        st.warning("지원되는 Voltage / Current / Power 정격 후보가 규칙 탐색에서 발견되지 않았습니다. 정격을 임의 생성하지 않습니다.")
        return
    st.write("로컬 LLM 제안 — 아직 승인되지 않음" if any(
        row.get("extraction_method") == "local_ollama" for row in candidates
    ) else "규칙 기반 후보 — 아직 승인되지 않음")
    st.dataframe([{"정격": row["rating_type"], "값": row["raw_value"], "종류": row["rating_kind"],
                   "페이지(0부터)": row["pdf_page_index"], "근거": row["source_text"],
                   "표 행·열 문맥": row.get("table_context")} for row in candidates], hide_index=True)


def render_candidate_source(st: Any, *, document: dict, selected: dict,
                            manufacturer: str, mpn: str, revision: str,
                            candidate_index: int, candidates: list[dict]) -> tuple[dict, str | None]:
    page = document["pages"][selected["pdf_page_index"]]
    st.markdown("**PDF 페이지 원문 (일부)**")
    st.code(page["text"][:12000] or "텍스트가 없습니다.")
    with st.expander("표 추출 결과 (행/열 보존)"):
        tables = [table for table in document["tables"]
                  if table["page_index"] == selected["pdf_page_index"]]
        if tables:
            for table in tables:
                st.caption(f"페이지 {table['page_index'] + 1} · 표 {table['table_index'] + 1}")
                st.dataframe(table["rows"], hide_index=True)
        else:
            st.info("이 페이지에서 표를 찾지 못했습니다. 문장 기반 후보만 검토하세요.")
    source_status = selected.get("evidence_status")
    st.write({"evidence_status": source_status or "PENDING_REVIEW",
              "raw_value": selected.get("raw_value"),
              "normalized_value": selected.get("normalized_value"),
              "normalized_unit": selected.get("normalized_unit"),
              "rating_kind": selected.get("rating_kind"), "conditions": selected.get("conditions"),
              "source_text": selected.get("source_text"),
              "source_page_index": selected.get("pdf_page_index"),
              "manufacturer": manufacturer, "mpn": mpn, "revision": revision,
              "llm_model": selected.get("llm_model") if selected.get("extraction_method") == "local_ollama" else None})
    if selected.get("table_context"):
        st.caption("표 기반 후보는 숫자 인용뿐 아니라 동일 표의 행·열·부품 모델 문맥을 함께 검토합니다.")
        st.json({"table_context": selected["table_context"]})
    if source_status in {None, "PENDING_REVIEW"}:
        selected = _attach_document(validate_candidate(
            {**selected, "source_document_sha256": document["sha256"]}, document["pages"],
            manufacturer=manufacturer, mpn=mpn,
        ), document)
        candidates[candidate_index] = selected
        source_status = selected["evidence_status"]
        st.warning(f"근거 검사 상태: {source_status} · {', '.join(selected.get('match_notes', [])) or '페이지 원문과 값/단위 일치'}")
    else:
        st.info(f"근거 검사 상태: {source_status} · {', '.join(selected.get('match_notes', [])) or '페이지 원문과 값/단위 일치'}")
    return selected, source_status
