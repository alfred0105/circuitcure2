"""Bounded local PDF text and optional table extraction."""
from __future__ import annotations

import copy
import hashlib
import io
from collections import OrderedDict
from typing import Any

MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 200
MAX_EXTRACTED_TEXT_CHARS = 4_000_000
MAX_TABLES = 200
MAX_TABLE_CELLS = 20_000
PARSER_VERSION = "pdf-extract-v2"
_CACHE_MAX_ENTRIES = 4
_document_cache: OrderedDict[tuple[str, str, int, int], dict[str, Any]] = OrderedDict()
_table_cache: OrderedDict[tuple[str, str, int], list[dict[str, Any]]] = OrderedDict()
_cache_stats = {"document_hits": 0, "document_misses": 0, "table_hits": 0, "table_misses": 0}


def cache_stats(*, reset: bool = False) -> dict[str, int]:
    stats = dict(_cache_stats)
    if reset:
        for key in _cache_stats:
            _cache_stats[key] = 0
    return stats


def clear_pdf_cache() -> None:
    _document_cache.clear()
    _table_cache.clear()


def _cache_put(cache: OrderedDict, key: tuple, value: Any) -> None:
    cache[key] = copy.deepcopy(value)
    cache.move_to_end(key)
    while len(cache) > _CACHE_MAX_ENTRIES:
        cache.popitem(last=False)

class DatasheetError(ValueError):
    """Invalid, unsupported, or over-limit local PDF."""

def _validate_pdf(data: bytes, max_bytes: int) -> str:
    if not isinstance(data, bytes) or not data:
        raise DatasheetError("The uploaded PDF is empty.")
    if len(data) > max_bytes:
        raise DatasheetError(f"PDF exceeds the {max_bytes // (1024 * 1024)} MiB size limit.")
    if not data.startswith(b"%PDF-"):
        raise DatasheetError("The uploaded file does not have a PDF signature.")
    return hashlib.sha256(data).hexdigest()


def _load_pdf_local(data: bytes, filename: str, *, max_bytes: int = MAX_PDF_BYTES,
                    max_pages: int = MAX_PDF_PAGES) -> dict[str, Any]:
    digest = _validate_pdf(data, max_bytes)
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DatasheetError('Install PDF support with: python -m pip install -e ".[datasheet]"') from exc
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise DatasheetError("Encrypted PDFs are not supported. Provide an unlocked local copy.")
        if not reader.pages:
            raise DatasheetError("PDF contains no pages.")
        if len(reader.pages) > max_pages:
            raise DatasheetError(f"PDF exceeds the {max_pages}-page processing limit.")
        pages = []
        total_chars = 0
        for index, page in enumerate(reader.pages):
            text = page.extract_text(extraction_mode="layout") or ""
            remaining = MAX_EXTRACTED_TEXT_CHARS - total_chars
            if len(text) > remaining:
                text = text[:remaining]
            pages.append({"page_index": index, "text": text})
            total_chars += len(text)
            if total_chars >= MAX_EXTRACTED_TEXT_CHARS:
                break
        metadata = {str(key).lstrip("/"): str(value) for key, value in (reader.metadata or {}).items() if value is not None}
    except DatasheetError:
        raise
    except Exception as exc:
        raise DatasheetError(f"Could not read PDF: {exc}") from exc
    extracted = sum(bool(page["text"].strip()) for page in pages)
    result = {"filename": filename, "sha256": digest, "size_bytes": len(data),
            "page_count": len(pages), "metadata": metadata, "pages": pages,
            "text_status": "TEXT_EXTRACTED" if extracted else "OCR_REQUIRED", "pages_with_text": extracted,
            "truncated_by_text_limit": total_chars >= MAX_EXTRACTED_TEXT_CHARS}
    return result

def _extract_pdf_tables_local(data: bytes, *, max_pages: int = MAX_PDF_PAGES) -> list[dict[str, Any]]:
    try:
        import pdfplumber
    except ImportError:
        return []
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if len(pdf.pages) > max_pages:
                raise DatasheetError(f"PDF exceeds the {max_pages}-page table processing limit.")
            tables = []
            cells_used = 0
            for index, page in enumerate(pdf.pages):
                for table_index, table in enumerate(page.extract_tables() or []):
                    if not table or len(tables) >= MAX_TABLES:
                        continue
                    normalized = []
                    for row in table:
                        clean_row = []
                        for cell in row:
                            if cells_used >= MAX_TABLE_CELLS:
                                break
                            clean_row.append((cell or "")[:1000])
                            cells_used += 1
                        if clean_row:
                            normalized.append(clean_row)
                        if cells_used >= MAX_TABLE_CELLS:
                            break
                    tables.append({"page_index": index, "table_index": table_index, "rows": normalized})
                    if cells_used >= MAX_TABLE_CELLS:
                        break
                if cells_used >= MAX_TABLE_CELLS or len(tables) >= MAX_TABLES:
                    break
        return tables
    except Exception as exc:
        raise DatasheetError(f"PDF table extraction failed: {exc}") from exc


def load_pdf(data: bytes, filename: str, *, max_bytes: int = MAX_PDF_BYTES,
             max_pages: int = MAX_PDF_PAGES) -> dict[str, Any]:
    digest = _validate_pdf(data, max_bytes)
    cache_key = (digest, PARSER_VERSION, max_bytes, max_pages)
    if cache_key in _document_cache:
        _cache_stats["document_hits"] += 1
        _document_cache.move_to_end(cache_key)
        document = copy.deepcopy(_document_cache[cache_key])
        document["filename"] = filename
        return document
    _cache_stats["document_misses"] += 1
    from .pdf_supervisor import run_pdf_worker
    document, worker = run_pdf_worker(data, "document", filename=filename,
                                      max_bytes=max_bytes, max_pages=max_pages)
    if document.get("sha256") != digest or not isinstance(document.get("pages"), list):
        raise DatasheetError("PDF worker returned an invalid or mismatched document result.")
    document["worker"] = worker
    _cache_put(_document_cache, cache_key, document)
    return document


def extract_pdf_tables(data: bytes) -> list[dict[str, Any]]:
    digest = _validate_pdf(data, MAX_PDF_BYTES)
    cache_key = (digest, PARSER_VERSION, MAX_TABLE_CELLS)
    if cache_key in _table_cache:
        _cache_stats["table_hits"] += 1
        _table_cache.move_to_end(cache_key)
        return copy.deepcopy(_table_cache[cache_key])
    _cache_stats["table_misses"] += 1
    from .pdf_supervisor import run_pdf_worker
    tables, _worker = run_pdf_worker(data, "tables", max_bytes=MAX_PDF_BYTES,
                                     max_pages=MAX_PDF_PAGES)
    if not isinstance(tables, list):
        raise DatasheetError("PDF worker returned an invalid table result.")
    _cache_put(_table_cache, cache_key, tables)
    return tables
