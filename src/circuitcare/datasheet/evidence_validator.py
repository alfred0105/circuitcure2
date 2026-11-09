"""Validate an extracted value and verbatim quote against its cited page."""
from __future__ import annotations
import re
import unicodedata
from typing import Any
from .rating_extractor import _NUMBER_UNIT, _PREFIX, _conditions

def validate_candidate(candidate: dict[str, Any], pages: list[dict[str, Any]], *, manufacturer: str, mpn: str) -> dict[str, Any]:
    page_index = candidate.get("pdf_page_index")
    if isinstance(page_index, bool) or not isinstance(page_index, int) or not 0 <= page_index < len(pages):
        return {**candidate, "evidence_status": "SOURCE_MISMATCH", "match_notes": ["PDF page index is invalid."]}
    page_text = pages[page_index].get("text", "")
    source = str(candidate.get("source_text") or "")
    if not source or _normalized(source) not in _normalized(page_text):
        return {**candidate, "evidence_status": "SOURCE_NOT_FOUND", "match_notes": ["Quoted evidence is not present on the cited PDF page."]}
    numbers = []
    for match in _NUMBER_UNIT.finditer(source):
        try:
            number = float(match.group("value").replace(" ", "").replace("−", "-")) * _PREFIX[match.group("prefix") or ""]
            if candidate.get("rating_key") == "max_abs_voltage_v":
                number = abs(number)
            numbers.append((number, match.group("unit").upper()[:1]))
        except (ValueError, KeyError):
            pass
    range_expression = re.search(r"(?:\d+(?:\.\d+)?\s*(?:m|u|µ|μ|k)?\s*[VAW])\s*(?:-|–|to)\s*\d+(?:\.\d+)?\s*(?:m|u|µ|μ|k)?\s*[VAW]", source, re.I)
    if range_expression or (len(numbers) > 1 and re.search(r"\b(?:range|from|to)\b|[-–]", source, re.I)):
        return {**candidate, "evidence_status": "SOURCE_AMBIGUOUS", "match_notes": ["The quoted evidence contains a range or multiple limits; the rating relationship needs human disambiguation."]}
    value = candidate.get("normalized_value")
    unit = str(candidate.get("normalized_unit", "")).upper()
    table_context = candidate.get("table_context")
    if isinstance(table_context, dict):
        context_fields = [(key, table_context.get(key)) for key in ("row_label", "variant", "column_header", "cell_text", "document_condition_context") if table_context.get(key)]
        absent = [str(field) for key, field in context_fields if not _table_context_found(key, str(field), page_text)]
        if absent:
            return {**candidate, "evidence_status": "SOURCE_MISMATCH",
                    "match_notes": ["Table row, model variant, or column context was not found on the cited page."]}
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not any(abs(number - value) <= max(1e-9, abs(value) * 1e-6) and found_unit == unit for number, found_unit in numbers):
        return {**candidate, "evidence_status": "SOURCE_MISMATCH", "match_notes": ["Normalized value and physical unit are not both present in the cited evidence."]}
    whole_doc = "\n".join(page.get("text", "") for page in pages)
    if manufacturer and not re.search(re.escape(manufacturer), whole_doc, re.I):
        return {**candidate, "evidence_status": "SOURCE_AMBIGUOUS", "match_notes": ["Manufacturer string was not found in extracted PDF text."]}
    conditions = candidate.get("conditions") or _conditions(source)
    if not conditions.get("duration"):
        return {**candidate, "conditions": conditions, "evidence_status": "CONDITION_INCOMPLETE",
                "match_notes": ["Rating duration/use condition is not explicit in the quoted evidence; reviewer input is required."],
                "manufacturer": manufacturer, "mpn": mpn}
    return {**candidate, "evidence_status": "SOURCE_MATCHED", "match_notes": [], "manufacturer": manufacturer, "mpn": mpn}

def _normalized(value: str) -> str:
    value = "".join(char for char in value if unicodedata.category(char) != "Cf")
    return re.sub(r"\s+", "", value).casefold().replace("−", "-")


def _table_context_found(kind: str, value: str, page_text: str) -> bool:
    normalized_page = _normalized(page_text)
    normalized_value = _normalized(value)
    if normalized_value in normalized_page:
        return True
    if kind == "row_label":
        terms = [word.casefold() for word in re.findall(r"[A-Za-z]+", value)
                 if word.casefold() not in {"p", "max", "rms", "ac", "dc"}]
        return len(terms) >= 2 and all(term in normalized_page for term in terms[:2])
    return False
