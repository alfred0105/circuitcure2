"""Conservative rule-based rating candidate discovery."""
from __future__ import annotations
import re
import unicodedata
from typing import Any

_RULES = [
    ("max_power_w", "power_rating", r"(?:rated\s+power|rated\s+dissipation|power\s+rating|maximum\s+power\s+dissipation|max(?:imum)?\s+dissipated\s+power|continuous\s+power|^power$)"),
    ("max_current_a", "current_rating", r"(?:maximum\s+continuous\s+current|continuous\s+current|rated\s+current|output\s+current|forward\s+current|peak\s+current)"),
    ("max_abs_voltage_v", "voltage_rating", r"(?:absolute\s+maximum\s+(?:input\s+)?voltage|maximum\s+input\s+voltage|maximum\s+working\s+voltage|working\s+voltage|rated\s+voltage|reverse\s+voltage|breakdown\s+voltage|operating\s+voltage|voltage\s+dc)"),
]
_NUMBER_UNIT = re.compile(r"(?P<value>[+−-]?\s*\d+(?:\.\d+)?)\s*(?P<prefix>m|u|µ|μ|k)?\s*(?P<unit>V(?:DC|AC)?|A(?:DC|AC)?|W)\b", re.I)
_PREFIX = {"": 1.0, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "k": 1e3}
_TARGET_UNIT = {"max_power_w": "W", "max_current_a": "A", "max_abs_voltage_v": "V"}

def find_rating_candidates(pages: list[dict[str, Any]], tables: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    results = []
    table_keys: set[tuple[int, str]] = set()
    for table in tables or []:
        rows = table.get("rows", [])
        for row_index, row in enumerate(rows):
            if len(row) < 2:
                continue
            if len(row) > 2:
                continue
            label, value_text = (_clean(str(row[0])), _clean(str(row[1])))
            for key, rating_type, keyword in _RULES:
                if not re.search(keyword, label, re.I):
                    continue
                # If a cell contains several voltage limits, retain only the value whose adjacent descriptor
                # says continuous/working. Overvoltage and withstand values are separate candidate classes.
                value_match = _NUMBER_UNIT.search(value_text)
                if key == "max_abs_voltage_v" and value_match:
                    descriptor = value_text[max(0, value_match.start() - 32):value_match.end() + 50]
                    if not re.search(r"working|continuous|rated", descriptor, re.I):
                        continue
                if not value_match or _canonical_unit(value_match.group("unit")) != _TARGET_UNIT[key]:
                    continue
                page_index = table["page_index"]
                page_text = next((page.get("text", "") for page in pages if page.get("page_index") == page_index), "")
                candidate = _make_candidate(key, rating_type, value_match, label + " " + value_text,
                                            page_index, "table_rule_based", page_text)
                if candidate:
                    results.append(candidate)
                    table_keys.add((page_index, key))
        for candidate in _matrix_candidates(table, pages):
            results.append(candidate)
            table_keys.add((candidate["pdf_page_index"], candidate["rating_key"]))
    for page in pages:
        lines = [_clean(line) for line in page.get("text", "").splitlines()]
        for line_no, line in enumerate(lines):
            for key, rating_type, keyword in _RULES:
                if (int(page.get("page_index", -1)), key) in table_keys:
                    continue
                if not re.search(keyword, line, re.I):
                    continue
                context = "\n".join(lines[max(0, line_no - 1):min(len(lines), line_no + 3)])
                for match in _NUMBER_UNIT.finditer(context):
                    unit = _canonical_unit(match.group("unit"))
                    if unit != _TARGET_UNIT[key]:
                        continue
                    snippet = context.strip()
                    candidate = _make_candidate(key, rating_type, match, snippet,
                                                int(page["page_index"]), "text_rule_based", page.get("text", ""))
                    if candidate:
                        results.append(candidate)
    unique = {}
    for row in results:
        unique[(row["rating_key"], row["normalized_value"], row["pdf_page_index"], row["source_text"])] = row
    return list(unique.values())

def _make_candidate(key: str, rating_type: str, match: re.Match, snippet: str, page_index: int,
                    method: str, page_text: str = "") -> dict[str, Any] | None:
    raw_number = match.group("value").replace(" ", "").replace("−", "-")
    try:
        value = float(raw_number) * _PREFIX[match.group("prefix") or ""]
    except (ValueError, KeyError):
        return None
    raw_signed_value = value
    if key == "max_abs_voltage_v":
        value = abs(value)
    if value <= 0:
        return None
    lower = snippet.casefold()
    kind = "peak" if re.search(r"peak|overload|pulse|surge", lower) else ("absolute_max" if "absolute maximum" in lower or "absolute max" in lower else "continuous")
    return {"rating_key": key, "rating_type": rating_type, "rating_kind": kind,
            "raw_value": match.group(0).strip(), "normalized_value": value,
            "normalized_unit": _TARGET_UNIT[key], "source_text": snippet.strip(),
            "pdf_page_index": page_index, "printed_page_number": _printed_page_number(page_text),
            "conditions": {**_conditions(snippet), "polarity": "negative" if raw_signed_value < 0 else _conditions(snippet).get("polarity")}, "extraction_method": method,
            "extraction_status": "candidate", "evidence_status": "PENDING_REVIEW", "human_approved": False}


def _canonical_unit(unit: str) -> str:
    normalized = unit.upper()
    return normalized[:1]


def _matrix_candidates(table: dict[str, Any], pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract individual rating cells while preserving row, variant, and column headers."""
    rows = table.get("rows", [])
    if not rows:
        return []
    model_header = []
    operation_header = []
    unit_column = None
    for row in rows[:8]:
        cleaned = [_clean(str(cell)) for cell in row]
        if unit_column is None:
            unit_column = next((index for index, cell in enumerate(cleaned) if cell.strip().casefold() == "unit"), None)
        if sum(bool(re.search(r"\b(?:MCS|MCT|MCU|MCA)\s*\d{4}\b", cell, re.I)) for cell in cleaned) >= 2:
            model_header = cleaned
        if any(re.search(r"precision|standard", cell, re.I) for cell in cleaned):
            operation_header = cleaned
    output = []
    active_key = active_type = active_label = None
    for row_index, row in enumerate(rows):
        cells = [_clean(str(cell)) for cell in row]
        if not cells:
            continue
        label = cells[0]
        match_rule = next((rule for rule in _RULES if re.search(rule[2], label, re.I)), None)
        if match_rule:
            active_key, active_type, active_label = match_rule[0], match_rule[1], label
        elif label and active_key and re.search(r"^[A-Za-z][A-Za-z0-9 /().,_-]*$", label) and not _NUMBER_UNIT.search(label):
            # A new descriptive row terminates a prior merged label, except rows that
            # are just continuation labels in the same table row family.
            if not re.search(r"(?:^|\s)(?:MCS|MCT|MCU|MCA)\s*\d{4}(?:$|\s)", label, re.I):
                if any(re.search(rule[2], label, re.I) for rule in _RULES):
                    continue
                if not active_label or not re.search(r"rated\s+dissipation|operating\s+voltage", active_label, re.I):
                    active_key = active_type = active_label = None
        if not active_key or row_index < 1:
            continue
        for column_index, cell in enumerate(cells):
            cell_text = cell
            number = _NUMBER_UNIT.search(cell_text)
            if not number and unit_column is not None and unit_column < len(cells) and cell and re.fullmatch(r"[+−-]?\s*\d+(?:\.\d+)?", cell):
                unit_text = cells[unit_column]
                if unit_text and unit_text.casefold() in {"v", "vdc", "vac", "a", "ma", "ua", "µa", "w", "mw"}:
                    cell_text = f"{cell} {unit_text}"
                    number = _NUMBER_UNIT.search(cell_text)
            if not number or _canonical_unit(number.group("unit")) != _TARGET_UNIT[active_key]:
                continue
            variant = ""
            if column_index < len(model_header) and re.search(r"\b(?:MCS|MCT|MCU|MCA)\s*\d{4}\b", model_header[column_index], re.I):
                variant = model_header[column_index]
            elif len(cells) > 2 and len(cells) > 1 and re.search(r"\b(?:MCS|MCT|MCU|MCA)\s*\d{4}\b", cells[1], re.I):
                variant = cells[1]
            column_name = operation_header[column_index] if column_index < len(operation_header) else ""
            page_index = int(table["page_index"])
            page_text = next((page.get("text", "") for page in pages if page.get("page_index") == page_index), "")
            default_condition_line = next((line.strip() for line in page_text.splitlines()
                if re.search(r"(?:T[_ ]?amb|ambient).*unless otherwise specified", line, re.I)), "")
            context = f"{active_label} {variant} {column_name} {default_condition_line}"
            candidate = _make_candidate(active_key, active_type, number, cell_text, page_index,
                                        "table_matrix_rule_based", page_text)
            if candidate:
                candidate["conditions"] = {**candidate["conditions"], **{key: value for key, value in _conditions(context).items() if value is not None}}
                label_text = (active_label or "").casefold()
                candidate["rating_kind"] = "peak" if re.search(r"peak|pulse|surge|overload", label_text) else ("absolute_max" if re.search(r"absolute\s+max", label_text) else "continuous")
                candidate["table_context"] = {"row_label": active_label, "variant": variant,
                    "column_header": column_name, "cell_text": cell_text,
                    "document_condition_context": default_condition_line or None,
                    "table_index": table.get("table_index"), "row_index": row_index,
                    "column_index": column_index}
                output.append(candidate)
    return output

def _clean(text: str) -> str:
    return "".join(char for char in text if unicodedata.category(char) != "Cf").replace("−", "-").strip()

def _printed_page_number(text: str) -> str | None:
    for line in reversed(text.splitlines()[-20:]):
        match = re.fullmatch(r"\s*(?:page\s*)?(\d{1,4})(?:\s+of\s+\d{1,4})?\s*", _clean(line), re.I)
        if match:
            return match.group(1)
    return None

def _conditions(text: str) -> dict[str, Any]:
    temp = re.search(r"(?:at\s+)?(-?\d+(?:\.\d+)?)\s*°?\s*C", text, re.I)
    if temp is None:
        temp = re.search(r"\bP\s*(?:\(\d+\)\s*)?70\b", text, re.I)
        temperature = 70.0 if temp else None
    else:
        temperature = float(temp.group(1))
    duration = "peak" if re.search(r"peak|pulse|surge|overload", text, re.I) else ("continuous" if re.search(r"continuous|rated|working|voltage\s+dc", text, re.I) else None)
    derating = next((line.strip() for line in text.splitlines() if re.search(r"derat|temperature", line, re.I)), None)
    signed = _NUMBER_UNIT.search(text)
    polarity = "negative" if signed and signed.group("value").replace(" ", "").startswith(("-", "−")) else None
    return {"temperature_c": temperature, "duration": duration, "polarity": polarity, "derating_text": derating}
