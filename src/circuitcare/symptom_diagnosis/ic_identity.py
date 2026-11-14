"""Conservative IC marking resolver; prefix hints are never confirmed identities."""

from __future__ import annotations

import re


_ROLE_HINTS = (
    (re.compile(r"^(?:STM32|STM8|ATMEGA|ATTINY|PIC(?:16|18|24|32)|ESP32|MSP430)"), "control"),
    (re.compile(r"^(?:W25|25Q|24C|AT24|M24C)"), "memory"),
    (re.compile(r"^(?:CP210|CH340|FT232|MAX232|TJA|SN65)"), "communication"),
    (re.compile(r"^(?:TPS|MP15|MP23|RT82|LM2596|LM2576|AMS1117)"), "power"),
    (re.compile(r"^(?:DRV|ULN|L293|A4988|TB6612)"), "driver"),
    (re.compile(r"^(?:LM358|LM393|TL072|TL074|OP07)"), "analog"),
)


def normalize_marking(marking: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (marking or "").upper())


def resolve_ic_identity(marking: str | None, catalog: dict[str, dict] | None = None) -> dict:
    """Exact catalog match is explicit; generic marking prefixes stay unverified hints."""
    normalized = normalize_marking(marking)
    if not normalized:
        return {"marking": marking, "normalized_marking": "", "status": "NO_MARKING",
                "role_candidates": [], "manufacturer": None, "package": None,
                "pin_count": None, "datasheet_url": None, "verified": False}
    catalog = catalog or {}
    for part_number, row in catalog.items():
        if normalize_marking(part_number) == normalized:
            verified = bool(
                row.get("verified") is True
                and row.get("verification_source") == "manufacturer_datasheet"
                and row.get("manufacturer") and row.get("package")
                and isinstance(row.get("pin_count"), int) and row.get("pin_count") > 0
                and str(row.get("datasheet_url", "")).startswith("https://")
            )
            return {"marking": marking, "normalized_marking": normalized,
                    "status": "CATALOG_VERIFIED" if verified else "CATALOG_MATCH_UNVERIFIED",
                    "part_number": part_number,
                    "role_candidates": list(row.get("roles", [])),
                    "manufacturer": row.get("manufacturer"), "package": row.get("package"),
                    "pin_count": row.get("pin_count"), "datasheet_url": row.get("datasheet_url"),
                    "verified": verified}
    roles = [role for pattern, role in _ROLE_HINTS if pattern.search(normalized)]
    return {"marking": marking, "normalized_marking": normalized,
            "status": "MARKING_PREFIX_HINT" if roles else "UNRESOLVED_MARKING",
            "role_candidates": roles, "manufacturer": None, "package": None,
            "pin_count": None, "datasheet_url": None, "verified": False}
