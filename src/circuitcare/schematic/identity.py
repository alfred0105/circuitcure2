"""Conservative identity extraction from KiCad S-expression schematics."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias, cast

MAX_SCHEMATIC_BYTES = 20_000_000
MAX_PROJECT_FILES = 32
MAX_SHEET_DEPTH = 8
SExpr: TypeAlias = str | list["SExpr"]


@dataclass(frozen=True)
class SymbolIdentity:
    component_ref: str
    symbol_uuid: str
    sheet_path: str
    unit: int
    library_id: str
    value: str
    status: str = "UNVERIFIED_MAPPING"


def _unescape(token: str) -> str:
    value = token[1:-1]
    output: list[str] = []
    i = 0
    escapes = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}
    while i < len(value):
        if value[i] == "\\" and i + 1 < len(value):
            i += 1
            output.append(escapes.get(value[i], value[i]))
        else:
            output.append(value[i])
        i += 1
    return "".join(output)


def parse_sexpr(text: str) -> list[SExpr]:
    """Parse KiCad's nested lists while respecting escaped strings."""
    stack: list[list[SExpr]] = []
    root: list[SExpr] = []
    current = root
    # Validate tokenization separately so ignored trailing or malformed quotes fail closed.
    token_pattern = re.compile(r'"(?:\\.|[^"\\])*"|[()]|[^\s()]+')
    cursor = 0
    for match in token_pattern.finditer(text):
        gap = text[cursor:match.start()]
        if '"' in gap:
            raise ValueError("Malformed quoted string in KiCad S-expression")
        token = match.group(0)
        cursor = match.end()
        if token == "(":
            child: list[SExpr] = []
            current.append(child)
            stack.append(current)
            current = child
        elif token == ")":
            if not stack:
                raise ValueError("Unexpected closing parenthesis in KiCad S-expression")
            current = stack.pop()
        else:
            if token.startswith('"'):
                if not token.endswith('"'):
                    raise ValueError("Malformed quoted string in KiCad S-expression")
                current.append(_unescape(token))
            else:
                current.append(token)
    if '"' in text[cursor:] or stack:
        raise ValueError("Incomplete KiCad S-expression")
    if len(root) != 1 or not isinstance(root[0], list):
        raise ValueError("Expected one KiCad S-expression root")
    return root[0]


def _children(node: list[SExpr], key: str) -> list[list[SExpr]]:
    return [item for item in node if isinstance(item, list) and item and item[0] == key]


def _child(node: list[SExpr], key: str) -> list[SExpr] | None:
    return next(iter(_children(node, key)), None)


def _atom(node: list[SExpr] | None, index: int = 1, default: str = "") -> str:
    if node is not None and len(node) > index and isinstance(node[index], str):
        return cast(str, node[index])
    return default


def _prop(node: list[SExpr], name: str) -> str | None:
    for prop in _children(node, "property"):
        if len(prop) >= 3 and prop[1] == name and isinstance(prop[2], str):
            return prop[2]
    return None


def _walk_sheets(path: Path, root: Path) -> tuple[list[SymbolIdentity], list[str]]:
    identities: list[SymbolIdentity] = []
    files, missing = discover_schematic_files(path, project_root=root)
    for source in files:
        tree = parse_sexpr(source.read_text(encoding="utf-8"))
        if not tree or tree[0] != "kicad_sch":
            raise ValueError("Expected KiCad schematic root")
        for symbol in _children(tree, "symbol"):
            ref = _prop(symbol, "Reference")
            if not ref or ref.startswith("#"):
                continue
            uuid = _atom(_child(symbol, "uuid"))
            lib = _atom(_child(symbol, "lib_id"))
            value = _prop(symbol, "Value") or ""
            default_unit = _atom(_child(symbol, "unit"), default="1")
            try:
                unit = int(default_unit)
            except ValueError:
                raise ValueError(f"Invalid symbol unit for {ref}") from None
            instances = _child(symbol, "instances")
            project_paths: list[list[SExpr]] = []
            if instances:
                for project in _children(instances, "project"):
                    project_paths.extend(_children(project, "path"))
            if not project_paths:
                project_paths = [["path", "/", ["reference", ref], ["unit", str(unit)]]]
            for instance_path in project_paths:
                inst_ref = _atom(_child(instance_path, "reference"), default=ref)
                inst_unit_raw = _atom(_child(instance_path, "unit"), default=str(unit))
                try:
                    inst_unit = int(inst_unit_raw)
                except ValueError:
                    raise ValueError(f"Invalid instance unit for {inst_ref}") from None
                sheet_path = _atom(instance_path, default="/")
                identities.append(SymbolIdentity(inst_ref, uuid, sheet_path, inst_unit, lib, value))
    # KiCad can reuse one symbol identity on different hierarchical paths (for a
    # multiply-instantiated sheet). A UUID collision within the same sheet path is real.
    seen: dict[tuple[str, str], tuple[str, int, str, str]] = {}
    ref_units: dict[tuple[str, str, int], str] = {}
    deduped: list[SymbolIdentity] = []
    for item in identities:
        if not item.symbol_uuid:
            raise ValueError(f"Symbol {item.component_ref} is missing its UUID")
        key = (item.symbol_uuid, item.sheet_path)
        identity = (item.component_ref, item.unit, item.library_id, item.value)
        previous = seen.get(key)
        if previous and previous != identity:
            raise ValueError(f"Conflicting symbol UUID {item.symbol_uuid} on {item.sheet_path}")
        ref_unit_key = (item.component_ref, item.sheet_path, item.unit)
        previous_uuid = ref_units.get(ref_unit_key)
        if previous_uuid and previous_uuid != item.symbol_uuid:
            raise ValueError(f"Duplicate symbol reference/unit {item.component_ref} unit {item.unit} on {item.sheet_path}")
        ref_units[ref_unit_key] = item.symbol_uuid
        if previous is None:
            seen[key] = identity
            deduped.append(item)
    return deduped, missing


def discover_schematic_files(path: str | Path, *, project_root: str | Path) -> tuple[list[Path], list[str]]:
    """Recursively list a bounded project schematic tree, returning incomplete reasons."""
    root = Path(project_root).resolve(strict=True)
    source = Path(path).resolve(strict=True)
    if source.suffix != ".kicad_sch" or not source.is_relative_to(root):
        raise ValueError("Schematic must be a .kicad_sch file inside the project root")
    files: list[Path] = []
    missing: list[str] = []
    visited: set[Path] = set()

    def walk(current: Path, depth: int) -> None:
        current = current.resolve(strict=False)
        if depth > MAX_SHEET_DEPTH or len(visited) >= MAX_PROJECT_FILES:
            missing.append("INCOMPLETE_PROJECT: sheet traversal limit reached")
            return
        if not current.is_relative_to(root):
            missing.append(f"INCOMPLETE_PROJECT: external sheet blocked: {current.name}")
            return
        if current in visited:
            return
        if not current.is_file():
            missing.append(f"INCOMPLETE_PROJECT: missing sheet: {current.name}")
            return
        if current.stat().st_size > MAX_SCHEMATIC_BYTES:
            missing.append(f"INCOMPLETE_PROJECT: oversized sheet: {current.name}")
            return
        visited.add(current)
        files.append(current)
        tree = parse_sexpr(current.read_text(encoding="utf-8"))
        if not tree or tree[0] != "kicad_sch":
            raise ValueError("Expected KiCad schematic root")
        for sheet in _children(tree, "sheet"):
            filename = _prop(sheet, "Sheet file")
            if not filename:
                missing.append(f"INCOMPLETE_PROJECT: sheet has no Sheet file property: {current.name}")
            elif Path(filename).suffix != ".kicad_sch":
                missing.append(f"INCOMPLETE_PROJECT: invalid child sheet file: {filename}")
            else:
                walk(current.parent / filename, depth + 1)

    walk(source, 0)
    return files, sorted(set(missing))


def read_symbol_identities(path: str | Path, *, project_root: str | Path) -> tuple[list[SymbolIdentity], list[str]]:
    """Read placed symbols recursively, bounding traversal to the project directory."""
    root = Path(project_root).resolve(strict=True)
    source = Path(path).resolve(strict=True)
    if source.suffix != ".kicad_sch" or not source.is_relative_to(root):
        raise ValueError("Schematic must be a .kicad_sch file inside the project root")
    if source.stat().st_size > MAX_SCHEMATIC_BYTES:
        raise ValueError("Invalid or oversized KiCad schematic")
    return _walk_sheets(source, root)


def map_component_refs(component_refs: list[str] | tuple[str, ...], identities: list[SymbolIdentity]) -> dict[str, dict]:
    """Match identifiers only; this does not assert electrical connectivity equivalence."""
    result: dict[str, dict[str, object]] = {}
    for ref in component_refs:
        matches = [item for item in identities if item.component_ref == ref]
        unique = {(item.symbol_uuid, item.sheet_path, item.unit) for item in matches}
        if matches and len(unique) == len(matches):
            instances = [asdict(item) | {"status": "VERIFIED_REFERENCE"} for item in matches]
            result[ref] = {"component_ref": ref, "status": "VERIFIED_REFERENCE",
                           "symbol_instances": instances}
        else:
            result[ref] = {"component_ref": ref, "status": "UNVERIFIED_MAPPING",
                           "match_count": len(matches)}
    return result
