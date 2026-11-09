#!/usr/bin/env python3
"""Create wire-removed KiCad inputs and component-connectivity targets.

The source ``.kicad_sch`` is read-only. Outputs go to a separate dataset
directory; no source design is edited in place.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
LABELS = (
    "IC", "LED", "battery", "buzzer", "capacitor", "connector", "diode",
    "display", "emi_filter", "ferrite_bead", "fuse", "inductor", "jumper",
    "oscillator", "potentiometer", "relay", "resistor", "switch", "transistor",
)
REMOVED_ROOT_NODES = {
    "wire", "bus", "bus_entry", "bus_alias", "junction", "no_connect", "label",
    "global_label", "hierarchical_label",
}
TOKEN_RE = re.compile(r'"(?:\\.|[^"\\])*"|[()]|[^\s()]+')


def decode_kicad_string(token: str) -> str:
    """Decode KiCad's C-style quoted strings without requiring strict JSON."""
    body = token[1:-1]
    output: list[str] = []
    index = 0
    escapes = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}
    while index < len(body):
        char = body[index]
        if char == "\\" and index + 1 < len(body):
            index += 1
            output.append(escapes.get(body[index], body[index]))
        else:
            output.append(char)
        index += 1
    return "".join(output)


def parse_expr(source: str) -> Any:
    """Parse one KiCad S-expression, decoding quoted strings with JSON rules."""
    tokens = TOKEN_RE.findall(source)
    pos = 0

    def parse() -> Any:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Unexpected end of S-expression")
        token = tokens[pos]
        pos += 1
        if token != "(":
            if token == ")":
                raise ValueError("Unexpected closing parenthesis")
            return decode_kicad_string(token) if token.startswith('"') else token
        result = []
        while pos < len(tokens) and tokens[pos] != ")":
            result.append(parse())
        if pos >= len(tokens):
            raise ValueError("Unclosed S-expression")
        pos += 1
        return result

    value = parse()
    if pos != len(tokens):
        raise ValueError("Unexpected trailing S-expression content")
    return value


def root_children(source: str) -> Iterator[tuple[str, int, int]]:
    """Yield root-level expression head and source offsets without rewriting."""
    start = source.find("(")
    if start < 0:
        raise ValueError("No KiCad S-expression root found")
    depth = 0
    in_string = False
    escaped = False
    item_start: int | None = None
    for index in range(start, len(source)):
        char = source[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "(":
            if depth == 1:
                item_start = index
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced closing parenthesis")
            if depth == 1 and item_start is not None:
                raw = source[item_start:index + 1]
                head = re.match(r"\(\s*([^\s()]+)", raw)
                if head:
                    yield head.group(1), item_start, index + 1
                item_start = None
            if depth == 0:
                return
    raise ValueError("Unclosed KiCad root expression")


def child(node: Any, name: str) -> list[Any] | None:
    if not isinstance(node, list):
        return None
    return next((part for part in node if isinstance(part, list) and part and part[0] == name), None)


def children(node: Any, name: str) -> list[list[Any]]:
    if not isinstance(node, list):
        return []
    return [part for part in node if isinstance(part, list) and part and part[0] == name]


def props(node: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in children(node, "property"):
        if len(item) >= 3 and isinstance(item[1], str) and isinstance(item[2], str):
            result[item[1]] = item[2]
    return result


def classify(ref: str, lib_id: str, value: str = "", description: str = "") -> str | None:
    ref_prefix = re.match(r"[A-Za-z]+", ref)
    prefix = ref_prefix.group(0).upper() if ref_prefix else ""
    key = f"{lib_id} {value} {description}".casefold().replace("_", " ")
    part = lib_id.rsplit(":", 1)[-1].casefold().replace("_", " ")
    text = f"{key} {part}"

    # Specific parts take precedence over broad reference designators.
    if any(term in text for term in ("ferrite bead", "ferritebead")) or prefix == "FB":
        return "ferrite_bead"
    if any(term in text for term in ("emi filter", "emc filter", "common mode choke")) or prefix == "FL":
        return "emi_filter"
    if any(term in text for term in ("potentiometer", "trimmer", "variable resistor")) or prefix in {"RV", "VR"}:
        return "potentiometer"
    if any(term in text for term in ("oscillator", "crystal", "resonator")) or prefix in {"Y", "X"}:
        return "oscillator"
    if any(term in text for term in ("display", "seven segment", "7 segment", "lcd", "oled")) or prefix in {"DS", "DISP"}:
        return "display"
    if any(term in text for term in ("buzzer", "piezo", "sounder")) or prefix in {"BZ", "BZR"}:
        return "buzzer"
    if any(term in text for term in ("battery", "cell holder", "coin cell")) or prefix in {"BT", "BAT"}:
        return "battery"
    if "relay" in text or prefix in {"K", "RY"}:
        return "relay"
    if any(term in text for term in ("fuse", "polyfuse", "resettable fuse")) or prefix in {"F", "FU"}:
        return "fuse"
    if "jumper" in text or prefix in {"JP", "JMP"}:
        return "jumper"
    if any(term in text for term in ("switch", "pushbutton", "push button", "tactile")) or prefix in {"SW", "S"}:
        return "switch"
    if any(term in text for term in ("transistor", "mosfet", "mosfet", "bjt", "jfet", "igbt")) or prefix == "Q":
        return "transistor"
    if any(term in text for term in ("led", "light emitting diode")) or prefix == "LED":
        return "LED"
    if "diode" in text or prefix == "D":
        return "diode"
    if any(term in text for term in ("capacitor", "device:c", "device:cp")) or prefix == "C":
        return "capacitor"
    if any(term in text for term in ("inductor", "choke")) or prefix == "L":
        return "inductor"
    if any(term in text for term in ("connector", "conn_", "usb", "header")) or prefix in {"J", "P", "CN"}:
        return "connector"
    if any(term in text for term in ("integrated circuit", "microcontroller", "amplifier", "logic gate", "processor", "memory")) or prefix in {"U", "IC"}:
        return "IC"
    if "resistor" in text or prefix == "R":
        return "resistor"
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wireless_copy(source: str) -> tuple[str, list[dict[str, str]]]:
    spans = list(root_children(source))
    remove_spans = [(start, end) for head, start, end in spans if head in REMOVED_ROOT_NODES]
    output = []
    cursor = 0
    for start, end in remove_spans:
        output.append(source[cursor:start])
        cursor = end
    output.append(source[cursor:])
    root = parse_expr(source)
    root_libs = child(root, "lib_symbols")
    lib_description: dict[str, str] = {}
    if root_libs:
        for symbol in children(root_libs, "symbol"):
            if len(symbol) > 1 and isinstance(symbol[1], str):
                lib_description[symbol[1]] = props(symbol).get("Description", "")
    instances: list[dict[str, str]] = []
    for symbol in children(root, "symbol"):
        lib_node = child(symbol, "lib_id")
        lib_id = str(lib_node[1]) if lib_node and len(lib_node) > 1 else ""
        properties = props(symbol)
        ref = properties.get("Reference", "").strip()
        if not ref or ref.startswith("#"):
            continue
        value = properties.get("Value", "")
        label = classify(ref, lib_id, value, lib_description.get(lib_id, ""))
        instances.append({"ref": ref, "lib_id": lib_id, "value": value, "class": label or ""})
    return "".join(output), instances


def export_netlist(cli: str, schematic: Path, output: Path) -> None:
    subprocess.run([cli, "sch", "export", "netlist", "--format", "kicadxml", "-o", str(output), str(schematic)],
                   check=True, capture_output=True, text=True)


def build_target_from_components_nets(components_raw: list[dict[str, Any]], nets_raw: list[dict[str, Any]],
                                      instances: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_ref = {item["ref"]: item for item in instances}
    exported: dict[str, dict[str, str]] = {}
    for comp in components_raw:
        ref = str(comp.get("ref", "")).strip()
        if not ref:
            continue
        lib_id = str(comp.get("lib_id", ""))
        record = by_ref.get(ref, {})
        value = str(comp.get("value", record.get("value", "")))
        label = record.get("class") or classify(ref, lib_id, value)
        exported[ref] = {"ref": ref, "value": value, "lib_id": lib_id, "class": label or ""}
    supported = {ref: item for ref, item in exported.items() if item["class"] in LABELS}
    if len(supported) < 2:
        raise ValueError("fewer than two components map to the 19-class taxonomy")
    nets = []
    for net in nets_raw:
        nodes = [{"ref": str(n.get("ref", "")).strip(), "pin": str(n.get("pin", "")).strip()}
                 for n in net.get("nodes", [])]
        known_nodes = [n for n in nodes if n["ref"] in supported]
        if len({n["ref"] for n in known_nodes}) >= 2:
            nets.append({"name": str(net.get("name", "")), "nodes": known_nodes})
    if not nets:
        raise ValueError("no net connects at least two taxonomy-labelled components")
    target = {"schema_version": "circuitcare-connectivity-target-v1",
              "components": sorted(supported.values(), key=lambda item: item["ref"]),
              "nets": nets}
    original = {"schema_version": "circuitcare-original-netlist-v1",
                "components": sorted(exported.values(), key=lambda item: item["ref"]),
                "nets": [{"name": str(net.get("name", "")), "nodes": [
                    {"ref": str(n.get("ref", "")), "pin": str(n.get("pin", ""))}
                    for n in net.get("nodes", [])
                ]} for net in nets_raw]}
    return target, original


def load_target_map(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("netlist map must be a JSON object keyed by source-relative schematic path")
    return raw


def normalize_netlist_json(raw: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise ValueError("netlist JSON must be an object")
    components = raw.get("components")
    nets = raw.get("nets")
    if not isinstance(components, list) or not isinstance(nets, list):
        raise ValueError("netlist JSON must contain components and nets arrays")
    return components, nets


def split_for_group(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory containing source .kicad_sch files")
    parser.add_argument("--output", type=Path, default=ROOT / "data/external/connectivity_pairs")
    parser.add_argument("--kicad-cli", default="kicad-cli", help="KiCad CLI executable")
    parser.add_argument("--netlist-map", type=Path,
                        help="Optional JSON map from source-relative .kicad_sch path to netlist_json")
    parser.add_argument("--target-format", choices=("netlist-json", "original-schematic"),
                        default="netlist-json",
                        help="Use exported connectivity JSON, or retain the original wired schematic as target")
    parser.add_argument("--limit", type=int, default=0, help="Maximum source files; 0 means all")
    parser.add_argument("--source-name", default="local-kicad-schematic")
    parser.add_argument("--license", default="UNSPECIFIED", help="Source dataset license identifier")
    parser.add_argument("--source-url", default="", help="Dataset or repository URL")
    parser.add_argument("--target-source-name", default="")
    parser.add_argument("--target-license", default="")
    parser.add_argument("--target-source-url", default="")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        parser.error(f"source directory does not exist: {source}")
    if source == output or source in output.parents:
        parser.error("output must not be inside the source tree")
    files = sorted(source.rglob("*.kicad_sch"))
    if args.limit > 0:
        files = files[:args.limit]
    if not files:
        parser.error(f"no .kicad_sch files found under {source}")
    for folder in ("inputs", "targets", "originals"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    manifest_path = output / "records.jsonl"
    counts = {"processed": 0, "skipped": 0, "errors": 0}
    skip_reasons: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    try:
        target_map = load_target_map(args.netlist_map.expanduser().resolve() if args.netlist_map else None)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(f"cannot read --netlist-map: {exc}")
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, path in enumerate(files, start=1):
            relative = path.relative_to(source).as_posix()
            sample_id = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
            try:
                raw = path.read_text(encoding="utf-8")
                cleaned, instances = wireless_copy(raw)
                target = original = None
                if args.target_format == "original-schematic":
                    labelled = [item for item in instances if item["class"] in LABELS]
                    if len(labelled) < 2:
                        raise ValueError("fewer than two components map to the 19-class taxonomy")
                    target_rel = f"targets/{sample_id}.kicad_sch"
                    (output / target_rel).write_text(raw, encoding="utf-8")
                else:
                    mapped = target_map.get(relative)
                    if mapped is not None:
                        components_raw, nets_raw = normalize_netlist_json(mapped)
                    else:
                        temp_xml = output / f".{sample_id}.netlist.xml"
                        try:
                            export_netlist(args.kicad_cli, path, temp_xml)
                            xml_root = ET.parse(temp_xml).getroot()
                            components_raw = []
                            for comp in xml_root.findall("./components/comp"):
                                lib = comp.find("libsource")
                                if lib is None:
                                    continue
                                components_raw.append({
                                    "ref": comp.get("ref", ""),
                                    "value": comp.findtext("value") or "",
                                    "lib_id": f"{lib.get('lib', '')}:{lib.get('part', '')}".strip(":"),
                                })
                            nets_raw = [{"name": net.get("name", ""), "nodes": [
                                {"ref": n.get("ref", ""), "pin": n.get("pin", "")}
                                for n in net.findall("node")
                            ]} for net in xml_root.findall("./nets/net")]
                        finally:
                            temp_xml.unlink(missing_ok=True)
                    target, original = build_target_from_components_nets(components_raw, nets_raw, instances)
                    target_rel = f"targets/{sample_id}.json"
                    (output / target_rel).write_text(
                        json.dumps(target, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    (output / "originals" / f"{sample_id}.netlist.json").write_text(
                        json.dumps(original, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                temp_input = output / "inputs" / f"{sample_id}.kicad_sch"
                temp_input.write_text(cleaned, encoding="utf-8")
                # Preserve the directory tree as the project group. Flat
                # sources fall back to the filename stem and mark that weaker
                # grouping explicitly in provenance.
                relative_parts = path.relative_to(source).parts
                group_id = "/".join(relative_parts[:-1]) if len(relative_parts) > 1 else path.stem
                classes = {item["ref"]: item["class"] for item in instances if item["class"]}
                for label in classes.values():
                    class_counts[label] = class_counts.get(label, 0) + 1
                record = {
                    "sample_id": sample_id, "source_path": relative,
                    "source_sha256": sha256(path), "input_path": f"inputs/{sample_id}.kicad_sch",
                    "target_path": target_rel,
                    "target_sha256": sha256(output / target_rel),
                    "target_format": args.target_format,
                    "classes": classes, "group_id": group_id,
                    "split_group_quality": "DIRECTORY_GROUP" if len(relative_parts) > 1 else "FILENAME_FALLBACK",
                    "split": split_for_group(group_id), "source_name": args.source_name,
                    "source_url": args.source_url, "license": args.license,
                    "target_source_name": args.target_source_name or args.source_name,
                    "target_source_url": args.target_source_url or args.source_url,
                    "target_license": args.target_license or args.license,
                    "removed_root_nodes": sorted(REMOVED_ROOT_NODES),
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["processed"] += 1
            except (OSError, ValueError, ET.ParseError, subprocess.SubprocessError) as exc:
                counts["skipped"] += 1
                if isinstance(exc, subprocess.CalledProcessError):
                    detail = (exc.stderr or str(exc)).strip()
                else:
                    detail = str(exc)
                reason = f"{type(exc).__name__}: {detail[:180]}"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                if counts["skipped"] <= 10:
                    print(f"SKIP {relative}: {detail}", file=sys.stderr)
            if index % 100 == 0:
                print(f"Progress {index}/{len(files)}; accepted={counts['processed']}, skipped={counts['skipped']}")
    (output / "summary.json").write_text(json.dumps({
        "schema_version": "circuitcare-connectivity-dataset-v1", "taxonomy": list(LABELS),
        "source_root": str(source), "source_name": args.source_name, "source_url": args.source_url,
        "license": args.license, "class_counts": class_counts, "skip_reasons": skip_reasons,
        "target_format": args.target_format, "target_source_name": args.target_source_name or args.source_name,
        "target_source_url": args.target_source_url or args.source_url,
        "target_license": args.target_license or args.license, **counts,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False))
    return 0 if counts["processed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
