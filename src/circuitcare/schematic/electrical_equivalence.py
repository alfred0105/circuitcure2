"""Conservative electrical-equivalence gate for KiCad schematics and XML netlists."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from circuitcare.schematic.exporter import find_kicad_cli
from circuitcare.schematic.identity import MAX_SCHEMATIC_BYTES, discover_schematic_files

MAX_NETLIST_BYTES = 20_000_000
VERIFIER_VERSION = "circuitcare-net-equivalence-v1"


@dataclass(frozen=True)
class ElectricalEquivalenceResult:
    status: str
    message: str
    cache_key: str | None = None
    unknown_metadata: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Netlist:
    components: dict[str, tuple[str, str | None, frozenset[str] | None]]
    net_names_by_pins: dict[frozenset[tuple[str, str]], str]


def _normal_net_name(name: str) -> str:
    """Drop only KiCad XML's root separator; retain hierarchical name segments."""
    name = name.strip()
    return name.removeprefix("/")


def _read_netlist(path: Path) -> _Netlist:
    if not path.is_file() or path.stat().st_size > MAX_NETLIST_BYTES:
        raise ValueError("XML netlist is missing or exceeds the 20 MB limit")
    root = ET.parse(path).getroot()
    if root.tag != "export":
        raise ValueError("Expected KiCad XML netlist root <export>")
    components: dict[str, tuple[str, str | None, frozenset[str] | None]] = {}
    for component in root.findall("./components/comp"):
        ref = (component.get("ref") or "").strip()
        if not ref or ref in components:
            raise ValueError("Missing or duplicate component Reference in XML netlist")
        lib = component.find("libsource")
        library_id = None
        if lib is not None:
            library_id = f"{lib.get('lib', '')}:{lib.get('part', '')}"
        component_pins = frozenset((pin.get("num") or "").strip()
                                    for pin in component.findall("./units/unit/pins/pin")
                                    if (pin.get("num") or "").strip())
        components[ref] = ((component.findtext("value") or "").strip(), library_id,
                            component_pins or None)
    if not components:
        raise ValueError("XML netlist contains no components")

    pin_to_group: dict[tuple[str, str], frozenset[tuple[str, str]]] = {}
    names_by_pins: dict[frozenset[tuple[str, str]], str] = {}
    pin_groups_by_name: dict[str, frozenset[tuple[str, str]]] = {}
    for net in root.findall("./nets/net"):
        raw_name = (net.get("name") or "").strip()
        if not raw_name:
            raise ValueError("Net has no stable name; electrical equivalence is unverified")
        name = _normal_net_name(raw_name)
        pins: list[tuple[str, str]] = []
        for node in net.findall("node"):
            ref = (node.get("ref") or "").strip()
            pin = (node.get("pin") or "").strip()
            if not ref or not pin or ref not in components:
                raise ValueError(f"Malformed or unknown pin on net {raw_name!r}")
            pins.append((ref, pin))
        if not pins:
            continue
        pin_group = frozenset(pins)
        if len(pin_group) != len(pins):
            raise ValueError(f"Duplicate pin entry on net {raw_name!r}")
        prior_group = pin_groups_by_name.get(name)
        if prior_group is not None and prior_group != pin_group:
            raise ValueError(f"Net identity {name!r} is assigned to multiple pin groups")
        pin_groups_by_name[name] = pin_group
        if pin_group in names_by_pins:
            raise ValueError("Multiple net names describe the same pin group")
        for pin_id in pin_group:
            if pin_id in pin_to_group:
                raise ValueError(f"Pin {pin_id[0]} pin {pin_id[1]} belongs to multiple nets")
            pin_to_group[pin_id] = pin_group
        names_by_pins[pin_group] = name
    if not names_by_pins:
        raise ValueError("XML netlist contains no pin-to-net connections")
    return _Netlist(components, names_by_pins)


def _compare(expected: _Netlist, actual: _Netlist) -> tuple[str, str, tuple[str, ...]]:
    if set(expected.components) != set(actual.components):
        return "MISMATCH", "Component Reference sets differ", ()
    for ref in sorted(expected.components):
        expected_value, expected_library, expected_pins = expected.components[ref]
        actual_value, actual_library, actual_pins = actual.components[ref]
        if expected_value != actual_value:
            return "MISMATCH", f"Component value differs for {ref}", ()
        # Library metadata is useful when present, but older supplied XML may omit it.
        if expected_library and actual_library and expected_library != actual_library:
            return "MISMATCH", f"Library ID differs for {ref}", ()
        if expected_pins is not None and actual_pins is not None and expected_pins != actual_pins:
            return "MISMATCH", f"Pin-number sets differ for {ref}", ()
    if set(expected.net_names_by_pins) != set(actual.net_names_by_pins):
        return "MISMATCH", "Pin-to-net connectivity differs (split, merge, or changed pin membership)", ()
    for pin_group, expected_name in expected.net_names_by_pins.items():
        actual_name = actual.net_names_by_pins[pin_group]
        if expected_name != actual_name:
            return "MISMATCH", f"Net identity differs for connected pin group: {expected_name!r} vs {actual_name!r}", ()
    unknown_items = []
    for ref, (_, library_id, component_pins) in expected.components.items():
        if library_id is None:
            unknown_items.append(f"{ref}: Library ID missing from supplied XML netlist")
        if component_pins is None:
            unknown_items.append(f"{ref}: Pin-number set missing from supplied XML netlist")
    unknown = tuple(unknown_items)
    message = "Component values, named nets, and pin-to-net groups match"
    if unknown:
        message += "; library metadata is UNKNOWN"
    return "VERIFIED_EQUIVALENT", message, unknown


def verify_schematic_netlist(schematic_path: str | Path, netlist_path: str | Path, *,
                             project_root: str | Path, cache_root: str | Path,
                             timeout: float = 30.0) -> ElectricalEquivalenceResult:
    """Export the schematic's KiCad XML netlist and compare it with the analysis netlist.

    Only validated verdicts are cached. CLI failures, timeouts, malformed exports, and
    incomplete projects remain NOT_VERIFIED/INCOMPLETE_PROJECT and are never cached.
    """
    try:
        root = Path(project_root).resolve(strict=True)
        schematic = Path(schematic_path).resolve(strict=True)
        supplied = Path(netlist_path).resolve(strict=True)
        cache = Path(cache_root).resolve()
        if not schematic.is_relative_to(root) or schematic.suffix != ".kicad_sch":
            return ElectricalEquivalenceResult("NOT_VERIFIED", "Schematic path is outside the project")
        if not supplied.is_relative_to(root) or supplied.suffix.lower() != ".xml":
            return ElectricalEquivalenceResult("NOT_VERIFIED", "Analysis netlist must be a project-contained XML file")
        if not cache.is_relative_to(root):
            return ElectricalEquivalenceResult("NOT_VERIFIED", "Equivalence cache must be inside the project")
        if supplied.stat().st_size > MAX_NETLIST_BYTES or schematic.stat().st_size > MAX_SCHEMATIC_BYTES:
            return ElectricalEquivalenceResult("NOT_VERIFIED", "Schematic or XML netlist exceeds the size limit")
        files, incomplete = discover_schematic_files(schematic, project_root=root)
    except (OSError, ValueError) as exc:
        return ElectricalEquivalenceResult("NOT_VERIFIED", str(exc))
    if incomplete:
        return ElectricalEquivalenceResult("INCOMPLETE_PROJECT", "; ".join(incomplete))

    cli = find_kicad_cli()
    if not cli:
        return ElectricalEquivalenceResult("NOT_VERIFIED", "KiCad CLI is unavailable")
    try:
        version_proc = subprocess.run([cli, "version"], capture_output=True, text=True,
                                      timeout=5, check=True, close_fds=False)
        cli_version = version_proc.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return ElectricalEquivalenceResult("NOT_VERIFIED", f"KiCad CLI version check failed: {exc}")

    digest = hashlib.sha256()
    for source in sorted(files):
        digest.update(source.relative_to(root).as_posix().encode())
        digest.update(source.read_bytes())
    digest.update(supplied.read_bytes())
    digest.update(f"{cli_version}|{VERIFIER_VERSION}".encode())
    cache_key = digest.hexdigest()
    try:
        cache.mkdir(parents=True, exist_ok=True)
        if not cache.is_relative_to(root):
            return ElectricalEquivalenceResult("NOT_VERIFIED", "Equivalence cache escaped the project root")
        cache_file = cache / f"{cache_key}.json"
        if cache_file.is_file() and cache_file.stat().st_size <= 64_000:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("cache_key") == cache_key and cached.get("status") in {"VERIFIED_EQUIVALENT", "MISMATCH"}:
                return ElectricalEquivalenceResult(cached["status"], cached["message"], cache_key,
                                                   tuple(cached.get("unknown_metadata", [])))
    except (OSError, ValueError, json.JSONDecodeError):
        pass

    try:
        expected = _read_netlist(supplied)
        with tempfile.TemporaryDirectory(prefix="circuitcare-netlist-") as temp_dir:
            exported = Path(temp_dir) / "schematic.xml"
            process = subprocess.run(
                [cli, "sch", "export", "netlist", "--format", "kicadxml", "--output",
                 str(exported), str(schematic)], cwd=str(root), capture_output=True, text=True,
                timeout=timeout, check=False, close_fds=False,
            )
            if process.returncode != 0:
                return ElectricalEquivalenceResult("NOT_VERIFIED", (process.stderr or process.stdout)[-2000:], cache_key)
            actual = _read_netlist(exported)
        status, message, unknown = _compare(expected, actual)
    except subprocess.TimeoutExpired:
        return ElectricalEquivalenceResult("NOT_VERIFIED", "KiCad netlist export timed out", cache_key)
    except (OSError, ET.ParseError, ValueError, json.JSONDecodeError) as exc:
        return ElectricalEquivalenceResult("NOT_VERIFIED", f"Netlist equivalence could not be established: {exc}", cache_key)

    result = ElectricalEquivalenceResult(status, message, cache_key, unknown)
    try:
        temp = cache / f".{cache_key}.{os.getpid()}.tmp"
        temp.write_text(json.dumps({"cache_key": cache_key, "status": status,
                                    "message": message, "unknown_metadata": unknown}, sort_keys=True),
                        encoding="utf-8")
        temp.replace(cache / f"{cache_key}.json")
    except OSError:
        pass
    return result
