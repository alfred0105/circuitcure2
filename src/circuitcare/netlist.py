"""Parse an exported KiCad XML schematic netlist.

This module deliberately does not assume that shared nets imply a fault
or electrical damage.
"""

from pathlib import Path
from xml.etree import ElementTree as ET

from .models import Circuit, Component, Connection


def _unique_net_name(base: str, reserved: set[str]) -> str:
    name = base
    suffix = 2
    while name in reserved:
        name = f"{base}_{suffix}"
        suffix += 1
    reserved.add(name)
    return name


def _pin_sort_key(pin: str) -> tuple[int, int | str]:
    try:
        return (0, int(pin))
    except ValueError:
        return (1, pin)


def load_netlist(path: str | Path) -> Circuit:
    """Read components and explicit pin-to-net membership from KiCad XML."""
    root = ET.parse(path).getroot()
    if root.tag != "export":
        raise ValueError("Expected KiCad XML netlist root <export>")

    components: dict[str, Component] = {}
    for item in root.findall("./components/comp"):
        ref = (item.get("ref") or "").strip()
        if not ref:
            raise ValueError("Component missing ref")
        if ref in components:
            raise ValueError(f"Duplicate component ref: {ref}")
        library = item.find("libsource")
        components[ref] = Component(
            ref=ref,
            value=(item.findtext("value") or "").strip(),
            library_id=(
                f"{library.get('lib', '')}:{library.get('part', '')}"
                if library is not None else ""
            ),
            footprint=(item.findtext("footprint") or "").strip() or None,
            datasheet=(item.findtext("datasheet") or "").strip() or None,
        )

    components = dict(sorted(components.items()))
    connections: list[Connection] = []
    nets = root.findall("./nets/net")
    reserved_names = {(net.get("name") or "").strip() for net in nets}
    named_net_codes: dict[str, str | None] = {}
    pin_nets: dict[tuple[str, str], str] = {}
    unnamed_index = 0

    for net in nets:
        name = (net.get("name") or "").strip()
        code = (net.get("code") or "").strip() or None
        if name:
            if name in named_net_codes:
                previous_code = named_net_codes[name]
                if previous_code != code:
                    raise ValueError(f"Duplicate net name {name!r} has conflicting codes")
            else:
                named_net_codes[name] = code
        elif code:
            name = _unique_net_name(f"__code_{code}", reserved_names)
        else:
            unnamed_index += 1
            name = _unique_net_name(f"__unnamed_{unnamed_index}", reserved_names)

        for node in net.findall("node"):
            ref = (node.get("ref") or "").strip()
            pin = (node.get("pin") or "").strip()
            if not ref or not pin:
                raise ValueError(f"Malformed pin entry on net {name}")
            if ref not in components:
                raise ValueError(f"Net {name} refers to unknown component {ref}")
            pin_id = (ref, pin)
            previous_net = pin_nets.get(pin_id)
            if previous_net is not None:
                if previous_net != name:
                    raise ValueError(
                        f"Conflicting net connections for {ref} pin {pin}: "
                        f"{previous_net!r} and {name!r}"
                    )
                continue
            pin_nets[pin_id] = name
            connections.append(Connection(ref=ref, pin=pin, net=name, net_code=code))

    if not components:
        raise ValueError("No components found in netlist")
    if not connections:
        raise ValueError("No pin-to-net connections found in netlist")
    connections.sort(key=lambda item: (item.ref, _pin_sort_key(item.pin), item.net, item.net_code or ""))
    return Circuit(components=components, connections=connections)
