"""Minimal typed domain objects for KiCad netlist ingestion."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Component:
    ref: str
    value: str
    library_id: str = ""
    footprint: str | None = None
    datasheet: str | None = None


@dataclass(frozen=True)
class Connection:
    ref: str
    pin: str
    net: str
    net_code: str | None = None


@dataclass
class Circuit:
    components: dict[str, Component]
    connections: list[Connection]

    @property
    def component_nets(self) -> dict[str, tuple[str, ...]]:
        """Return each component's connected nets in stable order."""
        nets: dict[str, set[str]] = {ref: set() for ref in self.components}
        for connection in self.connections:
            nets[connection.ref].add(connection.net)
        return {ref: tuple(sorted(names)) for ref, names in sorted(nets.items())}


@dataclass(frozen=True)
class CandidateImpact:
    target_ref: str
    shared_nets: tuple[str, ...]
    evidence_level: str = "connectivity_only"
