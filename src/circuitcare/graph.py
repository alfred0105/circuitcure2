"""Bipartite connectivity graph and connectivity-only path discovery."""

from collections import defaultdict, deque
from dataclasses import dataclass
from time import monotonic

from .models import CandidateImpact, Circuit

_NET_TYPES = {"ground", "power", "signal", "unknown"}


@dataclass(frozen=True)
class PathCandidate:
    target_ref: str
    connection_type: str
    hops: int
    path: tuple[str, ...]
    nets: tuple[str, ...]
    net_types: tuple[tuple[str, str], ...]
    ground_only_path: bool | None
    evidence_level: str = "connectivity_only"


@dataclass(frozen=True)
class PathSearchResult:
    paths: tuple[PathCandidate, ...]
    status: str
    expanded_states: int
    peak_queue_size: int
    total_paths: int
    limit_reason: str | None = None


def _component_node(reference: str) -> str:
    return f"component:{reference}"


def _net_node(name: str) -> str:
    return f"net:{name}"


def _normalize_net_types(net_types: dict[str, str] | None) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for name, kind in (net_types or {}).items():
        value = str(kind).strip().lower()
        if value not in _NET_TYPES:
            raise ValueError(f"Unsupported net type for {name!r}: {kind!r}")
        normalized[str(name)] = value
    return normalized


def build_graph(circuit: Circuit, net_types: dict[str, str] | None = None) -> dict:
    """Build a typed Component–Net bipartite graph using only netlist evidence."""
    type_by_net = _normalize_net_types(net_types)
    nodes: dict[str, dict] = {}
    edges_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)

    for reference in sorted(circuit.components):
        node = _component_node(reference)
        nodes[node] = {"type": "component", "reference": reference}

    for connection in circuit.connections:
        net = connection.net
        node = _net_node(net)
        nodes.setdefault(node, {
            "type": "net",
            "name": net,
            "net_type": type_by_net.get(net, "unknown"),
        })
        edges_by_pair[(_component_node(connection.ref), node)].add(connection.pin)

    edges = [
        {"component": component, "net": net, "pins": tuple(sorted(pins, key=_pin_sort_key))}
        for (component, net), pins in sorted(edges_by_pair.items())
    ]
    return {"nodes": nodes, "edges": edges}


def _pin_sort_key(pin: str) -> tuple[int, int | str]:
    try:
        return (0, int(pin))
    except ValueError:
        return (1, pin)


def direct_candidates(circuit: Circuit, failed_ref: str) -> list[CandidateImpact]:
    """Find components sharing one or more nets with a failed component."""
    if failed_ref not in circuit.components:
        raise ValueError(f"Unknown component reference: {failed_ref}")

    component_nets: dict[str, set[str]] = defaultdict(set)
    net_components: dict[str, set[str]] = defaultdict(set)
    for conn in circuit.connections:
        component_nets[conn.ref].add(conn.net)
        net_components[conn.net].add(conn.ref)

    shared: dict[str, set[str]] = defaultdict(set)
    for net in component_nets[failed_ref]:
        for neighbor in net_components[net]:
            if neighbor != failed_ref:
                shared[neighbor].add(net)

    return [
        CandidateImpact(target_ref=ref, shared_nets=tuple(sorted(nets)))
        for ref, nets in sorted(shared.items())
    ]


def find_paths_result(
    circuit: Circuit,
    failed_ref: str,
    max_hops: int = 2,
    *,
    net_types: dict[str, str] | None = None,
    exclude_net_types: set[str] | None = None,
    max_paths_per_candidate: int = 3,
    max_expanded_states: int = 10_000,
    max_queue_size: int = 5_000,
    max_total_paths: int = 5_000,
    time_budget_seconds: float | None = None,
) -> PathSearchResult:
    """Find deterministic simple paths, counting hops between components.

    A path is alternating component/net/component. Net type exclusions limit
    path traversal only; the graph and direct candidate data remain unchanged.
    """
    if failed_ref not in circuit.components:
        raise ValueError(f"Unknown component reference: {failed_ref}")
    if max_hops < 1:
        raise ValueError("max_hops must be at least 1")
    if max_paths_per_candidate < 1:
        raise ValueError("max_paths_per_candidate must be at least 1")
    if min(max_expanded_states, max_queue_size, max_total_paths) < 1:
        raise ValueError("Search budgets must be positive")
    if time_budget_seconds is not None and time_budget_seconds <= 0:
        raise ValueError("time_budget_seconds must be positive when provided")

    graph = build_graph(circuit, net_types)
    excluded = {kind.strip().lower() for kind in (exclude_net_types or set())}
    unsupported = excluded - _NET_TYPES
    if unsupported:
        raise ValueError(f"Unsupported excluded net type(s): {', '.join(sorted(unsupported))}")

    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in graph["edges"]:
        adjacency[edge["component"]].append(edge["net"])
        adjacency[edge["net"]].append(edge["component"])
    for neighbors in adjacency.values():
        neighbors.sort()

    start = _component_node(failed_ref)
    queue = deque([(start, (start,), frozenset({start}), 0)])
    paths: list[PathCandidate] = []
    path_counts: dict[str, int] = defaultdict(int)
    expanded = 0
    peak_queue = 1
    status = "COMPLETE"
    reason = None
    started = monotonic()
    depth_limited = False
    while queue:
        if expanded >= max_expanded_states:
            status, reason = "PARTIAL_RESOURCE_LIMIT", "max_expanded_states"
            break
        if time_budget_seconds is not None and monotonic() - started >= time_budget_seconds:
            status, reason = "PARTIAL_RESOURCE_LIMIT", "time_budget_seconds"
            break
        current, path, visited_components, hops = queue.popleft()
        expanded += 1
        if hops >= max_hops:
            if any(graph["nodes"].get(net, {}).get("net_type") not in excluded for net in adjacency[current]):
                depth_limited = True
            continue
        for net in adjacency[current]:
            net_info = graph["nodes"][net]
            if net in path or net_info["net_type"] in excluded:
                continue
            net_path = (*path, net)
            for component in adjacency[net]:
                if component == net or component in visited_components:
                    continue
                reference = graph["nodes"][component]["reference"]
                next_hops = hops + 1
                component_path = (*net_path, component)
                if reference != failed_ref and path_counts[reference] < max_paths_per_candidate:
                    if len(paths) >= max_total_paths:
                        status, reason = "PARTIAL_RESOURCE_LIMIT", "max_total_paths"
                        break
                    route_nets = tuple(graph["nodes"][item]["name"] for item in component_path if item.startswith("net:"))
                    route_types = tuple(
                        (graph["nodes"][item]["name"], graph["nodes"][item]["net_type"])
                        for item in component_path if item.startswith("net:")
                    )
                    is_ground_only = None if any(kind == "unknown" for _, kind in route_types) else all(
                        kind == "ground" for _, kind in route_types
                    )
                    paths.append(PathCandidate(
                        target_ref=reference,
                        connection_type="direct" if next_hops == 1 else "indirect",
                        hops=next_hops,
                        path=component_path,
                        nets=route_nets,
                        net_types=route_types,
                        ground_only_path=is_ground_only,
                    ))
                    path_counts[reference] += 1
                if next_hops < max_hops:
                    if len(queue) >= max_queue_size:
                        status, reason = "PARTIAL_RESOURCE_LIMIT", "max_queue_size"
                        break
                    queue.append((component, component_path, visited_components | {component}, next_hops))
                    peak_queue = max(peak_queue, len(queue))
                elif next_hops == max_hops and any(
                    graph["nodes"][candidate_net]["net_type"] not in excluded
                    for candidate_net in adjacency[component]
                ):
                    depth_limited = True
            if status != "COMPLETE":
                break
        if status != "COMPLETE":
            break

    if status == "COMPLETE" and queue:
        status, reason = "PARTIAL_RESOURCE_LIMIT", "unprocessed_queue"
    elif status == "COMPLETE" and depth_limited:
        status, reason = "PARTIAL_DEPTH_LIMIT", "max_hops"
    return PathSearchResult(tuple(sorted(paths, key=lambda item: (item.hops, item.target_ref, item.path))),
                            status, expanded, peak_queue, len(paths), reason)


def find_paths(
    circuit: Circuit,
    failed_ref: str,
    max_hops: int = 2,
    *,
    net_types: dict[str, str] | None = None,
    exclude_net_types: set[str] | None = None,
    max_paths_per_candidate: int = 3,
    max_expanded_states: int = 10_000,
    max_queue_size: int = 5_000,
    max_total_paths: int = 5_000,
    time_budget_seconds: float | None = None,
) -> list[PathCandidate]:
    """Backward-compatible path list API with bounded global search resources."""
    return list(find_paths_result(circuit, failed_ref, max_hops, net_types=net_types,
                                  exclude_net_types=exclude_net_types,
                                  max_paths_per_candidate=max_paths_per_candidate,
                                  max_expanded_states=max_expanded_states,
                                  max_queue_size=max_queue_size,
                                  max_total_paths=max_total_paths,
                                  time_budget_seconds=time_budget_seconds).paths)


def to_dot(graph: dict, failed_ref: str, paths: list[PathCandidate] | None = None) -> str:
    """Serialize a typed circuit graph as Graphviz DOT without external tools."""
    fault_node = _component_node(failed_ref)
    path_nodes = {node for item in (paths or []) for node in item.path}
    path_edges = {
        tuple(sorted((left, right)))
        for item in (paths or [])
        for left, right in zip(item.path, item.path[1:])
    }
    lines = [
        "graph CircuitCare {",
        '  graph [label="Connectivity only — not evidence of damage", labelloc="t"];',
        '  node [fontname="Helvetica"];',
        '  edge [fontname="Helvetica", fontsize=9];',
    ]

    for node, info in sorted(graph["nodes"].items()):
        if info["type"] == "component":
            label = info["reference"]
            shape = "box"
            fill = "#f4a3a3" if node == fault_node else ("#fff0bf" if node in path_nodes else "#eeeeee")
        else:
            kind = info["net_type"]
            label = f"{info['name']}\n({kind})"
            shape = "diamond"
            fill = "#cfe8ff" if node in path_nodes else "#eeeeee"
        lines.append(f"  {_dot_quote(node)} [label={_dot_quote(label)}, shape={shape}, style=filled, fillcolor={_dot_quote(fill)}];")

    for edge in graph["edges"]:
        component, net = edge["component"], edge["net"]
        attributes = [f'label={_dot_quote("pins " + ", ".join(edge["pins"]))}']
        if tuple(sorted((component, net))) in path_edges:
            attributes.extend(['color="#d27b00"', "penwidth=2"])
        lines.append(f"  {_dot_quote(component)} -- {_dot_quote(net)} [{', '.join(attributes)}];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _dot_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'
