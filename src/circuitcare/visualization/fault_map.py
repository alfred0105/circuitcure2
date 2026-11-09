"""Presentation data and DOT styling for the fault influence candidate map."""

from __future__ import annotations

import json
from typing import Any

from circuitcare.graph import build_graph, find_paths_result
from circuitcare.models import Circuit

RISK_COLORS = {
    "KNOWN_FAILED": "#9B59B6", "EXCEEDED": "#E74C3C", "NEAR_LIMIT": "#F39C12",
    "WITHIN_LIMIT": "#5DADE2", "UNKNOWN": "#AAB2B8",
}
PRIORITY_COLORS = ["#E8EDF2", "#D5E8F6", "#9CC7E8", "#5A9BD4", "#21618C"]


def build_fault_map(circuit: Circuit, failed_component: str, *, risk_assessment: dict | None = None,
                    inspection_ranking: dict | None = None, selected_component: str | None = None,
                    view_mode: str = "Electrical Stress View", max_hops: int = 3) -> dict[str, Any]:
    """Build a display/export payload without inventing risk or priority evidence."""
    if failed_component not in circuit.components:
        raise ValueError(f"Unknown component reference: {failed_component}")
    graph = build_graph(circuit)
    search = find_paths_result(circuit, failed_component, max_hops=max_hops)
    paths = search.paths
    risk_rows = {row["component_ref"]: row for row in (risk_assessment or {}).get("secondary_candidates", [])}
    priority_rows = {row["component_ref"]: row for row in (inspection_ranking or {}).get("posterior_weighted_inspection_order", [])}
    prior_priority_rows = {row["component_ref"]: row for row in (inspection_ranking or {}).get("prior_weighted_inspection_order", [])}
    minimum_path: dict[str, dict[str, Any]] = {}
    for path in paths:
        existing = minimum_path.get(path.target_ref)
        if existing is None or path.hops < existing["hops"]:
            minimum_path[path.target_ref] = {"connection_type": path.connection_type, "hops": path.hops,
                                             "nets": list(path.nets), "evidence_level": "connectivity_only"}
    node_data = []
    for ref, component in sorted(circuit.components.items()):
        stress = risk_rows.get(ref, {})
        priority = priority_rows.get(ref, {})
        prior_priority = prior_priority_rows.get(ref, {})
        node_data.append({
            "component_ref": ref, "value": component.value,
            "connectivity": ({"status": "failed_source", "hops": 0, "evidence_level": "user_selected_fault"}
                             if ref == failed_component else minimum_path.get(ref, {"status": "not_in_candidate_path", "hops": None,
                                                                                    "evidence_level": "connectivity_only"})),
            "risk_level": "KNOWN_FAILED" if ref == failed_component else stress.get("risk_level", "UNKNOWN"),
            "risk_evidence_level": stress.get("evidence_level"),
            "maximum_utilization_ratio": stress.get("maximum_utilization_ratio"),
            "electrical_priority_rank": stress.get("priority_rank"),
            "electrical_priority_score_not_probability": stress.get("priority_score"),
            "inspection_priority_rank": priority.get("priority_rank"),
            "inspection_priority_score_not_probability": priority.get("posterior_weighted_priority_score"),
            "prior_inspection_rank": prior_priority.get("priority_rank"),
            "prior_inspection_score_not_probability": prior_priority.get("posterior_weighted_priority_score"),
            "relevant_hypotheses": priority.get("scenario_evidence", []),
            "electrical_reason": stress.get("reason"),
            "damage_confirmed": False,
            "selected": ref == selected_component,
        })
    payload = {"schema_version": "1.0", "map_type": "fault_influence_candidate_map",
               "view_mode": view_mode, "failed_component": failed_component,
               "selected_component": selected_component, "nodes": node_data,
               "nets": [{"name": info["name"], "net_type": info["net_type"]}
                        for info in graph["nodes"].values() if info["type"] == "net"],
               "edges": [{"component_ref": edge["component"].split(":", 1)[1],
                          "net": edge["net"].split(":", 1)[1], "pins": list(edge["pins"])}
                         for edge in graph["edges"]],
               "paths": [{"target_ref": item.target_ref, "connection_type": item.connection_type,
                          "hops": item.hops, "nets": list(item.nets),
                          "path_nodes": [part.split(":", 1)[1] for part in item.path],
                          "path_sequence": list(item.path),
                          "evidence_level": "connectivity_only"}
                         for item in paths],
               "path_search": {"status": search.status, "expanded_states": search.expanded_states,
                               "peak_queue_size": search.peak_queue_size, "total_paths": search.total_paths,
                               "limit_reason": search.limit_reason},
               "damage_confirmed": False,
               "interpretation": "Connectivity paths are influence candidates, not directed power flow or proof of stress/damage. Electrical risk uses only the provided scenario-matched SPICE/rating result. Inspection priority is a separate conditional index, not damage probability."}
    return payload


def render_fault_map_dot(payload: dict[str, Any]) -> str:
    """Render the bipartite map with separate risk/priority color modes."""
    selected = payload.get("selected_component")
    failed = payload["failed_component"]
    nodes = {row["component_ref"]: row for row in payload["nodes"]}
    included_refs = {failed, selected} | {row["target_ref"] for row in payload["paths"]}
    highlighted_nets = {net for row in payload["paths"]
                        if row["target_ref"] == selected for net in row["nets"]}
    highlighted_edges = set()
    for route in payload["paths"]:
        if route["target_ref"] != selected:
            continue
        for left, right in zip(route["path_sequence"], route["path_sequence"][1:]):
            if left.startswith("component:") and right.startswith("net:"):
                highlighted_edges.add((left.split(":", 1)[1], right.split(":", 1)[1]))
            elif left.startswith("net:") and right.startswith("component:"):
                highlighted_edges.add((right.split(":", 1)[1], left.split(":", 1)[1]))
    lines = ["graph FaultInfluence {", '  graph [label="Connectivity candidate map · not a circuit schematic", labelloc="t"];',
             '  node [fontname="Helvetica", style=filled];', '  edge [fontname="Helvetica", fontsize=9];']
    for ref, row in sorted(nodes.items()):
        if payload["view_mode"] == "Inspection Priority View":
            rank = row.get("inspection_priority_rank")
            score = row.get("inspection_priority_score_not_probability")
            bucket = min(4, max(0, int(float(score) // 20))) if score is not None else None
            color = "#9B59B6" if ref == failed else (PRIORITY_COLORS[bucket] if bucket is not None else RISK_COLORS["UNKNOWN"])
        elif payload["view_mode"] == "Connection View":
            color = "#9B59B6" if ref == failed else ("#7FB3D5" if row["connectivity"].get("hops") is not None else "#E5E7E9")
        else:
            color = RISK_COLORS.get(row["risk_level"], RISK_COLORS["UNKNOWN"])
        rank = row.get("inspection_priority_rank") if payload["view_mode"] == "Inspection Priority View" else row.get("electrical_priority_rank")
        label = ref + (f"\\n#{rank}" if rank else "")
        extra = ', penwidth=3, color="#1B4F72"' if ref == selected else ""
        lines.append(f"  {_q('c:' + ref)} [label={_q(label)}, shape=box, fillcolor={_q(color)}{extra}];")
    for row in payload["nets"]:
        fill = "#D6EAF8" if row["name"] in highlighted_nets else "#F2F3F4"
        net_label = row["name"] + "\\n(" + row["net_type"] + ")"
        lines.append(f"  {_q('n:' + row['name'])} [label={_q(net_label)}, shape=diamond, fillcolor={_q(fill)}];")
    for edge in payload["edges"]:
        attrs = [f'label={_q("pins " + ", ".join(edge["pins"]))}']
        if (edge["component_ref"], edge["net"]) in highlighted_edges:
            attrs.extend(['color="#D35400"', "penwidth=3"])
        elif edge["component_ref"] not in included_refs:
            attrs.extend(['color="#D5D8DC"', 'fontcolor="#AAB2B8"'])
        lines.append(f"  {_q('c:' + edge['component_ref'])} -- {_q('n:' + edge['net'])} [{', '.join(attrs)}];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def fault_map_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _q(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
