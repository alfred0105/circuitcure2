#!/usr/bin/env python3
"""Create pin-only PCB inputs and isolated connection labels from acquired CAD."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import networkx as nx
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/pcb-photo-cad-2026-11-09"
DEST = SOURCE / "pin_only_training"


def dequote(value):
    if not isinstance(value, str):
        return value
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return bytes(value[1:-1], "utf-8").decode("unicode_escape")
    return value


def sexpr_tokens(text):
    token = re.compile(r'\s+|;[^\n]*|"(?:\\.|[^"\\])*"|[()]|[^\s();]+')
    for match in token.finditer(text):
        item = match.group(0)
        if item.isspace() or item.startswith(";"):
            continue
        yield item


def parse_sexpr(text):
    stack = []
    root = None
    for tok in sexpr_tokens(text):
        if tok == "(":
            node = []
            if stack:
                stack[-1].append(node)
            stack.append(node)
        elif tok == ")":
            if not stack:
                raise ValueError("unbalanced closing parenthesis in KiCad board")
            node = stack.pop()
            if not stack:
                if root is not None:
                    raise ValueError("multiple root expressions in KiCad board")
                root = node
        else:
            if not stack:
                raise ValueError("atom outside KiCad board root expression")
            stack[-1].append(tok)
    if stack or root is None:
        raise ValueError("incomplete KiCad board expression")
    return root


def atom(node):
    return node[0] if isinstance(node, list) and node else None


def children(node, key):
    return [x for x in node if isinstance(x, list) and atom(x) == key]


def first_child(node, key):
    return next((x for x in node if isinstance(x, list) and atom(x) == key), None)


def render_sexpr(node):
    if isinstance(node, list):
        return "(" + " ".join(render_sexpr(x) for x in node) + ")"
    return node


def reference_of_footprint(fp):
    for prop in children(fp, "property"):
        if len(prop) >= 3 and dequote(prop[1]) == "Reference":
            return dequote(prop[2])
    for item in children(fp, "fp_text"):
        if len(item) >= 3 and dequote(item[1]) == "reference":
            return dequote(item[2])
    return None


def xyrot(at_node):
    if not at_node or len(at_node) < 3:
        return 0.0, 0.0, 0.0
    try:
        return float(at_node[1]), float(at_node[2]), float(at_node[3]) if len(at_node) > 3 else 0.0
    except (TypeError, ValueError, IndexError):
        return 0.0, 0.0, 0.0


def rotate_xy(x, y, angle):
    theta = math.radians(angle)
    return x * math.cos(theta) - y * math.sin(theta), x * math.sin(theta) + y * math.cos(theta)


def classify_component(ref, value="", footprint=""):
    text = " ".join((ref or "", value or "", footprint or "")).lower()
    prefix = re.match(r"^([A-Z]+)(?:\$?\d|$)", (ref or "").upper())
    p = prefix.group(1) if prefix else ""
    if p in {"BAT", "BT"} or "battery" in text or "coin_cell" in text: return "battery"
    if p == "LED" or "led" in text: return "LED"
    if p in {"BZ", "BUZ", "SPK"} or "buzzer" in text: return "buzzer"
    if p == "FL" or "emi filter" in text or "emi_filter" in text: return "emi_filter"
    if p == "FB" or "ferrite" in text: return "ferrite_bead"
    if p in {"F", "FU"} or "fuse" in text: return "fuse"
    if p == "JP" or "jumper" in text: return "jumper"
    if p in {"RV", "POT"} or "potentiometer" in text or "trimmer" in text: return "potentiometer"
    if p in {"SW", "S"} or "switch" in text: return "switch"
    if p in {"K", "RL"} or "relay" in text: return "relay"
    if p == "Q" or "mosfet" in text or "transistor" in text: return "transistor"
    if p in {"DS", "DISP"} or "display" in text or "oled" in text or "lcd" in text: return "display"
    if p == "D" or "diode" in text: return "diode"
    if p == "L" or "inductor" in text: return "inductor"
    if p in {"Y", "X"} or "oscillator" in text or "crystal" in text: return "oscillator"
    if p == "R" or "resistor" in text: return "resistor"
    if p == "C" or "capacitor" in text: return "capacitor"
    if p in {"J", "P", "CN", "CONN"} or "connector" in text or "header" in text: return "connector"
    if p in {"U", "IC"} or "ic" in text or "integrated circuit" in text: return "IC"
    return "unknown"


def kicad_process(source: Path, output: Path):
    original = source.read_text(encoding="utf-8", errors="strict")
    root = parse_sexpr(original)
    if atom(root) != "kicad_pcb":
        raise ValueError("not a kicad_pcb root")
    nodes = {}
    nets = defaultdict(list)
    kept_footprints = []
    for fp in children(root, "footprint"):
        ref = reference_of_footprint(fp)
        pads = children(fp, "pad")
        if not ref or ref.startswith("#") or not pads:
            continue
        value_node = first_child(fp, "property")
        value = ""
        for prop in children(fp, "property"):
            if len(prop) >= 3 and dequote(prop[1]) == "Value":
                value = dequote(prop[2]); break
        footprint_name = dequote(fp[1]) if len(fp) > 1 else ""
        at = xyrot(first_child(fp, "at"))
        layer_node = first_child(fp, "layer")
        layer = dequote(layer_node[1]) if layer_node and len(layer_node) > 1 else "F.Cu"
        flip = layer.startswith("B.")
        kept_pads = []
        for pad in pads:
            number = dequote(pad[1]) if len(pad) > 1 else ""
            pat = xyrot(first_child(pad, "at"))
            px, py = (-pat[0] if flip else pat[0]), pat[1]
            dx, dy = rotate_xy(px, py, at[2])
            x, y = at[0] + dx, at[1] + dy
            size_node = first_child(pad, "size")
            size = [float(v) for v in size_node[1:3]] if size_node and len(size_node) > 2 else [0.0, 0.0]
            shape = dequote(pad[3]) if len(pad) > 3 else "unknown"
            node_id = f"{ref}:{number}"
            nodes[node_id] = {"id": node_id, "reference": ref, "pad_number": number,
                "component_value": value, "footprint": footprint_name,
                "component_type": classify_component(ref, value, footprint_name),
                "x_mm": x, "y_mm": y, "component_x_mm": at[0], "component_y_mm": at[1],
                "rotation_deg": at[2], "side": "B" if flip else "F",
                "pad_shape": shape, "pad_size_mm": size}
            net_node = first_child(pad, "net")
            if net_node and len(net_node) >= 3:
                net_name = dequote(net_node[2])
                if net_name:
                    nets[net_name].append(node_id)
            # Erase all net assignments from the training-input copy.
            pad[:] = [part for part in pad if not (isinstance(part, list) and atom(part) == "net")]
            kept_pads.append(pad)
        fp[:] = [part for part in fp if not (isinstance(part, list) and atom(part) == "pad")]
        fp.extend(kept_pads)
        kept_footprints.append(fp)

    clean = []
    for item in root:
        if not isinstance(item, list):
            clean.append(item); continue
        key = atom(item)
        if key in {"segment", "via", "zone", "net", "net_settings", "dimension", "image", "group", "target"}:
            continue
        if key == "footprint":
            if item in kept_footprints: clean.append(item)
            continue
        if key in {"gr_line", "gr_arc", "gr_rect", "gr_poly", "gr_circle", "gr_curve", "gr_text", "gr_text_box"}:
            layer = first_child(item, "layer")
            if not layer or len(layer) < 2 or dequote(layer[1]) != "Edge.Cuts":
                continue
        clean.append(item)
    root[:] = clean
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_sexpr(root) + "\n", encoding="utf-8")
    groups = [sorted(set(pins)) for pins in nets.values() if len(set(pins)) >= 2]
    return nodes, groups


def eagle_process(source: Path, output: Path):
    parser = ET.XMLParser()
    tree = ET.parse(source, parser=parser)
    root = tree.getroot()
    board = root.find("./drawing/board")
    if board is None:
        raise ValueError("Eagle XML has no drawing/board")
    package_map = {}
    for library in board.findall("./libraries/library"):
        lname = library.get("name", "")
        for package in library.findall("./packages/package"):
            package_map[(lname, package.get("name", ""))] = package
    elements = {}
    nodes = {}
    for element in board.findall("./elements/element"):
        ref = element.get("name", "")
        pkg = package_map.get((element.get("library", ""), element.get("package", "")))
        if not ref or pkg is None: continue
        try:
            ex, ey = float(element.get("x", "0")), float(element.get("y", "0"))
        except ValueError:
            continue
        rot_text = element.get("rot", "R0")
        mirrored = "M" in rot_text.upper()
        try:
            angle = float(re.search(r"(-?\d+(?:\.\d+)?)", rot_text).group(1))
        except (AttributeError, ValueError):
            angle = 0.0
        elements[ref] = element
        for pad in list(pkg.findall("pad")) + list(pkg.findall("smd")):
            name = pad.get("name", "")
            try:
                px, py = float(pad.get("x", "0")), float(pad.get("y", "0"))
            except ValueError:
                continue
            if mirrored: px = -px
            dx, dy = rotate_xy(px, py, angle)
            node_id = f"{ref}:{name}"
            nodes[node_id] = {"id": node_id, "reference": ref, "pad_number": name,
                "component_value": element.get("value", ""), "footprint": element.get("package", ""),
                "component_type": classify_component(ref, element.get("value", ""), element.get("package", "")),
                "x_mm": ex + dx, "y_mm": ey + dy, "component_x_mm": ex, "component_y_mm": ey,
                "rotation_deg": angle, "side": "B" if mirrored else "F",
                "pad_shape": pad.get("shape", pad.tag),
                "pad_size_mm": [float(pad.get("dx", pad.get("diameter", "0"))), float(pad.get("dy", pad.get("diameter", "0")))]}
    signal_groups = []
    signals = board.find("signals")
    if signals is not None:
        for signal in signals.findall("signal"):
            pins = []
            for contact in signal.findall("contactref"):
                node_id = f"{contact.get('element', '')}:{contact.get('pad', '')}"
                if node_id in nodes: pins.append(node_id)
            if len(set(pins)) >= 2: signal_groups.append(sorted(set(pins)))
        board.remove(signals)
    output.parent.mkdir(parents=True, exist_ok=True)
    xml = ET.tostring(root, encoding="unicode", short_empty_elements=True)
    output.write_text('<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE eagle SYSTEM "eagle.dtd">\n' + xml + "\n", encoding="utf-8")
    return nodes, signal_groups


def candidates_and_labels(nodes, net_groups, radius_mm, k_nearest):
    ids = sorted(nodes)
    if not ids:
        return [], [], [], [], None
    points = [(nodes[key]["x_mm"], nodes[key]["y_mm"]) for key in ids]
    tree = cKDTree(points)
    pair_set = set()
    for i, j in tree.query_pairs(radius_mm, output_type="set"):
        left, right = ids[i], ids[j]
        if nodes[left]["reference"] != nodes[right]["reference"]:
            pair_set.add((min(left, right), max(left, right)))
    if len(ids) > 1 and k_nearest > 0:
        distances, indices = tree.query(points, k=min(k_nearest + 1, len(ids)))
        for i in range(len(ids)):
            for j in indices[i][1:]:
                if i == j or nodes[ids[i]]["reference"] == nodes[ids[j]]["reference"]: continue
                pair_set.add((min(ids[i], ids[j]), max(ids[i], ids[j])))
    truth_net_map = {}
    truth_nets = []
    positive = set()
    for group in net_groups:
        members = sorted(set(pin for pin in group if pin in nodes))
        components = {nodes[pin]["reference"] for pin in members}
        if len(components) < 2:
            continue
        net_idx = len(truth_nets)
        truth_nets.append({"net_id": net_idx, "members": members})
        for i, left in enumerate(members):
            for right in members[i + 1:]:
                if nodes[left]["reference"] == nodes[right]["reference"]:
                    continue
                pair = (min(left, right), max(left, right))
                positive.add(pair)
                truth_net_map[pair] = net_idx
    truth_edges = [{"source": a, "target": b, "truth_net_id": truth_net_map[(a, b)]}
                   for a, b in sorted(positive)]
    edges = []
    labels = []
    for idx, (left, right) in enumerate(sorted(pair_set)):
        a, b = nodes[left], nodes[right]
        dx, dy = b["x_mm"] - a["x_mm"], b["y_mm"] - a["y_mm"]
        distance = math.hypot(dx, dy)
        edges.append({"edge_id": idx, "source": left, "target": right,
                      "distance_mm": distance, "dx_mm": dx, "dy_mm": dy,
                      "same_component": a["reference"] == b["reference"]})
        labels.append({"edge_id": idx, "label": int((left, right) in positive),
                       "truth_net_id": truth_net_map.get((left, right))})
    candidate_recall = len(positive.intersection(pair_set)) / len(positive) if positive else None
    return edges, labels, truth_edges, truth_nets, candidate_recall


def split_repositories(repositories, seed=42):
    groups = sorted(set(repositories))
    random.Random(seed).shuffle(groups)
    n = len(groups)
    n_test = max(1, round(n * .1)) if n >= 10 else 1
    n_val = max(1, round(n * .1)) if n >= 10 else 0
    test, val = set(groups[:n_test]), set(groups[n_test:n_test+n_val])
    return {repo: ("test" if repo in test else "validation" if repo in val else "train") for repo in groups}


def source_boards(manifest):
    assets = []
    for record in manifest.get("assets", []):
        cad = record.get("assets", {}).get("cad", {})
        if cad.get("status") == "downloaded" and Path(cad["path"]).suffix.lower() in {".brd", ".kicad_pcb"}:
            assets.append({"path": cad["path"], "repository": record.get("repository", "unknown"), "repo_url": record.get("repo_url"), "source_kind": "direct_csv"})
    for record in manifest.get("repository_discovery", {}).get("assets", []):
        cad = record.get("assets", {}).get("board", {})
        if cad.get("status") == "downloaded" and Path(cad["path"]).suffix.lower() in {".brd", ".kicad_pcb"}:
            assets.append({"path": cad["path"], "repository": record.get("repository", "unknown"), "repo_url": record.get("repo_url"), "source_kind": record.get("source_kind", "repository_discovery")})
    dedup = {}
    for item in assets:
        path = ROOT / item["path"]
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha not in dedup:
            item["sha256"] = sha
            dedup[sha] = item
        else:
            dedup[sha].setdefault("duplicate_sources", []).append(item)
    return list(dedup.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius-mm", type=float, default=25.0)
    parser.add_argument("--k-nearest", type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    boards = source_boards(manifest)
    split = split_repositories([x["repository"] for x in boards])
    DEST.mkdir(parents=True, exist_ok=True)
    for name in ("boards",): (DEST / name).mkdir(exist_ok=True)
    graphs_path, labels_path = DEST / "graphs.jsonl", DEST / "edge_labels.jsonl"
    summary = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "input_boards": len(boards),
        "radius_mm": args.radius_mm, "k_nearest": args.k_nearest,
        "input_schema": "nodes contain component/pad geometry only; edge labels are stored separately and never copied to input boards or graph features.",
        "split_policy": "repository-grouped 80/10/10 deterministic split (seed 42)", "processed": 0, "failed": [],
        "components": 0, "pins": 0, "candidate_edges": 0, "positive_edges": 0, "all_truth_edges": 0,
        "formats": defaultdict(int), "splits": defaultdict(int), "component_type_counts": defaultdict(int),
        "candidate_recall_sum": 0.0, "candidate_recall_boards": 0,
        "boards": [], "excluded": []}
    with graphs_path.open("w", encoding="utf-8") as gf, labels_path.open("w", encoding="utf-8") as lf:
        for index, item in enumerate(boards, 1):
            source = ROOT / item["path"]
            board_id = item["sha256"][:16]
            ext = source.suffix.lower()
            clean_path = DEST / "boards" / f"{board_id}.pin_only{ext}"
            try:
                nodes, groups = (eagle_process(source, clean_path) if ext == ".brd" else kicad_process(source, clean_path))
                edges, edge_labels, truth_edges, truth_nets, candidate_recall = candidates_and_labels(nodes, groups, args.radius_mm, args.k_nearest)
                positives = sum(x["label"] for x in edge_labels)
                if not nodes or not truth_edges:
                    reason = "no_component_pads" if not nodes else "no_connection_labels"
                    summary["excluded"].append({"source_path": item["path"], "repository": item["repository"],
                                                "reason": reason, "pin_count": len(nodes), "candidate_edge_count": len(edges)})
                    print(f"[{index}/{len(boards)}] EXCLUDED {item['repository']}: {reason}", flush=True)
                    continue
                graph = nx.Graph()
                for node_id, attributes in nodes.items(): graph.add_node(node_id, **attributes)
                for edge in edges:
                    graph.add_edge(edge["source"], edge["target"], edge_id=edge["edge_id"],
                                   distance_mm=edge["distance_mm"], dx_mm=edge["dx_mm"], dy_mm=edge["dy_mm"],
                                   same_component=edge["same_component"])
                graph_doc = {"graph_id": board_id, "format": "kicad" if ext == ".kicad_pcb" else "eagle",
                    "source_repository": item["repository"], "nodes": list(graph.nodes(data=True)),
                    "candidate_edges": list(graph.edges(data=True))}
                # Keep serialized graph explicit and stable rather than relying on NetworkX's node-link API version.
                graph_doc["candidate_edges"] = [dict(edge, source=u, target=v) for u, v, edge in graph.edges(data=True)]
                graph_doc["nodes"] = [{"id": n, **d} for n, d in graph.nodes(data=True)]
                label_doc = {"graph_id": board_id, "edge_labels": edge_labels,
                             "positive_edge_count": positives, "truth_edges": truth_edges,
                             "truth_nets": truth_nets, "candidate_recall": candidate_recall}
                common = {"graph_id": board_id, "source_path": item["path"], "source_repository": item["repository"],
                    "repo_url": item.get("repo_url"), "source_kind": item["source_kind"], "source_sha256": item["sha256"],
                    "format": "kicad" if ext == ".kicad_pcb" else "eagle",
                    "clean_board_path": f"boards/{clean_path.name}", "split": split[item["repository"]],
                    "component_count": len(set(n["reference"] for n in nodes.values())), "pin_count": len(nodes),
                    "candidate_edge_count": len(edges), "positive_edge_count": label_doc["positive_edge_count"],
                    "truth_edge_count": len(truth_edges), "candidate_recall": candidate_recall,
                    "target_net_group_count": len(truth_nets)}
                graph_doc.update(common)
                label_doc.update({"source_repository": item["repository"], "split": split[item["repository"]]})
                gf.write(json.dumps(graph_doc, ensure_ascii=False) + "\n")
                lf.write(json.dumps(label_doc, ensure_ascii=False) + "\n")
                summary["processed"] += 1
                summary["components"] += common["component_count"]
                summary["pins"] += common["pin_count"]
                unique_components = {}
                for node in nodes.values(): unique_components.setdefault(node["reference"], node["component_type"])
                for component_type in unique_components.values(): summary["component_type_counts"][component_type] += 1
                summary["candidate_edges"] += common["candidate_edge_count"]
                summary["positive_edges"] += common["positive_edge_count"]
                summary["all_truth_edges"] += common["truth_edge_count"]
                if candidate_recall is not None:
                    summary["candidate_recall_sum"] += candidate_recall
                    summary["candidate_recall_boards"] += 1
                summary["formats"][common["format"]] += 1
                summary["splits"][common["split"]] += 1
                summary["boards"].append(common)
                print(f"[{index}/{len(boards)}] {board_id} {ext} pins={len(nodes)} candidates={len(edges)} positives={common['positive_edge_count']}", flush=True)
            except Exception as exc:
                summary["failed"].append({"source_path": item["path"], "repository": item["repository"], "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{index}/{len(boards)}] FAILED {item['path']}: {type(exc).__name__}: {exc}", flush=True)
    summary["formats"] = dict(summary["formats"])
    summary["splits"] = dict(summary["splits"])
    summary["component_type_counts"] = dict(summary["component_type_counts"])
    summary["candidate_recall_micro"] = summary["positive_edges"] / summary["all_truth_edges"] if summary["all_truth_edges"] else None
    summary["candidate_recall_macro"] = summary["candidate_recall_sum"] / summary["candidate_recall_boards"] if summary["candidate_recall_boards"] else None
    summary["component_classes"] = {"taxonomy": ["IC", "LED", "battery", "buzzer", "capacitor", "connector", "diode", "display", "emi_filter", "ferrite_bead", "fuse", "inductor", "jumper", "oscillator", "potentiometer", "relay", "resistor", "switch", "transistor"],
                                    "assignment": "Heuristic from reference/value/footprint text; unknown is kept when evidence is insufficient."}
    summary["manifest_path"] = str((DEST / "dataset_manifest.json").relative_to(ROOT)).replace("\\", "/")
    (DEST / "dataset_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("input_boards", "processed", "excluded", "failed", "components", "pins", "candidate_edges", "positive_edges", "all_truth_edges", "candidate_recall_micro", "candidate_recall_macro", "formats", "splits")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
