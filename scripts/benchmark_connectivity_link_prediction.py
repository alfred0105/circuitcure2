#!/usr/bin/env python3
"""Build pin-link graphs and compare an ACLP-inspired GNN to LightGBM.

The GNN is an independent, compact graph link-prediction baseline. It is not
the official GNN-ACLP implementation or a reproduction of its experiments.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import itertools
import json
import math
import os
import subprocess
import tempfile
import traceback
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
PAIR_DIR = ROOT / "data/external/connectivity_pairs-5000-complete"
DATA_DIR = ROOT / "data/benchmarks/phase22_connectivity"
LABELS = ("IC", "LED", "battery", "buzzer", "capacitor", "connector", "diode", "display",
          "emi_filter", "ferrite_bead", "fuse", "inductor", "jumper", "oscillator",
          "potentiometer", "relay", "resistor", "switch", "transistor")
ELECTRICAL = {"input": 0, "output": 1, "bidirectional": 2, "tri_state": 3,
              "passive": 4, "free": 5, "unspecified": 6, "power_in": 7,
              "power_out": 8, "open_collector": 9, "open_emitter": 10,
              "no_connect": 11}


def load_parser():
    path = ROOT / "scripts/prepare_connectivity_pairs.py"
    spec = importlib.util.spec_from_file_location("prepare_connectivity_pairs", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


PARSER = load_parser()


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def symbol_pins(sch: Path, classes: dict[str, str]) -> list[dict[str, Any]]:
    root = PARSER.parse_expr(sch.read_text(encoding="utf-8"))
    library = PARSER.child(root, "lib_symbols")
    library_by_id = {str(s[1]): s for s in PARSER.children(library, "symbol") if len(s) > 1}
    result = []
    for instance in PARSER.children(root, "symbol"):
        props = PARSER.props(instance)
        ref = props.get("Reference", "").strip()
        label = classes.get(ref)
        if label not in LABELS or ref.startswith("#"):
            continue
        lib_node, at, unit_node = (PARSER.child(instance, k) for k in ("lib_id", "at", "unit"))
        if not lib_node or not at:
            continue
        lib_id = str(lib_node[1])
        lib = library_by_id.get(lib_id)
        if lib is None:
            continue
        x, y, angle = fnum(at[1]), fnum(at[2]), math.radians(fnum(at[3] if len(at) > 3 else 0))
        unit = int(fnum(unit_node[1] if unit_node and len(unit_node) > 1 else 1, 1))
        mirrors = PARSER.child(instance, "mirror")
        mirror = str(mirrors[1]) if mirrors and len(mirrors) > 1 else ""
        pin_defs = []
        for sub in PARSER.children(lib, "symbol"):
            # Nested names end in _<unit>_<representation>.
            try:
                sub_unit = int(str(sub[1]).rsplit("_", 2)[-2])
            except (ValueError, IndexError):
                sub_unit = unit
            if sub_unit not in (0, unit):
                continue
            for pin in PARSER.children(sub, "pin"):
                p_at = PARSER.child(pin, "at")
                p_num = PARSER.child(pin, "number")
                if not p_at or not p_num or len(p_at) < 3 or len(p_num) < 2:
                    continue
                lx, ly = fnum(p_at[1]), fnum(p_at[2])
                if mirror == "x":
                    ly = -ly
                elif mirror == "y":
                    lx = -lx
                # KiCad schematic coordinates have a downward-positive y axis.
                px = x + math.cos(angle) * lx + math.sin(angle) * ly
                py = y - math.sin(angle) * lx + math.cos(angle) * ly
                electrical = str(pin[1]) if len(pin) > 1 else "unspecified"
                name_node = PARSER.child(pin, "name")
                pin_defs.append({"ref": ref, "pin": str(p_num[1]), "x": px, "y": py,
                                 "class": label, "electrical": electrical,
                                 "pin_name": str(name_node[1]) if name_node and len(name_node) > 1 else "",
                                 "lib_id": lib_id, "unit": unit})
        # Pins can be repeated in alternate graphical representations; dedupe by number.
        unique = {}
        for pin in pin_defs:
            unique.setdefault(pin["pin"], pin)
        result.extend(unique.values())
    return result


def export_nets(cli: str, target: Path, xml_out: Path) -> tuple[list[dict[str, Any]], str]:
    run = subprocess.run([cli, "sch", "export", "netlist", "--format", "kicadxml", "-o", str(xml_out), str(target)],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    if run.returncode:
        raise RuntimeError((run.stderr or run.stdout or "kicad-cli export failed")[-1200:])
    root = ET.parse(xml_out).getroot()
    tool = root.findtext("./design/tool", default="")
    nets = []
    for net in root.findall("./nets/net"):
        code = net.attrib.get("code", "")
        name = net.attrib.get("name", "")
        nodes = [{"ref": n.attrib.get("ref", ""), "pin": n.attrib.get("pin", "")}
                 for n in net.findall("node")]
        nets.append({"net_id": f"{code}:{name}", "nodes": nodes})
    return nets, tool


def graph_for_sample(record: dict[str, Any], cli: str, k: int, radius: float | None,
                     tmp: Path) -> dict[str, Any]:
    target = PAIR_DIR / record["target_path"]
    schematic_input = PAIR_DIR / record["input_path"]
    classes = record.get("classes", {})
    pins = symbol_pins(schematic_input, classes)
    if len(pins) < 2:
        raise ValueError("fewer than two taxonomy pins parsed")
    nets, cli_version = export_nets(cli, target, tmp / f"{record['sample_id']}.xml")
    pin_keys = {(p["ref"], p["pin"]): i for i, p in enumerate(pins)}
    pair_labels: set[tuple[int, int]] = set()
    net_membership: dict[str, list[int]] = {}
    for net in nets:
        members = sorted({pin_keys[(n["ref"], n["pin"])] for n in net["nodes"]
                          if (n["ref"], n["pin"]) in pin_keys})
        if len(members) >= 2:
            net_membership[net["net_id"]] = members
            pair_labels.update((a, b) for a, b in itertools.combinations(members, 2))

    coords = np.array([[p["x"], p["y"]] for p in pins], dtype=float)
    tree = cKDTree(coords)
    candidate_set: set[tuple[int, int]] = set()
    candidate_ranks: dict[tuple[int, int], int] = {}
    # Query each pin's nearest cross-component pins, then symmetrize the k-NN graph.
    query_k = min(len(pins), max(k * 16 + 1, 128))
    distances, neighbors = tree.query(coords, k=query_k, distance_upper_bound=radius or np.inf)
    if query_k == 1:
        distances, neighbors = distances[:, None], neighbors[:, None]
    for i, (ds, js) in enumerate(zip(distances, neighbors)):
        found = 0
        for rank, (distance, j) in enumerate(zip(ds, js), start=1):
            j = int(j)
            if j >= len(pins) or j == i or pins[j]["ref"] == pins[i]["ref"]:
                continue
            edge = (min(i, j), max(i, j))
            candidate_set.add(edge)
            candidate_ranks[edge] = min(candidate_ranks.get(edge, rank), rank)
            found += 1
            if found >= k:
                break
    graph = nx.Graph(sample_id=record["sample_id"], group_id=record.get("group_id", ""),
                     split=record.get("split", "train"), k=k, radius=radius)
    for idx, pin in enumerate(pins):
        graph.add_node(idx, **pin)
    for i, j in sorted(candidate_set):
        dx, dy = coords[j] - coords[i]
        dist = float(math.hypot(dx, dy))
        graph.add_edge(i, j, distance=dist, dx=float(dx), dy=float(dy),
                       rank_i=float(candidate_ranks[(i, j)]),
                       label=int((i, j) in pair_labels))
    candidate_truth = pair_labels.intersection(candidate_set)
    return {"sample_id": record["sample_id"], "group_id": record.get("group_id", ""),
            "split": record.get("split", "train"), "nodes": [dict(graph.nodes[n]) for n in graph.nodes],
            "edges": [{"source": u, "target": v, **dict(attrs)} for u, v, attrs in graph.edges(data=True)],
            "truth_edges": [[u, v] for u, v in sorted(pair_labels)],
            "candidate_recall": len(candidate_truth) / len(pair_labels) if pair_labels else None,
            "net_count": len(net_membership), "cli_version": cli_version,
            "source_path": record["source_path"]}


def extract(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cli = args.kicad_cli
    if not cli:
        candidates = [os.environ.get("KICAD_CLI"), r"C:\Program Files\KiCad\10.0\bin\kicad-cli.exe",
                      str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/KiCad/10.0/bin/kicad-cli.exe")]
        cli = next((p for p in candidates if p and Path(p).is_file()), None)
    if not cli:
        raise SystemExit("kicad-cli not found; pass --kicad-cli PATH")
    records_path = PAIR_DIR / "records.jsonl"
    records = [json.loads(line) for line in records_path.open(encoding="utf-8") if line.strip()]
    if args.limit:
        records = records[:args.limit]
    failures = []
    successes = 0
    with tempfile.TemporaryDirectory(prefix="cc-net-export-") as temp_name, out.open("w", encoding="utf-8") as stream:
        temp = Path(temp_name)
        def one(record):
            return graph_for_sample(record, cli, args.k, args.radius, temp)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(one, record): record for record in records}
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                record = futures[future]
                try:
                    item = future.result()
                    stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
                    successes += 1
                except Exception as exc:
                    failures.append({"sample_id": record.get("sample_id"), "source_path": record.get("source_path"),
                                     "error": f"{type(exc).__name__}: {exc}"})
                if i % 100 == 0 or i == len(records):
                    stream.flush()
                    print(f"[{i}/{len(records)}] graph records={successes} failures={len(failures)}", flush=True)
    (out.parent / "extraction_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    recalls = [json.loads(line).get("candidate_recall") for line in out.open(encoding="utf-8")]
    recalls = [v for v in recalls if v is not None]
    summary = {"requested": len(records), "extracted": successes, "failed": len(failures), "k": args.k,
               "radius_mm": args.radius, "mean_candidate_recall": float(np.mean(recalls)) if recalls else None,
               "kicad_cli": cli}
    (out.parent / "extraction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def node_features(node: dict[str, Any]) -> np.ndarray:
    cls = np.zeros(len(LABELS), dtype=np.float32)
    if node.get("class") in LABELS:
        cls[LABELS.index(node["class"])] = 1
    electrical = np.zeros(len(ELECTRICAL), dtype=np.float32)
    electrical[ELECTRICAL.get(str(node.get("electrical", "")).casefold(), 6)] = 1
    pin = str(node.get("pin", ""))
    # Stable numeric cues without fitting a vocabulary across splits.
    try:
        pin_num = min(float(pin), 1000.0) / 1000.0
    except ValueError:
        pin_num = 0.0
    return np.concatenate((cls, electrical, np.array([float(node.get("x", 0)) / 1000,
        float(node.get("y", 0)) / 1000, pin_num, len(str(node.get("pin_name", ""))) / 32], dtype=np.float32)))


def edge_features(graph: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes = graph["nodes"]
    x = np.stack([node_features(n) for n in nodes])
    pairs = np.array([[e["source"], e["target"]] for e in graph["edges"]], dtype=np.int64)
    y = np.array([e["label"] for e in graph["edges"]], dtype=np.int64)
    ef = np.array([[math.log1p(e["distance"]), e["dx"] / 100, e["dy"] / 100,
                    abs(e["dx"]) / (e["distance"] + 1e-6),
                    float(nodes[e["source"]]["class"] == nodes[e["target"]]["class"]),
                    float(nodes[e["source"]]["ref"] == nodes[e["target"]]["ref"])]
                   for e in graph["edges"]], dtype=np.float32)
    return x, pairs, ef, y


def metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
    pred = (scores >= 0.5).astype(int)
    return {"candidate_edges": int(len(y)), "positive_edges": int(y.sum()),
            "positive_rate": float(y.mean()) if len(y) else None,
            "pr_auc": float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else None,
            "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else None,
            "f1_at_0_5": float(f1_score(y, pred, zero_division=0)),
            "precision_at_0_5": float(precision_score(y, pred, zero_division=0)),
            "recall_at_0_5": float(recall_score(y, pred, zero_division=0))}


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch import nn
    from lightgbm import LGBMClassifier
    from sklearn.utils.class_weight import compute_sample_weight

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    graph_path = Path(args.graphs)
    metadata = []
    with graph_path.open(encoding="utf-8") as graph_file:
        for line in graph_file:
            g = json.loads(line)
            metadata.append({"sample_id": g["sample_id"], "group_id": g["group_id"],
                             "split": g["split"], "candidate_recall": g["candidate_recall"]})
    by_split = {split: [m for m in metadata if m["split"] == split]
                for split in ("train", "validation", "test")}
    if not by_split["train"] or not by_split["test"]:
        raise SystemExit("Need non-empty train and test graph splits")
    rng = np.random.default_rng(args.seed)
    train_pool = by_split["train"]
    train_pool.sort(key=lambda item: item["sample_id"])
    selected_count = min(len(train_pool), args.gnn_max_graphs or len(train_pool))
    baseline_count = min(200, len(train_pool))
    baseline_indices = rng.choice(len(train_pool), baseline_count, replace=False)
    baseline_set = set(map(int, baseline_indices))
    remaining = np.array([i for i in range(len(train_pool)) if i not in baseline_set], dtype=np.int64)
    nested_order = np.concatenate((baseline_indices, rng.permutation(remaining)))
    selected = nested_order[:selected_count]
    selected_ids = {train_pool[int(i)]["sample_id"] for i in selected}
    train_rows = []
    xgb_rows, ygb_rows = [], []
    with graph_path.open(encoding="utf-8") as graph_file:
        for line in graph_file:
            g = json.loads(line)
            if g["sample_id"] not in selected_ids:
                continue
            row = edge_features(g)
            train_rows.append(row)
            x, pairs, ef, y = row
            if not len(y):
                continue
            positives = np.flatnonzero(y == 1)
            negatives = np.flatnonzero(y == 0)
            neg_count = min(len(negatives), max(args.min_negatives_per_graph,
                             len(positives) * args.negative_ratio), args.max_negatives_per_graph)
            if neg_count < len(negatives):
                seed_material = f"{args.seed}:{g['sample_id']}".encode("utf-8")
                graph_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
                graph_rng = np.random.default_rng(graph_seed)
                negatives = graph_rng.choice(negatives, neg_count, replace=False)
            keep = np.sort(np.concatenate((positives, negatives)))
            xgb_rows.append(np.concatenate((ef[keep], x[pairs[keep, 0]], x[pairs[keep, 1]]), axis=1))
            ygb_rows.append(y[keep])
    if not train_rows or not xgb_rows:
        raise SystemExit("Selected training graphs contain no usable candidate edges")
    Xtr, ytr = np.concatenate(xgb_rows), np.concatenate(ygb_rows)
    weights = compute_sample_weight("balanced", ytr)
    lgbm = LGBMClassifier(n_estimators=args.lgbm_trees, learning_rate=0.04, num_leaves=31,
                          max_depth=-1, subsample=0.85, colsample_bytree=0.85,
                          reg_lambda=1.0, random_state=args.seed, n_jobs=args.jobs, verbosity=-1)
    lgbm.fit(Xtr, ytr, sample_weight=weights)
    del Xtr, ytr, weights, xgb_rows, ygb_rows
    results = {"metadata": {"graph_records": len(metadata), "train_projects": len({g["group_id"] for g in by_split["train"]}),
        "validation_projects": len({g["group_id"] for g in by_split["validation"]}),
        "test_projects": len({g["group_id"] for g in by_split["test"]}),
        "split_counts": {k: len(v) for k, v in by_split.items()}, "seed": args.seed,
        "training_graphs_sampled_for_both_models": len(train_rows),
        "lightgbm_negative_to_positive_cap": args.negative_ratio,
        "lightgbm_max_sampled_negatives_per_graph": args.max_negatives_per_graph,
        "candidate_mean_recall_test": float(np.mean([g["candidate_recall"] for g in by_split["test"] if g["candidate_recall"] is not None]))}}

    # Compact message-passing GNN, trained in graph batches to bound memory.
    class ACLPGNN(nn.Module):
        def __init__(self, node_dim: int, edge_dim: int, hidden: int = 48):
            super().__init__()
            self.encoder = nn.Linear(node_dim, hidden)
            self.msg1 = nn.Linear(hidden * 2, hidden)
            self.msg2 = nn.Linear(hidden * 2, hidden)
            self.edge_head = nn.Sequential(nn.Linear(hidden * 2 + edge_dim, hidden), nn.ReLU(),
                                           nn.Dropout(0.15), nn.Linear(hidden, 1))

        def forward(self, x, edges, edge_attr):
            h = torch.relu(self.encoder(x))
            for layer in (self.msg1, self.msg2):
                src, dst = edges[:, 0], edges[:, 1]
                agg = torch.zeros_like(h)
                agg.index_add_(0, dst, h[src])
                agg.index_add_(0, src, h[dst])
                deg = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
                ones = torch.ones((len(src), 1), dtype=h.dtype, device=h.device)
                deg.index_add_(0, dst, ones); deg.index_add_(0, src, ones)
                h = torch.relu(layer(torch.cat([h, agg / deg.clamp_min(1)], dim=1)))
            return self.edge_head(torch.cat([h[edges[:, 0]], h[edges[:, 1]], edge_attr], dim=1)).squeeze(-1)

    def combine(rows):
        node_list, edge_list, attr_list, y_list = [], [], [], []
        offset = 0
        for x, pairs, ef, y in rows:
            node_list.append(x)
            if len(pairs):
                edge_list.append(pairs + offset); attr_list.append(ef); y_list.append(y)
            offset += len(x)
        return (np.concatenate(node_list), np.concatenate(edge_list), np.concatenate(attr_list), np.concatenate(y_list))

    node_dim = train_rows[0][0].shape[1]
    edge_dim = train_rows[0][2].shape[1]
    total_positive = sum(int(row[3].sum()) for row in train_rows)
    total_edges = sum(len(row[3]) for row in train_rows)
    model = ACLPGNN(node_dim, edge_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_weight = torch.tensor([(total_edges - total_positive) / max(total_positive, 1)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for epoch in range(args.epochs):
        order = np.random.default_rng(args.seed + epoch).permutation(len(train_rows))
        total_loss, seen_edges = 0.0, 0
        for start in range(0, len(order), args.graph_batch_size):
            batch_rows = [train_rows[int(i)] for i in order[start:start + args.graph_batch_size]]
            bx, be, ba, by = combine(batch_rows)
            tx, te, ta = torch.as_tensor(bx), torch.as_tensor(be), torch.as_tensor(ba)
            ty = torch.as_tensor(by, dtype=torch.float32)
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits = model(tx, te, ta)
            loss = loss_fn(logits, ty)
            loss.backward(); optimizer.step()
            total_loss += float(loss.detach()) * len(by)
            seen_edges += len(by)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"GNN epoch {epoch + 1}/{args.epochs} loss={total_loss / max(seen_edges, 1):.4f}", flush=True)
    model.eval()
    test_y_rows, gnn_score_rows, lgbm_score_rows = [], [], []
    with graph_path.open(encoding="utf-8") as graph_file, torch.no_grad():
        for line in graph_file:
            g = json.loads(line)
            if g["split"] != "test":
                continue
            x, pairs, ef, y = edge_features(g)
            if not len(y):
                continue
            features = np.concatenate((ef, x[pairs[:, 0]], x[pairs[:, 1]]), axis=1)
            lgbm_score_rows.append(lgbm.booster_.predict(features))
            gnn_score_rows.append(torch.sigmoid(model(torch.as_tensor(x), torch.as_tensor(pairs),
                                                       torch.as_tensor(ef))).numpy())
            test_y_rows.append(y)
    test_y = np.concatenate(test_y_rows)
    lgbm_scores = np.concatenate(lgbm_score_rows)
    gnn_scores = np.concatenate(gnn_score_rows)
    results["lightgbm"] = metrics(test_y, lgbm_scores)
    results["gnn_aclp_inspired"] = metrics(test_y, gnn_scores)
    results["gnn_aclp_inspired"]["training_graphs"] = len(train_rows)
    results["gnn_aclp_inspired"]["epochs"] = args.epochs
    results["methodology"] = "Same selected training schematics, cKDTree candidate links, project-group splits, and full test candidates. LightGBM uses endpoint/edge features with per-graph negative sampling capped by ratio and maximum; GNN uses all candidate edges in its sampled training graphs and two message-passing layers. GNN-ACLP-inspired, not a reproduction of the paper."
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "node_dim": int(node_dim),
                "edge_dim": int(edge_dim), "hidden_dim": 48, "seed": args.seed,
                "epochs": args.epochs}, model_dir / "gnn_aclp_inspired.pt")
    lgbm.booster_.save_model(str(model_dir / "lightgbm.txt"))
    results["model_artifacts"] = {"gnn": str(model_dir / "gnn_aclp_inspired.pt"),
                                  "lightgbm": str(model_dir / "lightgbm.txt")}
    if args.report:
        out = Path(args.report); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not getattr(args, "quiet", False):
        print(json.dumps(results, indent=2))
    return results


def sweep(args: argparse.Namespace) -> None:
    rows = []
    dataset_metadata = None
    for size in args.sizes:
        run_args = argparse.Namespace(**vars(args))
        run_args.command = "train"
        run_args.gnn_max_graphs = size
        run_args.report = None
        run_args.model_dir = str(Path(args.model_root) / str(size))
        run_args.quiet = True
        result = train(run_args)
        if dataset_metadata is None:
            dataset_metadata = result["metadata"]
        rows.append({"training_schematics": size,
                     "candidate_mean_recall_test": result["metadata"]["candidate_mean_recall_test"],
                     "lightgbm": result["lightgbm"], "gnn_aclp_inspired": result["gnn_aclp_inspired"]})
        print(f"[{len(rows)}/{len(args.sizes)}] train={size} "
              f"LightGBM PR-AUC={result['lightgbm']['pr_auc']:.4f} "
              f"GNN PR-AUC={result['gnn_aclp_inspired']['pr_auc']:.4f}", flush=True)
    extraction_path = DATA_DIR / "extraction_summary.json"
    extraction = json.loads(extraction_path.read_text(encoding="utf-8")) if extraction_path.exists() else {}
    report = {"experiment": {"seed": args.seed, "epochs": args.epochs,
              "graph_batch_size": args.graph_batch_size, "lightgbm_trees": args.lgbm_trees,
              "lightgbm_negative_to_positive_cap": args.negative_ratio,
              "lightgbm_max_sampled_negatives_per_graph": args.max_negatives_per_graph,
              "training_sample_policy": "Nested project-grouped train subset; first 200 graphs reused as the baseline subset.",
              "candidate_generation": {"k": extraction.get("k"), "radius_mm": extraction.get("radius_mm")},
              "dataset_counts": {"extracted_graphs": dataset_metadata["graph_records"],
                                 "project_splits": dataset_metadata["split_counts"],
                                 "projects_per_split": {"train": dataset_metadata["train_projects"],
                                     "validation": dataset_metadata["validation_projects"],
                                     "test": dataset_metadata["test_projects"]}}},
              "metrics": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Learning-curve results saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract", help="Export netlists and generate NetworkX/cKDTree pin candidate graphs")
    p.add_argument("--kicad-cli")
    p.add_argument("--output", default=str(DATA_DIR / "pin_candidate_graphs.jsonl"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--k", type=int, default=96)
    p.add_argument("--radius", type=float, default=250.0, help="candidate radius in schematic mm")
    p.add_argument("--workers", type=int, default=6)
    p.set_defaults(func=extract)
    p = sub.add_parser("train", help="Compare GNN-ACLP-inspired model with LightGBM")
    p.add_argument("--graphs", default=str(DATA_DIR / "pin_candidate_graphs.jsonl"))
    p.add_argument("--report", default=str(DATA_DIR / "model_comparison.json"))
    p.add_argument("--model-dir", default=str(ROOT / "models/phase22_connectivity"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--gnn-max-graphs", type=int, default=200)
    p.add_argument("--graph-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--lgbm-trees", type=int, default=400)
    p.add_argument("--negative-ratio", type=int, default=20)
    p.add_argument("--min-negatives-per-graph", type=int, default=16)
    p.add_argument("--max-negatives-per-graph", type=int, default=1000)
    p.add_argument("--jobs", type=int, default=-1)
    p.set_defaults(func=train)
    p = sub.add_parser("sweep", help="Run nested training-size experiments")
    p.add_argument("--graphs", default=str(DATA_DIR / "pin_candidate_graphs.jsonl"))
    p.add_argument("--output", default=str(DATA_DIR / "learning_curve.json"))
    p.add_argument("--model-root", default=str(ROOT / "models/phase22_connectivity/learning_curve"))
    p.add_argument("--sizes", nargs="+", type=int, default=[200, 500, 1000, 2000, 3920])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--graph-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--lgbm-trees", type=int, default=250)
    p.add_argument("--negative-ratio", type=int, default=20)
    p.add_argument("--min-negatives-per-graph", type=int, default=16)
    p.add_argument("--max-negatives-per-graph", type=int, default=1000)
    p.add_argument("--jobs", type=int, default=8)
    p.set_defaults(func=sweep)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
