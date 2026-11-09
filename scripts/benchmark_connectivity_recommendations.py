#!/usr/bin/env python3
"""Run ranking, hard-negative, net-aware, and adaptive candidate experiments.

Uses the released pin_candidate_graphs.jsonl without re-exporting schematics.
All evaluation remains on the fixed project-grouped test split.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/benchmarks/phase22_connectivity"
PROGRESS_FILE = DATA / "recommendation_progress.json"
SPEC = importlib.util.spec_from_file_location("benchmark_connectivity_link_prediction",
    ROOT / "scripts/benchmark_connectivity_link_prediction.py")
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(BASE)


class Progress:
    def __init__(self, count: int):
        self.count = count
        self.started = time.time()
        self.update("초기화", 0, 0, "데이터와 GPU 상태를 확인 중")

    def update(self, phase: str, phase_number: int, fraction: float, detail: str = ""):
        fraction = float(np.clip(fraction, 0, 1))
        overall = 100 * (max(phase_number - 1, 0) + fraction) / self.count
        DATA.mkdir(parents=True, exist_ok=True)
        payload = {"status": "완료" if phase == "완료" else "실행 중",
                   "phase": phase, "phase_number": phase_number if phase_number else None,
                   "phase_count": self.count, "phase_percent": fraction * 100,
                   "overall_percent": 100.0 if phase == "완료" else overall,
                   "elapsed_seconds": round(time.time() - self.started, 1),
                   "detail": detail,
                   "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")}
        temp = PROGRESS_FILE.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        for attempt in range(20):
            try:
                os.replace(temp, PROGRESS_FILE)
                break
            except PermissionError:
                if attempt == 19:
                    # The dashboard reads this small file concurrently on Windows.
                    PROGRESS_FILE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                else:
                    time.sleep(.05)


def lgb_progress(progress: Progress, phase: str, phase_no: int, start: float = 0.25, span: float = 0.7):
    class Callback:
        order = 50
        before_iteration = False
        def __call__(self, env):
            every = max(1, (env.end_iteration - env.begin_iteration) // 40)
            if (env.iteration + 1) % every == 0 or env.iteration + 1 == env.end_iteration:
                local = start + span * (env.iteration + 1) / env.end_iteration
                progress.update(phase, phase_no, local,
                    f"LightGBM tree {env.iteration + 1}/{env.end_iteration}")
    return Callback()


def save_partial(output: Path, results: dict[str, Any]):
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial.json")
    temp = partial.with_suffix(".tmp")
    temp.write_text(json.dumps(results, indent=2), encoding="utf-8")
    os.replace(temp, partial)


def pair_rows(g: dict[str, Any]):
    x, pairs, ef, y = BASE.edge_features(g)
    pairs = np.asarray(pairs, dtype=np.int64).reshape((-1, 2))
    ef = np.asarray(ef, dtype=np.float32).reshape((-1, 6))
    y = np.asarray(y, dtype=np.int64).reshape((-1,))
    feats = np.concatenate((ef, x[pairs[:, 0]], x[pairs[:, 1]]), axis=1)
    return x, pairs, ef, y, feats


def seed_for(seed: int, sample: str, suffix: str = "") -> int:
    raw = f"{seed}:{sample}:{suffix}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def net_ids(g: dict[str, Any]) -> tuple[np.ndarray, dict[int, list[int]]]:
    n = len(g["nodes"])
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in g.get("truth_edges", []):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    members: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        members[find(i)].append(i)
    active = {i: vs for i, vs in members.items() if len(vs) > 1}
    ids = np.full(n, -1, dtype=np.int64)
    for net_idx, vs in enumerate(active.values()):
        ids[vs] = net_idx
    return ids, {i: vs for i, vs in enumerate(active.values())}


def sampled_indices(g, y, scores, mode, seed, neg_ratio, neg_cap, net_positive_cap):
    rng = np.random.default_rng(seed_for(seed, g["sample_id"], mode))
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if mode == "net_cap":
        ids, nets = net_ids(g)
        selected = []
        p2n = defaultdict(list)
        for i in positive:
            a, b = (int(v) for v in (int(g["edges"][i]["source"]), int(g["edges"][i]["target"])))
            if ids[a] >= 0 and ids[a] == ids[b]:
                p2n[int(ids[a])].append(int(i))
            else:
                selected.append(int(i))
        for net_idx, indexes in p2n.items():
            if len(indexes) > net_positive_cap:
                selected.extend(rng.choice(indexes, net_positive_cap, replace=False).tolist())
            else:
                selected.extend(indexes)
        positive = np.array(sorted(selected), dtype=np.int64)
    neg_count = min(len(negative), max(16, len(positive) * neg_ratio), neg_cap)
    if mode in ("hard25", "hard50", "net_weight") and scores is not None and len(negative):
        frac = 0.25 if mode == "hard25" else (0.5 if mode == "hard50" else 0.0)
        hard_n = min(len(negative), int(round(neg_count * frac)))
        random_n = min(len(negative) - hard_n, neg_count - hard_n)
        hard = negative[np.argsort(scores[negative])[-hard_n:]] if hard_n else np.array([], dtype=np.int64)
        remaining = np.setdiff1d(negative, hard, assume_unique=False)
        random = rng.choice(remaining, random_n, replace=False) if random_n and len(remaining) > random_n else remaining
        negative = np.concatenate((hard, random))
    elif neg_count < len(negative):
        negative = rng.choice(negative, neg_count, replace=False)
    keep = np.sort(np.concatenate((positive, negative))).astype(np.int64)
    weights = np.ones(len(keep), dtype=np.float32)
    if mode == "net_weight":
        ids, nets = net_ids(g)
        for row, edge_idx in enumerate(keep):
            if y[edge_idx] != 1:
                continue
            edge = g["edges"][int(edge_idx)]
            a, b = int(edge["source"]), int(edge["target"])
            net_idx = int(ids[a])
            if net_idx >= 0 and net_idx == ids[b]:
                # Each net contributes approximately one unit of positive mass.
                weights[row] = 1.0 / max(len(nets[net_idx]) * (len(nets[net_idx]) - 1) / 2, 1)
        pos_mass = weights[y[keep] == 1].sum()
        if pos_mass:
            weights[y[keep] == 1] *= max(int((y[keep] == 1).sum()), 1) / pos_mass
    return keep, weights


def model_metrics(graphs, score_by_id, k_values=(1, 5)):
    from sklearn.metrics import average_precision_score, roc_auc_score
    ys, scores = [], []
    pin_hit = {k: [] for k in k_values}
    pin_precision = {k: [] for k in k_values}
    directed_hit5 = directed_truth5 = 0
    candidate_recalls = []
    for g in graphs:
        # Include truth links missed by candidate generation in the E2E denominator.
        directed_truth5 += 2 * len(g.get("truth_edges", []))
        x, pairs, ef, y, feats = pair_rows(g)
        if not len(y):
            continue
        s = score_by_id[g["sample_id"]]
        ys.append(y); scores.append(s)
        if g.get("candidate_recall") is not None:
            candidate_recalls.append(g["candidate_recall"])
        by_pin = defaultdict(list)
        for i, (a, b) in enumerate(pairs):
            by_pin[int(a)].append(i); by_pin[int(b)].append(i)
        for indexes in by_pin.values():
            positives = int(y[indexes].sum())
            if not positives:
                continue
            ordered = np.asarray(indexes)[np.argsort(-s[indexes], kind="stable")]
            for k in k_values:
                top = ordered[:k]
                hits = int(y[top].sum())
                pin_hit[k].append(float(hits > 0))
                pin_precision[k].append(hits / k)
            top5 = set(map(int, ordered[:5]))
            directed_hit5 += int(y[list(top5)].sum())
    y = np.concatenate(ys); s = np.concatenate(scores)
    out = {"candidate_edges": int(len(y)), "positive_edges": int(y.sum()),
           "pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
           "candidate_recall_mean": float(np.mean(candidate_recalls)) if candidate_recalls else None,
           "candidate_eligible_anchor_pins": len(pin_hit[1])}
    for k in k_values:
        out[f"pin_precision_at_{k}"] = float(np.mean(pin_precision[k])) if pin_precision[k] else None
        out[f"pin_recall_at_{k}"] = float(np.mean(pin_hit[k])) if pin_hit[k] else None
    out["end_to_end_recall_at_5"] = directed_hit5 / directed_truth5 if directed_truth5 else None
    return out


def directional_rank_rows(g, pairs, ef, y, feats, max_neg=50, seed=42):
    # One query per anchor pin. Each undirected candidate becomes a document for
    # both endpoints, matching the pin-level recommendation task.
    anchors = defaultdict(list)
    for i, (a, b) in enumerate(pairs):
        anchors[int(a)].append((int(b), i, 0))
        anchors[int(b)].append((int(a), i, 1))
    rng = np.random.default_rng(seed_for(seed, g["sample_id"], "ranker"))
    rows, labels, groups = [], [], []
    for anchor, docs in anchors.items():
        pos = [d for d in docs if y[d[1]] == 1]
        neg = [d for d in docs if y[d[1]] == 0]
        if not pos:
            continue
        if len(neg) > max_neg:
            neg = [neg[i] for i in rng.choice(len(neg), max_neg, replace=False)]
        chosen = pos + neg
        if len(chosen) < 2:
            continue
        groups.append(len(chosen))
        for target, edge_i, direction in chosen:
            # reverse displacement when the anchor was the original target
            feat = feats[edge_i].copy()
            if direction:
                feat[1] *= -1; feat[2] *= -1
            rows.append(feat); labels.append(int(y[edge_i]))
    return rows, labels, groups


def candidate_edges(nodes, k, radius, adaptive=False):
    coords = np.asarray([[float(n["x"]), float(n["y"])] for n in nodes], dtype=np.float64)
    if len(coords) < 2:
        return set()
    tree = cKDTree(coords)
    counts = np.asarray([len(tree.query_ball_point(p, 50.0)) - 1 for p in coords]) if adaptive else None
    candidate = set()
    for i, p in enumerate(coords):
        ki, ri = k, radius
        if adaptive:
            # Sparse areas get wider search; dense areas remain bounded.
            ki = int(np.clip(round(192 / (1 + math.sqrt(max(0, counts[i])) / 2)), 32, 192))
            ri = min(500.0, radius * (1.0 + 100.0 / (100.0 + counts[i])))
        js = tree.query_ball_point(p, ri)
        js = [j for j in js if j != i and nodes[j]["ref"] != nodes[i]["ref"]]
        js.sort(key=lambda j: float(np.sum((coords[j] - p) ** 2)))
        for j in js[:ki]:
            candidate.add((min(i, j), max(i, j)))
        if str(nodes[i].get("pin_name", "")).strip().casefold() in {"gnd", "ground", "vcc", "vdd", "vin", "vout", "+5v", "+3v3"}:
            js_long = tree.query_ball_point(p, max(1000.0, radius * 4))
            js_long = [j for j in js_long if j != i and nodes[j]["ref"] != nodes[i]["ref"]]
            js_long.sort(key=lambda j: float(np.sum((coords[j] - p) ** 2)))
            for j in js_long[:32]:
                candidate.add((min(i, j), max(i, j)))
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, default=DATA / "pin_candidate_graphs.jsonl")
    parser.add_argument("--output", type=Path, default=DATA / "recommendation_experiments.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=250)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-negatives", type=int, default=1000)
    parser.add_argument("--ranker-max-negatives-per-pin", type=int, default=50)
    parser.add_argument("--gnn-epochs", type=int, default=15)
    parser.add_argument("--graph-batch-size", type=int, default=8)
    parser.add_argument("--limit-train", type=int, default=0, help="debug only; 0 uses all train graphs")
    args = parser.parse_args()
    from lightgbm import LGBMClassifier, LGBMRanker
    from sklearn.utils.class_weight import compute_sample_weight

    progress = Progress(11)
    progress.update("그래프 데이터 읽기", 0, 0.5, str(args.graphs))
    graphs = [json.loads(line) for line in args.graphs.open(encoding="utf-8") if line.strip()]
    train = sorted([g for g in graphs if g["split"] == "train"], key=lambda g: g["sample_id"])
    validation = [g for g in graphs if g["split"] == "validation"]
    test = [g for g in graphs if g["split"] == "test"]
    if args.limit_train:
        train = train[:args.limit_train]
    print(f"graphs train={len(train)} validation={len(validation)} test={len(test)}", flush=True)
    progress.update("데이터 로드 완료", 1, 1, f"train={len(train)}, test={len(test)}")

    # Group-held-out folds produce out-of-fold hard-negative scores.
    groups = sorted({g["group_id"] for g in train})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(groups)
    fold_of = {group: i % args.folds for i, group in enumerate(groups)}
    oof = {}
    for fold in range(args.folds):
        phase_no = fold + 1
        progress.update(f"OOF hard-negative fold {fold+1}/{args.folds}", phase_no, 0.02,
                        "프로젝트 단위로 학습/예측 데이터를 나누는 중")
        xx, yy = [], []
        val_graphs = [g for g in train if fold_of[g["group_id"]] == fold]
        fit_graphs = [g for g in train if fold_of[g["group_id"]] != fold]
        for prep_i, g in enumerate(fit_graphs, start=1):
            _, _, _, y, feats = pair_rows(g)
            if not len(y): continue
            keep = sampled_indices(g, y, None, "base", args.seed, 20, args.max_negatives, 0)[0]
            xx.append(feats[keep]); yy.append(y[keep])
            if prep_i % 100 == 0 or prep_i == len(fit_graphs):
                progress.update(f"OOF hard-negative fold {fold+1}/{args.folds}", phase_no,
                    .02 + .20*prep_i/max(len(fit_graphs),1),
                    f"OOF 학습 feature {prep_i}/{len(fit_graphs)}")
        model = LGBMClassifier(n_estimators=args.trees, learning_rate=.04, num_leaves=31,
            max_depth=-1, subsample=.85, colsample_bytree=.85, reg_lambda=1.,
            random_state=args.seed + fold, n_jobs=8, verbosity=-1)
        yfit = np.concatenate(yy); model.fit(np.concatenate(xx), yfit,
            sample_weight=compute_sample_weight("balanced", yfit),
            callbacks=[lgb_progress(progress, f"OOF hard-negative fold {fold+1}/{args.folds}", phase_no)])
        for pred_i, g in enumerate(val_graphs, start=1):
            _, _, _, _, feats = pair_rows(g)
            if len(feats):
                oof[g["sample_id"]] = model.predict_proba(feats)[:, 1]
            if pred_i % 100 == 0 or pred_i == len(val_graphs):
                progress.update(f"OOF hard-negative fold {fold+1}/{args.folds}", phase_no,
                    .95 + .05*pred_i/max(len(val_graphs),1),
                    f"OOF 점수 {pred_i}/{len(val_graphs)}")
        del xx, yy, model
        print(f"OOF fold {fold+1}/{args.folds}: {len(val_graphs)} held-out schematics", flush=True)

    configs = [("H0_baseline", "base"), ("H1_random75_hard25", "hard25"),
               ("H2_random50_hard50", "hard50"), ("N1_net_weighted", "net_weight"),
               ("N2_net_positive_cap64", "net_cap")]
    score_sets = {}
    score_arrays = {}
    val_score_arrays = {}
    for config_idx, (name, mode) in enumerate(configs, start=4):
        progress.update(name, config_idx, 0.02, "샘플과 가중치를 구성 중")
        xx, yy, ww = [], [], []
        for prep_i, g in enumerate(train, start=1):
            _, _, _, y, feats = pair_rows(g)
            if not len(y): continue
            keep, weights = sampled_indices(g, y, oof.get(g["sample_id"]), mode, args.seed,
                20, args.max_negatives, 64)
            xx.append(feats[keep]); yy.append(y[keep]); ww.append(weights)
            if prep_i % 100 == 0 or prep_i == len(train):
                progress.update(name, config_idx, .02 + .20*prep_i/max(len(train),1),
                    f"샘플 구성 {prep_i}/{len(train)}")
        yfit = np.concatenate(yy)
        model = LGBMClassifier(n_estimators=args.trees, learning_rate=.04, num_leaves=31,
            max_depth=-1, subsample=.85, colsample_bytree=.85, reg_lambda=1.,
            random_state=args.seed, n_jobs=8, verbosity=-1)
        model.fit(np.concatenate(xx), yfit,
            sample_weight=compute_sample_weight("balanced", yfit) * np.concatenate(ww),
            callbacks=[lgb_progress(progress, name, config_idx)])
        scores, val_scores = {}, {}
        for g in validation + test:
            _, _, _, _, feats = pair_rows(g)
            if len(feats):
                target = val_scores if g["split"] == "validation" else scores
                target[g["sample_id"]] = model.predict_proba(feats)[:, 1]
        score_arrays[name] = scores
        val_score_arrays[name] = val_scores
        score_sets[name] = model_metrics(test, scores)
        save_partial(args.output, {"source": "connectivity-dataset-v1", "completed_model": name,
            "training_schematics": len(train), "validation_schematics": len(validation),
            "test_schematics": len(test), "models": score_sets})
        print(f"{name}: PR-AUC={score_sets[name]['pr_auc']:.4f}, P@1={score_sets[name]['pin_precision_at_1']:.4f}, E2E-R@5={score_sets[name]['end_to_end_recall_at_5']:.4f}", flush=True)
        del xx, yy, ww, model

    # LGBMRanker, one query per eligible anchor pin.
    rx, ry, rg = [], [], []
    for prep_i, g in enumerate(train, start=1):
        _, pairs, ef, y, feats = pair_rows(g)
        rows, labels, groupsizes = directional_rank_rows(g, pairs, ef, y, feats,
            args.ranker_max_negatives_per_pin, args.seed)
        rx.extend(rows); ry.extend(labels); rg.extend(groupsizes)
        if prep_i % 100 == 0 or prep_i == len(train):
            progress.update("LightGBM Ranker", 9, .02 + .20*prep_i/max(len(train),1),
                f"anchor-query feature {prep_i}/{len(train)}")
    ranker = LGBMRanker(objective="lambdarank", metric="ndcg", n_estimators=args.trees,
        learning_rate=.04, num_leaves=31, max_depth=-1, subsample=.85,
        colsample_bytree=.85, reg_lambda=1., random_state=args.seed, n_jobs=8, verbosity=-1)
    progress.update("LightGBM Ranker", 9, 0.02, f"anchor-query 수={len(rg):,}")
    ranker.fit(np.asarray(rx, dtype=np.float32), np.asarray(ry), group=rg,
        callbacks=[lgb_progress(progress, "LightGBM Ranker", 9)])
    rank_scores, rank_val_scores = {}, {}
    for g in validation + test:
        x, pairs, ef, y, feats = pair_rows(g)
        if len(feats):
            target = rank_val_scores if g["split"] == "validation" else rank_scores
            target[g["sample_id"]] = ranker.predict(feats)
    score_arrays["B_lightgbm_ranker"] = rank_scores
    val_score_arrays["B_lightgbm_ranker"] = rank_val_scores
    score_sets["B_lightgbm_ranker"] = model_metrics(test, rank_scores)
    save_partial(args.output, {"source": "connectivity-dataset-v1", "completed_model": "B_lightgbm_ranker",
        "training_schematics": len(train), "validation_schematics": len(validation),
        "test_schematics": len(test), "models": score_sets})
    print(f"B_lightgbm_ranker: PR-AUC={score_sets['B_lightgbm_ranker']['pr_auc']:.4f}, P@1={score_sets['B_lightgbm_ranker']['pin_precision_at_1']:.4f}, E2E-R@5={score_sets['B_lightgbm_ranker']['end_to_end_recall_at_5']:.4f}", flush=True)

    # Train the graph model on CUDA and retain predictions for validation-based
    # rank-normalized LGBM/GNN ensemble selection.
    import torch
    from torch import nn
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the requested GPU run, but torch.cuda.is_available() is false")
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    progress.update("GPU GNN 학습", 10, 0.02,
                    f"{torch.cuda.get_device_name(0)}; 그래프 feature 구성 중")
    gnn_train = []
    for gi, g in enumerate(train, start=1):
        x, pairs, ef, y, _ = pair_rows(g)
        if len(y):
            gnn_train.append((x.astype(np.float32), pairs, ef, y))
        if gi % 100 == 0 or gi == len(train):
            progress.update("GPU GNN 학습", 10, 0.02 + 0.12 * gi / len(train),
                f"학습 회로 feature {gi}/{len(train)}")
    node_dim, edge_dim = gnn_train[0][0].shape[1], gnn_train[0][2].shape[1]
    class EdgeGNN(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = 48
            self.encoder = nn.Linear(node_dim, hidden)
            self.msg1 = nn.Linear(hidden * 2, hidden)
            self.msg2 = nn.Linear(hidden * 2, hidden)
            self.edge_head = nn.Sequential(nn.Linear(hidden * 2 + edge_dim, hidden), nn.ReLU(),
                nn.Dropout(.15), nn.Linear(hidden, 1))
        def forward(self, x, edges, edge_attr):
            h = torch.relu(self.encoder(x))
            for layer in (self.msg1, self.msg2):
                src, dst = edges[:, 0], edges[:, 1]
                agg = torch.zeros_like(h)
                agg.index_add_(0, dst, h[src]); agg.index_add_(0, src, h[dst])
                deg = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
                ones = torch.ones((len(src), 1), dtype=h.dtype, device=h.device)
                deg.index_add_(0, dst, ones); deg.index_add_(0, src, ones)
                h = torch.relu(layer(torch.cat([h, agg / deg.clamp_min(1)], dim=1)))
            return self.edge_head(torch.cat([h[edges[:, 0]], h[edges[:, 1]], edge_attr], dim=1)).squeeze(-1)
    def combine(rows):
        xs, es, attrs, labels = [], [], [], []
        offset = 0
        for x, pairs, ef, y in rows:
            xs.append(x); es.append(pairs + offset); attrs.append(ef); labels.append(y)
            offset += len(x)
        return np.concatenate(xs), np.concatenate(es), np.concatenate(attrs), np.concatenate(labels)
    gnn = EdgeGNN().to(device)
    optimizer = torch.optim.AdamW(gnn.parameters(), lr=.001, weight_decay=1e-4)
    total_pos = sum(int(row[3].sum()) for row in gnn_train)
    total_edges = sum(len(row[3]) for row in gnn_train)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        [(total_edges-total_pos)/max(total_pos,1)], dtype=torch.float32, device=device))
    epochs = args.gnn_epochs
    for epoch in range(epochs):
        order = np.random.default_rng(args.seed + epoch).permutation(len(gnn_train))
        total_loss, seen = 0.0, 0
        for start in range(0, len(order), args.graph_batch_size):
            rows = [gnn_train[int(i)] for i in order[start:start+args.graph_batch_size]]
            bx, be, ba, by = combine(rows)
            tx = torch.as_tensor(bx, device=device); te = torch.as_tensor(be, device=device)
            ta = torch.as_tensor(ba, device=device); ty = torch.as_tensor(by, dtype=torch.float32, device=device)
            gnn.train(); optimizer.zero_grad(set_to_none=True)
            logits = gnn(tx, te, ta); loss = loss_fn(logits, ty)
            loss.backward(); optimizer.step()
            total_loss += float(loss.detach()) * len(by); seen += len(by)
        progress.update("GPU GNN 학습", 10, 0.14 + 0.72 * (epoch+1)/epochs,
            f"epoch {epoch+1}/{epochs}; loss={total_loss/max(seen,1):.4f}; VRAM={torch.cuda.memory_allocated()/1024**2:.0f} MiB")
        print(f"GPU GNN epoch {epoch+1}/{epochs} loss={total_loss/max(seen,1):.4f}", flush=True)
    gnn.eval()
    gnn_test, gnn_val = {}, {}
    with torch.no_grad():
        for gi, g in enumerate(validation + test, start=1):
            x, pairs, ef, y, _ = pair_rows(g)
            if not len(y): continue
            logits = gnn(torch.as_tensor(x, device=device), torch.as_tensor(pairs, device=device),
                         torch.as_tensor(ef, device=device))
            scores = torch.sigmoid(logits).cpu().numpy()
            target = gnn_val if g["split"] == "validation" else gnn_test
            target[g["sample_id"]] = scores
            if gi % 50 == 0 or gi == len(validation) + len(test):
                progress.update("GPU GNN 평가", 10, .86 + .13*gi/(len(validation)+len(test)),
                    f"validation/test graph {gi}/{len(validation)+len(test)}")
    score_arrays["GNN_CUDA"] = gnn_test; val_score_arrays["GNN_CUDA"] = gnn_val
    score_sets["GNN_CUDA"] = model_metrics(test, gnn_test)
    save_partial(args.output, {"source": "connectivity-dataset-v1", "completed_model": "GNN_CUDA",
        "training_schematics": len(train), "validation_schematics": len(validation),
        "test_schematics": len(test), "models": score_sets})
    print(f"GNN_CUDA: PR-AUC={score_sets['GNN_CUDA']['pr_auc']:.4f}, P@1={score_sets['GNN_CUDA']['pin_precision_at_1']:.4f}, E2E-R@5={score_sets['GNN_CUDA']['end_to_end_recall_at_5']:.4f}", flush=True)

    # Calibrate the ensemble mix on validation only. Convert each model's score
    # to endpoint-local percentile ranks before averaging to avoid scale mismatch.
    def endpoint_rank(g, scores):
        _, pairs, _, _, _ = pair_rows(g)
        out = np.zeros(len(pairs), dtype=np.float32)
        edge_lists = defaultdict(list)
        for i, (a,b) in enumerate(pairs):
            edge_lists[int(a)].append(i); edge_lists[int(b)].append(i)
        accum = np.zeros(len(pairs), dtype=np.float64); counts = np.zeros(len(pairs), dtype=np.float64)
        for inds in edge_lists.values():
            inds = np.asarray(inds); order = np.argsort(scores[inds], kind="stable")
            ranks = np.empty(len(inds), dtype=np.float64); ranks[order] = np.linspace(0,1,len(inds))
            accum[inds] += ranks; counts[inds] += 1
        return (accum / np.maximum(counts, 1)).astype(np.float32)
    alphas = (.25, .5, .75, .9)
    ensemble_validation = {}
    for alpha in alphas:
        blended = {}
        for g in validation:
            sid = g["sample_id"]
            if sid in val_score_arrays["H0_baseline"] and sid in gnn_val:
                blended[sid] = alpha*endpoint_rank(g,val_score_arrays["H0_baseline"][sid]) + (1-alpha)*endpoint_rank(g,gnn_val[sid])
        ensemble_validation[str(alpha)] = model_metrics(validation, blended)
    best_alpha = max(alphas, key=lambda a: ensemble_validation[str(a)]["pin_precision_at_1"] or -1)
    ensemble_test = {}
    for alpha in alphas:
        blended = {}
        for g in test:
            sid = g["sample_id"]
            if sid in score_arrays["H0_baseline"] and sid in gnn_test:
                blended[sid] = alpha*endpoint_rank(g,score_arrays["H0_baseline"][sid]) + (1-alpha)*endpoint_rank(g,gnn_test[sid])
        ensemble_test[str(alpha)] = model_metrics(test, blended)
    score_sets["E_lgbm_gnn_ensemble"] = {"validation_alpha_selection": ensemble_validation,
        "selected_alpha_by_validation_pin_precision_at_1": best_alpha,
        "test_by_alpha": ensemble_test, "selected_test_metrics": ensemble_test[str(best_alpha)]}
    save_partial(args.output, {"source": "connectivity-dataset-v1", "completed_model": "ensemble",
        "training_schematics": len(train), "validation_schematics": len(validation),
        "test_schematics": len(test), "models": score_sets})
    model_path = ROOT / "models/phase22_connectivity/recommendation_gnn_cuda.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": gnn.cpu().state_dict(), "node_dim": node_dim, "edge_dim": edge_dim,
        "hidden_dim": 48, "seed": args.seed, "epochs": epochs},
        model_path)
    del gnn_train
    print(f"E_lgbm_gnn_ensemble selected alpha={best_alpha:.2f}; test P@1={ensemble_test[str(best_alpha)]['pin_precision_at_1']:.4f}", flush=True)

    # Adaptive candidate-generation recall / density audit, separate from model scores.
    fixed_stats, adaptive_stats = [], []
    for gi, g in enumerate(test, start=1):
        if gi % 10 == 0 or gi == len(test):
            progress.update("Adaptive 후보 생성 평가", 11, gi / max(len(test), 1),
                f"test graph {gi}/{len(test)}")
        truth = {tuple(sorted(map(int, e))) for e in g.get("truth_edges", [])}
        fixed = candidate_edges(g["nodes"], 96, 250.0, adaptive=False)
        adaptive = candidate_edges(g["nodes"], 96, 250.0, adaptive=True)
        if truth:
            fixed_stats.append((len(truth & fixed) / len(truth), len(fixed)))
            adaptive_stats.append((len(truth & adaptive) / len(truth), len(adaptive)))
    result = {"source": "connectivity-dataset-v1", "seed": args.seed,
        "training_schematics": len(train), "validation_schematics": len(validation),
        "test_schematics": len(test), "oof_folds": args.folds,
        "candidate_recall_micro": {
            split: (sum(sum(int(e["label"]) for e in g["edges"]) for g in graphs if g["split"] == split)
                    / max(sum(len(g.get("truth_edges", [])) for g in graphs if g["split"] == split), 1))
            for split in ("train", "validation", "test")},
        "metrics_definition": {
            "pin_precision_at_k": "macro mean over anchor pins with at least one candidate positive in the generated set; hits among top K divided by K",
            "pin_recall_at_k": "macro fraction of anchor pins with at least one candidate positive that place any positive in top K",
            "end_to_end_recall_at_5": "micro recall of all directed truth links, including links omitted by candidate generation, among each anchor pin's top five candidates",
            "candidate_recall_mean": "mean per-schematic truth-pair recall before ranking"},
        "models": score_sets,
        "D_adaptive_candidates": {
            "fixed_k96_radius250": {"mean_candidate_recall": float(np.mean([x[0] for x in fixed_stats])),
                "mean_edges": float(np.mean([x[1] for x in fixed_stats]))},
            "adaptive_density_k_radius_plus_longrange_power_hint": {
                "mean_candidate_recall": float(np.mean([x[0] for x in adaptive_stats])),
                "mean_edges": float(np.mean([x[1] for x in adaptive_stats]))}},
        "limitations": ["H3 is represented by three-fold project-group out-of-fold hard-negative mining in H1/H2; no self-scored negatives are used.",
            "Net-aware labels are recovered from archived truth-edge cliques; N1 equalizes positive mass approximately per net and N2 caps sampled positives at 64 per net.",
            "Adaptive search is a candidate-recall audit using archived pin coordinates; no power/ground net metadata is available, so long-range search uses only common pin-name hints.",
            "LightGBM classifier and ranker use CPU because the installed Windows LightGBM wheel has no GPU learner; the graph model trains on CUDA." ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    progress.update("완료", 11, 1, str(args.output))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
