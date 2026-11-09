#!/usr/bin/env python3
"""Train PCB-only pin-link models and evaluate candidate/net reconstruction."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
PIN_DATA = ROOT / "data/external/pcb-photo-cad-2026-11-09/pin_only_training"
BENCH = ROOT / "data/benchmarks/pcb_pin_connectivity"
PROGRESS_FILE = ROOT / "data/benchmarks/phase22_connectivity/recommendation_progress.json"
OUTPUT = BENCH / "pcb_pin_connectivity_results.json"
TAXONOMY = ("IC", "LED", "battery", "buzzer", "capacitor", "connector", "diode", "display",
            "emi_filter", "ferrite_bead", "fuse", "inductor", "jumper", "oscillator",
            "potentiometer", "relay", "resistor", "switch", "transistor", "unknown")


class Progress:
    def __init__(self, count):
        self.count, self.started = count, time.time()
    def update(self, phase, number, fraction, detail=""):
        payload = {"status": "완료" if phase == "완료" else "실행 중", "phase": phase,
            "phase_number": number, "phase_count": self.count, "phase_percent": round(100*fraction, 1),
            "overall_percent": 100.0 if phase == "완료" else 100*(number-1+fraction)/self.count,
            "elapsed_seconds": round(time.time()-self.started, 1), "detail": detail,
            "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")}
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = PROGRESS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        try: os.replace(tmp, PROGRESS_FILE)
        except PermissionError: PROGRESS_FILE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"[{number}/{self.count} {fraction:.0%}] {phase}: {detail}", flush=True)


def candidate_pairs(nodes, radius_mm, k_nearest):
    if len(nodes) < 2: return []
    coords = np.asarray([[n["x_mm"], n["y_mm"]] for n in nodes], dtype=np.float64)
    refs = [n["reference"] for n in nodes]
    tree = cKDTree(coords)
    pairs = set()
    for i, j in tree.query_pairs(radius_mm, output_type="set"):
        if refs[i] != refs[j]: pairs.add((min(i, j), max(i, j)))
    qk = min(k_nearest+1, len(nodes))
    _, nearest = tree.query(coords, k=qk)
    if qk == 1: nearest = nearest[:, None]
    for i, row in enumerate(nearest):
        for j in row[1:]:
            j = int(j)
            if j != i and refs[i] != refs[j]: pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def load_raw():
    labels = {x["graph_id"]: x for x in
              (json.loads(line) for line in (PIN_DATA / "edge_labels.jsonl").open(encoding="utf-8"))}
    rows = []
    for line in (PIN_DATA / "graphs.jsonl").open(encoding="utf-8"):
        graph = json.loads(line)
        truth = labels.get(graph["graph_id"])
        if truth is None: raise ValueError(f"missing labels for {graph['graph_id']}")
        nodes = graph["nodes"]
        by_id = {node["id"]: i for i, node in enumerate(nodes)}
        truth_pairs = set()
        truth_net_by_pair = {}
        for item in truth["truth_edges"]:
            pair = tuple(sorted((by_id[item["source"]], by_id[item["target"]])))
            truth_pairs.add(pair); truth_net_by_pair[pair] = int(item["truth_net_id"])
        truth_nets = [[by_id[node_id] for node_id in group["members"] if node_id in by_id]
                      for group in truth.get("truth_nets", [])]
        rows.append({"graph_id": graph["graph_id"], "repository": graph["source_repository"],
            "split": graph["split"], "nodes_raw": nodes, "truth_pairs": truth_pairs,
            "truth_net_by_pair": truth_net_by_pair, "truth_nets": truth_nets})
    return rows


def candidate_audit(rows, configs):
    reports = []
    for radius, k in configs:
        stats = {split: {"truth": 0, "covered": 0, "candidates": 0, "boards": 0} for split in ("train", "validation")}
        for row in rows:
            if row["split"] not in stats: continue
            pairs = candidate_pairs(row["nodes_raw"], radius, k)
            y = sum(1 for p in pairs if p in row["truth_pairs"])
            s = stats[row["split"]]; s["truth"] += len(row["truth_pairs"]); s["covered"] += y
            s["candidates"] += len(pairs); s["boards"] += 1
        val = stats["validation"]
        reports.append({"radius_mm": radius, "k_nearest": k,
            "validation_candidate_recall_micro": val["covered"]/val["truth"] if val["truth"] else None,
            "validation_candidate_recall_macro": float(np.mean([
                sum(p in row["truth_pairs"] for p in candidate_pairs(row["nodes_raw"], radius, k))/len(row["truth_pairs"])
                for row in rows if row["split"] == "validation" and row["truth_pairs"]])) if any(row["split"] == "validation" and row["truth_pairs"] for row in rows) else None,
            "validation_mean_candidate_degree": 2*val["candidates"]/max(sum(len(r["nodes_raw"]) for r in rows if r["split"] == "validation"), 1),
            "validation_mean_undirected_edges_per_pin": val["candidates"]/max(sum(len(r["nodes_raw"]) for r in rows if r["split"] == "validation"), 1),
            "train_candidate_recall_micro": stats["train"]["covered"]/stats["train"]["truth"] if stats["train"]["truth"] else None})
    return reports


def node_features(nodes):
    count = len(nodes)
    coords = np.asarray([[n["x_mm"], n["y_mm"]] for n in nodes], dtype=np.float32)
    if count:
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        span = np.maximum(hi-lo, .1)
        pos = (coords-lo)/span
    else:
        lo, span, pos = np.zeros(2), np.ones(2), coords
    diag = max(float(np.linalg.norm(span)), 1.0)
    types = np.zeros((count, len(TAXONOMY)), dtype=np.float32)
    cont = np.zeros((count, 10), dtype=np.float32)
    refs = []
    for i, node in enumerate(nodes):
        typ = node.get("component_type", "unknown")
        types[i, TAXONOMY.index(typ) if typ in TAXONOMY else TAXONOMY.index("unknown")] = 1
        cx = (float(node.get("component_x_mm", node["x_mm"]))-lo[0])/span[0]
        cy = (float(node.get("component_y_mm", node["y_mm"]))-lo[1])/span[1]
        angle = math.radians(float(node.get("rotation_deg", 0)))
        size = node.get("pad_size_mm") or [0, 0]
        try: pad_num = min(abs(float(node.get("pad_number", 0))), 1000)/1000
        except (ValueError, TypeError): pad_num = 0.0
        cont[i] = [pos[i,0], pos[i,1], cx, cy, math.sin(angle), math.cos(angle),
                   float(node.get("side") == "B"), min(abs(float(size[0])), 20)/20,
                   min(abs(float(size[1])), 20)/20, pad_num]
        refs.append(node["reference"])
    return np.concatenate([types, cont], axis=1), coords, refs, diag


def prepare_graph(row, radius, k):
    nodes = row["nodes_raw"]
    x, coords, refs, diag = node_features(nodes)
    pairs = candidate_pairs(nodes, radius, k)
    eidx = np.asarray(pairs, dtype=np.int64).reshape((-1, 2))
    edge_geom = np.zeros((len(pairs), 6), dtype=np.float32)
    y = np.zeros(len(pairs), dtype=np.int8)
    net_id = np.full(len(pairs), -1, dtype=np.int32)
    for i, (a, b) in enumerate(pairs):
        dx, dy = coords[b]-coords[a]
        dist = float(math.hypot(float(dx), float(dy)))
        edge_geom[i] = [dx/diag, dy/diag, abs(dx)/diag, abs(dy)/diag,
                        min(dist/diag, 2), min(math.log1p(dist)/math.log1p(diag), 2)]
        if (a, b) in row["truth_pairs"]:
            y[i] = 1; net_id[i] = row["truth_net_by_pair"][(a, b)]
    return {**row, "x": x, "coords": coords, "refs": refs, "diag": diag,
        "pairs": eidx, "edge_geom": edge_geom, "y": y, "net_id": net_id,
        "candidate_recall": int(y.sum())/len(row["truth_pairs"]) if row["truth_pairs"] else None}


def edge_features(g, indices=None):
    if indices is None: indices = np.arange(len(g["pairs"]))
    pairs = g["pairs"][indices]
    a, b = g["x"][pairs[:,0]], g["x"][pairs[:,1]]
    return np.concatenate([g["edge_geom"][indices], a+b, np.abs(a-b)], axis=1).astype(np.float32)


def deterministic_rng(seed, graph_id, purpose):
    h = hashlib.sha256(f"{seed}:{graph_id}:{purpose}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def sample_training(g, seed, neg_ratio=4, neg_cap=10000):
    positive = np.flatnonzero(g["y"] == 1)
    negative = np.flatnonzero(g["y"] == 0)
    if len(negative):
        desired = min(len(negative), max(128, len(positive)*neg_ratio), neg_cap)
        rng = deterministic_rng(seed, g["graph_id"], "negative-pairs")
        negative = rng.choice(negative, desired, replace=False) if desired < len(negative) else negative
    keep = np.sort(np.concatenate([positive, negative])).astype(np.int64)
    if not len(keep): return keep, np.empty(0, dtype=np.float32)
    weights = np.zeros(len(keep), dtype=np.float32)
    pos_rows = np.flatnonzero(g["y"][keep] == 1)
    if len(pos_rows):
        by_net = defaultdict(list)
        for row in pos_rows:
            by_net[int(g["net_id"][keep[row]])].append(int(row))
        for indexes in by_net.values():
            weights[indexes] = 1/max(len(indexes), 1)
        weights[pos_rows] /= max(float(weights[pos_rows].sum()), 1e-9)
    neg_rows = np.flatnonzero(g["y"][keep] == 0)
    if len(neg_rows): weights[neg_rows] = 1/len(neg_rows)
    return keep, weights


def truth_degree_metrics(graphs, scores, k=5):
    all_truth = candidate_hits = 0
    e2e_hit = e2e_truth = oracle_hit = 0
    pin_recall, pin_precision = [], []
    for g in graphs:
        candidate_positive = set(tuple(p) for p in g["pairs"][g["y"] == 1].tolist())
        all_truth += len(g["truth_pairs"]); candidate_hits += len(candidate_positive & g["truth_pairs"])
        incident = defaultdict(list)
        for i, (a,b) in enumerate(g["pairs"]):
            incident[int(a)].append(i); incident[int(b)].append(i)
        truth_by_pin = defaultdict(set)
        for a,b in g["truth_pairs"]:
            truth_by_pin[a].add(b); truth_by_pin[b].add(a)
        for pin, truths in truth_by_pin.items():
            idx = np.asarray(incident.get(pin, []), dtype=np.int64)
            if not len(idx):
                pin_recall.append(0.0); pin_precision.append(0.0); e2e_truth += len(truths); continue
            pred = scores[g["graph_id"]]
            order = idx[np.argsort(-pred[idx], kind="stable")[:k]]
            hit = 0
            for ei in order:
                a,b = g["pairs"][ei]
                if (int(a),int(b)) in g["truth_pairs"]: hit += 1
            candidate_truths = sum(1 for ei in idx if g["y"][ei] == 1)
            oracle_hit += min(k, candidate_truths)
            e2e_hit += hit; e2e_truth += len(truths)
            pin_recall.append(hit/len(truths)); pin_precision.append(hit/k)
    return {"candidate_recall_micro": candidate_hits/all_truth if all_truth else None,
        "truth_edges_all": all_truth, "candidate_truth_edges": candidate_hits,
        f"end_to_end_recall_at_{k}_micro": e2e_hit/e2e_truth if e2e_truth else None,
        f"end_to_end_recall_at_{k}_macro": float(np.mean(pin_recall)) if pin_recall else None,
        f"oracle_recall_at_{k}_micro": oracle_hit/e2e_truth if e2e_truth else None,
        f"pin_precision_at_{k}_macro": float(np.mean(pin_precision)) if pin_precision else None,
        "eligible_anchor_pins": len(pin_recall)}


def pairwise_clusters(g, scores, threshold):
    parent = list(range(len(g["nodes_raw"])))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    for i,(a,b) in enumerate(g["pairs"]):
        if scores[g["graph_id"]][i] >= threshold: union(int(a),int(b))
    clusters=defaultdict(list)
    for i in range(len(parent)): clusters[find(i)].append(i)
    predicted=set()
    for members in clusters.values():
        for pos,a in enumerate(members):
            for b in members[pos+1:]:
                if g["refs"][a]!=g["refs"][b]: predicted.add((min(a,b),max(a,b)))
    return predicted, sum(len(x)>1 for x in clusters.values())


def net_pair_metrics(graphs, scores, threshold):
    tp=fp=fn=clusters=0
    for g in graphs:
        pred,nc=pairwise_clusters(g,scores,threshold); truth=g["truth_pairs"]
        tp+=len(pred & truth);fp+=len(pred-truth);fn+=len(truth-pred);clusters+=nc
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
    return {"threshold":float(threshold),"net_pair_precision":precision,"net_pair_recall":recall,
        "net_pair_f1":2*precision*recall/max(precision+recall,1e-12),"predicted_nontrivial_clusters":clusters,
        "true_cross_component_pairs":tp+fn,"predicted_transitive_pairs":tp+fp}


def threshold_search(validation, scores):
    values=np.concatenate([scores[g["graph_id"]] for g in validation if len(scores.get(g["graph_id"],[]))])
    if not len(values): return .5, []
    grid=sorted(set(float(x) for x in np.quantile(values,[.5,.7,.8,.9,.95,.97,.99,.995,.999])))
    rows=[net_pair_metrics(validation,scores,t) for t in grid]
    best=max(rows,key=lambda r:(r["net_pair_f1"],r["net_pair_precision"]))
    return best["threshold"],rows


def evaluate(graphs, scores, threshold):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y=np.concatenate([g["y"] for g in graphs if len(g["y"])])
    s=np.concatenate([scores[g["graph_id"]] for g in graphs if len(g["y"])])
    net=net_pair_metrics(graphs,scores,threshold)
    rank=truth_degree_metrics(graphs,scores,5)
    return {"candidate_pr_auc":float(average_precision_score(y,s)),"candidate_roc_auc":float(roc_auc_score(y,s)),**rank,"net_reconstruction":net}


def save_partial(data, stage):
    BENCH.mkdir(parents=True,exist_ok=True)
    payload={"stage":stage,"updated_at":datetime.now(timezone.utc).isoformat(),"results":data}
    tmp=OUTPUT.with_suffix(".partial.tmp");tmp.write_text(json.dumps(payload,indent=2),encoding="utf-8");os.replace(tmp,OUTPUT.with_suffix(".partial.json"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--trees",type=int,default=350)
    p.add_argument("--gnn-epochs",type=int,default=15);p.add_argument("--graph-batch-size",type=int,default=4)
    p.add_argument("--limit-train",type=int,default=0,help="debug only; default uses every PCB train graph")
    args=p.parse_args(); progress=Progress(9);BENCH.mkdir(parents=True,exist_ok=True)
    progress.update("PCB 핀 전용 데이터 읽기",1,.1,str(PIN_DATA))
    rows=load_raw()
    train_raw=[r for r in rows if r["split"]=="train"]
    val_raw=[r for r in rows if r["split"]=="validation"]
    test_raw=[r for r in rows if r["split"]=="test"]
    if args.limit_train: train_raw=train_raw[:args.limit_train]
    progress.update("후보 생성 audit",2,.05,"검증 프로젝트로 반경/k 설정을 비교 중")
    configs=[(25.,8),(50.,16),(100.,32),(150.,64)]
    audit=candidate_audit(rows,configs)
    # Choose on validation only: keep the smallest graph within 0.5 percentage
    # points of the best validation recall rather than making every pin pair a candidate.
    best_recall=max((a["validation_candidate_recall_micro"] or 0) for a in audit)
    eligible=[a for a in audit if (a["validation_candidate_recall_micro"] or 0)>=best_recall-.005
              and a["validation_mean_candidate_degree"]<=250]
    choice=min(eligible or audit,key=lambda a:(a["validation_mean_candidate_degree"],
                                                -(a["validation_candidate_recall_micro"] or 0)))
    radius,k=float(choice["radius_mm"]),int(choice["k_nearest"])
    progress.update("PCB 후보 그래프 구성",3,.05,f"선택: radius={radius:g}mm, k={k}; 검증 candidate recall={choice['validation_candidate_recall_micro']:.3f}")
    graphs=[prepare_graph(r,radius,k) for r in rows]
    train=[g for g in graphs if g["split"]=="train"]
    if args.limit_train: train=train[:args.limit_train]
    validation=[g for g in graphs if g["split"]=="validation"]
    test=[g for g in graphs if g["split"]=="test"]
    summaries={"source":"pcb_pin_only_training only; prior schematic/Phase 22 training records excluded",
      "input_manifest":str((PIN_DATA/"dataset_manifest.json").relative_to(ROOT)).replace("\\","/"),
      "pcb_graph_count":len(graphs),"train_graphs":len(train),"validation_graphs":len(validation),"test_graphs":len(test),
      "candidate_audit_validation":audit,"selected_candidate":{"radius_mm":radius,"k_nearest":k},
      "same_component_policy":"Excluded consistently from candidate edges, candidate labels, truth edges, and ranking anchors.",
      "target_policy":"same original net across different components; this is net membership, not a literal single copper segment.",
      "candidate_recall_train":truth_degree_metrics(train,{g["graph_id"]:np.zeros(len(g["pairs"])) for g in train},5)["candidate_recall_micro"],
      "candidate_recall_validation":truth_degree_metrics(validation,{g["graph_id"]:np.zeros(len(g["pairs"])) for g in validation},5)["candidate_recall_micro"],
      "candidate_recall_test":truth_degree_metrics(test,{g["graph_id"]:np.zeros(len(g["pairs"])) for g in test},5)["candidate_recall_micro"]}
    save_partial(summaries,"candidate generation")
    progress.update("LightGBM PCB 학습 feature 구성",4,.05,f"train={len(train)}, validation={len(validation)}, test={len(test)}")
    Xs,ys,ws=[],[],[]
    for gi,g in enumerate(train,1):
        idx,w=sample_training(g,args.seed)
        if len(idx):Xs.append(edge_features(g,idx));ys.append(g["y"][idx]);ws.append(w)
        if gi%25==0 or gi==len(train):progress.update("LightGBM PCB 학습 feature 구성",4,.05+.90*gi/max(len(train),1),f"회로 {gi}/{len(train)}")
    X=np.concatenate(Xs);y=np.concatenate(ys);weight=np.concatenate(ws)
    progress.update("LightGBM CPU 기준선 학습",5,.05,f"CUDA GPU는 GNN에 사용; LightGBM Windows wheel은 CPU learner; examples={len(y):,}")
    from lightgbm import LGBMClassifier, log_evaluation
    lgb=LGBMClassifier(n_estimators=args.trees,learning_rate=.04,num_leaves=31,max_depth=-1,
        subsample=.85,colsample_bytree=.85,reg_lambda=1.,random_state=args.seed,n_jobs=6,verbosity=-1)
    lgb.fit(X,y,sample_weight=weight,callbacks=[log_evaluation(period=0)])
    model_dir=BENCH/"models";model_dir.mkdir(parents=True,exist_ok=True)
    lgb_path=model_dir/"lightgbm_pcb_pin_only.txt";lgb.booster_.save_model(str(lgb_path))
    def score_lgb(graph_list, phase_no):
        scored={}
        for i,g in enumerate(graph_list,1):
            feats=edge_features(g)
            scored[g["graph_id"]]=lgb.predict_proba(feats)[:,1] if len(feats) else np.zeros(0,dtype=np.float32)
            if i%4==0 or i==len(graph_list):progress.update("LightGBM 평가",phase_no,.05+.9*i/max(len(graph_list),1),f"회로 {i}/{len(graph_list)}")
        return scored
    lgb_val=score_lgb(validation,6);lgb_test=score_lgb(test,6)
    lgb_threshold,lgb_val_thresholds=threshold_search(validation,lgb_val)
    lgb_result={"training_device":"CPU (LightGBM wheel has no CUDA learner)","training_examples":int(len(y)),
        "validation":evaluate(validation,lgb_val,lgb_threshold),"test":evaluate(test,lgb_test,lgb_threshold),
        "net_threshold_calibration_validation":lgb_val_thresholds}
    summaries["LightGBM"]=lgb_result
    summaries.setdefault("model_artifacts",{})["LightGBM"]=str(lgb_path.relative_to(ROOT)).replace("\\","/")
    save_partial(summaries,"LightGBM complete")
    progress.update("GPU GNN 초기화",7,.02,"CUDA 모델과 PCB 그래프 feature 준비")
    import torch
    from torch import nn
    if not torch.cuda.is_available():raise RuntimeError("CUDA is required, but torch.cuda.is_available() is false")
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);torch.set_num_threads(4)
    device=torch.device("cuda:0");node_dim=train[0]["x"].shape[1];edge_dim=train[0]["edge_geom"].shape[1]
    class PCBEdgeGNN(nn.Module):
        def __init__(self):
            super().__init__();h=48
            self.enc=nn.Linear(node_dim,h);self.msg1=nn.Linear(h*2,h);self.msg2=nn.Linear(h*2,h)
            self.head=nn.Sequential(nn.Linear(h*2+edge_dim,96),nn.ReLU(),nn.Dropout(.15),nn.Linear(96,1))
        def forward(self,x,all_edges,query_edges,edge_attr):
            h=torch.relu(self.enc(x));src,dst=all_edges[:,0],all_edges[:,1]
            for layer in (self.msg1,self.msg2):
                agg=torch.zeros_like(h);deg=torch.zeros((len(h),1),device=h.device,dtype=h.dtype)
                agg.index_add_(0,dst,h[src]);agg.index_add_(0,src,h[dst])
                one=torch.ones((len(src),1),device=h.device,dtype=h.dtype)
                deg.index_add_(0,dst,one);deg.index_add_(0,src,one)
                h=torch.relu(layer(torch.cat([h,agg/deg.clamp_min(1)],dim=1)))
            a,b=h[query_edges[:,0]],h[query_edges[:,1]]
            return self.head(torch.cat([a+b,torch.abs(a-b),edge_attr],dim=1)).squeeze(-1)
    train_samples=[]
    for g in train:
        idx,w=sample_training(g,args.seed)
        train_samples.append((g,idx,w))
    model=PCBEdgeGNN().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    progress.update("GPU GNN 학습",7,.05,f"{torch.cuda.get_device_name(0)}; epochs={args.gnn_epochs}; batch={args.graph_batch_size}")
    for epoch in range(args.gnn_epochs):
        order=np.random.default_rng(args.seed+epoch).permutation(len(train_samples));loss_sum=0.;n_seen=0
        for start in range(0,len(order),args.graph_batch_size):
            batch=[train_samples[int(i)] for i in order[start:start+args.graph_batch_size]]
            xs=[];all_edges=[];queries=[];edge_attrs=[];ys=[];weights=[];offset=0
            for g,idx,w in batch:
                xs.append(g["x"]);all_edges.append(g["pairs"]+offset);queries.append(g["pairs"][idx]+offset)
                edge_attrs.append(g["edge_geom"][idx]);ys.append(g["y"][idx]);weights.append(w);offset+=len(g["x"])
            bx=torch.as_tensor(np.concatenate(xs),device=device);be=torch.as_tensor(np.concatenate(all_edges),device=device)
            qe=torch.as_tensor(np.concatenate(queries),device=device);ea=torch.as_tensor(np.concatenate(edge_attrs),device=device)
            by=torch.as_tensor(np.concatenate(ys),dtype=torch.float32,device=device);bw=torch.as_tensor(np.concatenate(weights),device=device)
            model.train();opt.zero_grad(set_to_none=True);logits=model(bx,be,qe,ea)
            loss=(nn.functional.binary_cross_entropy_with_logits(logits,by,reduction="none")*bw).sum()/len(batch)
            loss.backward();opt.step();loss_sum+=float(loss.detach());n_seen+=1
        progress.update("GPU GNN 학습",7,.08+.82*(epoch+1)/args.gnn_epochs,
            f"epoch {epoch+1}/{args.gnn_epochs}; weighted loss={loss_sum/max(n_seen,1):.4f}; VRAM={torch.cuda.memory_allocated()/1024**2:.0f} MiB")
    def score_gnn(graph_list, phase_no):
        result={};model.eval()
        with torch.no_grad():
            for i,g in enumerate(graph_list,1):
                if len(g["pairs"]):
                    logits=model(torch.as_tensor(g["x"],device=device),torch.as_tensor(g["pairs"],device=device),
                        torch.as_tensor(g["pairs"],device=device),torch.as_tensor(g["edge_geom"],device=device))
                    result[g["graph_id"]]=torch.sigmoid(logits).cpu().numpy()
                else:result[g["graph_id"]]=np.zeros(0,dtype=np.float32)
                if i%4==0 or i==len(graph_list):progress.update("GPU GNN 평가",phase_no,.05+.90*i/max(len(graph_list),1),f"회로 {i}/{len(graph_list)}")
        return result
    gnn_val=score_gnn(validation,8);gnn_test=score_gnn(test,8)
    gnn_threshold,gnn_val_thresholds=threshold_search(validation,gnn_val)
    gnn_result={"training_device":torch.cuda.get_device_name(0),"epochs":args.gnn_epochs,
        "validation":evaluate(validation,gnn_val,gnn_threshold),"test":evaluate(test,gnn_test,gnn_threshold),
        "net_threshold_calibration_validation":gnn_val_thresholds}
    gnn_path=model_dir/"gnn_cuda_pcb_pin_only.pt"
    torch.save({"state_dict":model.state_dict(),"node_dim":node_dim,"edge_dim":edge_dim,
        "taxonomy":TAXONOMY,"candidate":{"radius_mm":radius,"k_nearest":k},"seed":args.seed,
        "epochs":args.gnn_epochs,"training_device":torch.cuda.get_device_name(0)},gnn_path)
    summaries.setdefault("model_artifacts",{})["GNN_CUDA"]=str(gnn_path.relative_to(ROOT)).replace("\\","/")
    summaries["GNN_CUDA"]=gnn_result
    summaries["evaluation_notes"]={"oracle_recall_definition":"Best possible Recall@5 after candidate generation: assumes all candidate true links rank ahead of false links, with same-component pairs consistently excluded.",
        "end_to_end_recall_definition":"Top-k hits among candidate links divided by all eligible cross-component same-net truth links, including links omitted by candidate generation.",
        "net_metrics":"Predicted edges are thresholded using validation-only calibration, then connected components impose transitivity; pairwise net precision/recall/F1 are reported on test.",
        "candidate_choice":"On validation, select the lowest mean candidate-degree setting within 0.5 percentage points of best recall, capped at 250; mean degree is 2E/N.",
        "no_image_claim":"This benchmark uses PCB CAD placement/pad geometry and PCB CAD net labels. It does not establish accuracy from photographs.",
        "prior_training_excluded":True}
    save_partial(summaries,"complete")
    OUTPUT.write_text(json.dumps(summaries,indent=2),encoding="utf-8")
    try:OUTPUT.with_suffix(".partial.json").unlink()
    except FileNotFoundError:pass
    progress.update("완료",9,1,str(OUTPUT))
    print(json.dumps({"candidate":summaries["selected_candidate"],"candidate_recall_test":summaries["candidate_recall_test"],
        "LightGBM_test":lgb_result["test"],"GNN_test":gnn_result["test"],"output":str(OUTPUT)},ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
