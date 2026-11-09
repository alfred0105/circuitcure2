#!/usr/bin/env python3
"""Re-train PCB-only link models with corrected candidate/cluster evaluation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/benchmarks/pcb_pin_connectivity"
PIN_DATA = ROOT / "data/external/pcb-photo-cad-2026-11-09/pin_only_training"
OUTPUT = BENCH / "pcb_pin_connectivity_review2_results.json"
PROGRESS_FILE = ROOT / "data/benchmarks/phase22_connectivity/recommendation_progress.json"
SPEC = importlib.util.spec_from_file_location("pcb_connectivity_base", ROOT / "scripts/benchmark_pcb_pin_connectivity.py")
BASE = importlib.util.module_from_spec(SPEC); assert SPEC and SPEC.loader; SPEC.loader.exec_module(BASE)


class Progress:
    def __init__(self, count): self.count, self.started = count, time.time()
    def update(self, phase, n, fraction, detail=""):
        payload={"status":"완료" if phase=="완료" else "실행 중","phase":phase,"phase_number":n,
          "phase_count":self.count,"phase_percent":round(100*fraction,1),
          "overall_percent":100.0 if phase=="완료" else 100*(n-1+fraction)/self.count,
          "elapsed_seconds":round(time.time()-self.started,1),"detail":detail,
          "updated_at":datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")}
        PROGRESS_FILE.parent.mkdir(parents=True,exist_ok=True);tmp=PROGRESS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload,ensure_ascii=False),encoding="utf-8")
        try:os.replace(tmp,PROGRESS_FILE)
        except PermissionError:PROGRESS_FILE.write_text(json.dumps(payload,ensure_ascii=False),encoding="utf-8")
        print(f"[{n}/{self.count} {fraction:.0%}] {phase}: {detail}",flush=True)


def corrected_candidate_audit(rows, configs):
    reports=[]
    for radius,k in configs:
        state={s:{"truth":0,"covered":0,"edges":0,"pins":0,"recalls":[]} for s in ("train","validation")}
        for row in rows:
            if row["split"] not in state:continue
            pairs=BASE.candidate_pairs(row["nodes_raw"],radius,k)
            truth_count=len(row["truth_pairs"]);covered=sum(pair in row["truth_pairs"] for pair in pairs)
            s=state[row["split"]];s["truth"]+=truth_count;s["covered"]+=covered;s["edges"]+=len(pairs)
            s["pins"]+=len(row["nodes_raw"])
            if truth_count:s["recalls"].append(covered/truth_count)
        val=state["validation"];tr=state["train"]
        reports.append({"radius_mm":radius,"k_nearest":k,
          "validation_candidate_recall_micro":val["covered"]/max(val["truth"],1),
          "validation_candidate_recall_macro":float(np.mean(val["recalls"])) if val["recalls"] else None,
          "validation_mean_candidate_degree":2*val["edges"]/max(val["pins"],1),
          "validation_mean_undirected_edges_per_pin":val["edges"]/max(val["pins"],1),
          "train_candidate_recall_micro":tr["covered"]/max(tr["truth"],1),
          "train_mean_candidate_degree":2*tr["edges"]/max(tr["pins"],1)})
    return reports


def hash_tokens(value, width, field):
    out=np.zeros(width,dtype=np.float32)
    tokens=re.findall(r"[a-z0-9]+",str(value or "").casefold())
    for token in set(tokens):
        raw=hashlib.blake2b(f"{field}:{token}".encode(),digest_size=8).digest()
        out[int.from_bytes(raw[:4],"little")%width]=1.0
    return out


def semantic_node_features(nodes, base_x):
    # Stable text hashing adds package/value/pad-label cues without exporting any net name.
    extra=[]
    for n in nodes:
        extra.append(np.concatenate([
          hash_tokens(n.get("component_value"),48,"value"),
          hash_tokens(n.get("footprint"),64,"footprint"),
          hash_tokens(n.get("pad_number"),24,"pad"),
          hash_tokens(n.get("pad_shape"),12,"shape")]))
    return np.concatenate([base_x,np.asarray(extra,dtype=np.float32)],axis=1)


def message_edges(g,k=16):
    n=len(g["nodes_raw"])
    if n<2:return np.empty((0,2),dtype=np.int64)
    tree=cKDTree(g["coords"]);q=min(k+1,n)
    _,ids=tree.query(g["coords"],k=q)
    if q==1:ids=ids[:,None]
    refs=g["refs"];pairs=set()
    for i,row in enumerate(ids):
        for j in row[1:]:
            j=int(j)
            if j!=i and refs[i]!=refs[j]:pairs.add((min(i,j),max(i,j)))
    return np.asarray(sorted(pairs),dtype=np.int64).reshape((-1,2))


def power_role(node):
    tokens=set(re.findall(r"[a-z0-9]+",str(node.get("pad_number","")).casefold()))
    if tokens & {"gnd","ground","agnd","dgnd","pgnd"}:return "ground"
    if tokens & {"vcc","vdd","vin","vbus","vbat","3v3","5v"}:return "supply"
    return None


def make_edge_features(g,indices=None,feature_key="x"):
    if indices is None:indices=np.arange(len(g["pairs"]))
    pairs=g["pairs"][indices];x=g[feature_key]
    a,b=x[pairs[:,0]],x[pairs[:,1]]
    return np.concatenate([g["edge_geom"][indices],a+b,np.abs(a-b)],axis=1).astype(np.float32)


def union_clusters(g,scores,threshold,mode="threshold",topk=3):
    count=len(g["nodes_raw"]);parent=list(range(count));members={i:{i} for i in range(count)}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    score_map={tuple(map(int,p)):float(s) for p,s in zip(g["pairs"],scores)}
    roles=[power_role(n) for n in g["nodes_raw"]]
    selected=[]
    if mode=="mutual_topk":
        incident=defaultdict(list)
        for i,(a,b) in enumerate(g["pairs"]):
            incident[int(a)].append(i);incident[int(b)].append(i)
        chosen=set()
        for node,inds in incident.items():
            order=sorted(inds,key=lambda i:(-float(scores[i]),i))[:topk]
            for i in order:chosen.add(i)
        # Keep only edges selected by both endpoints.
        endpoint_top=[set() for _ in range(count)]
        for node,inds in incident.items():
            for i in sorted(inds,key=lambda z:(-float(scores[z]),z))[:topk]:endpoint_top[node].add(i)
        selected=[i for i,(a,b) in enumerate(g["pairs"]) if i in endpoint_top[int(a)] and i in endpoint_top[int(b)]]
    else:
        selected=[i for i,s in enumerate(scores) if float(s)>=threshold]
        selected.sort(key=lambda i:(-float(scores[i]),i))
    merges=0;rejected=0
    for edge_idx in selected:
        a,b=map(int,g["pairs"][edge_idx]);ra,rb=find(a),find(b)
        if ra==rb:continue
        if mode=="constrained":
            left_roles={roles[x] for x in members[ra] if roles[x]};right_roles={roles[x] for x in members[rb] if roles[x]}
            if ("ground" in left_roles and "supply" in right_roles) or ("supply" in left_roles and "ground" in right_roles):
                rejected+=1;continue
        if mode=="constrained" and len(members[ra])+len(members[rb])>2:
            left,right=members[ra],members[rb]
            cross=[score_map.get((min(u,v),max(u,v)),0.0) for u in left for v in right if g["refs"][u]!=g["refs"][v]]
            if not cross:rejected+=1;continue
            support=sum(s>=threshold for s in cross);fraction=support/len(cross);mean=float(np.mean(cross))
            needed=max(2,math.ceil(.5*len(cross)))
            if fraction<.5 or support<needed or mean<threshold*.8:
                rejected+=1;continue
        if len(members[ra])<len(members[rb]):ra,rb=rb,ra
        parent[rb]=ra;members[ra]|=members.pop(rb);merges+=1
    clusters=defaultdict(list)
    for i in range(count):clusters[find(i)].append(i)
    pred=set();nontrivial=0;false_merges=0;overmerged=0
    truth_net_by_node={}
    for net_id,group in enumerate(g["truth_nets"]):
        for node in group:truth_net_by_node[int(node)]=net_id
    for vs in clusters.values():
        if len(vs)>1:nontrivial+=1
        nets={truth_net_by_node[v] for v in vs if v in truth_net_by_node}
        if len(nets)>1:false_merges+=1;overmerged+=len(nets)-1
        for ix,u in enumerate(vs):
            for v in vs[ix+1:]:
                if g["refs"][u]!=g["refs"][v]:pred.add((min(u,v),max(u,v)))
    return pred,nontrivial,false_merges,overmerged,rejected


def cluster_metrics(graphs,scores,threshold,mode="threshold",topk=3):
    tp=fp=fn=clusters=false_merges=overmerged=rejected=0
    for g in graphs:
        pred,nc,fm,om,rj=union_clusters(g,scores[g["graph_id"]],threshold,mode,topk)
        truth=g["truth_pairs"];tp+=len(pred&truth);fp+=len(pred-truth);fn+=len(truth-pred)
        clusters+=nc;false_merges+=fm;overmerged+=om;rejected+=rj
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
    return {"method":mode,"threshold":float(threshold),"topk":topk,
      "net_pair_precision":precision,"net_pair_recall":recall,
      "net_pair_f1":2*precision*recall/max(precision+recall,1e-12),
      "predicted_transitive_pairs":tp+fp,"true_cross_component_pairs":tp+fn,
      "predicted_nontrivial_clusters":clusters,"false_merge_clusters":false_merges,
      "overmerged_truth_net_count":overmerged,"rejected_cluster_merges":rejected}


def calibrate_cluster(validation,scores,mode):
    vals=np.concatenate([scores[g["graph_id"]] for g in validation if len(scores.get(g["graph_id"],[]))])
    if not len(vals):return {"threshold":.5,"topk":3,"validation":[]}
    if mode=="mutual_topk":
        rows=[cluster_metrics(validation,scores,.5,mode,k) for k in (1,2,3,5,8,12)]
        best=max(rows,key=lambda x:(x["net_pair_f1"],x["net_pair_precision"]))
    else:
        qs=[.85,.90,.95,.97,.98,.99,.995,.999]
        grid=sorted(set(float(x) for x in np.quantile(vals,qs)))
        rows=[cluster_metrics(validation,scores,t,mode,3) for t in grid]
        best=max(rows,key=lambda x:(x["net_pair_f1"],x["net_pair_precision"]))
    return {"threshold":best["threshold"],"topk":best["topk"],"best_validation":best,"grid":rows}


def score_metrics(graphs,scores):
    from sklearn.metrics import average_precision_score,roc_auc_score
    ys=[];ss=[]
    for g in graphs:
        if len(g["y"]):ys.append(g["y"]);ss.append(scores[g["graph_id"]])
    y=np.concatenate(ys);s=np.concatenate(ss)
    return {"candidate_pr_auc":float(average_precision_score(y,s)),
            "candidate_roc_auc":float(roc_auc_score(y,s)),
            **BASE.truth_degree_metrics(graphs,scores,5)}


def fit_lgb(train,seed,feature_key,trees,progress=None,phase=1):
    from lightgbm import LGBMClassifier,log_evaluation
    xs=[];ys=[];ws=[]
    for i,g in enumerate(train,1):
        idx,w=BASE.sample_training(g,seed)
        if len(idx):xs.append(make_edge_features(g,idx,feature_key));ys.append(g["y"][idx]);ws.append(w)
        if progress and (i%30==0 or i==len(train)):
            progress.update("LightGBM 특징 구성",phase,.05+.90*i/max(len(train),1),f"{feature_key} 회로 {i}/{len(train)}")
    X=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws)
    model=LGBMClassifier(n_estimators=trees,learning_rate=.04,num_leaves=31,max_depth=-1,
        subsample=.85,colsample_bytree=.85,reg_lambda=1.,random_state=seed,n_jobs=6,verbosity=-1)
    model.fit(X,y,sample_weight=w,callbacks=[log_evaluation(period=0)])
    return model,int(len(y))


def predict_lgb(model,graphs,feature_key,progress=None,phase=1):
    out={}
    for i,g in enumerate(graphs,1):
        X=make_edge_features(g,feature_key=feature_key)
        out[g["graph_id"]]=model.predict_proba(X)[:,1] if len(X) else np.zeros(0,dtype=np.float32)
        if progress and (i%4==0 or i==len(graphs)):
            progress.update("LightGBM 평가",phase,.05+.9*i/max(len(graphs),1),f"회로 {i}/{len(graphs)}")
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--trees",type=int,default=350);p.add_argument("--gnn-epochs",type=int,default=15)
    p.add_argument("--graph-batch-size",type=int,default=4);args=p.parse_args()
    progress=Progress(10);BENCH.mkdir(parents=True,exist_ok=True)
    progress.update("PCB 학습 데이터 로드",1,.1,"pin-only PCB CAD dataset만 읽음; 과거 schematic data 제외")
    raw=BASE.load_raw();configs=[(25.,8),(50.,16),(100.,32),(150.,64)]
    progress.update("후보 설정 audit",2,.03,"candidate recall과 평균 핀 차수(2E/N)를 같이 비교")
    audit=corrected_candidate_audit(raw,configs)
    best=max(x["validation_candidate_recall_micro"] for x in audit)
    eligible=[x for x in audit if x["validation_candidate_recall_micro"]>=best-.005 and x["validation_mean_candidate_degree"]<=250]
    choice=min(eligible or audit,key=lambda x:(x["validation_mean_candidate_degree"],-x["validation_candidate_recall_micro"]))
    radius,k=float(choice["radius_mm"]),int(choice["k_nearest"])
    progress.update("패드·부품 특징 그래프 생성",3,.05,f"{radius:g}mm / k={k}; 검증 recall={choice['validation_candidate_recall_micro']:.4f}")
    graphs=[]
    for i,row in enumerate(raw,1):
        g=BASE.prepare_graph(row,radius,k)
        g["x_base"]=g["x"];g["x_semantic"]=semantic_node_features(g["nodes_raw"],g["x_base"])
        g["message_pairs"]=message_edges(g,16)
        graphs.append(g)
        if i%40==0 or i==len(raw):progress.update("패드·부품 특징 그래프 생성",3,.05+.9*i/len(raw),f"보드 {i}/{len(raw)}")
    train=[g for g in graphs if g["split"]=="train"];validation=[g for g in graphs if g["split"]=="validation"];test=[g for g in graphs if g["split"]=="test"]
    summaries={"source":"pin-only PCB CAD data only; the prior schematic/Phase 22 training records were not loaded",
      "pcb_graph_count":len(graphs),"train_graphs":len(train),"validation_graphs":len(validation),"test_graphs":len(test),
      "candidate_audit_validation":audit,"selected_candidate":{"radius_mm":radius,"k_nearest":k,
        "mean_candidate_degree_validation":choice["validation_mean_candidate_degree"]},
      "target":"same-net membership between different components; not a literal direct copper segment",
      "same_component_policy":"Excluded consistently from candidate and truth pairs; multiple IC pads on one ground net remain members of that truth net and are not discarded from grouping.",
      "degree_metric":"Mean candidate degree is 2E/N. The prior report's 104.2 was E/N; corrected degree is approximately 208.4.",
      "trace_feature_policy":"No original copper-track/via features are included: raw CAD routing would reveal the target and would not simulate image-derived trace masks.",
      "semantic_features":"Stable hashed component value, footprint, pad number/name, and pad shape; no net names, net IDs, or source-repository IDs enter model features.",
      "gnn_message_graph":"16-nearest cross-component pad graph, separate from the wider candidate graph, to reduce dense-neighbor aggregation noise.",
      "candidate_recall":{"train":BASE.truth_degree_metrics(train,{g["graph_id"]:np.zeros(len(g["pairs"])) for g in train},5)["candidate_recall_micro"],
          "validation":BASE.truth_degree_metrics(validation,{g["graph_id"]:np.zeros(len(g["pairs"])) for g in validation},5)["candidate_recall_micro"],
          "test":BASE.truth_degree_metrics(test,{g["graph_id"]:np.zeros(len(g["pairs"])) for g in test},5)["candidate_recall_micro"]}}
    BENCH.mkdir(parents=True,exist_ok=True)
    def save_partial(stage):
        temp=OUTPUT.with_suffix(".partial.tmp");temp.write_text(json.dumps({"stage":stage,"results":summaries},indent=2),encoding="utf-8");os.replace(temp,OUTPUT.with_suffix(".partial.json"))
    save_partial("candidate selection")
    # Baselines have no trainable parameters.
    distance_val={};distance_test={};random_val={};random_test={}
    for g in validation+test:
        dist=np.linalg.norm(g["coords"][g["pairs"][:,0]]-g["coords"][g["pairs"][:,1]],axis=1) if len(g["pairs"]) else np.zeros(0)
        ds=1/(1+dist);rs=BASE.deterministic_rng(args.seed,g["graph_id"],"random-baseline").random(len(ds))
        (distance_val if g["split"]=="validation" else distance_test)[g["graph_id"]]=ds
        (random_val if g["split"]=="validation" else random_test)[g["graph_id"]]=rs
    models={"distance_only":(distance_val,distance_test),"random":(random_val,random_test)}
    # Geometry/type baseline and expanded package/value/pad semantic features.
    for name,key,phase in (("LightGBM_geometry","x_base",4),("LightGBM_semantic","x_semantic",5)):
        progress.update(f"{name} 학습",phase,.02,"net-balanced positives + sampled negatives")
        model,n_examples=fit_lgb(train,args.seed,key,args.trees,progress,phase)
        val_scores=predict_lgb(model,validation,key,progress,6);test_scores=predict_lgb(model,test,key,progress,6)
        model_dir=BENCH/"models";model_dir.mkdir(parents=True,exist_ok=True)
        dest=model_dir/(("lightgbm_pcb_geometry_review2.txt") if key=="x_base" else "lightgbm_pcb_semantic_review2.txt")
        model.booster_.save_model(str(dest));summaries.setdefault("model_artifacts",{})[name]=str(dest.relative_to(ROOT)).replace("\\","/")
        summaries[name]={"device":"CPU; installed Windows LightGBM learner has no CUDA support","training_examples":n_examples}
        models[name]=(val_scores,test_scores);save_partial(name+" trained")
    progress.update("GPU GNN 학습",7,.02,"RTX 4060 Ti; semantic pad features + 16-NN message graph")
    import torch
    from torch import nn
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);torch.set_num_threads(4);device=torch.device("cuda:0")
    node_dim=train[0]["x_semantic"].shape[1];edge_dim=train[0]["edge_geom"].shape[1]
    class EdgeAwareGNN(nn.Module):
        def __init__(self):
            super().__init__();h=48
            self.encoder=nn.Sequential(nn.Linear(node_dim,96),nn.ReLU(),nn.Dropout(.10),nn.Linear(96,h),nn.ReLU())
            self.msg1=nn.Linear(2*h,h);self.msg2=nn.Linear(2*h,h)
            self.edge=nn.Sequential(nn.Linear(2*h+edge_dim,96),nn.ReLU(),nn.Dropout(.15),nn.Linear(96,1))
        def forward(self,x,msg_edges,q_edges,q_geom):
            h=self.encoder(x);src,dst=msg_edges[:,0],msg_edges[:,1]
            for layer in (self.msg1,self.msg2):
                agg=torch.zeros_like(h);deg=torch.zeros((len(h),1),device=h.device,dtype=h.dtype)
                agg.index_add_(0,dst,h[src]);agg.index_add_(0,src,h[dst]);one=torch.ones((len(src),1),device=h.device,dtype=h.dtype)
                deg.index_add_(0,dst,one);deg.index_add_(0,src,one);h=torch.relu(layer(torch.cat([h,agg/deg.clamp_min(1)],1)))
            a,b=h[q_edges[:,0]],h[q_edges[:,1]]
            return self.edge(torch.cat([a+b,torch.abs(a-b),q_geom],1)).squeeze(-1)
    samples=[]
    for g in train:
        idx,w=BASE.sample_training(g,args.seed);samples.append((g,idx,w))
    model=EdgeAwareGNN().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    for epoch in range(args.gnn_epochs):
        order=np.random.default_rng(args.seed+epoch).permutation(len(samples));loss_sum=0.;steps=0
        for start in range(0,len(order),args.graph_batch_size):
            batch=[samples[int(i)] for i in order[start:start+args.graph_batch_size]]
            xs=[];msgs=[];queries=[];geoms=[];ys=[];weights=[];offset=0
            for g,idx,w in batch:
                xs.append(g["x_semantic"]);msgs.append(g["message_pairs"]+offset);queries.append(g["pairs"][idx]+offset)
                geoms.append(g["edge_geom"][idx]);ys.append(g["y"][idx]);weights.append(w);offset+=len(g["x"])
            tx=torch.as_tensor(np.concatenate(xs),device=device);tm=torch.as_tensor(np.concatenate(msgs),device=device)
            tq=torch.as_tensor(np.concatenate(queries),device=device);tg=torch.as_tensor(np.concatenate(geoms),device=device)
            ty=torch.as_tensor(np.concatenate(ys),dtype=torch.float32,device=device);tw=torch.as_tensor(np.concatenate(weights),device=device)
            model.train();opt.zero_grad(set_to_none=True);logits=model(tx,tm,tq,tg)
            loss=(nn.functional.binary_cross_entropy_with_logits(logits,ty,reduction="none")*tw).sum()/len(batch)
            loss.backward();opt.step();loss_sum+=float(loss.detach());steps+=1
        progress.update("GPU GNN 학습",7,.05+.85*(epoch+1)/args.gnn_epochs,
          f"epoch {epoch+1}/{args.gnn_epochs}; weighted loss={loss_sum/max(steps,1):.4f}; VRAM={torch.cuda.memory_allocated()/1024**2:.0f} MiB")
    def score_gnn(group):
        out={};model.eval()
        with torch.no_grad():
            for i,g in enumerate(group,1):
                q=torch.as_tensor(g["pairs"],device=device);geom=torch.as_tensor(g["edge_geom"],device=device)
                msg=torch.as_tensor(g["message_pairs"],device=device);x=torch.as_tensor(g["x_semantic"],device=device)
                out[g["graph_id"]]=torch.sigmoid(model(x,msg,q,geom)).cpu().numpy() if len(q) else np.zeros(0)
                if i%4==0 or i==len(group):progress.update("GNN 평가",8,.05+.9*i/max(len(group),1),f"보드 {i}/{len(group)}")
        return out
    gnn_val=score_gnn(validation);gnn_test=score_gnn(test);models["GNN_semantic_CUDA"]=(gnn_val,gnn_test)
    gpath=BENCH/"models/gnn_cuda_pcb_semantic_review2.pt"
    torch.save({"state_dict":model.state_dict(),"node_dim":node_dim,"edge_dim":edge_dim,"candidate":{"radius_mm":radius,"k_nearest":k},
        "message_k":16,"taxonomy":BASE.TAXONOMY,"seed":args.seed,"epochs":args.gnn_epochs},gpath)
    summaries.setdefault("model_artifacts",{})["GNN_semantic_CUDA"]=str(gpath.relative_to(ROOT)).replace("\\","/")
    summaries["GNN_semantic_CUDA"]={"device":torch.cuda.get_device_name(0),"epochs":args.gnn_epochs};save_partial("models trained")
    progress.update("거리·랜덤·모델 기준선 평가",9,.02,"거리-only와 random 기준선 포함")
    all_metrics={}
    for model_idx,(name,(val_scores,test_scores)) in enumerate(models.items()):
        calibrated={}
        for mode_idx,mode in enumerate(("threshold","mutual_topk","constrained")):
            progress.update("넷 복원 방식 validation 비교",9,.05+.75*(model_idx*3+mode_idx)/max(len(models)*3,1),f"{name} · {mode}")
            calibrated[mode]=calibrate_cluster(validation,val_scores,mode)
        test_clusters={}
        for mode,config in calibrated.items():
            test_clusters[mode]=cluster_metrics(test,test_scores,config["threshold"],mode,config["topk"])
        val_clusters={mode:cfg["best_validation"] for mode,cfg in calibrated.items()}
        all_metrics[name]={"validation_candidate_metrics":score_metrics(validation,val_scores),
          "test_candidate_metrics":score_metrics(test,test_scores),"cluster_calibration_validation":val_clusters,
          "test_cluster_metrics":test_clusters}
    for name in ("LightGBM_geometry","LightGBM_semantic","GNN_semantic_CUDA"):
        summaries[name].update(all_metrics[name])
    summaries["baselines"]={name:all_metrics[name] for name in ("distance_only","random")}
    summaries["evaluation_notes"]={"net_membership_and_direct_connection":"This run predicts same-net membership, not direct copper path. Direct trace labels need a separate route-graph target.",
      "clustering":"Compared threshold union, mutual top-k, and support-constrained agglomerative merges. Thresholds/top-k are selected on validation only. Multi-pin component nets are preserved as truth groups.",
      "candidate_degree":"Reported as 2E/N, the actual average number of adjacent candidate pins per pin.",
      "traces_and_vias":"Not used as model inputs because the original CAD routing would leak the net target and is not equivalent to a mask detected from a photograph.",
      "image_evaluation":"No photo-to-pad registration or image-derived traces were evaluated; this remains PCB-CAD-to-CAD.",
      "prior_training_excluded":True}
    summaries["run"]={"seed":args.seed,"lightgbm_trees":args.trees,"gnn_epochs":args.gnn_epochs,
      "training_scope":"All training examples came from the PCB pin-only archive; validation/test were repository grouped."}
    save_partial("complete");OUTPUT.write_text(json.dumps(summaries,indent=2),encoding="utf-8")
    try:OUTPUT.with_suffix(".partial.json").unlink()
    except FileNotFoundError:pass
    progress.update("완료",10,1,str(OUTPUT))
    print(json.dumps({"selected_candidate":summaries["selected_candidate"],"metrics":{k:v["test_candidate_metrics"] for k,v in all_metrics.items()},
      "clusters":{k:v["test_cluster_metrics"] for k,v in all_metrics.items()},"output":str(OUTPUT)},ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
