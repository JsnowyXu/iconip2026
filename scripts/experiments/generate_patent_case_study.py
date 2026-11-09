#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dhc.data.unified_dataset import load_unified_pt
from dhc.eval.evaluator import replay_history
from dhc.models.dhc_model import DHCModel


def load_model(run_dir: Path, data):
    model_stats = json.loads((run_dir / "model_stats.json").read_text(encoding="utf-8"))
    args = model_stats.get("args", {})

    num_multiclass = 0
    num_multilabel = 0
    if data.node_multiclass is not None:
        num_multiclass = int(data.node_multiclass["labels"].max().item()) + 1
    if data.node_multilabel is not None:
        num_multilabel = int(data.node_multilabel["labels"].shape[1])

    model = DHCModel(
        num_nodes=data.num_nodes,
        num_rels=data.num_rels,
        num_node_types=data.num_node_types,
        node_feat=data.node_feat,
        node_type=data.node_type,
        edge_feat_dim=data.edge_feat.shape[1],
        mem_dim=int(args.get("mem_dim", 128)),
        rel_dim=int(args.get("rel_dim", 64)),
        time_dim=int(args.get("time_dim", 32)),
        msg_dim=int(args.get("msg_dim", 128)),
        community_dim=int(args.get("community_dim", 128)),
        num_communities=int(args.get("num_communities", 64)),
        num_multiclass=num_multiclass,
        num_multilabel=num_multilabel,
        use_relation_memory=not bool(args.get("no_relation_memory", False)),
        use_community=not bool(args.get("no_community", False)),
        community_assign_temperature=float(args.get("community_assign_temperature", 1.0)),
        community_assign_topk=int(args.get("community_assign_topk", 0)),
    ).to(data.node_feat.device)

    state = torch.load(run_dir / "best_model.pt", map_location=data.node_feat.device)
    model.load_state_dict(state)
    model.eval()
    return model


def infer_target_rel_new(data_obj: dict, target_rel_name: str = "patent_ipc") -> int:
    extra = data_obj.get("meta", {}).get("extra", {})
    rel_map = extra.get("relation_map", {}) if isinstance(extra, dict) else {}
    target_old = int(rel_map.get(target_rel_name, 1))

    cr = extra.get("cross_relation_boost", {}) if isinstance(extra, dict) else {}
    if isinstance(cr, dict):
        new_map = cr.get("new_relation_map", {})
        if isinstance(new_map, dict) and str(target_old) in new_map:
            return int(new_map[str(target_old)])

    src_rel_ids = extra.get("source_rel_ids", []) if isinstance(extra, dict) else []
    if isinstance(src_rel_ids, list) and len(src_rel_ids) > 0:
        vals = sorted(int(x) for x in src_rel_ids)
        if target_old in vals:
            return int(vals.index(target_old))

    return 0


@torch.no_grad()
def predict_target_test_scores(data_path: Path, run_dir: Path, target_rel_name: str = "patent_ipc"):
    device = torch.device("cpu")
    data = load_unified_pt(str(data_path), device=device)
    data_obj = torch.load(data_path, map_location="cpu")
    target_rel_new = infer_target_rel_new(data_obj, target_rel_name=target_rel_name)

    model = load_model(run_dir=run_dir, data=data)

    hist = torch.cat([data.train_idx, data.val_idx], dim=0)
    replay_history(model, data, history_idx=hist, batch_size=512, device=device)

    test_idx = data.test_idx
    test_rel = data.rel[test_idx]
    use_idx = test_idx[test_rel == int(target_rel_new)]

    src = data.src[use_idx]
    dst = data.dst[use_idx]
    rel = data.rel[use_idx]
    ts = data.ts[use_idx]
    score = torch.sigmoid(model.predict_links(src, dst, rel, ts)).detach().cpu().numpy().astype(np.float64)

    out = pd.DataFrame(
        {
            "src": src.detach().cpu().numpy().astype(np.int64),
            "dst": dst.detach().cpu().numpy().astype(np.int64),
            "ts": ts.detach().cpu().numpy().astype(np.float64),
            "score": score,
        }
    )
    out["key"] = out.apply(lambda r: f"{int(r.src)}|{int(r.dst)}|{r.ts:.6f}", axis=1)
    return out


def build_context(raw_obj: dict):
    extra = raw_obj.get("meta", {}).get("extra", {})
    rel_map = extra.get("relation_map", {})
    rel_pat_assignee = int(rel_map.get("patent_assignee", 0))
    rel_assignee_ipc = int(rel_map.get("assignee_ipc", 2))

    src = raw_obj["edges"]["src"].numpy().astype(np.int64)
    dst = raw_obj["edges"]["dst"].numpy().astype(np.int64)
    rel = raw_obj["edges"]["rel"].numpy().astype(np.int64)

    patent_to_assignees = defaultdict(list)
    assignee_to_ipc = defaultdict(list)

    for s, d, r in zip(src, dst, rel):
        if int(r) == rel_pat_assignee:
            patent_to_assignees[int(s)].append(int(d))
        elif int(r) == rel_assignee_ipc:
            assignee_to_ipc[int(s)].append(int(d))

    pat_attr = extra.get("patent_attributes", {})
    ipc_attr = extra.get("ipc_attributes", {})

    pat_title = {}
    p_nodes = pat_attr.get("node_ids", [])
    titles = pat_attr.get("title", [])
    for n, t in zip(p_nodes, titles):
        pat_title[int(n)] = str(t)

    ipc_node_to_code = {}
    ipc_node_to_name = {}
    for nid, code, name in zip(ipc_attr.get("node_ids", []), ipc_attr.get("codes", []), ipc_attr.get("names", [])):
        ipc_node_to_code[int(nid)] = str(code)
        ipc_node_to_name[int(nid)] = str(name)

    return patent_to_assignees, assignee_to_ipc, pat_title, ipc_node_to_code, ipc_node_to_name


def main() -> None:
    ap = argparse.ArgumentParser("Generate patent case study from base vs cross-relation-boost runs")
    ap.add_argument("--raw-data", required=True, type=str, help="raw_cn_unified.pt or raw_us_unified.pt")
    ap.add_argument("--base-data", required=True, type=str, help="*_edge_patent_ipc.pt")
    ap.add_argument("--boost-data", required=True, type=str, help="*_edge_patent_ipc_plus_assignee_ipc.pt")
    ap.add_argument("--base-run", required=True, type=str)
    ap.add_argument("--boost-run", required=True, type=str)
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_df = predict_target_test_scores(Path(args.base_data), Path(args.base_run))
    boost_df = predict_target_test_scores(Path(args.boost_data), Path(args.boost_run))

    m = base_df[["key", "src", "dst", "ts", "score"]].rename(columns={"score": "base_score"}).merge(
        boost_df[["key", "score"]].rename(columns={"score": "boost_score"}),
        on="key",
        how="inner",
    )
    if m.empty:
        raise RuntimeError("no matched target test edges found between base and boost views")

    m["delta"] = m["boost_score"] - m["base_score"]

    raw_obj = torch.load(args.raw_data, map_location="cpu")
    patent_to_assignees, assignee_to_ipc, pat_title, ipc_node_to_code, ipc_node_to_name = build_context(raw_obj)

    cases = []
    top_gain = m.sort_values("delta", ascending=False).head(max(1, int(args.top_k)))
    top_drop = m.sort_values("delta", ascending=True).head(max(1, int(args.top_k // 3)))

    for tag, sub in [("gain", top_gain), ("drop", top_drop)]:
        for _, r in sub.iterrows():
            p = int(r["src"])
            c = int(r["dst"])
            asgs = patent_to_assignees.get(p, [])
            cc = Counter()
            for a in asgs:
                cc.update(assignee_to_ipc.get(a, []))
            top_ctx = []
            for ipc_nid, cnt in cc.most_common(5):
                code = ipc_node_to_code.get(int(ipc_nid), f"node_{int(ipc_nid)}")
                nm = ipc_node_to_name.get(int(ipc_nid), "")
                top_ctx.append(f"{code}({cnt}){':' + nm if nm else ''}")

            cases.append(
                {
                    "type": tag,
                    "src_patent_node": p,
                    "dst_ipc_node": c,
                    "ipc_code": ipc_node_to_code.get(c, ""),
                    "ipc_name": ipc_node_to_name.get(c, ""),
                    "base_score": float(r["base_score"]),
                    "boost_score": float(r["boost_score"]),
                    "delta": float(r["delta"]),
                    "num_assignees": int(len(asgs)),
                    "assignee_ipc_context_top5": " | ".join(top_ctx),
                    "patent_title": pat_title.get(p, "")[:180],
                }
            )

    case_df = pd.DataFrame(cases)
    case_df.to_csv(out_dir / "case_study_edges.csv", index=False)

    md_lines = [
        "# Patent Case Study: Base vs Cross-Relation Boost",
        "",
        "- Base data: `{}`".format(args.base_data),
        "- Boost data: `{}`".format(args.boost_data),
        "- Base run: `{}`".format(args.base_run),
        "- Boost run: `{}`".format(args.boost_run),
        "",
        "## Aggregate Delta",
        "",
    ]
    agg = m[["base_score", "boost_score", "delta"]].describe().T
    md_lines.append(agg.to_markdown())
    md_lines.append("")
    md_lines.append("## Top Gain Cases")
    md_lines.append("")
    md_lines.append(case_df[case_df["type"] == "gain"].to_markdown(index=False))
    md_lines.append("")
    md_lines.append("## Top Drop Cases")
    md_lines.append("")
    md_lines.append(case_df[case_df["type"] == "drop"].to_markdown(index=False))

    (out_dir / "case_study_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
