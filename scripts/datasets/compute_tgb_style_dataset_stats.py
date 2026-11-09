#!/usr/bin/env python3
"""Compute TGB2-style temporal dataset statistics for DHC task views."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


Triple = Tuple[int, int, int]


DEFAULT_TASKS = [
    (
        "dblp",
        "data/processed/paper_tasks/dblp_edge_transductive.pt",
    ),
    (
        "aminer_paper_citation",
        "data/processed/paper_tasks/aminer_paper_citation_edge_transductive.pt",
    ),
    (
        "patent_us",
        "data/processed/tasks/patent_us_edge_org_codeIII_transductive.pt",
    ),
    (
        "patent_in",
        "data/processed/tasks/patent_in_edge_org_codeIII_transductive.pt",
    ),
    (
        "patent_cn",
        "data/processed/tasks/patent_cn_edge_org_codeIII_transductive.pt",
    ),
    (
        "thgl_software",
        "data/processed/tgb2_tasks/thgl_software_edge_transductive.pt",
    ),
]


def tensor_to_list(x: torch.Tensor) -> List[int]:
    if x.dtype.is_floating_point:
        return [int(round(v)) for v in x.cpu().tolist()]
    return [int(v) for v in x.cpu().tolist()]


def infer_time_unit(times: List[int]) -> int:
    uniq = sorted(set(times))
    if len(uniq) < 2:
        return 1
    diffs = [b - a for a, b in zip(uniq, uniq[1:]) if b > a]
    if not diffs:
        return 1
    counts = defaultdict(int)
    for d in diffs:
        counts[d] += 1
    mode_diff = max(counts.items(), key=lambda kv: kv[1])[0]
    gcd = diffs[0]
    for d in diffs[1:]:
        gcd = math.gcd(gcd, d)
        if gcd == 1:
            break
    # Large Unix timestamps stored as float32 can create artificial 128-second
    # granularity. In that case the modal positive gap better recovers the
    # native discretization used by the task view.
    if max(uniq) > 100_000_000 and gcd < 3600 <= mode_diff:
        return int(mode_diff)
    return max(1, int(gcd))


def original_timestamp_column(data: dict, expected_len: int) -> Tuple[List[int], str]:
    extra = data.get("meta", {}).get("extra", {})
    csv_path = extra.get("original_edgelist_csv")
    if not csv_path:
        return [], "task_view"
    path = Path(csv_path)
    if not path.exists():
        return [], "task_view"
    values: List[int] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "timestamp" not in (reader.fieldnames or []):
            return [], "task_view"
        for row in reader:
            values.append(int(float(row["timestamp"])))
    if len(values) != expected_len:
        return [], "task_view"
    return values, "original_edgelist_csv"


def build_triple_times(src: List[int], dst: List[int], rel: List[int], ts: List[int]) -> Dict[Triple, set]:
    triple_times: Dict[Triple, set] = defaultdict(set)
    for s, d, r, t in zip(src, dst, rel, ts):
        triple_times[(s, r, d)].add(t)
    return triple_times


def max_consecutive_run(values: Iterable[int], unit: int) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    best = cur = 1
    prev = ordered[0]
    for t in ordered[1:]:
        if t == prev:
            continue
        if t - prev == unit:
            cur += 1
        else:
            best = max(best, cur)
            cur = 1
        prev = t
    return max(best, cur)


def compute_one(name: str, path: Path) -> dict:
    data = torch.load(path, map_location="cpu")
    edges = data["edges"]
    src = tensor_to_list(edges["src"])
    dst = tensor_to_list(edges["dst"])
    rel = tensor_to_list(edges["rel"])
    ts = tensor_to_list(edges["ts"])
    original_ts, time_source = original_timestamp_column(data, len(ts))
    if original_ts:
        ts = original_ts
    num_edges = len(src)
    num_nodes = int(data.get("num_nodes", max(max(src), max(dst)) + 1))
    num_rels = int(data.get("num_rels", len(set(rel))))
    num_node_types = int(data.get("num_node_types", 0))
    avg_degree = 2.0 * num_edges / num_nodes if num_nodes else 0.0

    split = data.get("split", {})
    train_idx = tensor_to_list(split.get("train", torch.arange(0)))
    test_idx = tensor_to_list(split.get("test", torch.arange(0)))
    train_nodes = set()
    for i in train_idx:
        train_nodes.add(src[i])
        train_nodes.add(dst[i])
    test_nodes = set()
    for i in test_idx:
        test_nodes.add(src[i])
        test_nodes.add(dst[i])
    inductive_test_node_prop = (
        len(test_nodes - train_nodes) / len(test_nodes) if test_nodes else 0.0
    )

    unit = infer_time_unit(ts)
    triple_times = build_triple_times(src, dst, rel, ts)

    rec = 0
    drec = 0
    for i in test_idx:
        times = triple_times[(src[i], rel[i], dst[i])]
        t = ts[i]
        if any(prev_t < t for prev_t in times):
            rec += 1
        if (t - unit) in times:
            drec += 1
    rec_degree = rec / len(test_idx) if test_idx else 0.0
    drec_degree = drec / len(test_idx) if test_idx else 0.0

    con_runs = [max_consecutive_run(times, unit) for times in triple_times.values()]
    consecutiveness = sum(con_runs) / len(con_runs) if con_runs else 0.0

    active_ts = len(set(ts))
    if ts:
        span_timesteps = int((max(ts) - min(ts)) // unit) + 1
    else:
        span_timesteps = 0
    nodes_per_ts: Dict[int, set] = defaultdict(set)
    edges_per_ts: Dict[int, int] = defaultdict(int)
    for s, d, t in zip(src, dst, ts):
        edges_per_ts[t] += 1
        nodes_per_ts[t].add(s)
        nodes_per_ts[t].add(d)
    mean_edges_per_ts = num_edges / span_timesteps if span_timesteps else 0.0
    mean_nodes_per_ts = (
        sum(len(nodes) for nodes in nodes_per_ts.values()) / span_timesteps
        if span_timesteps
        else 0.0
    )

    return {
        "dataset": name,
        "task_file": str(path),
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree_2e_over_n": avg_degree,
        "num_edge_types": num_rels,
        "num_node_types": num_node_types,
        "active_timesteps": active_ts,
        "time_unit": unit,
        "time_source": time_source,
        "span_timesteps": span_timesteps,
        "inductive_test_nodes": inductive_test_node_prop,
        "drec": drec_degree,
        "rec": rec_degree,
        "con": consecutiveness,
        "mean_edges_per_ts": mean_edges_per_ts,
        "mean_nodes_per_ts": mean_nodes_per_ts,
        "train_edges": len(train_idx),
        "test_edges": len(test_idx),
    }


def write_latex(rows: List[dict], out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Dataset statistics following the TGB2.0 protocol. Induct. Nodes denotes the proportion of test nodes unseen during training. DRec, Rec and Con are direct recurrency, recurrency and consecutiveness. Mean Edges/Ts. and Mean Nodes/Ts. are averaged over the full temporal span at the inferred native time unit.}",
        r"\label{tab:dataset-tgb-style-stats}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Dataset & Nodes & Edges & Avg. Deg. & Induct. Nodes & DRec & Rec & Con & Edges/Ts. & Nodes/Ts. \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {:,} & {:,} & {:.2f} & {:.3f} & {:.3f} & {:.3f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
                row["dataset"].replace("_", r"\_"),
                row["num_nodes"],
                row["num_edges"],
                row["avg_degree_2e_over_n"],
                row["inductive_test_nodes"],
                row["drec"],
                row["rec"],
                row["con"],
                row["mean_edges_per_ts"],
                row["mean_nodes_per_ts"],
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--out-dir",
        default="results/dataset_tgb_style_stats",
        help="Output directory relative to repo root unless absolute.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, rel_path in DEFAULT_TASKS:
        path = repo_root / rel_path
        print(f"computing {name}: {path}")
        rows.append(compute_one(name, path))

    csv_path = out_dir / "dataset_tgb_style_stats.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_latex(rows, out_dir / "dataset_tgb_style_stats_table.tex")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
