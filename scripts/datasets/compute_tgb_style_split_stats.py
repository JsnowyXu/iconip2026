#!/usr/bin/env python3
"""Compute TGB2-style statistics separately for train/val/test splits."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from compute_tgb_style_dataset_stats import (
    build_triple_times,
    infer_time_unit,
    max_consecutive_run,
    original_timestamp_column,
    tensor_to_list,
)


DEFAULT_PROTOCOL_TASKS = [
    ("dblp", "inductive", "data/processed/paper_tasks/dblp_edge_inductive.pt"),
    ("dblp", "transductive", "data/processed/paper_tasks/dblp_edge_transductive.pt"),
    (
        "aminer_paper_citation",
        "inductive",
        "data/processed/paper_tasks/aminer_paper_citation_edge_inductive.pt",
    ),
    (
        "aminer_paper_citation",
        "transductive",
        "data/processed/paper_tasks/aminer_paper_citation_edge_transductive.pt",
    ),
    (
        "patent_us",
        "inductive",
        "data/processed/tasks/patent_us_edge_org_codeIII_inductive.pt",
    ),
    (
        "patent_us",
        "transductive",
        "data/processed/tasks/patent_us_edge_org_codeIII_transductive.pt",
    ),
    (
        "patent_in",
        "inductive",
        "data/processed/tasks/patent_in_edge_org_codeIII_inductive.pt",
    ),
    (
        "patent_in",
        "transductive",
        "data/processed/tasks/patent_in_edge_org_codeIII_transductive.pt",
    ),
    (
        "patent_cn",
        "inductive",
        "data/processed/tasks/patent_cn_edge_org_codeIII_inductive.pt",
    ),
    (
        "patent_cn",
        "transductive",
        "data/processed/tasks/patent_cn_edge_org_codeIII_transductive.pt",
    ),
    (
        "thgl_software",
        "inductive",
        "data/processed/tgb2_tasks/thgl_software_edge_inductive.pt",
    ),
    (
        "thgl_software",
        "transductive",
        "data/processed/tgb2_tasks/thgl_software_edge_transductive.pt",
    ),
]


def split_rows_for_dataset(name: str, protocol: str, path: Path) -> List[dict]:
    data = torch.load(path, map_location="cpu")
    edges = data["edges"]
    src = tensor_to_list(edges["src"])
    dst = tensor_to_list(edges["dst"])
    rel = tensor_to_list(edges["rel"])
    ts = tensor_to_list(edges["ts"])
    original_ts, time_source = original_timestamp_column(data, len(ts))
    if original_ts:
        ts = original_ts
    node_type = data.get("node_type")
    split = data["split"]
    unit = infer_time_unit(ts)
    full_triple_times = build_triple_times(src, dst, rel, ts)

    train_idx = tensor_to_list(split["train"])
    train_nodes = set()
    for i in train_idx:
        train_nodes.add(src[i])
        train_nodes.add(dst[i])

    rows: List[dict] = []
    for split_name in ["train", "val", "test"]:
        idx = tensor_to_list(split[split_name])
        split_nodes = set()
        split_rels = set()
        split_node_types = set()
        split_times = []
        split_triple_times: Dict[Tuple[int, int, int], set] = defaultdict(set)
        edges_per_ts: Dict[int, int] = defaultdict(int)
        nodes_per_ts: Dict[int, set] = defaultdict(set)

        rec = 0
        drec = 0
        for i in idx:
            s = src[i]
            d = dst[i]
            r = rel[i]
            t = ts[i]
            split_nodes.add(s)
            split_nodes.add(d)
            split_rels.add(r)
            split_times.append(t)
            split_triple_times[(s, r, d)].add(t)
            edges_per_ts[t] += 1
            nodes_per_ts[t].add(s)
            nodes_per_ts[t].add(d)
            if node_type is not None:
                split_node_types.add(int(node_type[s]))
                split_node_types.add(int(node_type[d]))

            all_times = full_triple_times[(s, r, d)]
            if any(prev_t < t for prev_t in all_times):
                rec += 1
            if (t - unit) in all_times:
                drec += 1

        num_edges = len(idx)
        num_nodes = len(split_nodes)
        active_timesteps = len(set(split_times))
        span_timesteps = (
            int((max(split_times) - min(split_times)) // unit) + 1
            if split_times
            else 0
        )
        con_runs = [max_consecutive_run(times, unit) for times in split_triple_times.values()]
        rows.append(
            {
                "dataset": name,
                "protocol": protocol,
                "split": split_name,
                "task_file": str(path),
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "avg_degree_2e_over_n": 2.0 * num_edges / num_nodes if num_nodes else 0.0,
                "num_edge_types": len(split_rels),
                "num_node_types": len(split_node_types),
                "active_timesteps": active_timesteps,
                "time_unit": unit,
                "time_source": time_source,
                "span_timesteps": span_timesteps,
                "inductive_nodes_vs_train": (
                    len(split_nodes - train_nodes) / len(split_nodes)
                    if split_nodes and split_name != "train"
                    else 0.0
                ),
                "drec": drec / num_edges if num_edges else 0.0,
                "rec": rec / num_edges if num_edges else 0.0,
                "con": sum(con_runs) / len(con_runs) if con_runs else 0.0,
                "mean_edges_per_ts": num_edges / span_timesteps if span_timesteps else 0.0,
                "mean_nodes_per_ts": (
                    sum(len(nodes) for nodes in nodes_per_ts.values()) / span_timesteps
                    if span_timesteps
                    else 0.0
                ),
                "ts_min": min(split_times) if split_times else "",
                "ts_max": max(split_times) if split_times else "",
            }
        )
    return rows


def write_latex(rows: List[dict], out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Train/validation/test split statistics. The split-level columns follow the TGB2-style dataset statistics; Induct. Nodes is measured against the training split.}",
        r"\label{tab:dataset-split-tgb-style-stats}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lllrrrrrrrrr}",
        r"\toprule",
        r"Dataset & Protocol & Split & Nodes & Edges & Avg. Deg. & Induct. Nodes & DRec & Rec & Con & Edges/Ts. & Nodes/Ts. \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {} & {:,} & {:,} & {:.2f} & {:.3f} & {:.3f} & {:.3f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
                row["dataset"].replace("_", r"\_"),
                row["protocol"],
                row["split"],
                row["num_nodes"],
                row["num_edges"],
                row["avg_degree_2e_over_n"],
                row["inductive_nodes_vs_train"],
                row["drec"],
                row["rec"],
                row["con"],
                row["mean_edges_per_ts"],
                row["mean_nodes_per_ts"],
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""])
    out_path.write_text("\n".join(lines))


def write_readme(rows: List[dict], out_path: Path) -> None:
    by_dataset: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in rows:
        by_dataset[(row["dataset"], row["protocol"])].append(row)

    lines = [
        "# Train/validation/test split statistics",
        "",
        "This directory reports how each formal edge-prediction dataset is split into training, validation and test sets for both inductive and transductive protocols, together with TGB2-style statistics computed separately on each split.",
        "",
        "## Split construction",
        "",
        "The split membership is not recomputed here; it is read directly from the processed task dictionary field `split`, whose keys are `train`, `val` and `test`. These split arrays contain edge indices in the processed temporal graph.",
        "",
        "The transductive protocol is chronological: earlier edges form the training context, later edges form validation and test windows. The inductive protocol additionally applies task-specific unseen-node filtering or context construction, so its train/validation/test edge indices can differ from the transductive protocol. Actual edge counts can deviate from an exact 70/15/15 ratio because the preprocessing preserves task-view constraints such as timestamp windows, relation masks and dataset-specific filtering.",
        "",
        "In the current processed files, `thgl_software` has identical split indices for inductive and transductive task views. The other five datasets do not: `dblp` and `aminer_paper_citation` keep the same total edge set but differ in split membership, while `patent_us`, `patent_in` and `patent_cn` differ in both selected edge set size and split membership.",
        "",
        "For `thgl_software`, timestamps are read from the original TGB2.0 edgelist CSV to avoid `float32` precision loss in large Unix timestamps. Other datasets use the processed task-view timestamps.",
        "",
        "## Dataset split summary",
        "",
        "| Dataset | Protocol | Train edges | Val edges | Test edges | Train span | Val span | Test span |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (dataset, protocol), ds_rows in by_dataset.items():
        row_map = {r["split"]: r for r in ds_rows}
        lines.append(
            "| {} | {} | {:,} | {:,} | {:,} | {}-{} | {}-{} | {}-{} |".format(
                dataset,
                protocol,
                row_map["train"]["num_edges"],
                row_map["val"]["num_edges"],
                row_map["test"]["num_edges"],
                row_map["train"]["ts_min"],
                row_map["train"]["ts_max"],
                row_map["val"]["ts_min"],
                row_map["val"]["ts_max"],
                row_map["test"]["ts_min"],
                row_map["test"]["ts_max"],
            )
        )

    lines.extend(
        [
            "",
            "## CSV columns",
            "",
            "| Column | Meaning and statistical procedure |",
            "| --- | --- |",
            "| `dataset` | Dataset name used in the formal experiments. |",
            "| `protocol` | Edge-prediction protocol: `inductive` or `transductive`. |",
            "| `split` | One of `train`, `val` or `test`. |",
            "| `task_file` | Processed `.pt` task view from which edges and split indices are read. |",
            "| `num_nodes` | Number of unique endpoint nodes appearing in this split. |",
            "| `num_edges` | Number of temporal edges in this split. |",
            "| `avg_degree_2e_over_n` | `2 * num_edges / num_nodes`, computed inside the split. |",
            "| `num_edge_types` | Number of relation IDs appearing in this split. |",
            "| `num_node_types` | Number of node types touched by split endpoint nodes. |",
            "| `active_timesteps` | Number of timestamps in the split that contain at least one edge. |",
            "| `time_unit` | Native timestep size inferred from the full dataset timestamp sequence and reused for all splits. |",
            "| `time_source` | `original_edgelist_csv` for raw TGB2 timestamps, otherwise `task_view`. |",
            "| `span_timesteps` | `(max(ts)-min(ts))/time_unit + 1` within the split, including empty timesteps. |",
            "| `inductive_nodes_vs_train` | For `val` and `test`, the proportion of split nodes unseen in the training split: `|split_nodes - train_nodes| / |split_nodes|`. For `train`, this is `0`. |",
            "| `drec` | Fraction of split edges `(s,r,o,t)` whose triple `(s,r,o)` also appears at `t - time_unit` in the full temporal graph. |",
            "| `rec` | Fraction of split edges `(s,r,o,t)` whose triple `(s,r,o)` appears at any earlier timestamp `< t` in the full temporal graph. |",
            "| `con` | Average maximum consecutive active run length across unique triples inside this split. |",
            "| `mean_edges_per_ts` | `num_edges / span_timesteps` within the split. |",
            "| `mean_nodes_per_ts` | Sum of active endpoint-node counts over observed timestamps divided by `span_timesteps`. |",
            "| `ts_min` | Minimum timestamp in the split. |",
            "| `ts_max` | Maximum timestamp in the split. |",
            "",
            "## Reproduction",
            "",
            "```bash",
            "/home/xuxj/.conda/envs/pytorch/bin/python scripts/datasets/compute_tgb_style_split_stats.py \\",
            "  --repo-root /home/xuxj/2026/dyn_relation_memory \\",
            "  --out-dir results/dataset_tgb_style_split_stats",
            "```",
            "",
        ]
    )
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", default="results/dataset_tgb_style_split_stats")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for name, protocol, rel_path in DEFAULT_PROTOCOL_TASKS:
        path = repo_root / rel_path
        print(f"computing {name} {protocol}: {path}")
        rows.extend(split_rows_for_dataset(name, protocol, path))

    csv_path = out_dir / "dataset_tgb_style_split_stats.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_latex(rows, out_dir / "dataset_tgb_style_split_stats_table.tex")
    write_readme(rows, out_dir / "README.md")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
