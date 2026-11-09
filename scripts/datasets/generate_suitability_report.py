#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def judge(row):
    reasons = []
    has_time = row["time_span"] > 0
    is_hetero = row["num_relations"] > 1 or row["num_node_types"] > 1
    enough_scale = row["num_edges"] >= 5000

    link_ok = has_time and enough_scale
    node_ok = link_ok and row["has_multiclass_label"]
    hotspot_ok = has_time and enough_scale and is_hetero

    if has_time:
        reasons.append("含时间信息")
    if is_hetero:
        reasons.append("含多类型关系/节点")
    if enough_scale:
        reasons.append("样本规模满足动态任务")
    if not reasons:
        reasons.append("信息不足")

    return link_ok, node_ok, hotspot_ok, "、".join(reasons)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stats-csv", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.stats_csv)
    lines = []
    lines.append("# 数据集适用性评估\n")
    lines.append("| 数据集 | 动态链接预测 | 节点分类 | 热点推荐 | 结论依据 |")
    lines.append("|---|---|---|---|---|")

    for _, row in df.iterrows():
        link_ok, node_ok, hot_ok, reason = judge(row)
        lines.append(
            f"| {row['dataset']} | {'适合' if link_ok else '受限'} | {'适合' if node_ok else '受限'} | {'适合' if hot_ok else '受限'} | {reason} |"
        )

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
