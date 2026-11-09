#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def f4(x):
    try:
        if pd.isna(x):
            return "--"
        return f"{float(x):.4f}"
    except Exception:
        return "--"


def esc(s):
    return str(s).replace("_", "\\_")


def build_dhc_table(df: pd.DataFrame) -> str:
    dhc = df[df["model"].astype(str).str.startswith("dhc_")].copy()
    rows = []
    rows.append("\\begin{table}[ht]")
    rows.append("\\centering")
    rows.append("\\caption{DHC results on patent task views (executed outputs).}")
    rows.append("\\begin{tabular}{lccccc}")
    rows.append("\\toprule")
    rows.append("Task & MRR & AUC & AP & Micro-F1 & Epoch(s) \\\\")
    rows.append("\\midrule")
    for _, r in dhc.iterrows():
        task = esc(r["task"])
        rows.append(
            f"{task} & {f4(r.get('test_mrr'))} & {f4(r.get('test_auc'))} & "
            f"{f4(r.get('test_ap'))} & {f4(r.get('node_ml_micro_f1'))} & {f4(r.get('avg_epoch_time_sec'))} \\\\"
        )
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\end{table}")
    return "\n".join(rows)


def build_edge_baseline_table(df: pd.DataFrame) -> str:
    tgb2 = df[(df["task"] == "tgb2_raw_cn_assignee_ipc") & (df["split"] == "test")][["task", "auc", "ap"]]
    tgb2_p = df[(df["task"] == "tgb2_raw_cn_patent_ipc") & (df["split"] == "test")][["task", "auc", "ap"]]
    dyg_tgn = df[(df["task"] == "dyglib_tgn_raw_cn_assignee_ipc") & (df["split"] == "test")][["task", "auc", "ap"]]
    dyg_tgn_p = df[(df["task"] == "dyglib_tgn_raw_cn_patent_ipc") & (df["split"] == "test")][["task", "auc", "ap"]]
    tgn_ext = df[(df["task"] == "tgn_external_raw_cn_assignee_ipc") & (df["split"] == "old_nodes")][["task", "auc", "ap"]]
    tgn_ext_new = df[(df["task"] == "tgn_external_raw_cn_assignee_ipc") & (df["split"] == "new_nodes")][["task", "auc", "ap"]]
    tgn_ext_p = df[(df["task"] == "tgn_external_raw_cn_patent_ipc") & (df["split"] == "old_nodes")][["task", "auc", "ap"]]
    tgn_ext_p_new = df[(df["task"] == "tgn_external_raw_cn_patent_ipc") & (df["split"] == "new_nodes")][["task", "auc", "ap"]]
    zoo = df[df["task"] == "dyglib_model_zoo_raw_cn_assignee_ipc"][["model", "test_auc", "test_ap"]]

    rows = []
    rows.append("\\begin{table}[ht]")
    rows.append("\\centering")
    rows.append("\\caption{Baseline link prediction results on patent edge tasks (executed outputs).}")
    rows.append("\\begin{tabular}{lccc}")
    rows.append("\\toprule")
    rows.append("Model & Task & AUC & AP \\\\")
    rows.append("\\midrule")

    if len(tgb2) > 0:
        r = tgb2.iloc[0]
        rows.append(f"TGB2-custom & assignee$\\rightarrow$ipc & {f4(r['auc'])} & {f4(r['ap'])} \\\\")
    if len(tgb2_p) > 0:
        r = tgb2_p.iloc[0]
        rows.append(f"TGB2-custom & patent$\\rightarrow$ipc & {f4(r['auc'])} & {f4(r['ap'])} \\\\")
    if len(dyg_tgn) > 0:
        r = dyg_tgn.iloc[0]
        rows.append(f"DyGLib-TGN-custom & assignee$\\rightarrow$ipc & {f4(r['auc'])} & {f4(r['ap'])} \\\\")
    if len(dyg_tgn_p) > 0:
        r = dyg_tgn_p.iloc[0]
        rows.append(f"DyGLib-TGN-custom & patent$\\rightarrow$ipc & {f4(r['auc'])} & {f4(r['ap'])} \\\\")
    if len(tgn_ext) > 0:
        r = tgn_ext.iloc[0]
        rows.append(f"TGN-external(old) & assignee$\\rightarrow$ipc & {f4(r['auc'])} & {f4(r['ap'])} \\\\")
    if len(tgn_ext_new) > 0:
        r = tgn_ext_new.iloc[0]
        rows.append(f"TGN-external(new) & assignee$\\rightarrow$ipc & {f4(r['auc'])} & {f4(r['ap'])} \\\\")
    if len(tgn_ext_p) > 0:
        r = tgn_ext_p.iloc[0]
        rows.append(f"TGN-external(old) & patent$\\rightarrow$ipc & {f4(r['auc'])} & {f4(r['ap'])} \\\\")
    if len(tgn_ext_p_new) > 0:
        r = tgn_ext_p_new.iloc[0]
        rows.append(f"TGN-external(new) & patent$\\rightarrow$ipc & {f4(r['auc'])} & {f4(r['ap'])} \\\\")
    for _, r in zoo.iterrows():
        model_name = esc(r["model"])
        rows.append(
            f"DyGLib-{model_name} & assignee$\\rightarrow$ipc & "
            f"{f4(r.get('test_auc'))} & {f4(r.get('test_ap'))} \\\\"
        )

    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\end{table}")
    return "\n".join(rows)


def build_node_baseline_table(df: pd.DataFrame) -> str:
    node = df[df["task"] == "dyglib_model_zoo_raw_cn_node_task"][["model", "test_auc", "test_ap", "node_ml_micro_f1"]]

    rows = []
    rows.append("\\begin{table}[ht]")
    rows.append("\\centering")
    rows.append("\\caption{DyGLib model-zoo on node-task view (executed outputs).}")
    rows.append("\\begin{tabular}{lccc}")
    rows.append("\\toprule")
    rows.append("Model & Link AUC & Link AP & Node Micro-F1 \\\\")
    rows.append("\\midrule")
    for _, r in node.iterrows():
        model_name = esc(r["model"])
        rows.append(
            f"{model_name} & {f4(r.get('test_auc'))} & "
            f"{f4(r.get('test_ap'))} & {f4(r.get('node_ml_micro_f1'))} \\\\"
        )
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\end{table}")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", required=True, type=str)
    ap.add_argument("--out-tex", required=True, type=str)
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    parts = [
        "% Auto-generated by scripts/experiments/render_patent_tables_to_latex.py",
        build_dhc_table(df),
        "",
        build_edge_baseline_table(df),
        "",
        build_node_baseline_table(df),
        "",
    ]
    out = Path(args.out_tex)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
