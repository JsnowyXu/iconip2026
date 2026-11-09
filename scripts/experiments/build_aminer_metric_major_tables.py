#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


TASK_ORDER = [
    "aminer_paper_citation_edge_transductive",
    "aminer_paper_citation_edge_inductive",
    "aminer_paper_coauthor_edge_transductive",
    "aminer_paper_coauthor_edge_inductive",
    "aminer_paper_node_multilabel",
]

TASK_LABEL = {
    "aminer_paper_citation_edge_transductive": "Citation-Trans",
    "aminer_paper_citation_edge_inductive": "Citation-Ind",
    "aminer_paper_coauthor_edge_transductive": "Coauthor-Trans",
    "aminer_paper_coauthor_edge_inductive": "Coauthor-Ind",
    "aminer_paper_node_multilabel": "Node-ML",
}

MODEL_LABEL = {
    "DHC": "DHC",
    "tgn": "TGN",
    "tgn_edgetype": "TGN-edges",
    "sthn": "STHN",
    "jodie": "JODIE",
    "dyrep": "DyRep",
    "tgat": "TGAT",
    "cawn": "CAWN",
    "tcl": "TCL",
    "graphmixer": "GraphMixer",
    "dygformer": "DyGFormer",
}


def normalize_model_name(x: str) -> str:
    key = str(x or "").strip()
    if key in MODEL_LABEL:
        return MODEL_LABEL[key]
    low = key.lower()
    return MODEL_LABEL.get(low, key)


def load_dhc(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df = df.copy()
    df["source"] = "dhc"
    df["model_label"] = "DHC"
    df["best_test_mrr"] = pd.to_numeric(df.get("test_mrr"), errors="coerce")
    df["best_test_auc"] = pd.to_numeric(df.get("test_auc"), errors="coerce")
    df["best_test_ap"] = pd.to_numeric(df.get("test_ap"), errors="coerce")
    df["best_node_ml_micro_f1"] = pd.to_numeric(df.get("test_node_ml_micro_f1"), errors="coerce")
    df["best_node_ml_macro_f1"] = pd.to_numeric(df.get("test_node_ml_macro_f1"), errors="coerce")
    return df[
        [
            "dataset",
            "model_label",
            "task_name",
            "task_type",
            "relation_target",
            "best_test_mrr",
            "best_test_auc",
            "best_test_ap",
            "best_node_ml_micro_f1",
            "best_node_ml_macro_f1",
            "source",
        ]
    ]


def load_baseline(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df = df.copy()
    df["source"] = "baseline"
    df["model_label"] = df["model"].map(normalize_model_name)
    return df[
        [
            "dataset",
            "model_label",
            "task_name",
            "task_type",
            "relation_target",
            "best_test_mrr",
            "best_test_auc",
            "best_test_ap",
            "best_node_ml_micro_f1",
            "best_node_ml_macro_f1",
            "source",
        ]
    ]


def write_metric_table(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    use = df[["model_label", "task_name", metric]].copy()
    use = use[pd.to_numeric(use[metric], errors="coerce").notna()].copy()
    if use.empty:
        return
    use["task_name"] = pd.Categorical(use["task_name"], categories=TASK_ORDER, ordered=True)
    use = use.sort_values(["model_label", "task_name"])
    wide = use.pivot_table(index="model_label", columns="task_name", values=metric, aggfunc="max")
    wide = wide.reindex(columns=[x for x in TASK_ORDER if x in wide.columns])
    wide.columns = [TASK_LABEL.get(x, x) for x in wide.columns]
    wide = wide.sort_index()
    wide.to_csv(out_dir / f"{metric}_table.csv")

    md_lines = [f"# {metric} table", ""]
    md_lines.append("| Model | " + " | ".join(wide.columns.tolist()) + " |")
    md_lines.append("| --- | " + " | ".join(["---:"] * len(wide.columns)) + " |")
    for model, row in wide.iterrows():
        vals = []
        for col in wide.columns:
            v = row[col]
            vals.append("" if pd.isna(v) else f"{float(v):.6f}")
        md_lines.append("| " + str(model) + " | " + " | ".join(vals) + " |")
    (out_dir / f"{metric}_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser("Build AMiner metric-major tables")
    ap.add_argument("--dhc-summary-csv", type=str, default="")
    ap.add_argument("--baseline-summary-csv", type=str, default="")
    ap.add_argument("--out-dir", type=str, default="/home/xuxj/2026/dyn_relation_memory/results/aminer_metric_tables")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    if args.dhc_summary_csv:
        frames.append(load_dhc(Path(args.dhc_summary_csv)))
    if args.baseline_summary_csv:
        frames.append(load_baseline(Path(args.baseline_summary_csv)))
    frames = [x for x in frames if not x.empty]
    if not frames:
        print("no input rows found")
        return

    df = pd.concat(frames, axis=0, ignore_index=True)
    df = df[df["dataset"] == "aminer_paper"].copy()
    df.to_csv(out_dir / "aminer_all_results_long.csv", index=False)

    for metric in [
        "best_test_mrr",
        "best_test_auc",
        "best_test_ap",
        "best_node_ml_micro_f1",
        "best_node_ml_macro_f1",
    ]:
        write_metric_table(df, metric=metric, out_dir=out_dir)

    print(f"saved AMiner metric-major tables to {out_dir}")


if __name__ == "__main__":
    main()
