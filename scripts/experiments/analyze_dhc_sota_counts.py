#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


CURRENT_NAME = "Current(dyn_hetero_community)"


def parse_cell(text: str) -> Dict[str, float]:
    if not isinstance(text, str) or text.strip() in {"", "-", "nan", "None"}:
        return {}
    out: Dict[str, float] = {}
    m = re.search(r"MRR/AUC/AP=([0-9.]+)/([0-9.]+)/([0-9.]+)", text)
    if m:
        out["mrr"] = float(m.group(1))
        out["auc"] = float(m.group(2))
        out["ap"] = float(m.group(3))
    m = re.search(r"AUC/AP=([0-9.]+)/([0-9.]+)", text)
    if m:
        out.setdefault("auc", float(m.group(1)))
        out.setdefault("ap", float(m.group(2)))
    m = re.search(r"F1micro=([0-9.]+)", text)
    if m:
        out["f1micro"] = float(m.group(1))
    m = re.search(r"F1macro=([0-9.]+)", text)
    if m:
        out["f1macro"] = float(m.group(1))
    m = re.search(r"mAP=([0-9.]+)", text)
    if m:
        out["map"] = float(m.group(1))
    return out


def collect_rows(tables_dir: Path) -> pd.DataFrame:
    files = ["table_transductive.csv", "table_inductive.csv", "table_multilabel.csv"]
    rows: List[Dict] = []
    for fn in files:
        fp = tables_dir / fn
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        datasets = [c for c in df.columns if c != "method"]
        for dataset in datasets:
            metrics_rows = []
            for _, r in df.iterrows():
                m = parse_cell(r[dataset])
                if m:
                    metrics_rows.append((str(r["method"]), m))
            metric_names = sorted({k for _, m in metrics_rows for k in m})
            for metric in metric_names:
                vals = [(meth, mm[metric]) for meth, mm in metrics_rows if metric in mm]
                if not vals:
                    continue
                best = max(v for _, v in vals)
                curr_vals = [v for meth, v in vals if meth == CURRENT_NAME]
                if not curr_vals:
                    continue
                cur = curr_vals[0]
                ties = [meth for meth, v in vals if abs(v - best) <= 1e-12]
                rows.append(
                    {
                        "table": fn.replace("table_", "").replace(".csv", ""),
                        "dataset": dataset,
                        "metric": metric,
                        "current": float(cur),
                        "best": float(best),
                        "is_best": bool(abs(cur - best) <= 1e-12),
                        "n_ties": int(len(ties)),
                        "ties": "|".join(ties),
                    }
                )
    return pd.DataFrame(rows)


def render_md(df: pd.DataFrame, out_md: Path) -> None:
    comparable = df[df["metric"].isin(["auc", "ap", "f1micro"])].copy()
    wins = int(comparable["is_best"].sum()) if not comparable.empty else 0
    total = int(len(comparable))
    lines = [
        "# DHC SOTA Count Summary",
        "",
        f"- Comparable metrics (AUC/AP/F1micro): **{wins}/{total}**",
        "",
    ]
    if not comparable.empty:
        g = comparable.groupby("table")["is_best"].agg(["sum", "count"]).reset_index()
        lines.append("## By Task Type")
        lines.append("")
        lines.append("| table | wins | total |")
        lines.append("|---|---:|---:|")
        for _, r in g.iterrows():
            lines.append(f"| {r['table']} | {int(r['sum'])} | {int(r['count'])} |")
        lines.append("")

        not_best = comparable[~comparable["is_best"]].sort_values(["table", "dataset", "metric"])
        lines.append("## Not Best")
        lines.append("")
        lines.append("| table | dataset | metric | current | best | best method |")
        lines.append("|---|---|---|---:|---:|---|")
        for _, r in not_best.iterrows():
            lines.append(
                f"| {r['table']} | {r['dataset']} | {r['metric']} | "
                f"{r['current']:.4f} | {r['best']:.4f} | {r['ties']} |"
            )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tables-dir",
        type=str,
        default="/home/xuxj/2026/dyn_hetero_community/results/dataset_statistics",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="/home/xuxj/2026/dyn_hetero_community/results/dataset_statistics/dhc_sota_breakdown.csv",
    )
    ap.add_argument(
        "--out-md",
        type=str,
        default="/home/xuxj/2026/dyn_hetero_community/results/dataset_statistics/dhc_sota_summary.md",
    )
    args = ap.parse_args()

    tables_dir = Path(args.tables_dir)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    df = collect_rows(tables_dir)
    if df.empty:
        print("[analyze_dhc_sota_counts] no rows parsed")
        return
    df.to_csv(out_csv, index=False)
    render_md(df, out_md)
    comparable = df[df["metric"].isin(["auc", "ap", "f1micro"])]
    print(
        f"[analyze_dhc_sota_counts] comparable wins={int(comparable['is_best'].sum())}/{len(comparable)} "
        f"csv={out_csv} md={out_md}"
    )


if __name__ == "__main__":
    main()
