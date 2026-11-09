#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_METRICS = [
    "test_mrr",
    "test_auc",
    "test_ap",
    "node_ml_micro_f1",
    "node_ml_map",
]


def find_extremes(df: pd.DataFrame, low_thr: float, high_thr: float) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        for m in TARGET_METRICS:
            if m not in r or pd.isna(r[m]):
                continue
            v = float(r[m])
            if v <= low_thr or v >= high_thr:
                rows.append(
                    {
                        "name": r.get("name", r.get("run_name", "")),
                        "dataset": r.get("dataset", ""),
                        "task_type": r.get("task_type", ""),
                        "metric": m,
                        "value": v,
                        "extreme": "low" if v <= low_thr else "high",
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser("Check near-0 / near-1 metric extremes")
    ap.add_argument("--inputs", nargs="+", required=True, help="csv files, e.g., cuda summaries/grid summary")
    ap.add_argument("--out-csv", required=True, type=str)
    ap.add_argument("--out-md", required=True, type=str)
    ap.add_argument("--low-thr", type=float, default=0.01)
    ap.add_argument("--high-thr", type=float, default=0.99)
    args = ap.parse_args()

    frames = []
    for p in args.inputs:
        fp = Path(p)
        if not fp.exists():
            continue
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        df["source_csv"] = str(fp)
        frames.append(df)

    if not frames:
        raise RuntimeError("no valid input csv found")

    all_df = pd.concat(frames, axis=0, ignore_index=True)
    ex = find_extremes(all_df, low_thr=float(args.low_thr), high_thr=float(args.high_thr))

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    ex.to_csv(out_csv, index=False)

    lines = [
        "# Metric Extremes Report",
        "",
        f"- low threshold: <= {args.low_thr}",
        f"- high threshold: >= {args.high_thr}",
        f"- num extreme rows: {len(ex)}",
        "",
    ]
    if len(ex) > 0:
        lines.append(ex.sort_values(["metric", "dataset", "value"]).to_markdown(index=False))
        lines.extend(
            [
                "",
                "## Interpretation Tips",
                "",
                "- Near-0 micro-F1 with high AUC/AP often indicates threshold mismatch under class imbalance.",
                "- Near-1 metrics can indicate either solved task or label/setting degeneracy; check label distribution.",
                "- Verify with class stats in `results/dataset_statistics/node_multilabel_stats.csv`.",
            ]
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_csv)
    print(out_md)


if __name__ == "__main__":
    main()
