#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

pd.set_option("display.max_colwidth", None)


def to_tex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    tex = df.to_latex(index=False, escape=True, longtable=False)
    body = []
    body.append("\\begin{table*}[t]")
    body.append("\\centering")
    body.append("\\scriptsize")
    body.append("\\setlength{\\tabcolsep}{3pt}")
    body.append(f"\\caption{{{caption}}}")
    body.append(f"\\label{{{label}}}")
    body.append("\\resizebox{\\textwidth}{!}{%")
    body.append(tex.rstrip())
    body.append("}%")
    body.append("\\end{table*}")
    return "\n".join(body) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        default="/home/xuxj/2026/dyn_hetero_community/results/dataset_statistics",
        type=str,
    )
    ap.add_argument(
        "--out-dir",
        default="/home/xuxj/2026/dyn_hetero_community/paper_draft/tables",
        type=str,
    )
    args = ap.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    specs = [
        (
            inp / "table_transductive.csv",
            out / "baseline_transductive_table.tex",
            "Transductive Link Prediction on Five Datasets (available methods in the current workspace; `-` denotes unavailable or excluded results)",
            "tab:baseline_transductive",
        ),
        (
            inp / "table_inductive.csv",
            out / "baseline_inductive_table.tex",
            "Inductive Link Prediction on Five Datasets (available methods in the current workspace; `-` denotes unavailable or excluded results)",
            "tab:baseline_inductive",
        ),
        (
            inp / "table_multilabel.csv",
            out / "baseline_multilabel_table.tex",
            "Node Multi-label Classification on Five Datasets (available methods in the current workspace; `-` denotes unavailable or excluded results)",
            "tab:baseline_multilabel",
        ),
    ]

    generated = []
    for csv_path, tex_path, caption, label in specs:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing input: {csv_path}")
        df = pd.read_csv(csv_path)
        tex = to_tex_table(df, caption=caption, label=label)
        tex_path.write_text(tex, encoding="utf-8")
        generated.append(tex_path)

    bundle = out / "baseline_task_tables.tex"
    bundle.write_text(
        "\n".join(
            [
                "% Auto-generated bundle for baseline task tables",
                "\\input{tables/baseline_transductive_table}",
                "\\input{tables/baseline_inductive_table}",
                "\\input{tables/baseline_multilabel_table}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("generated:")
    for p in generated + [bundle]:
        print(p)


if __name__ == "__main__":
    main()
