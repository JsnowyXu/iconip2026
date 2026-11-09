#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=str)
    p.add_argument("--out", required=True, type=str)
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    plt.figure(figsize=(7, 4))
    if "loss" in df.columns:
        plt.plot(df["epoch"], df["loss"], label="train loss", color="#1d3557")
    if "val_mrr" in df.columns:
        plt.plot(df["epoch"], df["val_mrr"], label="val MRR", color="#e63946")
    if "val_auc" in df.columns:
        plt.plot(df["epoch"], df["val_auc"], label="val AUC", color="#2a9d8f")
    plt.xlabel("epoch")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160)


if __name__ == "__main__":
    main()
