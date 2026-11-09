#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd


def export_tgn(obj, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    src = obj["edges"]["src"].numpy()
    dst = obj["edges"]["dst"].numpy()
    ts = obj["edges"]["ts"].numpy()
    ef = obj["edges"]["edge_feat"].numpy()

    csv_path = out_dir / f"{name}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("u,i,ts,label,f0,f1\n")
        for i in range(len(src)):
            f.write(f"{int(src[i])},{int(dst[i])},{float(ts[i])},1,{float(ef[i,0])},{float(ef[i,1])}\n")

    # Directly provide preprocessed format expected by utils/data_processing.py
    df = pd.DataFrame({"u": src + 1, "i": dst + 1, "ts": ts, "label": 1, "idx": range(1, len(src) + 1)})
    df.to_csv(out_dir / f"ml_{name}.csv", index=False)
    np.save(out_dir / f"ml_{name}.npy", np.vstack([np.zeros((1, ef.shape[1]), dtype=np.float32), ef]))
    node_feat = obj["node_feat"].numpy().astype(np.float32)
    np.save(out_dir / f"ml_{name}_node.npy", np.vstack([np.zeros((1, node_feat.shape[1]), dtype=np.float32), node_feat]))



def export_tgb2(obj, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "u": obj["edges"]["src"].numpy(),
            "i": obj["edges"]["dst"].numpy(),
            "ts": obj["edges"]["ts"].numpy(),
            "edge_type": obj["edges"]["rel"].numpy(),
            "label": 1,
            "idx": range(obj["edges"]["src"].shape[0]),
        }
    )
    df.to_csv(out_dir / f"{name}_edgelist.csv", index=False)
    pd.DataFrame({"node_id": range(obj["num_nodes"]), "node_type": obj["node_type"].numpy()}).to_csv(
        out_dir / f"{name}_nodetype.csv", index=False
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=str)
    p.add_argument("--name", required=True, type=str)
    p.add_argument("--out-dir", required=True, type=str)
    args = p.parse_args()

    obj = torch.load(args.input, map_location="cpu")
    base = Path(args.out_dir)
    export_tgn(obj, base / "tgn", args.name)
    export_tgb2(obj, base / "tgb2", args.name)
    print(f"Exported baseline adapter files into {base}")


if __name__ == "__main__":
    main()
