#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch


def summarize_task(path: Path) -> Dict:
    obj = torch.load(path, map_location="cpu")
    src = obj["edges"]["src"]
    dst = obj["edges"]["dst"]
    split = obj["split"]
    tr = split["train"].long()
    va = split["val"].long()
    te = split["test"].long()
    extra = obj.get("meta", {}).get("extra", {})

    train_src = set(src[tr].tolist()) if tr.numel() > 0 else set()
    test_src = set(src[te].tolist()) if te.numel() > 0 else set()
    train_dst = set(dst[tr].tolist()) if tr.numel() > 0 else set()
    test_dst = set(dst[te].tolist()) if te.numel() > 0 else set()

    task_name = path.stem
    is_node_task = "node" in task_name
    return {
        "task": task_name,
        "path": str(path),
        "num_edges": int(src.numel()),
        "num_rels": int(obj["num_rels"]),
        "train_edges": int(tr.numel()),
        "val_edges": int(va.numel()),
        "test_edges": int(te.numel()),
        "edge_task_ready": bool(tr.numel() > 0 and va.numel() > 0 and te.numel() > 0),
        "train_test_src_overlap": int(len(train_src & test_src)),
        "train_test_dst_overlap": int(len(train_dst & test_dst)),
        "has_node_multilabel": bool(obj.get("node_multilabel") is not None),
        "node_label_classes": int(obj["node_multilabel"]["labels"].shape[1]) if obj.get("node_multilabel") is not None else 0,
        "relation_source": extra.get("relation_source", ""),
        "task_view_name": extra.get("task_view_name", ""),
        "node_protocol_note": "implicit_from_edge_history" if is_node_task else "",
    }


def to_markdown(rows: List[Dict]) -> str:
    lines = [
        "# Task View Audit",
        "",
        "| task | edges | rels | train | val | test | edge_ready | src_overlap | dst_overlap | relation_source | note |",
        "|:--|--:|--:|--:|--:|--:|:--|--:|--:|:--|:--|",
    ]
    for r in rows:
        lines.append(
            "| {task} | {num_edges} | {num_rels} | {train_edges} | {val_edges} | {test_edges} | {edge_task_ready} | "
            "{train_test_src_overlap} | {train_test_dst_overlap} | {relation_source} | {node_protocol_note} |".format(
                **r
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="/home/xuxj/2026/dyn_hetero_community")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    task_dirs = [
        project_root / "data" / "processed" / "paper_tasks",
        project_root / "data" / "processed" / "tasks",
    ]
    rows: List[Dict] = []
    for task_dir in task_dirs:
        if not task_dir.exists():
            continue
        for path in sorted(task_dir.glob("*.pt")):
            rows.append(summarize_task(path))

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(to_markdown(rows), encoding="utf-8")

    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
