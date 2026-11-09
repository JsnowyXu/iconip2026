#!/usr/bin/env python3
import json
from pathlib import Path

import torch


DATASETS = ["patent_us", "patent_cn", "patent_in"]


def main() -> None:
    project_root = Path("/home/xuxj/2026/dyn_relation_memory")
    processed_root = project_root / "data" / "processed"
    task_root = processed_root / "tasks"
    report_root = project_root / "data" / "reports"

    lines = ["# patent Dataset Note", "", "## Main Processing Steps", ""]
    steps = [
        "1. Read the full raw CSV together with `clsIII-reInd.csv` and the IPC subclass table.",
        "2. Split multi-valued `orgID` and `codeIIIId` fields, and align organization names when present.",
        "3. Build a heterogeneous temporal graph with node types `patent`, `org`, and `codeIII`.",
        "4. Add relation types `patent_org`, `patent_codeIII`, `org_codeIII`, and co-assignee `org_coorg` edges.",
        "5. Initialize node text features from patent text, organization names, and IPC subclass descriptions.",
        "6. Export unified `.pt`, task-view `.pt`, TGN-style adapters, and TGB2/STHN adapters.",
        "7. Draw dataset analysis figures and save per-task statistics for reruns.",
    ]
    for ds in DATASETS:
        lines.append(f"### {ds}")
        lines.extend(steps)
        lines.append("")

    lines.extend(["## Unified Graph Statistics", ""])
    for ds in DATASETS:
        unified_path = processed_root / f"{ds}_unified.pt"
        if not unified_path.exists():
            continue
        obj = torch.load(unified_path, map_location="cpu")
        rel = obj["edges"]["rel"].numpy()
        node_type = obj["node_type"].numpy()
        labels = obj["node_multilabel"]["labels"].numpy()
        extra = obj["meta"].get("extra", {})
        rel_inv = {int(v): k for k, v in extra.get("relation_map", {}).items()}
        node_type_inv = {int(v): k for k, v in extra.get("node_type_map", {}).items()}
        label_names = extra.get("label_names", [])

        lines.append(f"### {ds}")
        lines.append("")
        lines.append("| Item | Value |")
        lines.append("| --- | ---: |")
        rows = [
            ("Nodes", obj["num_nodes"]),
            ("Edges", obj["edges"]["src"].shape[0]),
            ("Node types", obj["num_node_types"]),
            ("Relation types", obj["num_rels"]),
            ("Node feature dim", obj["node_feat"].shape[1]),
            ("Edge feature dim", obj["edges"]["edge_feat"].shape[1]),
            ("Labeled patents", obj["node_multilabel"]["node_ids"].shape[0]),
            ("Multilabel classes", obj["node_multilabel"]["labels"].shape[1]),
        ]
        for k, v in rows:
            lines.append(f"| {k} | {v} |")
        lines.append("")

        lines.append("| Node type id | Node type name | Count |")
        lines.append("| --- | --- | ---: |")
        for nid in sorted(node_type_inv):
            lines.append(f"| {nid} | {node_type_inv[nid]} | {int((node_type == nid).sum())} |")
        lines.append("")

        lines.append("| Relation id | Relation name | Count |")
        lines.append("| --- | --- | ---: |")
        for rid in sorted(rel_inv):
            lines.append(f"| {rid} | {rel_inv[rid]} | {int((rel == rid).sum())} |")
        lines.append("")

        lines.append("| Label id | Label name | Positive count |")
        lines.append("| --- | --- | ---: |")
        for i in range(min(labels.shape[1], 30)):
            name = label_names[i] if i < len(label_names) else {}
            label = f"{name.get('codeIIIId', i)} / {name.get('codeIII', '')}".strip()
            lines.append(f"| {i} | {label} | {int(labels[:, i].sum())} |")
        if labels.shape[1] > 30:
            lines.append(f"| ... | ... | {labels.shape[1] - 30} more labels omitted |")
        lines.append("")

        lines.append("Task views:")
        for suffix in ["edge_org_codeIII_transductive", "edge_org_codeIII_inductive", "node_patent_multilabel"]:
            lines.append(f"- `{task_root / f'{ds}_{suffix}.pt'}`")
        lines.append("")

        lines.append("Analysis outputs:")
        for name in ["degree_distribution.png", "temporal_density.png", "relation_distribution.png", "node_type_distribution.png", "community_sketch.png"]:
            lines.append(f"- `{report_root / ds / name}`")
        lines.append("")

    lines.extend(
        [
            "## DHC Representation Usage Check",
            "- Edge prediction does use other edge types as temporal context.",
            "- The patent task views keep the full heterogeneous stream and only restrict supervision with `edge_label` on `org_codeIII`.",
            "- In `dhc/models/dhc_model.py`, link prediction consumes `rel_emb`, relation memory, edge features, and time encoding, not only node states.",
            "- Node multilabel classification already replays the temporal edge stream before calling `predict_node_heads(...)`.",
            "- `predict_node_heads(...)` uses `_node_repr(...)`, which fuses node memory, recent-neighbor context, relation-aware neighbor messages, and community context.",
            "- Current conclusion: edge-type and edge-feature information is already used for both edge prediction and node classification; no immediate DHC patch is required before first-round patent experiments.",
            "",
        ]
    )

    (project_root / "patent.md").write_text("\n".join(lines), encoding="utf-8")
    print(project_root / "patent.md")


if __name__ == "__main__":
    main()
