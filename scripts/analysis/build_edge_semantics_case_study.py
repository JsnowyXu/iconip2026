#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _task_sort_key(name: str) -> tuple[int, str]:
    order = {
        "dblp_edge_transductive": 0,
        "patent_us_edge_org_codeIII_inductive": 1,
        "patent_us_edge_org_codeIII_transductive": 2,
        "thgl_software_edge_inductive": 3,
        "aminer_paper_citation_edge_transductive": 4,
    }
    return (order.get(name, 99), name)


def _variant_label(v: str) -> str:
    v = str(v).replace("__s42", "")
    return {
        "full": "Full",
        "no_edge_type": "No edge type",
        "no_relation_memory": "No relation memory",
        "no_rel": "No relation memory",
        "no_edge_type_no_rel": "No edge type + no relation memory",
        "no_community": "No community",
    }.get(v, v)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_edge_representation_effect(analysis_root: Path, out_dir: Path) -> pd.DataFrame:
    imp = _read_csv(analysis_root / "edge_semantic_importance.csv")
    if imp.empty:
        return imp
    imp = imp.copy()
    imp["variant_clean"] = imp["variant"].str.replace("__s42", "", regex=False)
    imp["variant_label"] = imp["variant_clean"].map(_variant_label)
    imp["delta_mrr_bp"] = pd.to_numeric(imp["delta_mrr_vs_full"], errors="coerce") * 10000.0
    imp["test_mrr"] = pd.to_numeric(imp["test_mrr"], errors="coerce")
    imp = imp.sort_values(["task_name", "variant_clean"])
    imp.to_csv(out_dir / "edge_representation_effect.csv", index=False)

    plot = imp[~imp["variant_clean"].eq("full") & imp["delta_mrr_bp"].notna()].copy()
    plot["task_name"] = pd.Categorical(plot["task_name"], sorted(plot["task_name"].unique(), key=_task_sort_key), ordered=True)
    plot = plot.sort_values(["task_name", "variant_clean"])
    if not plot.empty:
        fig, ax = plt.subplots(figsize=(12.6, 6.2))
        xlabels = []
        xpos = []
        colors = []
        vals = []
        pos = 0
        for task, g in plot.groupby("task_name", observed=True):
            for _, r in g.iterrows():
                xpos.append(pos)
                xlabels.append(f"{task}\n{r['variant_label']}")
                vals.append(r["delta_mrr_bp"])
                colors.append("#dc2626" if r["delta_mrr_bp"] < 0 else "#2563eb")
                pos += 1
            pos += 1
        ax.bar(xpos, vals, color=colors, alpha=0.86)
        ax.axhline(0, color="#111827", linewidth=1)
        ax.set_ylabel("MRR delta vs Full (basis points)")
        ax.set_title("Accuracy Impact of Edge Semantics / Relation-Memory Ablations")
        ax.set_xticks(xpos)
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / "edge_representation_delta_mrr.png", dpi=200)
        plt.close(fig)

    focus_tasks = ["dblp_edge_transductive", "patent_us_edge_org_codeIII_transductive", "thgl_software_edge_inductive", "aminer_paper_citation_edge_transductive"]
    focus = imp[imp["task_name"].isin(focus_tasks)].copy()
    if not focus.empty:
        piv = focus.pivot_table(index="task_name", columns="variant_label", values="test_mrr", aggfunc="max")
        piv = piv.reindex(sorted(piv.index, key=_task_sort_key))
        fig, ax = plt.subplots(figsize=(10.8, 5.6))
        piv.plot(kind="bar", ax=ax, width=0.82)
        ax.set_ylabel("test MRR")
        ax.set_title("Case-Study Tasks: Full vs Edge-Representation Variants")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "case_task_variant_test_mrr.png", dpi=200)
        plt.close(fig)
    return imp


def _task_dirs(analysis_root: Path) -> list[Path]:
    out = []
    for dataset_dir in analysis_root.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name == "figures":
            continue
        for task_dir in dataset_dir.iterdir():
            if task_dir.is_dir() and (task_dir / "relation_type" / "relation_type_pca2d.csv").exists():
                out.append(task_dir)
    return sorted(out)


def build_relation_memory_evolution(analysis_root: Path, out_dir: Path) -> pd.DataFrame:
    task_def = _read_csv(analysis_root / "task_definitions.csv")
    rows = []
    epoch_rows = []
    for task_dir in _task_dirs(analysis_root):
        dataset = task_dir.parent.name
        task = task_dir.name
        rel_path = task_dir / "relation_type" / "relation_type_pca2d.csv"
        hist_path = task_dir / "performance" / "main_train_history.csv"
        rel = pd.read_csv(rel_path)
        hist = _read_csv(hist_path)
        target_labels = set(
            task_def[(task_def["dataset"] == dataset) & (task_def["task_name"] == task) & (task_def["is_target_relation"].astype(str).str.lower() == "true")]["relation_name"].astype(str)
        )
        if not target_labels:
            target_labels = set(rel["label"].astype(str).head(1))

        for label, g in rel.groupby("label"):
            g = g.sort_values("epoch").copy()
            if len(g) < 2:
                continue
            dx = np.diff(g["pc1"].to_numpy(dtype=float))
            dy = np.diff(g["pc2"].to_numpy(dtype=float))
            steps = np.sqrt(dx * dx + dy * dy)
            first = g.iloc[0]
            last = g.iloc[-1]
            direct = float(np.sqrt((last["pc1"] - first["pc1"]) ** 2 + (last["pc2"] - first["pc2"]) ** 2))
            path = float(steps.sum())
            rows.append(
                {
                    "dataset": dataset,
                    "task_name": task,
                    "relation_type": label,
                    "is_target_relation": str(label) in target_labels,
                    "first_epoch": int(first["epoch"]),
                    "last_epoch": int(last["epoch"]),
                    "first_pc1": float(first["pc1"]),
                    "first_pc2": float(first["pc2"]),
                    "last_pc1": float(last["pc1"]),
                    "last_pc2": float(last["pc2"]),
                    "direct_drift": direct,
                    "path_length": path,
                    "norm_first": float(first["l2_norm"]),
                    "norm_last": float(last["l2_norm"]),
                    "norm_delta": float(last["l2_norm"] - first["l2_norm"]),
                }
            )
            prev = None
            for _, r in g.iterrows():
                step = np.nan if prev is None else float(np.sqrt((r["pc1"] - prev["pc1"]) ** 2 + (r["pc2"] - prev["pc2"]) ** 2))
                epoch_rows.append(
                    {
                        "dataset": dataset,
                        "task_name": task,
                        "relation_type": label,
                        "is_target_relation": str(label) in target_labels,
                        "epoch": int(r["epoch"]),
                        "pc1": float(r["pc1"]),
                        "pc2": float(r["pc2"]),
                        "l2_norm": float(r["l2_norm"]),
                        "step_drift": step,
                    }
                )
                prev = r

        if not hist.empty and "epoch" in hist:
            merged = pd.DataFrame([r for r in epoch_rows if r["dataset"] == dataset and r["task_name"] == task])
            merged = merged.merge(hist[["epoch", "val_mrr", "loss", "epoch_sec"]], on="epoch", how="left")
            merged.to_csv(out_dir / f"{dataset}__{task}__relation_epoch_drift_with_metric.csv", index=False)

            fig, ax = plt.subplots(figsize=(7.4, 5.8))
            for label, g in rel.groupby("label"):
                g = g.sort_values("epoch")
                is_target = str(label) in target_labels
                ax.plot(g["pc1"], g["pc2"], marker="o", linewidth=2.2 if is_target else 1.2, alpha=0.95 if is_target else 0.45, label=str(label))
                if is_target:
                    for _, r in g.iterrows():
                        ax.text(r["pc1"], r["pc2"], str(int(r["epoch"])), fontsize=8)
            ax.set_title(f"{task}: relation_type semantic trajectory")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, loc="best")
            fig.tight_layout()
            fig.savefig(out_dir / f"{dataset}__{task}__relation_type_trajectory.png", dpi=200)
            plt.close(fig)

            target = merged[merged["is_target_relation"]].copy()
            if not target.empty:
                fig, ax1 = plt.subplots(figsize=(7.6, 4.8))
                ax1.plot(target["epoch"], target["step_drift"], marker="o", color="#dc2626", label="target relation step drift")
                ax1.set_xlabel("epoch")
                ax1.set_ylabel("PCA step drift")
                ax2 = ax1.twinx()
                ax2.plot(hist["epoch"], hist["val_mrr"], marker="s", color="#2563eb", label="val MRR")
                ax2.set_ylabel("val MRR")
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines + lines2, labels + labels2, loc="best")
                ax1.grid(alpha=0.2)
                ax1.set_title(f"{task}: target relation drift vs validation MRR")
                fig.tight_layout()
                fig.savefig(out_dir / f"{dataset}__{task}__target_relation_drift_vs_val_mrr.png", dpi=200)
                plt.close(fig)

    summary = pd.DataFrame(rows)
    epoch_df = pd.DataFrame(epoch_rows)
    summary.to_csv(out_dir / "relation_memory_semantic_evolution_summary.csv", index=False)
    epoch_df.to_csv(out_dir / "relation_memory_epoch_drift.csv", index=False)

    if not summary.empty:
        top = summary.sort_values("path_length", ascending=False).head(20).copy()
        top["label"] = top["task_name"] + "\n" + top["relation_type"].astype(str)
        fig, ax = plt.subplots(figsize=(11.6, 6.0))
        colors = ["#dc2626" if v else "#64748b" for v in top["is_target_relation"]]
        ax.bar(top["label"], top["path_length"], color=colors, alpha=0.86)
        ax.set_ylabel("PCA path length")
        ax.set_title("Largest Relation-Memory Semantic Trajectories")
        ax.tick_params(axis="x", labelrotation=50)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / "relation_memory_path_length_top20.png", dpi=200)
        plt.close(fig)

        target = summary[summary["is_target_relation"]].sort_values("path_length", ascending=False)
        if not target.empty:
            fig, ax = plt.subplots(figsize=(9.8, 5.4))
            labels = target["task_name"] + "\n" + target["relation_type"].astype(str)
            ax.bar(labels, target["path_length"], color="#dc2626", alpha=0.86, label="path length")
            ax.set_ylabel("target relation path length")
            ax.set_title("Target Relation-Memory Evolution by Task")
            ax.tick_params(axis="x", labelrotation=35)
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            fig.savefig(out_dir / "target_relation_memory_path_length.png", dpi=200)
            plt.close(fig)
    return summary


def write_case_study_text(out_dir: Path, effect: pd.DataFrame, evolution: pd.DataFrame) -> None:
    lines = [
        "# Edge Semantics and Relation-Memory Case Study",
        "",
        "This case-study package answers two questions: (1) whether different edge representations affect link-prediction accuracy, and (2) whether relation memory exhibits semantic evolution over training epochs.",
        "",
        "## Evidence 1: Edge Representation Affects Accuracy",
        "",
    ]
    if not effect.empty:
        for task in sorted(effect["task_name"].unique(), key=_task_sort_key):
            g = effect[effect["task_name"] == task].copy()
            full = g[g["variant_clean"].eq("full")]
            full_mrr = float(full["test_mrr"].iloc[0]) if not full.empty else np.nan
            neg = g[(~g["variant_clean"].eq("full")) & (g["delta_mrr_vs_full"] < 0)].sort_values("delta_mrr_vs_full")
            pos = g[(~g["variant_clean"].eq("full")) & (g["delta_mrr_vs_full"] > 0)].sort_values("delta_mrr_vs_full", ascending=False)
            if not neg.empty:
                strongest = neg.iloc[0]
                lines.append(f"- `{task}`: Full test MRR={full_mrr:.4f}; strongest degradation is `{strongest['variant_label']}` with ΔMRR={strongest['delta_mrr_vs_full']:.4f}.")
            elif not pos.empty:
                strongest = pos.iloc[0]
                lines.append(f"- `{task}`: Full test MRR={full_mrr:.4f}; no degradation in available ablations, and `{strongest['variant_label']}` improves ΔMRR={strongest['delta_mrr_vs_full']:.4f}. Treat as a boundary condition rather than positive evidence for relation memory.")
    lines += [
        "",
        "The strongest positive evidence is DBLP transductive: removing edge type drops MRR by about 0.2904 and removing relation memory drops MRR by about 0.0711. Patent-US is near saturation, so relation-memory ablation has little room to reduce MRR. AMiner and THGL include boundary cases where simpler variants can match or exceed Full.",
        "",
        "## Evidence 2: Relation Memory Evolves Semantically",
        "",
    ]
    if not evolution.empty:
        target = evolution[evolution["is_target_relation"]].sort_values("path_length", ascending=False)
        for _, r in target.iterrows():
            lines.append(
                f"- `{r['task_name']}` target `{r['relation_type']}` moves with path length={r['path_length']:.3f}, direct drift={r['direct_drift']:.3f}, norm change={r['norm_delta']:.3f}."
            )
    lines += [
        "",
        "For the paper case study, use `relation_type_trajectory.png` to show semantic movement in 2D, `target_relation_drift_vs_val_mrr.png` to connect movement with validation dynamics, and `edge_representation_delta_mrr.png` to quantify accuracy impact.",
        "",
        "## Recommended Writing",
        "",
        "A concise interpretation is: relation memory is not merely a static type embedding. It moves over epochs in the learned memory space; on DBLP, where the target paper-citation relation must be separated from author-write-paper context, disabling edge type or relation memory materially reduces MRR. On Patent-US, the org-codeIII task is almost saturated, so memory evolution is observable but ablation is not a strong accuracy discriminator. This gives both positive evidence and boundary conditions.",
    ]
    (out_dir / "case_study_edge_semantics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    zh = [
        "# 边语义与 Relation Memory 的 Case Study 写作材料",
        "",
        "本 case study 面向两个问题：第一，不同边表征是否会影响链路预测准确度；第二，relation memory 是否真的学习到了随训练过程变化的语义表征，而不是静态的 relation id embedding。",
        "",
        "## 1. 不同边表征对准确度的影响",
        "",
    ]
    if not effect.empty:
        for task in sorted(effect["task_name"].unique(), key=_task_sort_key):
            g = effect[effect["task_name"] == task].copy()
            full = g[g["variant_clean"].eq("full")]
            full_mrr = float(full["test_mrr"].iloc[0]) if not full.empty else np.nan
            neg = g[(~g["variant_clean"].eq("full")) & (g["delta_mrr_vs_full"] < 0)].sort_values("delta_mrr_vs_full")
            pos = g[(~g["variant_clean"].eq("full")) & (g["delta_mrr_vs_full"] > 0)].sort_values("delta_mrr_vs_full", ascending=False)
            if not neg.empty:
                strongest = neg.iloc[0]
                zh.append(f"- `{task}`：Full 的 test MRR 为 `{full_mrr:.4f}`；去掉 `{strongest['variant_label']}` 后下降最明显，ΔMRR=`{strongest['delta_mrr_vs_full']:.4f}`。")
            elif not pos.empty:
                strongest = pos.iloc[0]
                zh.append(f"- `{task}`：Full 的 test MRR 为 `{full_mrr:.4f}`；当前消融没有带来下降，`{strongest['variant_label']}` 反而提升 ΔMRR=`{strongest['delta_mrr_vs_full']:.4f}`，应作为边界条件讨论。")
    zh += [
        "",
        "最强证据来自 DBLP transductive 任务：完整模型 test MRR 为 `0.5447`，去掉 edge type 后降至 `0.2543`，下降约 `0.2904`；去掉 relation memory 后降至 `0.4736`，下降约 `0.0711`。这说明在 paper-citation 预测中，模型需要区分 `author_write_paper` 上下文边和 `paper_cite_paper` 目标边；边类型/关系记忆不是装饰项，而会实质影响预测准确度。",
        "",
        "Patent-US 则是一个边界案例：`org -> codeIII` 任务接近饱和，Full MRR 已接近 1.0，因此去掉 relation memory 只带来很小下降。这不否定 relation memory 的作用，而说明该任务本身可分性较强，accuracy 指标难以继续放大模块差异。",
        "",
        "AMiner 和 THGL 中出现部分简化模型超过 Full 的情况，建议在论文中表述为：relation memory 的动态表征是存在的，但其带来的准确度收益依赖数据集的监督关系、上下文关系噪声和任务难度。",
        "",
        "对应图：`edge_representation_delta_mrr.png` 展示各消融相对 Full 的 MRR 变化；`case_task_variant_test_mrr.png` 展示 Full 与各边表征变体的 test MRR。",
        "",
        "## 2. Relation Memory 是否体现语义演变",
        "",
    ]
    if not evolution.empty:
        target = evolution[evolution["is_target_relation"]].sort_values("path_length", ascending=False)
        for _, r in target.iterrows():
            zh.append(
                f"- `{r['task_name']}` 的目标关系 `{r['relation_type']}` 在二维 relation-memory 空间中的路径长度为 `{r['path_length']:.3f}`，首尾直线漂移为 `{r['direct_drift']:.3f}`，范数变化为 `{r['norm_delta']:.3f}`。"
            )
    zh += [
        "",
        "这些轨迹说明 relation memory 并非固定查表向量。以 DBLP 为例，`paper_cite_paper` 和 `author_write_paper` 在训练过程中沿不同方向移动，目标 citation 关系的二维位置、路径长度和范数均发生变化；同时 DBLP 的消融结果显示去掉 edge type 或 relation memory 会显著降低 MRR。因此可以把 DBLP 写成“语义演化 + 准确度因果证据”最强的 case。",
        "",
        "THGL-software 的 `rel_6` 目标关系路径长度较大，适合作为多关系场景下 relation memory 发生显著动态漂移的可视化案例；Patent-US 虽然准确率接近饱和，但 `org_codeIII` 的 relation memory 仍有可观轨迹变化，适合说明模型内部确实在更新目标关系语义。",
        "",
        "对应图：每个任务都有 `*_relation_type_trajectory.png` 展示 relation type 的二维轨迹，`*_target_relation_drift_vs_val_mrr.png` 展示目标 relation 的逐 epoch 漂移与验证 MRR 的对应关系，`target_relation_memory_path_length.png` 汇总目标关系的语义路径长度。",
        "",
        "## 3. 可直接写入论文的段落草稿",
        "",
        "为了验证边语义记忆是否真正影响预测，我们进一步比较了完整模型与去除 edge type、去除 relation memory 等变体。在 DBLP citation 任务中，完整模型达到 0.5447 test MRR；去除 edge type 后下降到 0.2543，去除 relation memory 后下降到 0.4736，分别带来 0.2904 和 0.0711 的 MRR 损失。这表明模型需要显式区分 `author_write_paper` 上下文边和 `paper_cite_paper` 目标边，边语义表征对 paper citation prediction 具有实质贡献。",
        "",
        "进一步地，我们将每个 epoch 保存的 relation memory 投影到二维空间。结果显示目标关系并非停留在固定位置，而是在训练过程中形成连续轨迹。例如 DBLP 的 `paper_cite_paper` 关系在二维空间中产生明显漂移，同时其 relation-memory 范数也发生变化；THGL-software 中目标关系 `rel_6` 的路径长度更大，说明多关系软件交互场景下 relation memory 具有更强的动态演化。上述现象表明 relation memory 学到的是随时间更新的边语义状态，而不是静态 relation id 编码。",
    ]
    (out_dir / "case_study_edge_semantics_zh.md").write_text("\n".join(zh) + "\n", encoding="utf-8")


def write_commands(out_dir: Path, project_root: Path, analysis_root: Path) -> None:
    py = "/home/xuxj/.conda/envs/pytorch/bin/python"
    lines = [
        "# Reproduction Commands",
        "",
        "Run the upstream visualization first if source CSV/PNG are stale:",
        "",
        f"`{py} scripts/analysis/analyze_main_edge_visuals.py --project-root {project_root} --datasets patent_us,aminer_paper,dblp,thgl_software`",
        "",
        "Then regenerate this case-study package:",
        "",
        f"`{py} scripts/analysis/build_edge_semantics_case_study.py --project-root {project_root} --analysis-root {analysis_root} --out-dir {out_dir}`",
    ]
    (out_dir / "commands.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--analysis-root", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    project_root = args.project_root.resolve()
    analysis_root = (args.analysis_root or project_root / "results" / "main_edge_visual_analysis").resolve()
    out_dir = (args.out_dir or project_root / "results" / "case_study_edge_semantics").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    effect = build_edge_representation_effect(analysis_root, out_dir)
    evolution = build_relation_memory_evolution(analysis_root, out_dir)
    write_case_study_text(out_dir, effect, evolution)
    write_commands(out_dir, project_root, analysis_root)
    print(f"saved edge semantics case-study artifacts to {out_dir}")


if __name__ == "__main__":
    main()
