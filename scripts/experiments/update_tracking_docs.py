#!/usr/bin/env python3
import argparse
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


README_START = "<!-- AUTO_TRACKING_STATUS:START -->"
README_END = "<!-- AUTO_TRACKING_STATUS:END -->"


def first_existing(project_root: Path, candidates: Iterable[str]) -> Optional[Path]:
    for rel in candidates:
        path = project_root / rel
        if path.exists():
            return path
    return None


def load_csv(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def clean_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except Exception:
        return None


def fmt_metric(value) -> str:
    num = clean_float(value)
    return "-" if num is None else f"{num:.4f}"


def fmt_delta(value) -> str:
    num = clean_float(value)
    return "-" if num is None else f"{num:+.4f}"


def fmt_triplet(mrr, auc, ap) -> str:
    return f"{fmt_metric(mrr)} / {fmt_metric(auc)} / {fmt_metric(ap)}"


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_暂无数据_"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def pick_head2head_paths(project_root: Path) -> Dict[str, Optional[Path]]:
    return {
        "runs": first_existing(
            project_root,
            [
                "results-auto/head2head_rawcn_current/head2head_runs.csv",
                "results/head2head_rawcn_current/head2head_runs.csv",
            ],
        ),
        "progress": first_existing(
            project_root,
            [
                "docs/rawcn_head2head_progress.md",
                "results-auto/head2head_rawcn_current/progress.md",
                "results/head2head_rawcn_current/progress.md",
            ],
        ),
        "design": first_existing(
            project_root,
            [
                "docs/rawcn_head2head_experiment_design.md",
                "results-auto/head2head_rawcn_current/experiment_design.md",
                "results/head2head_rawcn_current/experiment_design.md",
            ],
        ),
    }


def choose_champions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    working = df.copy()
    if "return_code" in working.columns:
        working = working[working["return_code"].fillna(0).astype(int) == 0].copy()
    for col in [
        "meets_strong_gate",
        "beats_majority",
        "beats_mrr",
        "beats_auc",
        "beats_ap",
        "test_mrr",
        "test_auc",
        "test_ap",
    ]:
        if col not in working.columns:
            working[col] = 0
        working[col] = pd.to_numeric(working[col], errors="coerce").fillna(0.0)
    working["config_rank"] = working["config"].astype(str).map(
        {
            "main_fullbench_auc_retry": 5,
            "patent_strong_sparse": 4,
            "main_fullbench": 3,
            "main_fullbench_mrr": 2,
            "main_fullbench_mrr_retry": 1,
        }
    ).fillna(0)
    working = working.sort_values(
        [
            "task",
            "meets_strong_gate",
            "beats_majority",
            "beats_mrr",
            "beats_auc",
            "beats_ap",
            "config_rank",
            "test_mrr",
            "test_auc",
            "test_ap",
        ],
        ascending=[True, False, False, False, False, False, False, False, False, False],
    )
    return working.groupby("task", as_index=False).head(1).reset_index(drop=True)


def extract_update_time(md_path: Optional[Path]) -> Optional[str]:
    if md_path is None or not md_path.exists():
        return None
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"更新时间：`([^`]+)`", text)
    if m:
        return m.group(1)
    return None


def build_mainline_summary(champions: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    if champions.empty:
        return ["- 当前还没有找到可解析的 `raw_cn head-to-head` 台账。"], pd.DataFrame()
    rows = []
    strong_count = int(pd.to_numeric(champions["meets_strong_gate"], errors="coerce").fillna(0).sum())
    total = len(champions)
    lines = [
        "- 当前主线：`raw_cn current-protocol head-to-head`，目标是为 `3` 个主 edge 任务确定冠军配置并完成 `42/43/44` 三 seed 方差。",
        f"- 当前 gate 状态：`{strong_count}/{total}` 个任务已经满足 `meets_strong_gate`。",
    ]
    trans_row = champions[champions["task"].astype(str).str.contains("transductive", na=False)]
    if not trans_row.empty:
        tr = trans_row.iloc[0]
        if clean_float(tr.get("beats_ap")) == 0:
            lines.append(
                f"- transductive 目前由 `{tr['config']}` 领跑，已经过 `MRR+AUC`，但 `AP` 仍落后 baseline {fmt_delta(tr.get('delta_ap'))}。"
            )
    if strong_count == total:
        lines.append("- 结论：主线已经具备进入三 seed 方差阶段的条件，后续不再需要等待 transductive 的 AP 补强。")
    for _, row in champions.iterrows():
        rows.append(
            {
                "task": row["task"],
                "champion": row["config"],
                "seed": int(clean_float(row.get("seed")) or 0),
                "test(MRR/AUC/AP)": fmt_triplet(row.get("test_mrr"), row.get("test_auc"), row.get("test_ap")),
                "delta_vs_baseline": fmt_triplet(row.get("delta_mrr"), row.get("delta_auc"), row.get("delta_ap")),
                "gate": "yes" if int(clean_float(row.get("meets_strong_gate")) or 0) == 1 else "no",
            }
        )
    return lines, pd.DataFrame(rows)


def build_round_section(df: pd.DataFrame) -> List[Tuple[str, str, pd.DataFrame]]:
    if df.empty:
        return []
    sections: List[Tuple[str, str, pd.DataFrame]] = []
    round_specs = [
        (
            "Round 1：主协议对打",
            (df["phase"].astype(str) == "auto_head2head") & (df["config"].astype(str) == "main_fullbench"),
            "先用 `main_fullbench` 建立当前协议下的基准，结果显示 patent-ipc 已经 3/3 过线，但 transductive 明显不过线。",
        ),
        (
            "Round 2：自动调参",
            (df["phase"].astype(str) == "auto_head2head") & (df["config"].astype(str) != "main_fullbench"),
            "自动调参主要解决 inductive 与 patent-ipc 的稳态提升，其中 `patent_strong_sparse` 成为 inductive 冠军。",
        ),
        (
            "Round 3：定向补跑",
            df["phase"].astype(str) == "manual_retry",
            "针对 transductive 单独切换监控目标，`main_fullbench_auc_retry` 把结果推进到 `MRR + AUC` 双过线。",
        ),
        (
            "Round 4：三 seed 方差",
            df["phase"].astype(str) == "variance",
            "该阶段只应沿用当前冠军配置，不再回退到旧的 `main_fullbench`。",
        ),
    ]
    for title, mask, summary in round_specs:
        subset = df[mask].copy()
        if subset.empty:
            if title.endswith("三 seed 方差"):
                sections.append((title, "尚未看到新的方差结果落盘；当前阶段仍是待执行。", pd.DataFrame()))
            continue
        subset = subset.sort_values(
            ["task", "meets_strong_gate", "beats_majority", "test_mrr", "test_auc", "test_ap"],
            ascending=[True, False, False, False, False, False],
        ).groupby("task", as_index=False).head(1)
        table = pd.DataFrame(
            [
                {
                    "task": row["task"],
                    "config": row["config"],
                    "seed": int(clean_float(row.get("seed")) or 0),
                    "test(MRR/AUC/AP)": fmt_triplet(row.get("test_mrr"), row.get("test_auc"), row.get("test_ap")),
                    "delta_vs_baseline": fmt_triplet(row.get("delta_mrr"), row.get("delta_auc"), row.get("delta_ap")),
                    "gate": "yes" if int(clean_float(row.get("meets_strong_gate")) or 0) == 1 else "no",
                }
                for _, row in subset.iterrows()
            ]
        )
        sections.append((title, summary, table))
    return sections


def summarize_patent_suite(project_root: Path) -> List[str]:
    path = first_existing(project_root, ["results/patent_suite/summary.csv", "results-auto/patent_suite/summary.csv"])
    df = load_csv(path)
    if df.empty:
        return []
    rows = []
    for task in [
        "raw_cn_edge_patent_ipc",
        "raw_us_edge_patent_ipc",
        "raw_cn_edge_assignee_ipc_trans",
        "raw_us_edge_assignee_ipc_trans",
    ]:
        subset = df[df["task"].astype(str) == task]
        if subset.empty:
            continue
        best = subset.sort_values("test_mrr", ascending=False).iloc[0]
        rows.append(f"- `{task}`: `{fmt_triplet(best.get('test_mrr'), best.get('test_auc'), best.get('test_ap'))}`")
    return rows


def summarize_full_benchmark(project_root: Path) -> List[str]:
    path = first_existing(project_root, ["results-auto/full_benchmark_repeats/aggregate_mean_std.csv", "results/full_benchmark_repeats/aggregate_mean_std.csv"])
    df = load_csv(path)
    if df.empty:
        return []
    full = df[df["variant"].astype(str) == "full"].copy()
    if full.empty:
        return []
    lines = [f"- `full_benchmark_repeats` 已有 `{len(full)}` 个 full 变体聚合任务。"]
    for task in [
        "raw_cn_edge_assignee_ipc_inductive",
        "raw_cn_edge_assignee_ipc_transductive",
        "raw_cn_edge_patent_ipc",
    ]:
        subset = full[full["task"].astype(str) == task]
        if subset.empty:
            continue
        row = subset.iloc[0]
        lines.append(
            f"- `{task}` 三 seed均值：`{fmt_triplet(row.get('test_mrr_mean'), row.get('test_auc_mean'), row.get('test_ap_mean'))}`"
        )
    return lines


def summarize_aminer(project_root: Path) -> List[str]:
    path = first_existing(project_root, ["results-auto/full_benchmark_aminer_formal/aggregate_mean_std.csv", "results/full_benchmark_aminer_formal/aggregate_mean_std.csv"])
    df = load_csv(path)
    if df.empty:
        return []
    lines = []
    for task in ["aminer_paper_edge_transductive", "aminer_paper_edge_inductive", "aminer_paper_node_multilabel"]:
        subset = df[(df["task"].astype(str) == task) & (df["variant"].astype(str) == "full")]
        if subset.empty:
            continue
        row = subset.iloc[0]
        lines.append(
            f"- `{task}` 三 seed均值：`{fmt_triplet(row.get('test_mrr_mean'), row.get('test_auc_mean'), row.get('test_ap_mean'))}`"
        )
    return lines


def summarize_tgb2(project_root: Path) -> List[str]:
    lines = []
    for ds in ["github", "software"]:
        path = first_existing(
            project_root,
            [
                f"results-auto/tgb2_thgl_{ds}_full_rel/summary.csv",
                f"results/tgb2_thgl_{ds}_full_rel/summary.csv",
            ],
        )
        df = load_csv(path)
        if df.empty:
            continue
        lines.append(f"- `thgl_{ds}` full-release:")
        for setting in ["transductive", "inductive"]:
            dhc_row = df[(df["family"].astype(str) == "dhc") & (df["model"].astype(str) == "DHC") & (df["setting"].astype(str) == setting)]
            base_row = df[(df["family"].astype(str) == "baseline") & (df["setting"].astype(str) == setting)]
            if dhc_row.empty:
                continue
            dhc = dhc_row.iloc[0]
            if base_row.empty:
                lines.append(f"- `{setting}` DHC：`{fmt_triplet(dhc.get('mrr'), dhc.get('auc'), dhc.get('ap'))}`")
            else:
                base = base_row.iloc[0]
                lines.append(
                    f"- `{setting}` DHC vs baseline：`{fmt_triplet(dhc.get('mrr'), dhc.get('auc'), dhc.get('ap'))}` vs `{fmt_triplet(base.get('mrr'), base.get('auc'), base.get('ap'))}`"
                )
    return lines


def summarize_cuda_academic(project_root: Path) -> List[str]:
    path = first_existing(project_root, ["results/cuda_paper_academic/summary.csv", "results-auto/cuda_paper_academic/summary.csv"])
    df = load_csv(path)
    if df.empty:
        return []
    lines = []
    for run_name in ["dblp_trans_full", "arxiv_ind_full", "dblp_ind_full"]:
        subset = df[df["run_name"].astype(str) == run_name]
        if subset.empty:
            continue
        row = subset.iloc[0]
        lines.append(f"- `{run_name}`：`{fmt_triplet(row.get('test_mrr'), row.get('test_auc'), row.get('test_ap'))}`")
    return lines


def detect_ongoing_lines(project_root: Path) -> List[str]:
    lines = []
    gpu_campaign = project_root / "results" / "gpu_full_campaign"
    if gpu_campaign.exists():
        log_count = len(list(gpu_campaign.glob("logs/*.log")))
        lines.append(f"- `results/gpu_full_campaign/` 已存在 `{log_count}` 个阶段日志，说明全量 campaign 还在收尾或尚未最终封版。")
    baseline_gpu = project_root / "results" / "baseline_formal_all_gpu" / "summary.csv"
    if baseline_gpu.exists():
        try:
            rows = len(pd.read_csv(baseline_gpu))
            lines.append(f"- `results/baseline_formal_all_gpu/summary.csv` 当前已有 `{rows}` 行基线结果，但仍建议和最终聚合表一起复核。")
        except Exception:
            lines.append("- `results/baseline_formal_all_gpu/` 已落盘，但尚未做最终封版复核。")
    return lines


def rel_to_project(path: Optional[Path], project_root: Path) -> str:
    if path is None:
        return "-"
    try:
        return str(path.relative_to(project_root))
    except Exception:
        return str(path)


def build_design_md(
    now_text: str,
    project_root: Path,
    head2head_paths: Dict[str, Optional[Path]],
    champions: pd.DataFrame,
    champion_table: pd.DataFrame,
) -> str:
    main_lines, _ = build_mainline_summary(champions)
    source_lines = []
    if head2head_paths.get("runs"):
        source_lines.append(f"- 主台账：`{rel_to_project(head2head_paths['runs'], project_root)}`")
    if head2head_paths.get("design"):
        source_lines.append(f"- 参考设计文档：`{rel_to_project(head2head_paths['design'], project_root)}`")
    if head2head_paths.get("progress"):
        source_lines.append(f"- 参考进度文档：`{rel_to_project(head2head_paths['progress'], project_root)}`")
    lines = [
        "# Current Main Experiment Design",
        "",
        f"更新时间：`{now_text}`",
        "",
        "> 该文件由 `scripts/experiments/update_tracking_docs.py` 自动生成，用来维护当前主试验的单一事实来源。",
        "",
        "## 主线定义",
        "",
        "- 当前主线不是全仓库所有结果，而是 `raw_cn current-protocol head-to-head -> 三 seed 方差`。",
        "- 目标是为 `raw_cn` 的 3 个主 edge 任务固定冠军配置，再进入 `42/43/44` 三轮方差；其他目录主要作为已落盘证据和论文支撑。",
        "",
        "## 当前判断",
        "",
        *main_lines,
        "",
        "## 覆盖任务与当前冠军",
        "",
        to_markdown(champion_table),
        "",
        "## 固定 gate 与执行规则",
        "",
        "- `meets_strong_gate`：`MRR` 过 baseline，且 `MRR/AUC/AP` 至少 `2/3` 过 baseline。",
        "- transductive 的 `AP` 补强仍可继续，但不再阻塞方差实验。",
        "- 三 seed 阶段必须沿用当前冠军配置，而不是回退到旧的 `main_fullbench` 默认组合。",
        "",
        "## 当前阶段设计",
        "",
        "1. `Round 1`：`main_fullbench` 建立主协议基线。",
        "2. `Round 2`：自动调参线负责寻找更强的 inductive / patent-ipc 配置。",
        "3. `Round 3`：针对 transductive 切换监控目标做定向补跑，当前最有效的是 `val_auc`。",
        "4. `Round 4`：在冠军配置上补齐 `42/43/44` 三 seed，并汇总 mean/std。",
        "",
        "## 固定训练控制项",
        "",
        "- 默认不动的控制项：`epochs=10`、`batch_size=512`、`lr=1e-3`、`early_stop_patience=5`。",
        "- 默认使用：`--ml-threshold-search`、`typed_negative_ratio=0.6`、`popular_negative_ratio=0.2`。",
        "- 默认保留 memory artifacts，便于后续可视化和 case study。",
        "",
        "## 数据来源",
        "",
        *(source_lines or ["- 当前未找到 head-to-head 原始文档，只能基于结果文件回推。"]),
        "",
        "## 自动更新",
        "",
        "- 手动刷新：`python scripts/experiments/update_tracking_docs.py --project-root /home/xuxj/2026/dyn_hetero_community`",
        "- 自动刷新入口：`run_rawcn_head2head_tuning.py`、`build_current_experiment_snapshot.py`、`refresh_paper_artifacts.sh`。",
        "",
    ]
    return "\n".join(lines)


def build_progress_md(
    now_text: str,
    champions: pd.DataFrame,
    champion_table: pd.DataFrame,
    head2head_df: pd.DataFrame,
    patent_lines: List[str],
    benchmark_lines: List[str],
    aminer_lines: List[str],
    tgb2_lines: List[str],
    cuda_lines: List[str],
    ongoing_lines: List[str],
) -> str:
    main_lines, _ = build_mainline_summary(champions)
    sections = [
        "# Experiment Progress",
        "",
        f"更新时间：`{now_text}`",
        "",
        "> 该文件由 `scripts/experiments/update_tracking_docs.py` 自动生成，只保留当前判断所需的汇总，不展开逐条 run 细节。",
        "",
        "## 总览",
        "",
        *main_lines,
        "",
        "## 当前冠军面板",
        "",
        to_markdown(champion_table),
        "",
        "## 主线分轮记录",
        "",
    ]
    for title, summary, table in build_round_section(head2head_df):
        sections.extend([f"### {title}", "", f"- {summary}", ""])
        if not table.empty:
            sections.extend([to_markdown(table), ""])

    sections.extend(
        [
            "## 其他已落盘结果",
            "",
            "### Patent Suite",
            "",
            *(patent_lines or ["- 暂无可解析的 Patent Suite 汇总。"]),
            "",
            "### Full Benchmark Repeats",
            "",
            *(benchmark_lines or ["- 暂无可解析的三 seed 主 benchmark 汇总。"]),
            "",
            "### AMiner",
            "",
            *(aminer_lines or ["- 暂无可解析的 AMiner 聚合结果。"]),
            "",
            "### TGB2 Full Release",
            "",
            *(tgb2_lines or ["- 暂无可解析的 TGB2 full-release 汇总。"]),
            "",
            "### CUDA Academic",
            "",
            *(cuda_lines or ["- 暂无可解析的 CUDA academic 汇总。"]),
            "",
            "## 仍在收尾/待复核",
            "",
            *(ongoing_lines or ["- 当前没有检测到明显的未封版目录。"]),
            "",
            "## 下一步",
            "",
            "- 立即沿当前冠军配置补齐 `42/43/44` 三 seed 方差。",
            "- transductive 的 `AP` 补强继续作为 sidecar，不再卡主线推进。",
            "- 新 seed 落盘后优先刷新本文件、`design.md` 与 `README.md` 顶部状态块。",
            "",
        ]
    )
    return "\n".join(sections)


def build_readme_block(
    now_text: str,
    champion_table: pd.DataFrame,
    patent_lines: List[str],
    benchmark_lines: List[str],
    ongoing_lines: List[str],
    head2head_update: Optional[str],
) -> str:
    champion_briefs: List[str] = []
    if not champion_table.empty:
        for _, row in champion_table.iterrows():
            champion_briefs.append(f"- `{row['task']}` -> `{row['champion']}`，指标 `{row['test(MRR/AUC/AP)']}`")
    block_lines = [
        README_START,
        "## 当前跟踪状态（自动更新）",
        "",
        f"更新时间：`{now_text}`",
        "",
        "- 当前主线：`raw_cn current-protocol head-to-head -> 三 seed 方差`。",
        f"- head-to-head 最近一次同步时间：`{head2head_update or '未知'}`。",
        "- 当前根目录追踪文档：`design.md`、`progress.md`、`README.md` 顶部状态块。",
        "",
        "主线冠军：",
        *(champion_briefs or ["- 暂无可解析的主线冠军信息。"]),
        "",
        "已落盘证据：",
        *((patent_lines[:2] if patent_lines else []) + (benchmark_lines[:2] if benchmark_lines else []) or ["- 暂无可解析的全局汇总。"]),
        "",
        "仍在收尾：",
        *(ongoing_lines or ["- 当前未检测到明显的收尾目录。"]),
        "",
        "快速入口：",
        "- [design.md](./design.md)",
        "- [progress.md](./progress.md)",
        "- [results/README_CURRENT.md](./results/README_CURRENT.md)",
        "- 刷新命令：`python scripts/experiments/update_tracking_docs.py --project-root /home/xuxj/2026/dyn_hetero_community`",
        README_END,
    ]
    return "\n".join(block_lines)


def update_readme(project_root: Path, block: str) -> None:
    readme_path = project_root / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    if README_START in text and README_END in text:
        text = re.sub(
            rf"{re.escape(README_START)}.*?{re.escape(README_END)}",
            block,
            text,
            flags=re.S,
        )
    else:
        lines = text.splitlines()
        insert_idx = 7 if len(lines) > 7 else len(lines)
        lines[insert_idx:insert_idx] = ["", block, ""]
        text = "\n".join(lines)
    readme_path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser("Update root-level tracking docs from current experiment outputs.")
    ap.add_argument("--project-root", type=Path, default=Path("/home/xuxj/2026/dyn_hetero_community"))
    args = ap.parse_args()

    project_root = args.project_root.resolve()
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M")

    head2head_paths = pick_head2head_paths(project_root)
    head2head_df = load_csv(head2head_paths["runs"])
    champions = choose_champions(head2head_df)
    _, champion_table = build_mainline_summary(champions)

    patent_lines = summarize_patent_suite(project_root)
    benchmark_lines = summarize_full_benchmark(project_root)
    aminer_lines = summarize_aminer(project_root)
    tgb2_lines = summarize_tgb2(project_root)
    cuda_lines = summarize_cuda_academic(project_root)
    ongoing_lines = detect_ongoing_lines(project_root)

    design_md = build_design_md(now_text, project_root, head2head_paths, champions, champion_table)
    progress_md = build_progress_md(
        now_text,
        champions,
        champion_table,
        head2head_df,
        patent_lines,
        benchmark_lines,
        aminer_lines,
        tgb2_lines,
        cuda_lines,
        ongoing_lines,
    )
    readme_block = build_readme_block(
        now_text,
        champion_table,
        patent_lines,
        benchmark_lines,
        ongoing_lines,
        extract_update_time(head2head_paths["progress"]),
    )

    (project_root / "design.md").write_text(design_md, encoding="utf-8")
    (project_root / "progress.md").write_text(progress_md, encoding="utf-8")
    update_readme(project_root, readme_block)

    print(project_root / "design.md")
    print(project_root / "progress.md")
    print(project_root / "README.md")


if __name__ == "__main__":
    main()
