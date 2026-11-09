#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


ALL_COLUMNS = [
    "experiment_group",
    "dataset",
    "task",
    "task_type",
    "protocol",
    "relation_target",
    "method_family",
    "method",
    "variant",
    "runs",
    "status",
    "mrr",
    "mrr_std",
    "auc",
    "auc_std",
    "ap",
    "ap_std",
    "node_ml_auc",
    "node_ml_auc_std",
    "node_ml_micro_f1",
    "node_ml_micro_f1_std",
    "node_ml_macro_f1",
    "node_ml_macro_f1_std",
    "node_ml_map",
    "node_ml_map_std",
    "node_ml_precision_at_1",
    "node_ml_precision_at_1_std",
    "node_ml_precision_at_3",
    "node_ml_precision_at_3_std",
    "node_ml_precision_at_5",
    "node_ml_precision_at_5_std",
    "node_ml_recall_at_1",
    "node_ml_recall_at_1_std",
    "node_ml_recall_at_3",
    "node_ml_recall_at_3_std",
    "node_ml_recall_at_5",
    "node_ml_recall_at_5_std",
    "node_ml_ndcg_at_1",
    "node_ml_ndcg_at_1_std",
    "node_ml_ndcg_at_3",
    "node_ml_ndcg_at_3_std",
    "node_ml_ndcg_at_5",
    "node_ml_ndcg_at_5_std",
    "ml_threshold",
    "avg_epoch_sec",
    "elapsed_sec",
    "source_group",
    "source_path",
    "notes",
]


def clean_float(value):
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def init_row() -> Dict:
    return {col: None for col in ALL_COLUMNS}


def infer_protocol(task_type: str) -> str:
    task_type = str(task_type or "")
    if task_type == "edge_inductive":
        return "inductive"
    if task_type == "edge_transductive":
        return "transductive"
    if task_type == "node_multilabel":
        return "node_multilabel"
    return task_type or "unknown"


def infer_task_type(task: str) -> str:
    task = str(task or "")
    if "node_" in task and "multilabel" in task:
        return "node_multilabel"
    if "edge_inductive" in task or task.endswith("_inductive"):
        return "edge_inductive"
    if "edge_transductive" in task or task.endswith("_transductive"):
        return "edge_transductive"
    return "unknown"


def infer_dataset(task: str) -> str:
    task = str(task or "")
    for prefix in [
        "aminer_paper",
        "amazon_movies",
        "dblp",
        "patent_cn",
        "patent_in",
        "patent_us",
    ]:
        if task.startswith(prefix):
            return prefix
    return task.split("_edge_")[0].split("_node_")[0]


def infer_relation_target(task: str) -> Optional[str]:
    task = str(task or "")
    if "citation" in task:
        return "paper_cite_paper"
    if "coauthor" in task:
        return "author_coauthor_author"
    if "org_codeIII" in task:
        return "org_codeIII"
    if "node_multilabel" in task:
        return "all_relations"
    return None


def normalize_method(family: str, model: Optional[str]) -> str:
    family = str(family or "").strip()
    model = str(model or "").strip()
    if family == "dyglib_model_zoo":
        return model or "DyGLib-model-zoo"
    if family == "dyglib_custom":
        return "DyGLib-custom"
    if family == "repeatmixer":
        return "RepeatMixer"
    if family == "tgb2_custom":
        if model == "tgn_edgetype":
            return "TGB2-custom(TGN-edgetype)"
        return f"TGB2-custom({model})" if model else "TGB2-custom"
    if family in {"tgn", "tgn_edges", "sthn"}:
        return model or family.upper()
    return model or family or "Unknown"


def load_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_manifest_lookup(manifest_path: Path) -> Dict[str, Dict]:
    if not manifest_path.exists():
        return {}
    df = pd.read_csv(manifest_path)
    lookup: Dict[str, Dict] = {}
    for _, r in df.iterrows():
        out_dir = Path(str(r.get("out_dir", ""))).name if "out_dir" in df.columns else ""
        run_name = str(r.get("run_name", out_dir or ""))
        key = out_dir or run_name
        lookup[key] = r.to_dict()
    return lookup


def collect_main_from_metrics_dir(metrics_root: Path, source_group: str, default_variant: str = "full") -> List[Dict]:
    if not metrics_root.exists():
        return []
    manifest_lookup = build_manifest_lookup(metrics_root / "run_manifest.csv")
    rows: List[Dict] = []
    for metrics_path in sorted(metrics_root.glob("*/metrics.json")):
        run_name = metrics_path.parent.name
        meta = manifest_lookup.get(run_name, {})
        metrics = load_json(metrics_path)
        test = metrics.get("test", {}) if isinstance(metrics, dict) else {}
        node_eval = metrics.get("node_eval", {}) if isinstance(metrics, dict) else {}

        task = meta.get("task", meta.get("task_name", run_name.split("__")[0]))
        task_type = meta.get("task_type", infer_task_type(task))
        dataset = meta.get("dataset", infer_dataset(task))
        relation_target = meta.get("relation_target", infer_relation_target(task))

        row = init_row()
        row.update(
            {
                "experiment_group": "main",
                "dataset": dataset,
                "task": task,
                "task_type": task_type,
                "protocol": infer_protocol(task_type),
                "relation_target": relation_target,
                "method_family": "dhc",
                "method": "DHC",
                "variant": meta.get("variant", default_variant),
                "runs": 1,
                "status": "ok" if clean_float(meta.get("return_code", 0)) in (None, 0.0) else str(meta.get("status", "failed")),
                "mrr": clean_float(test.get("mrr")),
                "auc": clean_float(test.get("auc")),
                "ap": clean_float(test.get("ap")),
                "node_ml_auc": clean_float(node_eval.get("node_ml_auc")),
                "node_ml_micro_f1": clean_float(node_eval.get("node_ml_micro_f1")),
                "node_ml_macro_f1": clean_float(node_eval.get("node_ml_macro_f1")),
                "node_ml_map": clean_float(node_eval.get("node_ml_map")),
                "node_ml_precision_at_1": clean_float(node_eval.get("node_ml_precision_at_1")),
                "node_ml_precision_at_3": clean_float(node_eval.get("node_ml_precision_at_3")),
                "node_ml_precision_at_5": clean_float(node_eval.get("node_ml_precision_at_5")),
                "node_ml_recall_at_1": clean_float(node_eval.get("node_ml_recall_at_1")),
                "node_ml_recall_at_3": clean_float(node_eval.get("node_ml_recall_at_3")),
                "node_ml_recall_at_5": clean_float(node_eval.get("node_ml_recall_at_5")),
                "node_ml_ndcg_at_1": clean_float(node_eval.get("node_ml_ndcg_at_1")),
                "node_ml_ndcg_at_3": clean_float(node_eval.get("node_ml_ndcg_at_3")),
                "node_ml_ndcg_at_5": clean_float(node_eval.get("node_ml_ndcg_at_5")),
                "ml_threshold": clean_float(
                    node_eval.get("node_ml_threshold", metrics.get("multilabel_threshold_calibration", {}).get("best_threshold"))
                ),
                "avg_epoch_sec": clean_float(metrics.get("avg_epoch_time_sec")),
                "elapsed_sec": clean_float(meta.get("elapsed_sec")),
                "source_group": source_group,
                "source_path": str(metrics_path),
                "notes": run_name,
            }
        )
        rows.append(row)
    return rows


def collect_baseline_from_metrics_long(path: Path, source_group: str) -> List[Dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows: List[Dict] = []
    for _, r in df.iterrows():
        if str(r.get("split", "")).strip() and str(r.get("split", "")).strip() != "test":
            continue
        row = init_row()
        row.update(
            {
                "experiment_group": "baseline",
                "dataset": infer_dataset(r.get("task_name")),
                "task": r.get("task_name"),
                "task_type": r.get("task_type", infer_task_type(r.get("task_name"))),
                "protocol": infer_protocol(r.get("task_type")),
                "relation_target": r.get("relation_target"),
                "method_family": r.get("family"),
                "method": normalize_method(r.get("family"), r.get("model")),
                "variant": "default",
                "runs": 1,
                "status": r.get("status", "ok"),
                "mrr": clean_float(r.get("mrr")),
                "auc": clean_float(r.get("auc")),
                "ap": clean_float(r.get("ap")),
                "node_ml_micro_f1": clean_float(r.get("node_ml_micro_f1")),
                "node_ml_macro_f1": clean_float(r.get("node_ml_macro_f1")),
                "node_ml_precision_at_1": clean_float(r.get("node_ml_precision_at_1")),
                "node_ml_precision_at_3": clean_float(r.get("node_ml_precision_at_3")),
                "node_ml_precision_at_5": clean_float(r.get("node_ml_precision_at_5")),
                "node_ml_recall_at_1": clean_float(r.get("node_ml_recall_at_1")),
                "node_ml_recall_at_3": clean_float(r.get("node_ml_recall_at_3")),
                "node_ml_recall_at_5": clean_float(r.get("node_ml_recall_at_5")),
                "node_ml_ndcg_at_1": clean_float(r.get("node_ml_ndcg_at_1")),
                "node_ml_ndcg_at_3": clean_float(r.get("node_ml_ndcg_at_3")),
                "node_ml_ndcg_at_5": clean_float(r.get("node_ml_ndcg_at_5")),
                "avg_epoch_sec": clean_float(r.get("avg_epoch_sec")),
                "source_group": source_group,
                "source_path": r.get("source_path", str(path)),
            }
        )
        rows.append(row)
    return rows


def collect_baseline_from_metrics_dir(metrics_dir: Path, source_group: str) -> List[Dict]:
    if not metrics_dir.exists():
        return []
    rows: List[Dict] = []
    for csv_path in sorted(metrics_dir.glob("*.csv")):
        if csv_path.name.startswith("_smoke_"):
            continue
        stem = csv_path.stem
        if "__" not in stem:
            continue
        family, task = stem.split("__", 1)
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        dataset = infer_dataset(task)
        task_type = infer_task_type(task)

        if "split" in df.columns:
            df = df[df["split"] == "test"].copy()
        for _, r in df.iterrows():
            row = init_row()
            row.update(
                {
                    "experiment_group": "baseline",
                    "dataset": dataset,
                    "task": task,
                    "task_type": task_type,
                    "protocol": infer_protocol(task_type),
                    "relation_target": infer_relation_target(task),
                    "method_family": family,
                    "method": normalize_method(family, r.get("model")),
                    "variant": "default",
                    "runs": 1,
                    "status": r.get("status", "ok"),
                    "mrr": clean_float(r.get("test_mrr", r.get("mrr"))),
                    "auc": clean_float(r.get("test_auc", r.get("auc"))),
                    "ap": clean_float(r.get("test_ap", r.get("ap"))),
                    "node_ml_micro_f1": clean_float(r.get("node_ml_micro_f1")),
                    "node_ml_macro_f1": clean_float(r.get("node_ml_macro_f1")),
                    "node_ml_precision_at_1": clean_float(r.get("node_ml_precision_at_1")),
                    "node_ml_precision_at_3": clean_float(r.get("node_ml_precision_at_3")),
                    "node_ml_precision_at_5": clean_float(r.get("node_ml_precision_at_5")),
                    "node_ml_recall_at_1": clean_float(r.get("node_ml_recall_at_1")),
                    "node_ml_recall_at_3": clean_float(r.get("node_ml_recall_at_3")),
                    "node_ml_recall_at_5": clean_float(r.get("node_ml_recall_at_5")),
                    "node_ml_ndcg_at_1": clean_float(r.get("node_ml_ndcg_at_1")),
                    "node_ml_ndcg_at_3": clean_float(r.get("node_ml_ndcg_at_3")),
                    "node_ml_ndcg_at_5": clean_float(r.get("node_ml_ndcg_at_5")),
                    "avg_epoch_sec": clean_float(r.get("avg_epoch_sec")),
                    "elapsed_sec": clean_float(r.get("total_sec")),
                    "source_group": source_group,
                    "source_path": str(csv_path),
                }
            )
            rows.append(row)
    return rows


def build_output(project_root: Path, out_csv: Path) -> Path:
    rows: List[Dict] = []

    rows += collect_main_from_metrics_dir(
        project_root / "results" / "paper_dhc_formal",
        "paper_dhc_formal",
    )
    rows += collect_main_from_metrics_dir(
        project_root / "results" / "aminer_dhc_formal",
        "aminer_dhc_formal",
    )
    rows += collect_main_from_metrics_dir(
        project_root / "results" / "patent_dhc_formal",
        "patent_dhc_formal",
    )

    rows += collect_baseline_from_metrics_dir(
        project_root / "results" / "patent_baseline_formal" / "metrics",
        "patent_baseline_formal",
    )
    rows += collect_baseline_from_metrics_dir(
        project_root / "results" / "baseline_formal_aminer_custom" / "metrics",
        "baseline_formal_aminer_custom",
    )
    rows += collect_baseline_from_metrics_dir(
        project_root / "results" / "baseline_formal_dblp_amazon" / "metrics",
        "baseline_formal_dblp_amazon",
    )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No rows collected for dyn_relation_memory export.")

    # When a richer source exists, prefer it over the same task/method from a thinner source.
    source_rank = {
        "paper_dhc_formal": 0,
        "aminer_dhc_formal": 0,
        "patent_dhc_formal": 0,
        "patent_baseline_formal": 0,
        "baseline_formal_aminer_custom": 0,
        "baseline_formal_dblp_amazon": 0,
    }
    df["source_rank"] = df["source_group"].map(source_rank).fillna(99)
    df = df.sort_values(
        ["dataset", "task_type", "task", "experiment_group", "method", "source_rank"]
    )
    df = df.drop_duplicates(
        subset=["experiment_group", "dataset", "task", "task_type", "method", "variant"],
        keep="first",
    ).copy()
    df = df.drop(columns=["source_rank"])

    df = df[ALL_COLUMNS].sort_values(
        ["dataset", "task_type", "task", "experiment_group", "method"]
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    df[df["experiment_group"] == "main"].to_csv(
        out_csv.with_name("main_experiments_table.csv"), index=False
    )
    df[df["experiment_group"] == "baseline"].to_csv(
        out_csv.with_name("baseline_experiments_table.csv"), index=False
    )
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export dyn_relation_memory main experiments and baselines into one big CSV."
    )
    ap.add_argument(
        "--project-root",
        type=str,
        default="/home/xuxj/2026/dyn_relation_memory",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="",
    )
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    out_csv = (
        Path(args.out_csv).resolve()
        if args.out_csv.strip()
        else project_root / "results" / "experiment_snapshot" / "main_baseline_metrics_big_table.csv"
    )
    written = build_output(project_root=project_root, out_csv=out_csv)
    print(written)


if __name__ == "__main__":
    main()
