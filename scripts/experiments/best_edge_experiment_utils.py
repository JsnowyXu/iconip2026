#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
EXPERIMENTS_ROOT = SCRIPT_PATH.parent
for _path in [PROJECT_ROOT, EXPERIMENTS_ROOT]:
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

try:
    from edge_benchmark_common import DATASET_ORDER, TaskSpec, build_edge_task_specs
except ModuleNotFoundError:
    from scripts.experiments.edge_benchmark_common import DATASET_ORDER, TaskSpec, build_edge_task_specs


BEST_SOURCE_ROOTS: List[Tuple[str, str]] = [
    ("edge_only_main", "results/edge_only_formal/main"),
    ("targeted_tuning", "results/targeted_edge_tuning_20260423"),
    ("targeted_tuning_followup", "results/targeted_edge_tuning_20260423_followup"),
]

FOCUS_DATASETS = ["patent_cn", "patent_in", "patent_us", "thgl_software"]
FOCUS_TASK_TYPES = ["edge_inductive", "edge_transductive"]


@dataclass(frozen=True)
class BestRun:
    task: TaskSpec
    source_group: str
    metrics_path: Path
    run_dir: Path
    run_name: str
    test_mrr: float
    run_args: Dict[str, object]


def dataset_rank(dataset: str) -> int:
    try:
        return DATASET_ORDER.index(dataset)
    except ValueError:
        return len(DATASET_ORDER)


def parse_task_name_from_run(run_name: str) -> str:
    if run_name.endswith("__seed42"):
        return run_name[:-8]
    if "__" not in run_name:
        return run_name
    parts = run_name.split("__")
    if parts[-1].startswith("s"):
        return "__".join(parts[:-2])
    return "__".join(parts[:-1])


def build_task_lookup(project_root: Path) -> Dict[str, TaskSpec]:
    return {task.task_name: task for task in build_edge_task_specs(project_root)}


def iter_candidate_metrics(project_root: Path) -> Iterable[Tuple[str, Path]]:
    project_root = project_root.resolve()
    for source_group, rel_root in BEST_SOURCE_ROOTS:
        root = project_root / rel_root
        if not root.exists():
            continue
        for metrics_path in sorted(root.glob("*/metrics.json")):
            yield source_group, metrics_path


def load_metrics(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def full_modules_enabled(run_args: Dict[str, object]) -> bool:
    return not any(
        bool(run_args.get(flag, False))
        for flag in ["no_relation_memory", "no_edge_type_features", "no_community"]
    )


def load_best_runs(
    project_root: Path,
    datasets: Optional[Sequence[str]] = None,
    task_types: Optional[Sequence[str]] = None,
    require_community_enabled: bool = False,
    require_full_modules: bool = False,
) -> List[BestRun]:
    project_root = project_root.resolve()
    task_lookup = build_task_lookup(project_root)
    dataset_filter = set(datasets or [])
    task_type_filter = set(task_types or [])
    best_by_task: Dict[str, BestRun] = {}

    for source_group, metrics_path in iter_candidate_metrics(project_root):
        run_name = metrics_path.parent.name
        task_name = parse_task_name_from_run(run_name)
        task = task_lookup.get(task_name)
        if task is None:
            continue
        if dataset_filter and task.dataset not in dataset_filter:
            continue
        if task_type_filter and task.task_type not in task_type_filter:
            continue

        obj = load_metrics(metrics_path)
        test = obj.get("test", {}) if isinstance(obj, dict) else {}
        test_mrr = test.get("mrr")
        if test_mrr is None:
            continue
        run_args = obj.get("run_args", {}) if isinstance(obj, dict) else {}
        if require_full_modules and not full_modules_enabled(run_args):
            continue
        if require_community_enabled and bool(run_args.get("no_community", False)):
            continue

        cand = BestRun(
            task=task,
            source_group=source_group,
            metrics_path=metrics_path,
            run_dir=metrics_path.parent,
            run_name=run_name,
            test_mrr=float(test_mrr),
            run_args=dict(run_args),
        )
        prev = best_by_task.get(task_name)
        if prev is None or cand.test_mrr > prev.test_mrr:
            best_by_task[task_name] = cand

    rows = list(best_by_task.values())
    rows.sort(key=lambda row: (dataset_rank(row.task.dataset), row.task.task_type, row.task.task_name))
    return rows


def module_locations() -> Dict[str, List[Tuple[str, str]]]:
    return {
        "relation_memory": [
            ("dhc/train/run_train.py", "CLI switch `--no-relation-memory` is parsed and passed into `DHCModel(use_relation_memory=...)`."),
            ("dhc/models/memory.py", "`NodeRelationMemory` stores node memory, relation memory, and temporal update state."),
            ("dhc/models/dhc_model.py", "`__init__(...)` wires relation memory in, and `_rel_vec(...)`, `_recent_neighbor_context(...)`, `forward_batch(...)`, `update_state(...)` consume it."),
        ],
        "edge_type": [
            ("dhc/train/run_train.py", "CLI switch `--no-edge-type-features` is parsed and passed into `DHCModel(use_edge_type_features=...)`."),
            ("dhc/models/dhc_model.py", "`__init__(...)` expands raw edge feature width, and `_augment_edge_feat(...)` injects one-hot edge-type features into messages and recent-neighbor cache."),
        ],
        "community": [
            ("dhc/train/run_train.py", "CLI switch `--no-community` is parsed and passed into `DHCModel(use_community=...)`."),
            ("dhc/models/community.py", "`CommunityMemory` implements soft assignment, memory update, and contrastive objective."),
            ("dhc/models/dhc_model.py", "`__init__(...)` constructs `CommunityMemory`, and `_node_repr(...)`, `_encode_link_query_nodes(...)`, `forward_batch(...)`, `update_state(...)` use community context and memory."),
        ],
        "neighbor_context": [
            ("dhc/train/run_train.py", "`--recent-neighbor-size` and `--neighbor-aggregation` control local temporal neighbor context."),
            ("dhc/models/recent_neighbors.py", "`RecentNeighborStore` keeps recent interaction context."),
            ("dhc/models/dhc_model.py", "`_recent_neighbor_context(...)` supports `attn` and `mean` aggregation."),
        ],
    }


def render_module_locations_markdown() -> str:
    lines = ["# Module Locations", ""]
    for module_name, entries in module_locations().items():
        lines.append(f"## {module_name}")
        lines.append("")
        for path, note in entries:
            lines.append(f"- `{path}`: {note}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def anchor_run_args(best_run: BestRun, force_full_modules: bool = False) -> Dict[str, object]:
    args = dict(best_run.run_args)
    if force_full_modules:
        args["no_relation_memory"] = False
        args["no_edge_type_features"] = False
        args["no_community"] = False
    args.setdefault("save_memory_artifacts", True)
    return args


def to_flag(name: str) -> str:
    return "--" + str(name).replace("_", "-")


_SCALAR_KEYS: List[str] = [
    "seed",
    "epochs",
    "batch_size",
    "lr",
    "early_stop_patience",
    "early_stop_min_delta",
    "monitor_metric",
    "task_type",
    "mem_dim",
    "rel_dim",
    "time_dim",
    "msg_dim",
    "community_dim",
    "num_communities",
    "community_assign_temperature",
    "community_assign_topk",
    "recent_neighbor_size",
    "neighbor_aggregation",
    "lambda_cl",
    "lambda_community_entropy",
    "lambda_mc",
    "lambda_ml",
    "relation_loss_weights",
    "ml_pos_weight_max",
    "ml_pos_weight_power",
    "ml_threshold_num_candidates",
    "ml_calib_max_nodes",
    "node_eval_max_nodes",
    "typed_negative_ratio",
    "popular_negative_ratio",
]

_BOOL_KEYS: List[str] = [
    "no_recent_neighbors",
    "no_edge_type_features",
    "no_relation_memory",
    "no_community",
    "ml_pos_weight",
    "ml_threshold_search",
    "save_memory_artifacts",
    "no_lightweight_checkpoint",
]


def build_train_command(
    python_bin: str,
    task: TaskSpec,
    out_dir: Path,
    run_args: Dict[str, object],
    gpu_id: int,
) -> List[str]:
    cmd: List[str] = [
        str(python_bin),
        "-m",
        "dhc.train.run_train",
        "--data",
        str(task.data_path),
        "--out-dir",
        str(out_dir),
        "--gpu-id",
        str(int(gpu_id)),
    ]
    for key in _SCALAR_KEYS:
        value = run_args.get(key)
        if value is None:
            continue
        cmd.extend([to_flag(key), str(value)])
    for key in _BOOL_KEYS:
        if bool(run_args.get(key, False)):
            cmd.append(to_flag(key))
    return cmd


def best_runs_to_rows(best_runs: Sequence[BestRun]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in best_runs:
        rows.append(
            {
                "dataset": item.task.dataset,
                "task_name": item.task.task_name,
                "task_type": item.task.task_type,
                "best_test_mrr": item.test_mrr,
                "source_group": item.source_group,
                "run_name": item.run_name,
                "metrics_path": str(item.metrics_path),
                "run_dir": str(item.run_dir),
            }
        )
    return rows
