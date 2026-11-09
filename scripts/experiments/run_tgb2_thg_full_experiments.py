#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def resolve_processed_root(project_root: Path) -> Path:
    primary = project_root / "data" / "processed"
    fallback = project_root / "data" / "processed-auto"
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary


def resolve_results_root(project_root: Path) -> Path:
    primary = project_root / "results"
    fallback = project_root / "results-auto"
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary


def dataset_specs(project_root: Path, repo_root: Path) -> Dict[str, Dict]:
    processed = resolve_processed_root(project_root)
    results = resolve_results_root(project_root)
    task_root = processed / "tgb2_tasks"
    return {
        "thgl_github": {
            "display_name": "thgl_github",
            "label": "github",
            "official_dataset_name": "thgl-github",
            "raw_edgelist": repo_root / "datasets" / "tgb2" / "thgl-github" / "thgl-github_edgelist.csv",
            "raw_nodetype": repo_root / "datasets" / "tgb2" / "thgl-github" / "thgl-github_nodetype.csv",
            "unified_pt": processed / "thgl_github_tgb2_full_unified.pt",
            "task_prefix": "thgl_github_tgb2_full",
            "task_dir": task_root,
            "results_dir": results / "tgb2_thgl_github_full_rel",
            "batch_size": 512,
            "tgb2_batch_size": 256,
        },
        "thgl_software": {
            "display_name": "thgl_software",
            "label": "software",
            "official_dataset_name": "thgl-software",
            "raw_edgelist": repo_root / "datasets" / "tgb2" / "thgl-software" / "thgl-software_edgelist.csv",
            "raw_nodetype": repo_root / "datasets" / "tgb2" / "thgl-software" / "thgl-software_nodetype.csv",
            "unified_pt": processed / "thgl_software_tgb2_full_unified.pt",
            "task_prefix": "thgl_software_tgb2_full",
            "task_dir": task_root,
            "results_dir": results / "tgb2_thgl_software_full_rel",
            "batch_size": 512,
            "tgb2_batch_size": 512,
        },
    }


def run_sync(cmd: List[str], cwd: Path, env: Dict[str, str]) -> int:
    print(f"[run] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    return int(proc.returncode)


def tail_last_line(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size <= 0:
                return ""
            off = min(size, 4096)
            f.seek(-off, os.SEEK_END)
            txt = f.read().decode("utf-8", errors="ignore")
        lines = [x.strip() for x in txt.splitlines() if x.strip()]
        return lines[-1] if lines else ""
    except Exception:
        return ""


def ensure_preprocessed(spec: Dict, project_root: Path, python_bin: str, env: Dict[str, str], force: bool) -> None:
    import_script = project_root / "scripts" / "datasets" / "import_tgb2_thg.py"
    task_script = project_root / "scripts" / "datasets" / "build_paper_task_views.py"
    prefix = spec["task_prefix"]
    unified_pt = spec["unified_pt"]
    task_dir = spec["task_dir"]
    trans_pt = task_dir / f"{prefix}_edge_transductive.pt"
    ind_pt = task_dir / f"{prefix}_edge_inductive.pt"

    if force or not unified_pt.exists():
        cmd = [
            python_bin,
            str(import_script),
            "--dataset-name",
            spec["display_name"],
            "--edgelist-csv",
            str(spec["raw_edgelist"]),
            "--nodetype-csv",
            str(spec["raw_nodetype"]),
            "--out-pt",
            str(unified_pt),
            "--max-edges",
            "-1",
        ]
        rc = run_sync(cmd, cwd=project_root, env=env)
        if rc != 0:
            raise RuntimeError(f"failed importing {spec['display_name']}")
    else:
        print(f"[prep] skip existing unified: {unified_pt}")

    if force or not (trans_pt.exists() and ind_pt.exists()):
        cmd = [
            python_bin,
            str(task_script),
            "--input",
            str(unified_pt),
            "--out-dir",
            str(task_dir),
            "--name-prefix",
            prefix,
        ]
        rc = run_sync(cmd, cwd=project_root, env=env)
        if rc != 0:
            raise RuntimeError(f"failed building task views for {spec['display_name']}")
    else:
        print(f"[prep] skip existing task views: {prefix}")


def task_paths(spec: Dict) -> Dict[str, Path]:
    prefix = spec["task_prefix"]
    base = spec["task_dir"]
    return {
        "trans": base / f"{prefix}_edge_transductive.pt",
        "ind": base / f"{prefix}_edge_inductive.pt",
    }


def build_jobs(spec: Dict, args, gpu_ids: List[int]) -> List[Dict]:
    paths = task_paths(spec)
    res = spec["results_dir"]
    jobs = [
        {
            "name": "tgb2_custom_trans",
            "kind": "baseline",
            "data": paths["trans"],
            "out_path": res / "tgb2_custom_trans.csv",
            "cmd": [
                args.python_bin,
                str(args.project_root / "scripts" / "baselines" / "run_tgb2_custom_smoke.py"),
                "--data",
                str(paths["trans"]),
                "--epochs",
                str(args.baseline_epochs),
                "--batch-size",
                str(spec["tgb2_batch_size"]),
                "--gpu-id",
                "0",
                "--eval-num-negs",
                str(args.eval_num_negs),
                "--out",
                str(res / "tgb2_custom_trans.csv"),
            ],
        },
        {
            "name": "tgb2_custom_ind",
            "kind": "baseline",
            "data": paths["ind"],
            "out_path": res / "tgb2_custom_ind.csv",
            "cmd": [
                args.python_bin,
                str(args.project_root / "scripts" / "baselines" / "run_tgb2_custom_smoke.py"),
                "--data",
                str(paths["ind"]),
                "--epochs",
                str(args.baseline_epochs),
                "--batch-size",
                str(spec["tgb2_batch_size"]),
                "--gpu-id",
                "0",
                "--eval-num-negs",
                str(args.eval_num_negs),
                "--out",
                str(res / "tgb2_custom_ind.csv"),
            ],
        },
        {
            "name": "tgb2_custom_edgetype_trans",
            "kind": "baseline",
            "data": paths["trans"],
            "out_path": res / "tgb2_custom_edgetype_trans.csv",
            "cmd": [
                args.python_bin,
                str(args.project_root / "scripts" / "baselines" / "run_tgb2_custom_smoke.py"),
                "--data",
                str(paths["trans"]),
                "--model",
                "tgn_edgetype",
                "--epochs",
                str(args.baseline_epochs),
                "--batch-size",
                str(spec["tgb2_batch_size"]),
                "--gpu-id",
                "0",
                "--eval-num-negs",
                str(args.eval_num_negs),
                "--out",
                str(res / "tgb2_custom_edgetype_trans.csv"),
            ],
        },
        {
            "name": "tgb2_custom_edgetype_ind",
            "kind": "baseline",
            "data": paths["ind"],
            "out_path": res / "tgb2_custom_edgetype_ind.csv",
            "cmd": [
                args.python_bin,
                str(args.project_root / "scripts" / "baselines" / "run_tgb2_custom_smoke.py"),
                "--data",
                str(paths["ind"]),
                "--model",
                "tgn_edgetype",
                "--epochs",
                str(args.baseline_epochs),
                "--batch-size",
                str(spec["tgb2_batch_size"]),
                "--gpu-id",
                "0",
                "--eval-num-negs",
                str(args.eval_num_negs),
                "--out",
                str(res / "tgb2_custom_edgetype_ind.csv"),
            ],
        },
        {
            "name": "sthn_trans",
            "kind": "baseline",
            "data": paths["trans"],
            "out_path": res / "sthn_trans.csv",
            "cmd": [
                args.python_bin,
                str(args.project_root / "scripts" / "baselines" / "run_tgb2_official_sthn_smoke.py"),
                "--data",
                str(paths["trans"]),
                "--epochs",
                str(args.baseline_epochs),
                "--batch-size",
                str(spec["tgb2_batch_size"]),
                "--gpu-id",
                "0",
                "--dataset-name",
                str(spec["official_dataset_name"]),
                "--out",
                str(res / "sthn_trans.csv"),
            ],
        },
        {
            "name": "sthn_ind",
            "kind": "baseline",
            "data": paths["ind"],
            "out_path": res / "sthn_ind.csv",
            "cmd": [
                args.python_bin,
                str(args.project_root / "scripts" / "baselines" / "run_tgb2_official_sthn_smoke.py"),
                "--data",
                str(paths["ind"]),
                "--epochs",
                str(args.baseline_epochs),
                "--batch-size",
                str(spec["tgb2_batch_size"]),
                "--gpu-id",
                "0",
                "--dataset-name",
                str(spec["official_dataset_name"]),
                "--out",
                str(res / "sthn_ind.csv"),
            ],
        },
        {
            "name": "dhc_trans",
            "kind": "dhc",
            "data": paths["trans"],
            "out_path": res / "dhc_trans" / "metrics.json",
            "cmd": [
                args.python_bin,
                "-m",
                "dhc.train.run_train",
                "--data",
                str(paths["trans"]),
                "--out-dir",
                str(res / "dhc_trans"),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(spec["batch_size"]),
                "--lr",
                str(args.lr),
                "--early-stop-patience",
                str(args.early_stop_patience),
                "--monitor-metric",
                "val_ap",
                "--lambda-ml",
                "0.0",
                "--node-eval-max-nodes",
                str(args.node_eval_max_nodes),
                "--gpu-id",
                "0",
            ],
        },
        {
            "name": "dhc_ind",
            "kind": "dhc",
            "data": paths["ind"],
            "out_path": res / "dhc_ind" / "metrics.json",
            "cmd": [
                args.python_bin,
                "-m",
                "dhc.train.run_train",
                "--data",
                str(paths["ind"]),
                "--out-dir",
                str(res / "dhc_ind"),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(spec["batch_size"]),
                "--lr",
                str(args.lr),
                "--early-stop-patience",
                str(args.early_stop_patience),
                "--monitor-metric",
                "val_ap",
                "--lambda-ml",
                "0.0",
                "--node-eval-max-nodes",
                str(args.node_eval_max_nodes),
                "--gpu-id",
                "0",
            ],
        },
        {
            "name": "dhc_trans_no_rel",
            "kind": "dhc",
            "data": paths["trans"],
            "out_path": res / "dhc_trans_no_rel" / "metrics.json",
            "cmd": [
                args.python_bin,
                "-m",
                "dhc.train.run_train",
                "--data",
                str(paths["trans"]),
                "--out-dir",
                str(res / "dhc_trans_no_rel"),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(spec["batch_size"]),
                "--lr",
                str(args.lr),
                "--early-stop-patience",
                str(args.early_stop_patience),
                "--monitor-metric",
                "val_ap",
                "--lambda-ml",
                "0.0",
                "--node-eval-max-nodes",
                str(args.node_eval_max_nodes),
                "--no-relation-memory",
                "--gpu-id",
                "0",
            ],
        },
        {
            "name": "dhc_trans_no_comm",
            "kind": "dhc",
            "data": paths["trans"],
            "out_path": res / "dhc_trans_no_comm" / "metrics.json",
            "cmd": [
                args.python_bin,
                "-m",
                "dhc.train.run_train",
                "--data",
                str(paths["trans"]),
                "--out-dir",
                str(res / "dhc_trans_no_comm"),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(spec["batch_size"]),
                "--lr",
                str(args.lr),
                "--early-stop-patience",
                str(args.early_stop_patience),
                "--monitor-metric",
                "val_ap",
                "--lambda-ml",
                "0.0",
                "--node-eval-max-nodes",
                str(args.node_eval_max_nodes),
                "--no-community",
                "--lambda-cl",
                "0.0",
                "--gpu-id",
                "0",
            ],
        },
    ]
    for idx, job in enumerate(jobs):
        if bool(getattr(args, "save_memory_artifacts", False)) and job["kind"] == "dhc":
            job["cmd"].append("--save-memory-artifacts")
        job["gpu"] = gpu_ids[idx % len(gpu_ids)] if gpu_ids else None
    return jobs


def run_jobs(spec: Dict, jobs: List[Dict], project_root: Path, env: Dict[str, str], max_parallel: int, poll_sec: int, skip_existing: bool) -> None:
    results_dir = spec["results_dir"]
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    running: List[Dict] = []
    queue = jobs[:]

    while queue or running:
        while queue and len(running) < max_parallel:
            job = queue.pop(0)
            out_path = Path(job["out_path"])
            if skip_existing and out_path.exists():
                print(f"[jobs] skip existing {spec['display_name']}::{job['name']}")
                continue
            if not Path(job["data"]).exists():
                raise FileNotFoundError(job["data"])

            log_path = logs_dir / f"{job['name']}.log"
            log_f = log_path.open("w", encoding="utf-8")
            job_env = env.copy()
            if job["gpu"] is not None:
                job_env["CUDA_VISIBLE_DEVICES"] = str(int(job["gpu"]))
            print(f"[jobs] launch {spec['display_name']}::{job['name']} gpu={job['gpu']}")
            proc = subprocess.Popen(
                job["cmd"],
                cwd=str(project_root),
                env=job_env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            running.append(
                {
                    "job": job,
                    "proc": proc,
                    "log_f": log_f,
                    "log_path": log_path,
                    "start": time.time(),
                }
            )

        time.sleep(max(1, poll_sec))
        still_running: List[Dict] = []
        for item in running:
            rc = item["proc"].poll()
            if rc is None:
                still_running.append(item)
                continue
            item["log_f"].close()
            elapsed = time.time() - item["start"]
            print(f"[jobs] done {spec['display_name']}::{item['job']['name']} rc={rc} elapsed={elapsed:.1f}s")
            if rc != 0:
                raise RuntimeError(f"job failed: {item['job']['name']} (see {item['log_path']})")
        running = still_running
        if running:
            print(f"[jobs] active={len(running)} queued={len(queue)}")
            for item in running:
                tail = tail_last_line(item["log_path"])
                if tail:
                    print(f"  - {spec['display_name']}::{item['job']['name']}: {tail}")


def load_test_row(csv_path: Path) -> Dict[str, float]:
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("split") == "test":
            return {
                "mrr": float(row["mrr"]),
                "auc": float(row["auc"]),
                "ap": float(row["ap"]),
                "avg_epoch_sec": float(row.get("avg_epoch_sec", 0.0)),
            }
    raise ValueError(f"missing test row in {csv_path}")


def load_dhc_metrics(metrics_path: Path) -> Dict[str, float]:
    obj = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "mrr": float(obj["test"]["mrr"]),
        "auc": float(obj["test"]["auc"]),
        "ap": float(obj["test"]["ap"]),
        "avg_epoch_sec": float(obj["avg_epoch_time_sec"]),
    }


def collect_metadata(spec: Dict) -> Dict:
    unified = torch.load(spec["unified_pt"], map_location="cpu")
    paths = task_paths(spec)
    trans = torch.load(paths["trans"], map_location="cpu")
    ind = torch.load(paths["ind"], map_location="cpu")

    def task_meta(obj: Dict) -> Dict:
        extra = obj.get("meta", {}).get("extra", {})
        return {
            "num_nodes": int(obj["num_nodes"]),
            "num_edges": int(obj["edges"]["src"].numel()),
            "num_task_relations": int(obj["num_rels"]),
            "train_edges": int(obj["split"]["train"].numel()),
            "val_edges": int(obj["split"]["val"].numel()),
            "test_edges": int(obj["split"]["test"].numel()),
            "source_rel_ids": list(extra.get("source_rel_ids", [])),
        }

    meta = {
        "dataset": spec["display_name"],
        "label": spec["label"],
        "raw_edgelist": str(spec["raw_edgelist"]),
        "raw_nodetype": str(spec["raw_nodetype"]),
        "unified_pt": str(spec["unified_pt"]),
        "task_prefix": spec["task_prefix"],
        "raw_num_nodes": int(unified["num_nodes"]),
        "raw_num_edges": int(unified["edges"]["src"].numel()),
        "raw_num_relations": int(unified["num_rels"]),
        "raw_num_node_types": int(unified["num_node_types"]),
        "raw_meta_extra": unified.get("meta", {}).get("extra", {}),
        "transductive_task": task_meta(trans),
        "inductive_task": task_meta(ind),
    }
    return meta


def write_summary(spec: Dict) -> None:
    results_dir = spec["results_dir"]
    records: List[Dict] = []

    for setting, stem in (("transductive", "trans"), ("inductive", "ind")):
        base = load_test_row(results_dir / f"tgb2_custom_{stem}.csv")
        edge = load_test_row(results_dir / f"tgb2_custom_edgetype_{stem}.csv")
        sthn = load_test_row(results_dir / f"sthn_{stem}.csv")
        dhc = load_dhc_metrics(results_dir / f"dhc_{stem}" / "metrics.json")
        records.append({"family": "baseline", "model": "TGB2-TGN", "setting": setting, **base})
        records.append({"family": "baseline", "model": "TGB2-TGN_edgetype", "setting": setting, **edge})
        records.append({"family": "baseline", "model": "STHN", "setting": setting, **sthn})
        records.append({"family": "dhc", "model": "DHC", "setting": setting, **dhc})
    no_rel = load_dhc_metrics(results_dir / "dhc_trans_no_rel" / "metrics.json")
    no_comm = load_dhc_metrics(results_dir / "dhc_trans_no_comm" / "metrics.json")
    records.append({"family": "dhc", "model": "DHC-no-rel", "setting": "transductive", **no_rel})
    records.append({"family": "dhc", "model": "DHC-no-comm", "setting": "transductive", **no_comm})

    summary_csv = results_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["family", "model", "setting", "mrr", "auc", "ap", "avg_epoch_sec"],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    meta = collect_metadata(spec)
    (results_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# {spec['display_name']} full relation-specific summary",
        "",
        f"- Raw graph: {meta['raw_num_edges']} edges, {meta['raw_num_nodes']} nodes, {meta['raw_num_relations']} relations, {meta['raw_num_node_types']} node types",
        f"- Transductive task view: {meta['transductive_task']['num_edges']} edges, source_rel_ids={meta['transductive_task']['source_rel_ids']}",
        f"- Inductive task view: {meta['inductive_task']['num_edges']} edges, source_rel_ids={meta['inductive_task']['source_rel_ids']}",
        "",
        "| family | model | setting | mrr | auc | ap | avg_epoch_sec |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['family']} | {row['model']} | {row['setting']} | "
            f"{row['mrr']:.4f} | {row['auc']:.4f} | {row['ap']:.4f} | {row['avg_epoch_sec']:.2f} |"
        )
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_tables(project_root: Path, python_bin: str) -> None:
    render_script = project_root / "scripts" / "experiments" / "render_tgb2_thg_tables_to_latex.py"
    results_root = resolve_results_root(project_root)
    args = [
        python_bin,
        str(render_script),
        "--dataset-root",
        f"thgl_github={results_root / 'tgb2_thgl_github_full_rel'}",
        "--dataset-root",
        f"thgl_software={results_root / 'tgb2_thgl_software_full_rel'}",
        "--main-out",
        str(project_root / "paper_draft" / "tables" / "tgb2_thg_main.tex"),
        "--ablation-out",
        str(project_root / "paper_draft" / "tables" / "tgb2_thg_ablation.tex"),
    ]
    rc = subprocess.run(args, cwd=str(project_root)).returncode
    if rc != 0:
        raise RuntimeError("failed rendering ESWA TGB2 tables")

    args = [
        python_bin,
        str(render_script),
        "--dataset-root",
        f"thgl_github={results_root / 'tgb2_thgl_github_full_rel'}",
        "--dataset-root",
        f"thgl_software={results_root / 'tgb2_thgl_software_full_rel'}",
        "--main-out",
        str(project_root / "paper_draft" / "ICONIP" / "tables" / "tgb2_thg_main.tex"),
        "--ablation-out",
        str(project_root / "paper_draft" / "ICONIP" / "tables" / "tgb2_thg_ablation.tex"),
    ]
    rc = subprocess.run(args, cwd=str(project_root)).returncode
    if rc != 0:
        raise RuntimeError("failed rendering ICONIP TGB2 tables")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path("/home/xuxj/2026/dyn_hetero_community"))
    p.add_argument("--repo-root", type=Path, default=Path("/home/xuxj/2026"))
    p.add_argument("--python-bin", type=str, default="python")
    p.add_argument("--datasets", type=str, default="thgl_github,thgl_software")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--baseline-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--early-stop-patience", type=int, default=2)
    p.add_argument("--eval-num-negs", type=int, default=20)
    p.add_argument("--node-eval-max-nodes", type=int, default=200000)
    p.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    p.add_argument("--max-parallel", type=int, default=2)
    p.add_argument("--poll-sec", type=int, default=20)
    p.add_argument("--force-preprocess", action="store_true")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--save-memory-artifacts", action="store_true", default=False)
    p.add_argument("--no-render", action="store_true", help="Skip LaTeX table refresh after experiments.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.repo_root = args.repo_root.resolve()
    specs = dataset_specs(project_root=args.project_root, repo_root=args.repo_root)
    selected = [x.strip() for x in args.datasets.split(",") if x.strip()]
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]

    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = (
        f"{args.project_root / 'src'}:{args.repo_root / 'baselines' / 'src' / 'TGB2-main' / 'TGB2-main'}:"
        f"{base_env.get('PYTHONPATH', '')}"
    )
    base_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    base_env["OMP_NUM_THREADS"] = "1"
    base_env["MKL_NUM_THREADS"] = "1"
    base_env["OPENBLAS_NUM_THREADS"] = "1"
    base_env["NUMEXPR_NUM_THREADS"] = "1"
    base_env["PYTHONUNBUFFERED"] = "1"

    for name in selected:
        if name not in specs:
            raise KeyError(f"unknown dataset: {name}")
        spec = specs[name]
        spec["batch_size"] = int(args.batch_size)
        spec["results_dir"].mkdir(parents=True, exist_ok=True)
        ensure_preprocessed(
            spec=spec,
            project_root=args.project_root,
            python_bin=args.python_bin,
            env=base_env,
            force=bool(args.force_preprocess),
        )
        jobs = build_jobs(spec=spec, args=args, gpu_ids=gpu_ids)
        run_jobs(
            spec=spec,
            jobs=jobs,
            project_root=args.project_root,
            env=base_env,
            max_parallel=args.max_parallel,
            poll_sec=args.poll_sec,
            skip_existing=bool(args.skip_existing),
        )
        write_summary(spec)

    if not args.no_render:
        render_tables(project_root=args.project_root, python_bin=args.python_bin)


if __name__ == "__main__":
    main()
