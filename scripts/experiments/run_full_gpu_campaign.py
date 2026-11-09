#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List


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


def run_stage(name: str, cmd: List[str], cwd: Path, env: Dict[str, str], log_path: Path) -> Dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[campaign] stage={name} cmd={' '.join(cmd)}")
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT)
    return {
        "stage": name,
        "return_code": int(proc.returncode),
        "elapsed_sec": float(time.time() - t0),
        "log_path": str(log_path),
        "cmd": " ".join(cmd),
    }


def load_manifest_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_stage_manifest(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "return_code", "elapsed_sec", "log_path", "cmd"])
        w.writeheader()
        for row in rows:
            w.writerow(row)


def ensure_cross_relation_views(project_root: Path, python_bin: str, env: Dict[str, str]) -> None:
    processed_root = resolve_processed_root(project_root)
    task_root = processed_root / "tasks"
    for ds in ["raw_cn", "raw_us"]:
        input_pt = processed_root / f"{ds}_unified.pt"
        out_pt = task_root / f"{ds}_edge_patent_ipc_plus_assignee_ipc.pt"
        if out_pt.exists():
            continue
        cmd = [
            python_bin,
            str(project_root / "scripts" / "datasets" / "build_cross_relation_patent_view.py"),
            "--input",
            str(input_pt),
            "--out-dir",
            str(task_root),
            "--name-prefix",
            ds,
            "--context-only-aux",
        ]
        result = run_stage(
            name=f"build_cross_relation_view::{ds}",
            cmd=cmd,
            cwd=project_root,
            env=env,
            log_path=resolve_results_root(project_root) / "gpu_full_campaign" / "logs" / f"build_cross_relation_{ds}.log",
        )
        if result["return_code"] != 0:
            raise RuntimeError(f"failed building cross relation view for {ds}")


def collect_postprocess_jobs(project_root: Path, results_root: Path, memory_seed: int) -> List[Dict[str, str]]:
    jobs: List[Dict[str, str]] = []
    processed_root = resolve_processed_root(project_root)

    fb_manifest = results_root / "full_benchmark_repeats_gpu" / "run_manifest.csv"
    for row in load_manifest_rows(fb_manifest):
        if int(row.get("return_code", 1)) != 0:
            continue
        if row.get("variant") != "full":
            continue
        if int(row.get("seed", -1)) != int(memory_seed):
            continue
        out_dir = Path(row["out_dir"])
        if not (out_dir / "metrics.json").exists():
            continue
        jobs.append({"data": row["data"], "run_dir": row["out_dir"]})

    for ds in ["thgl_github", "thgl_software"]:
        task_prefix = f"{ds}_tgb2_full"
        data_root = resolve_processed_root(project_root) / "tgb2_tasks"
        results_dir = results_root / f"tgb2_{ds}_full_rel"
        for split, run_name in [("transductive", "dhc_trans"), ("inductive", "dhc_ind")]:
            run_dir = results_dir / run_name
            data = data_root / f"{task_prefix}_edge_{'transductive' if split == 'transductive' else 'inductive'}.pt"
            if run_dir.exists() and data.exists():
                jobs.append({"data": str(data), "run_dir": str(run_dir)})

    cross_root = results_root / "iter_opt" / "cross_relation_boost"
    for ds in ["raw_cn", "raw_us"]:
        for tag in ["base", "boost"]:
            run_dir = cross_root / f"{ds}_patent_ipc_{tag}"
            data = processed_root / "tasks" / (f"{ds}_edge_patent_ipc.pt" if tag == "base" else f"{ds}_edge_patent_ipc_plus_assignee_ipc.pt")
            if run_dir.exists() and data.exists():
                jobs.append({"data": str(data), "run_dir": str(run_dir)})
    return jobs


def run_case_studies(project_root: Path, results_root: Path, python_bin: str, env: Dict[str, str], stage_rows: List[Dict]) -> None:
    processed_root = resolve_processed_root(project_root)
    cross_root = results_root / "iter_opt" / "cross_relation_boost"
    out_root = results_root / "case_studies"
    for ds in ["raw_cn", "raw_us"]:
        cmd = [
            python_bin,
            str(project_root / "scripts" / "experiments" / "generate_patent_case_study.py"),
            "--raw-data",
            str(processed_root / f"{ds}_unified.pt"),
            "--base-data",
            str(processed_root / "tasks" / f"{ds}_edge_patent_ipc.pt"),
            "--boost-data",
            str(processed_root / "tasks" / f"{ds}_edge_patent_ipc_plus_assignee_ipc.pt"),
            "--base-run",
            str(cross_root / f"{ds}_patent_ipc_base"),
            "--boost-run",
            str(cross_root / f"{ds}_patent_ipc_boost"),
            "--out-dir",
            str(out_root / ds),
        ]
        row = run_stage(
            name=f"case_study::{ds}",
            cmd=cmd,
            cwd=project_root,
            env=env,
            log_path=results_root / "gpu_full_campaign" / "logs" / f"case_study_{ds}.log",
        )
        stage_rows.append(row)
        if row["return_code"] != 0:
            raise RuntimeError(f"case study failed for {ds}")


def run_postprocess(project_root: Path, results_root: Path, python_bin: str, env: Dict[str, str], memory_seed: int, stage_rows: List[Dict]) -> None:
    seen = set()
    for job in collect_postprocess_jobs(project_root=project_root, results_root=results_root, memory_seed=memory_seed):
        key = (job["data"], job["run_dir"])
        if key in seen:
            continue
        seen.add(key)
        row = run_stage(
            name=f"postprocess::{Path(job['run_dir']).name}",
            cmd=[
                python_bin,
                str(project_root / "scripts" / "experiments" / "postprocess_dhc_run.py"),
                "--project-root",
                str(project_root),
                "--python-bin",
                python_bin,
                "--data",
                str(job["data"]),
                "--run-dir",
                str(job["run_dir"]),
            ],
            cwd=project_root,
            env=env,
            log_path=results_root / "gpu_full_campaign" / "logs" / f"postprocess_{Path(job['run_dir']).name}.log",
        )
        stage_rows.append(row)
        if row["return_code"] != 0:
            raise RuntimeError(f"postprocess failed for {job['run_dir']}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Run the full multi-GPU experiment campaign with diagnostics")
    ap.add_argument("--project-root", type=Path, default=Path("/home/xuxj/2026/dyn_hetero_community"))
    ap.add_argument("--repo-root", type=Path, default=Path("/home/xuxj/2026"))
    ap.add_argument("--python-bin", type=str, default="python")
    ap.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--baseline-epochs", type=int, default=10)
    ap.add_argument("--dhc-epochs", type=int, default=20)
    ap.add_argument("--thg-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--baseline-batch-size", type=int, default=256)
    ap.add_argument("--poll-sec", type=int, default=20)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--memory-seed", type=int, default=42)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    repo_root = args.repo_root.resolve()
    results_root = resolve_results_root(project_root)
    campaign_root = results_root / "gpu_full_campaign"
    campaign_root.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = f"{project_root / 'src'}:{base_env.get('PYTHONPATH', '')}"
    base_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    base_env["OMP_NUM_THREADS"] = "1"
    base_env["MKL_NUM_THREADS"] = "1"
    base_env["OPENBLAS_NUM_THREADS"] = "1"
    base_env["NUMEXPR_NUM_THREADS"] = "1"
    base_env["MKL_THREADING_LAYER"] = "GNU"
    base_env["PYTHONUNBUFFERED"] = "1"

    ensure_cross_relation_views(project_root=project_root, python_bin=args.python_bin, env=base_env)

    stage_rows: List[Dict] = []
    datasets = "raw_cn,raw_us,arxiv,dblp,amazon_movies,aminer_paper"

    stage_rows.append(
        run_stage(
            name="baseline_formal_all",
            cmd=[
                args.python_bin,
                str(project_root / "scripts" / "baselines" / "run_all_baselines_formal.py"),
                "--project-root",
                str(project_root),
                "--repo-root",
                str(repo_root),
                "--python-bin",
                args.python_bin,
                "--out-root",
                str(results_root / "baseline_formal_all_gpu"),
                "--datasets",
                datasets,
                "--max-parallel",
                str(args.max_parallel),
                "--epochs",
                str(args.baseline_epochs),
                "--batch-size",
                str(args.baseline_batch_size),
                "--poll-sec",
                str(args.poll_sec),
                "--gpu-ids",
                str(args.gpu_ids),
            ],
            cwd=project_root,
            env=base_env,
            log_path=campaign_root / "logs" / "baseline_formal_all.log",
        )
    )
    if stage_rows[-1]["return_code"] != 0:
        raise RuntimeError("baseline_formal_all failed")

    stage_rows.append(
        run_stage(
            name="full_benchmark_repeats",
            cmd=[
                args.python_bin,
                str(project_root / "scripts" / "experiments" / "run_full_benchmark_repeats.py"),
                "--project-root",
                str(project_root),
                "--python-bin",
                args.python_bin,
                "--out-root",
                str(results_root / "full_benchmark_repeats_gpu"),
                "--datasets",
                datasets,
                "--seeds",
                str(args.seeds),
                "--epochs",
                str(args.dhc_epochs),
                "--batch-size",
                str(args.batch_size),
                "--max-parallel",
                str(args.max_parallel),
                "--poll-sec",
                str(args.poll_sec),
                "--mode",
                "both",
                "--ablation-datasets",
                "arxiv,dblp,raw_cn,raw_us,aminer_paper",
                "--gpu-ids",
                str(args.gpu_ids),
                "--include-single-type-datasets",
                "--save-memory-artifacts",
                "--save-memory-variants",
                "full",
            ]
            + (["--skip-existing"] if args.skip_existing else ["--no-skip-existing"]),
            cwd=project_root,
            env=base_env,
            log_path=campaign_root / "logs" / "full_benchmark_repeats.log",
        )
    )
    if stage_rows[-1]["return_code"] != 0:
        raise RuntimeError("full_benchmark_repeats failed")

    stage_rows.append(
        run_stage(
            name="tgb2_thg_full",
            cmd=[
                args.python_bin,
                str(project_root / "scripts" / "experiments" / "run_tgb2_thg_full_experiments.py"),
                "--project-root",
                str(project_root),
                "--repo-root",
                str(repo_root),
                "--python-bin",
                args.python_bin,
                "--datasets",
                "thgl_github,thgl_software",
                "--epochs",
                str(args.thg_epochs),
                "--baseline-epochs",
                str(args.baseline_epochs),
                "--batch-size",
                str(args.batch_size),
                "--gpu-ids",
                str(args.gpu_ids),
                "--max-parallel",
                str(args.max_parallel),
                "--poll-sec",
                str(args.poll_sec),
                "--save-memory-artifacts",
            ]
            + (["--skip-existing"] if args.skip_existing else ["--no-skip-existing"]),
            cwd=project_root,
            env=base_env,
            log_path=campaign_root / "logs" / "tgb2_thg_full.log",
        )
    )
    if stage_rows[-1]["return_code"] != 0:
        raise RuntimeError("tgb2_thg_full failed")

    stage_rows.append(
        run_stage(
            name="cross_relation_boost",
            cmd=[
                args.python_bin,
                str(project_root / "scripts" / "experiments" / "run_cross_relation_boost_experiments.py"),
                "--project-root",
                str(project_root),
                "--python-bin",
                args.python_bin,
                "--datasets",
                "raw_cn,raw_us",
                "--epochs",
                str(args.dhc_epochs),
                "--batch-size",
                str(args.batch_size),
                "--gpu-ids",
                str(args.gpu_ids),
                "--max-parallel",
                str(args.max_parallel),
                "--save-memory-artifacts",
            ],
            cwd=project_root,
            env=base_env,
            log_path=campaign_root / "logs" / "cross_relation_boost.log",
        )
    )
    if stage_rows[-1]["return_code"] != 0:
        raise RuntimeError("cross_relation_boost failed")

    run_case_studies(project_root=project_root, results_root=results_root, python_bin=args.python_bin, env=base_env, stage_rows=stage_rows)
    run_postprocess(
        project_root=project_root,
        results_root=results_root,
        python_bin=args.python_bin,
        env=base_env,
        memory_seed=int(args.memory_seed),
        stage_rows=stage_rows,
    )

    append_stage_manifest(stage_rows, campaign_root / "stage_manifest.csv")
    print(campaign_root / "stage_manifest.csv")


if __name__ == "__main__":
    main()
