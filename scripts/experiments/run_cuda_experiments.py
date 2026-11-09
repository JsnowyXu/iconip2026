#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import time
from pathlib import Path


def build_experiments(data_root: Path):
    # task: transductive, inductive, node_multilabel + ablations
    return [
        {
            "name": "cn_trans_full",
            "data": data_root / "raw_cn_edge_assignee_ipc_transductive.pt",
            "gpu": 0,
            "extra": [],
        },
        {
            "name": "cn_ind_full",
            "data": data_root / "raw_cn_edge_assignee_ipc_inductive.pt",
            "gpu": 1,
            "extra": [],
        },
        {
            "name": "cn_node_full",
            "data": data_root / "raw_cn_node_patent_multilabel.pt",
            "gpu": 2,
            "extra": [],
        },
        {
            "name": "us_trans_full",
            "data": data_root / "raw_us_edge_assignee_ipc_transductive.pt",
            "gpu": 3,
            "extra": [],
        },
        {
            "name": "us_ind_full",
            "data": data_root / "raw_us_edge_assignee_ipc_inductive.pt",
            "gpu": 0,
            "extra": [],
        },
        {
            "name": "us_node_full",
            "data": data_root / "raw_us_node_patent_multilabel.pt",
            "gpu": 1,
            "extra": [],
        },
        # Ablations on CN transductive
        {
            "name": "cn_trans_no_rel",
            "data": data_root / "raw_cn_edge_assignee_ipc_transductive.pt",
            "gpu": 2,
            "extra": ["--no-relation-memory"],
        },
        {
            "name": "cn_trans_no_comm",
            "data": data_root / "raw_cn_edge_assignee_ipc_transductive.pt",
            "gpu": 3,
            "extra": ["--no-community"],
        },
        {
            "name": "cn_trans_no_rel_no_comm",
            "data": data_root / "raw_cn_edge_assignee_ipc_transductive.pt",
            "gpu": 0,
            "extra": ["--no-relation-memory", "--no-community"],
        },
        {
            "name": "cn_trans_no_cl",
            "data": data_root / "raw_cn_edge_assignee_ipc_transductive.pt",
            "gpu": 1,
            "extra": ["--lambda-cl", "0.0"],
        },
        # Ablations on CN node task
        {
            "name": "cn_node_no_rel",
            "data": data_root / "raw_cn_node_patent_multilabel.pt",
            "gpu": 2,
            "extra": ["--no-relation-memory"],
        },
        {
            "name": "cn_node_no_comm",
            "data": data_root / "raw_cn_node_patent_multilabel.pt",
            "gpu": 3,
            "extra": ["--no-community"],
        },
    ]


def tail_last_line(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size <= 0:
                return ""
            offset = min(4096, size)
            f.seek(-offset, os.SEEK_END)
            txt = f.read().decode("utf-8", errors="ignore")
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        return lines[-1] if lines else ""
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="/home/xuxj/2026/dyn_hetero_community")
    ap.add_argument("--python-bin", default="python")
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--early-stop-patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--poll-sec", type=int, default=20)
    ap.add_argument("--gpu-ids", default="", help="Comma-separated physical GPU ids. Empty keeps the original mapping.")
    args = ap.parse_args()

    proj = Path(args.project_root).resolve()
    data_root = proj / "data" / "processed" / "tasks"
    out_root = proj / "results" / "cuda_paper"
    log_root = out_root / "logs"
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    exps = build_experiments(data_root=data_root)
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    queue = list(exps)
    running = []
    finished = []

    print(f"[scheduler] total_experiments={len(queue)} max_parallel={args.max_parallel}")
    while queue or running:
        # Launch new jobs.
        while queue and len(running) < args.max_parallel:
            exp = queue.pop(0)
            exp_name = exp["name"]
            exp_out = out_root / exp_name
            exp_out.mkdir(parents=True, exist_ok=True)
            log_path = log_root / f"{exp_name}.log"
            log_f = log_path.open("w", encoding="utf-8")

            env = os.environ.copy()
            env["PYTHONPATH"] = f"{proj / 'src'}:{env.get('PYTHONPATH', '')}"
            assigned_gpu = gpu_ids[exp["gpu"] % len(gpu_ids)] if gpu_ids else exp["gpu"]
            env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"

            cmd = [
                args.python_bin,
                "-m",
                "dhc.train.run_train",
                "--data",
                str(exp["data"]),
                "--out-dir",
                str(exp_out),
                "--epochs",
                str(args.epochs),
                "--early-stop-patience",
                str(args.early_stop_patience),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--gpu-id",
                "0",
                "--ml-threshold-search",
            ] + exp["extra"]
            print(f"[scheduler] launch {exp_name} on GPU{assigned_gpu}: {' '.join(cmd)}")
            p = subprocess.Popen(cmd, cwd=str(proj), env=env, stdout=log_f, stderr=subprocess.STDOUT)
            running.append(
                {
                    "exp": exp,
                    "assigned_gpu": assigned_gpu,
                    "proc": p,
                    "log_path": log_path,
                    "log_f": log_f,
                    "start_time": time.time(),
                }
            )

        # Poll running jobs.
        time.sleep(max(1, args.poll_sec))
        still_running = []
        for item in running:
            p = item["proc"]
            rc = p.poll()
            if rc is None:
                still_running.append(item)
            else:
                item["log_f"].close()
                elapsed = time.time() - item["start_time"]
                print(f"[scheduler] done {item['exp']['name']} rc={rc} elapsed={elapsed:.1f}s")
                finished.append(
                    {
                        "name": item["exp"]["name"],
                        "gpu": item["assigned_gpu"],
                        "data": str(item["exp"]["data"]),
                        "return_code": int(rc),
                        "elapsed_sec": float(elapsed),
                        "log_path": str(item["log_path"]),
                        "out_dir": str(out_root / item["exp"]["name"]),
                        "extra": " ".join(item["exp"]["extra"]),
                    }
                )
        running = still_running

        if running:
            print(f"[scheduler] active={len(running)} queued={len(queue)}")
            for item in running:
                tail = tail_last_line(item["log_path"])
                if tail:
                    print(f"  - {item['exp']['name']}: {tail}")

    # Save manifest.
    manifest = out_root / "run_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["name", "gpu", "data", "return_code", "elapsed_sec", "log_path", "out_dir", "extra"],
        )
        w.writeheader()
        for row in finished:
            w.writerow(row)
    print(f"[scheduler] manifest: {manifest}")


if __name__ == "__main__":
    main()
