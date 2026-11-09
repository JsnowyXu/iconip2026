#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import time
from pathlib import Path


def exps(data_root: Path, include_aminer: bool = False):
    items = [
        {"name": "dblp_trans_full", "data": data_root / "dblp_edge_transductive.pt", "gpu": 0, "extra": []},
        {"name": "dblp_ind_full", "data": data_root / "dblp_edge_inductive.pt", "gpu": 1, "extra": []},
        {"name": "dblp_node_full", "data": data_root / "dblp_node_multilabel.pt", "gpu": 2, "extra": []},
        {"name": "arxiv_trans_full", "data": data_root / "arxiv_edge_transductive.pt", "gpu": 3, "extra": []},
        {"name": "arxiv_ind_full", "data": data_root / "arxiv_edge_inductive.pt", "gpu": 0, "extra": []},
        {"name": "arxiv_node_full", "data": data_root / "arxiv_node_multilabel.pt", "gpu": 1, "extra": []},
        # ablations on dblp transductive
        {"name": "dblp_trans_no_rel", "data": data_root / "dblp_edge_transductive.pt", "gpu": 2, "extra": ["--no-relation-memory"]},
        {"name": "dblp_trans_no_comm", "data": data_root / "dblp_edge_transductive.pt", "gpu": 3, "extra": ["--no-community"]},
        {
            "name": "dblp_trans_no_rel_no_comm",
            "data": data_root / "dblp_edge_transductive.pt",
            "gpu": 0,
            "extra": ["--no-relation-memory", "--no-community"],
        },
        {"name": "dblp_trans_no_cl", "data": data_root / "dblp_edge_transductive.pt", "gpu": 1, "extra": ["--lambda-cl", "0.0"]},
        {"name": "dblp_node_no_rel", "data": data_root / "dblp_node_multilabel.pt", "gpu": 2, "extra": ["--no-relation-memory"]},
        {"name": "dblp_node_no_comm", "data": data_root / "dblp_node_multilabel.pt", "gpu": 3, "extra": ["--no-community"]},
    ]
    if include_aminer:
        items.extend(
            [
                {"name": "aminer_trans_full", "data": data_root / "aminer_paper_edge_transductive.pt", "gpu": 2, "extra": []},
                {"name": "aminer_ind_full", "data": data_root / "aminer_paper_edge_inductive.pt", "gpu": 3, "extra": []},
                {"name": "aminer_node_full", "data": data_root / "aminer_paper_node_multilabel.pt", "gpu": 0, "extra": []},
            ]
        )
    return items


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="/home/xuxj/2026/dyn_hetero_community")
    ap.add_argument("--python-bin", default="python")
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--early-stop-patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--poll-sec", type=int, default=20)
    ap.add_argument("--gpu-ids", default="", help="Comma-separated physical GPU ids. Empty keeps the original mapping.")
    ap.add_argument("--include-aminer", action="store_true", help="Also schedule aminer_paper paper-task runs when task files exist.")
    args = ap.parse_args()

    proj = Path(args.project_root).resolve()
    data_root = proj / "data" / "processed" / "paper_tasks"
    out_root = proj / "results" / "cuda_paper_academic"
    log_root = out_root / "logs"
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    queue = exps(data_root=data_root, include_aminer=args.include_aminer)
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    running = []
    finished = []

    print(f"[scheduler] total_experiments={len(queue)}")
    while queue or running:
        while queue and len(running) < args.max_parallel:
            exp = queue.pop(0)
            if not exp["data"].exists():
                print(f"[scheduler] skip {exp['name']}: missing data {exp['data']}")
                finished.append(
                    {
                        "name": exp["name"],
                        "gpu": exp["gpu"],
                        "data": str(exp["data"]),
                        "return_code": 99,
                        "elapsed_sec": 0.0,
                        "log_path": "",
                        "out_dir": str(out_root / exp["name"]),
                        "extra": " ".join(exp["extra"]),
                    }
                )
                continue
            out_dir = out_root / exp["name"]
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_root / f"{exp['name']}.log"
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
                str(out_dir),
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
            print(f"[scheduler] launch {exp['name']} on GPU{assigned_gpu}")
            p = subprocess.Popen(cmd, cwd=str(proj), env=env, stdout=log_f, stderr=subprocess.STDOUT)
            running.append(
                {
                    "exp": exp,
                    "assigned_gpu": assigned_gpu,
                    "proc": p,
                    "log_f": log_f,
                    "log_path": log_path,
                    "start": time.time(),
                }
            )

        time.sleep(max(1, args.poll_sec))
        nr = []
        for it in running:
            rc = it["proc"].poll()
            if rc is None:
                nr.append(it)
            else:
                it["log_f"].close()
                elapsed = time.time() - it["start"]
                print(f"[scheduler] done {it['exp']['name']} rc={rc} elapsed={elapsed:.1f}s")
                finished.append(
                    {
                        "name": it["exp"]["name"],
                        "gpu": it["assigned_gpu"],
                        "data": str(it["exp"]["data"]),
                        "return_code": int(rc),
                        "elapsed_sec": float(elapsed),
                        "log_path": str(it["log_path"]),
                        "out_dir": str(out_root / it["exp"]["name"]),
                        "extra": " ".join(it["exp"]["extra"]),
                    }
                )
        running = nr
        if running:
            print(f"[scheduler] active={len(running)} queued={len(queue)}")
            for it in running:
                tail = tail_last_line(it["log_path"])
                if tail:
                    print(f"  - {it['exp']['name']}: {tail}")

    man = out_root / "run_manifest.csv"
    with man.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["name", "gpu", "data", "return_code", "elapsed_sec", "log_path", "out_dir", "extra"],
        )
        w.writeheader()
        for r in finished:
            w.writerow(r)
    print(f"[scheduler] manifest: {man}")


if __name__ == "__main__":
    main()
