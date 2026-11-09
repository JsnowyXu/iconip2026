#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

import torch


ROOT_TGB2 = Path("/home/xuxj/2026/baselines/src/TGB2-main/TGB2-main")


def run_cmd(cmd, cwd: Path, env):
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    return proc


def ensure_sampler_core(tgb2_root: Path, env) -> None:
    modules_dir = tgb2_root / "modules"
    existing = list(modules_dir.glob("sampler_core*.so"))
    if existing:
        return
    cmd = [sys.executable, "sthn_sampler_setup.py", "build_ext", "--inplace"]
    run_cmd(cmd, cwd=modules_dir, env=env)


def prepare_dataset(task_pt: Path, tgb2_root: Path, dataset_name: str, max_edges: int, env) -> None:
    prepare_py = Path(__file__).resolve().parent / "prepare_tgb2_official_dataset.py"
    cmd = [
        sys.executable,
        str(prepare_py),
        "--input",
        str(task_pt),
        "--tgb2-root",
        str(tgb2_root),
        "--dataset-name",
        dataset_name,
        "--max-edges",
        str(max_edges),
    ]
    run_cmd(cmd, cwd=tgb2_root, env=env)


def parse_metric_cell(value: str) -> float:
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    if match is None:
        return float("nan")
    return float(match.group(0))


def load_last_result(results_json: Path):
    payload = json.loads(results_json.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not payload:
            raise RuntimeError(f"empty results file: {results_json}")
        return payload[-1]
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--dataset-name", default="thgl-github")
    ap.add_argument("--max-edges", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    task_pt = Path(args.data).resolve()
    tgb2_root = ROOT_TGB2.resolve()
    if not task_pt.exists():
        raise FileNotFoundError(task_pt)
    if not tgb2_root.exists():
        raise FileNotFoundError(tgb2_root)

    obj = torch.load(task_pt, map_location="cpu")
    num_edges = int(obj["edges"]["src"].numel())
    max_edges = int(args.max_edges) if int(args.max_edges) > 0 else num_edges

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{tgb2_root}:{tgb2_root / 'modules'}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    local_gpu_id = int(args.gpu_id)
    if torch.cuda.is_available():
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        local_gpu_id = 0

    include_dir = sysconfig.get_config_var("INCLUDEPY")
    include_candidates = [
        Path(str(include_dir)) if include_dir else None,
        Path(sys.prefix) / "include" / f"python{sys.version_info.major}.{sys.version_info.minor}",
        Path("/opt/anaconda3/include") / f"python{sys.version_info.major}.{sys.version_info.minor}",
        Path("/usr/include") / f"python{sys.version_info.major}.{sys.version_info.minor}",
    ]
    include_candidates = [p for p in include_candidates if p is not None and (p / "Python.h").exists()]
    if include_candidates:
        include_flag = f"-I{include_candidates[0]}"
        env["CFLAGS"] = f"{include_flag} {env.get('CFLAGS', '')}".strip()
        env["CPPFLAGS"] = f"{include_flag} {env.get('CPPFLAGS', '')}".strip()

    ensure_sampler_core(tgb2_root=tgb2_root, env=env)
    prepare_dataset(task_pt=task_pt, tgb2_root=tgb2_root, dataset_name=args.dataset_name, max_edges=max_edges, env=env)

    example_dir = tgb2_root / "examples" / "linkproppred" / args.dataset_name
    results_json = example_dir / "saved_results" / f"STHN_{args.dataset_name}_results.json"
    log_path = Path(args.out).with_suffix(".log")
    results_json.parent.mkdir(parents=True, exist_ok=True)
    if results_json.exists():
        results_json.unlink()

    cmd = [
        sys.executable,
        str(example_dir / "sthn.py"),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--device",
        str(local_gpu_id),
        "--num_run",
        "1",
        "--seed",
        "1",
    ]
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(tgb2_root), env=env, capture_output=True, text=True)
    elapsed = time.time() - start
    log_path.write_text(proc.stdout + "\n\n[stderr]\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"STHN run failed ({proc.returncode}), see log: {log_path}\n"
            f"stderr:\n{proc.stderr}"
        )
    if not results_json.exists():
        raise RuntimeError(f"STHN results file not found: {results_json}")

    result = load_last_result(results_json)
    val_mrr = parse_metric_cell(result.get("val mrr", "nan"))
    test_mrr = parse_metric_cell(result.get("test mrr", "nan"))
    val_auc = parse_metric_cell(result.get("val auc", "nan"))
    test_auc = parse_metric_cell(result.get("test auc", "nan"))
    val_ap = parse_metric_cell(result.get("val ap", "nan"))
    test_ap = parse_metric_cell(result.get("test ap", "nan"))
    total_params = None
    trainable_params = None
    match = re.search(r"Trainable Parameters:\s*(\d+)", proc.stdout)
    if match is not None:
        total_params = int(match.group(1))
        trainable_params = total_params

    avg_epoch_sec = float(elapsed / max(int(args.epochs), 1))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "model",
                "mrr",
                "auc",
                "ap",
                "avg_epoch_sec",
                "total_params",
                "trainable_params",
                "device",
                "eval_num_negs",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "split": "val",
                "model": "sthn",
                "mrr": val_mrr,
                "auc": val_auc,
                "ap": val_ap,
                "avg_epoch_sec": avg_epoch_sec,
                "total_params": total_params if total_params is not None else "",
                "trainable_params": trainable_params if trainable_params is not None else "",
                "device": f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu",
                "eval_num_negs": 20,
            }
        )
        w.writerow(
            {
                "split": "test",
                "model": "sthn",
                "mrr": test_mrr,
                "auc": test_auc,
                "ap": test_ap,
                "avg_epoch_sec": avg_epoch_sec,
                "total_params": total_params if total_params is not None else "",
                "trainable_params": trainable_params if trainable_params is not None else "",
                "device": f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu",
                "eval_num_negs": 20,
            }
        )
    print(
        f"model=sthn test_mrr={test_mrr:.6f} test_auc={test_auc:.6f} test_ap={test_ap:.6f} val_mrr={val_mrr:.6f} "
        f"avg_epoch_sec={avg_epoch_sec:.2f} log={log_path}"
    )


if __name__ == "__main__":
    main()
