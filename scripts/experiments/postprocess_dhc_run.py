#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path


def run_step(cmd, cwd: Path, env: dict) -> None:
    print("[postprocess]", " ".join(str(x) for x in cmd))
    rc = subprocess.run(cmd, cwd=str(cwd), env=env).returncode
    if rc != 0:
        raise RuntimeError(f"postprocess failed: {' '.join(str(x) for x in cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser("Generate memory/community diagnostics for a DHC run directory")
    ap.add_argument("--project-root", required=True, type=str)
    ap.add_argument("--python-bin", default="python", type=str)
    ap.add_argument("--data", required=True, type=str)
    ap.add_argument("--run-dir", required=True, type=str)
    ap.add_argument("--max-community-nodes", type=int, default=5000)
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    data_path = Path(args.data).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root / 'src'}:{env.get('PYTHONPATH', '')}"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "1")
    env["OPENBLAS_NUM_THREADS"] = env.get("OPENBLAS_NUM_THREADS", "1")
    env["NUMEXPR_NUM_THREADS"] = env.get("NUMEXPR_NUM_THREADS", "1")

    trace_path = run_dir / "memory_trace.pt"
    if trace_path.exists():
        run_step(
            [
                args.python_bin,
                str(project_root / "scripts" / "visualization" / "plot_memory_evolution.py"),
                "--trace",
                str(trace_path),
                "--data",
                str(data_path),
                "--out-dir",
                str(run_dir / "memory_viz"),
            ],
            cwd=project_root,
            env=env,
        )
        run_step(
            [
                args.python_bin,
                str(project_root / "scripts" / "visualization" / "plot_community_vector_evolution.py"),
                "--trace",
                str(trace_path),
                "--out-dir",
                str(run_dir / "community_viz"),
            ],
            cwd=project_root,
            env=env,
        )

    if (run_dir / "best_model.pt").exists():
        run_step(
            [
                args.python_bin,
                str(project_root / "scripts" / "experiments" / "analyze_soft_community_assignments.py"),
                "--data",
                str(data_path),
                "--run-dir",
                str(run_dir),
                "--out-dir",
                str(run_dir / "community_viz"),
                "--max-nodes",
                str(int(args.max_community_nodes)),
            ],
            cwd=project_root,
            env=env,
        )

    print(run_dir)


if __name__ == "__main__":
    main()
