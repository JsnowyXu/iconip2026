#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from edge_benchmark_common import append_master_log, dump_json


def remove_path(path: Path, removed: list) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    removed.append(str(path))


def main() -> None:
    ap = argparse.ArgumentParser("Clean non-main baseline/intermediate outputs for the edge-only rerun")
    ap.add_argument("--project-root", type=Path, default=Path("/home/xuxj/2026/dyn_relation_memory"))
    args = ap.parse_args()

    project_root = args.project_root.resolve()
    results_root = project_root / "results"
    removed = []

    targets = [
        results_root / "baseline_formal_aminer_custom",
        results_root / "baseline_formal_dblp_amazon",
        results_root / "baseline_formal_all",
        results_root / "baseline_formal_all_gpu",
        results_root / "patent_baseline_formal",
        results_root / "experiment_snapshot",
        results_root / "patent_dhc_reindexed",
        results_root / "tgb2_thgl_github_full_rel",
        results_root / "tgb2_thgl_software_full_rel",
        results_root / "gpu_full_campaign",
        results_root / "edge_only_formal",
        project_root / "data" / "reports",
        project_root / "data" / "interim" / "baseline_adapters",
    ]

    for target in targets:
        remove_path(target, removed)

    out_root = results_root / "edge_only_formal"
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "project_root": str(project_root),
        "removed_paths": removed,
    }
    dump_json(out_root / "cleanup_manifest.json", manifest)
    append_master_log(
        project_root,
        [
            "- cleanup completed for non-main baseline/intermediate outputs",
            f"- removed path count: `{len(removed)}`",
            f"- manifest: `{out_root / 'cleanup_manifest.json'}`",
        ],
    )
    print(out_root / "cleanup_manifest.json")


if __name__ == "__main__":
    main()
