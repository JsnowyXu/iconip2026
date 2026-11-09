#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch


DATASET_ORDER = [
    "patent_cn",
    "patent_in",
    "patent_us",
    "thgl_github",
    "thgl_software",
    "amazon_movies",
    "aminer_paper",
    "dblp",
]

PROCESS_LOG_FILES = {
    "patent_cn": "patent_cn_process.md",
    "patent_in": "patent_in_process.md",
    "patent_us": "patent_us_process.md",
    "thgl_github": "thgl_github_process.md",
    "thgl_software": "thgl_software_process.md",
    "amazon_movies": "amazon_movies_process.md",
    "aminer_paper": "aminer_paper_process.md",
    "dblp": "dblp_process.md",
}


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    task_name: str
    task_type: str
    relation_target: str
    data_path: Path
    sthn_dataset_slot: str


def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_header(path: Path, title: str) -> None:
    if path.exists():
        return
    path.write_text(f"# {title}\n\n", encoding="utf-8")


def append_lines(path: Path, title: str, lines: Iterable[str]) -> None:
    ensure_header(path, title)
    block = [f"## {now_string()}"]
    block.extend(str(line) for line in lines)
    block.append("")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(block))


def append_master_log(project_root: Path, lines: Iterable[str]) -> None:
    append_lines(project_root / "edge_benchmark_process.md", "Edge Benchmark Process Log", lines)


def append_dataset_log(project_root: Path, dataset: str, lines: Iterable[str]) -> None:
    file_name = PROCESS_LOG_FILES.get(dataset, f"{dataset}_process.md")
    append_lines(project_root / file_name, f"{dataset} Process Log", lines)


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


def run_sync(cmd: List[str], cwd: Path, env: Dict[str, str]) -> None:
    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if int(result.returncode) != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


def maybe_load_target_rel(task_path: Path) -> Optional[int]:
    if not task_path.exists():
        return None
    try:
        obj = torch.load(task_path, map_location="cpu")
    except Exception:
        return None
    meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
    extra = meta.get("extra", {}) if isinstance(meta, dict) else {}
    ctx = extra.get("context_protocol", {}) if isinstance(extra, dict) else {}
    try:
        return int(ctx.get("target_original_rel_id"))
    except Exception:
        return None


def relation_target_name(default_name: str, task_path: Path) -> str:
    rel_id = maybe_load_target_rel(task_path)
    if rel_id is None:
        return default_name
    if default_name == "dominant_relation":
        return f"rel_{rel_id}"
    return default_name


def ensure_thgl_tasks(project_root: Path, repo_root: Path, python_bin: str, env: Dict[str, str]) -> None:
    processed_root = resolve_processed_root(project_root)
    task_root = processed_root / "tgb2_tasks"
    task_root.mkdir(parents=True, exist_ok=True)
    import_script = project_root / "scripts" / "datasets" / "import_tgb2_thg.py"
    task_script = project_root / "scripts" / "datasets" / "build_paper_task_views.py"

    specs = {
        "thgl_github": {
            "dataset_name": "thgl_github",
            "edgelist": repo_root / "datasets" / "tgb2" / "thgl-github" / "thgl-github_edgelist.csv",
            "nodetype": repo_root / "datasets" / "tgb2" / "thgl-github" / "thgl-github_nodetype.csv",
            "unified": processed_root / "thgl_github_unified.pt",
        },
        "thgl_software": {
            "dataset_name": "thgl_software",
            "edgelist": repo_root / "datasets" / "tgb2" / "thgl-software" / "thgl-software_edgelist.csv",
            "nodetype": repo_root / "datasets" / "tgb2" / "thgl-software" / "thgl-software_nodetype.csv",
            "unified": processed_root / "thgl_software_unified.pt",
        },
    }

    for dataset, spec in specs.items():
        trans_pt = task_root / f"{dataset}_edge_transductive.pt"
        ind_pt = task_root / f"{dataset}_edge_inductive.pt"
        if not spec["unified"].exists():
            cmd = [
                python_bin,
                str(import_script),
                "--dataset-name",
                str(spec["dataset_name"]),
                "--edgelist-csv",
                str(spec["edgelist"]),
                "--nodetype-csv",
                str(spec["nodetype"]),
                "--out-pt",
                str(spec["unified"]),
                "--max-edges",
                "-1",
            ]
            run_sync(cmd, cwd=project_root, env=env)
            append_dataset_log(
                project_root,
                dataset,
                [
                    "- built THGL unified dataset",
                    f"- command: `{ ' '.join(cmd) }`",
                    f"- output: `{spec['unified']}`",
                ],
            )
        if not (trans_pt.exists() and ind_pt.exists()):
            cmd = [
                python_bin,
                str(task_script),
                "--input",
                str(spec["unified"]),
                "--out-dir",
                str(task_root),
                "--name-prefix",
                dataset,
            ]
            run_sync(cmd, cwd=project_root, env=env)
            append_dataset_log(
                project_root,
                dataset,
                [
                    "- built THGL task views",
                    f"- command: `{ ' '.join(cmd) }`",
                    f"- outputs: `{trans_pt}`, `{ind_pt}`",
                ],
            )


def build_edge_task_specs(project_root: Path) -> List[TaskSpec]:
    processed_root = resolve_processed_root(project_root)
    paper_root = processed_root / "paper_tasks"
    patent_root = processed_root / "tasks"
    thgl_root = processed_root / "tgb2_tasks"

    raw_specs = [
        TaskSpec(
            dataset="amazon_movies",
            task_name="amazon_movies_edge_inductive",
            task_type="edge_inductive",
            relation_target="dominant_relation",
            data_path=paper_root / "amazon_movies_edge_inductive.pt",
            sthn_dataset_slot="thgl-github",
        ),
        TaskSpec(
            dataset="amazon_movies",
            task_name="amazon_movies_edge_transductive",
            task_type="edge_transductive",
            relation_target="dominant_relation",
            data_path=paper_root / "amazon_movies_edge_transductive.pt",
            sthn_dataset_slot="thgl-github",
        ),
        TaskSpec(
            dataset="aminer_paper",
            task_name="aminer_paper_citation_edge_inductive",
            task_type="edge_inductive",
            relation_target="paper_cite_paper",
            data_path=paper_root / "aminer_paper_citation_edge_inductive.pt",
            sthn_dataset_slot="thgl-github",
        ),
        TaskSpec(
            dataset="aminer_paper",
            task_name="aminer_paper_citation_edge_transductive",
            task_type="edge_transductive",
            relation_target="paper_cite_paper",
            data_path=paper_root / "aminer_paper_citation_edge_transductive.pt",
            sthn_dataset_slot="thgl-github",
        ),
        TaskSpec(
            dataset="dblp",
            task_name="dblp_edge_inductive",
            task_type="edge_inductive",
            relation_target="dominant_relation",
            data_path=paper_root / "dblp_edge_inductive.pt",
            sthn_dataset_slot="thgl-github",
        ),
        TaskSpec(
            dataset="dblp",
            task_name="dblp_edge_transductive",
            task_type="edge_transductive",
            relation_target="dominant_relation",
            data_path=paper_root / "dblp_edge_transductive.pt",
            sthn_dataset_slot="thgl-github",
        ),
        TaskSpec(
            dataset="patent_cn",
            task_name="patent_cn_edge_org_codeIII_inductive",
            task_type="edge_inductive",
            relation_target="org_codeIII",
            data_path=patent_root / "patent_cn_edge_org_codeIII_inductive.pt",
            sthn_dataset_slot="thgl-github",
        ),
        TaskSpec(
            dataset="patent_cn",
            task_name="patent_cn_edge_org_codeIII_transductive",
            task_type="edge_transductive",
            relation_target="org_codeIII",
            data_path=patent_root / "patent_cn_edge_org_codeIII_transductive.pt",
            sthn_dataset_slot="thgl-github",
        ),
        TaskSpec(
            dataset="patent_in",
            task_name="patent_in_edge_org_codeIII_inductive",
            task_type="edge_inductive",
            relation_target="org_codeIII",
            data_path=patent_root / "patent_in_edge_org_codeIII_inductive.pt",
            sthn_dataset_slot="thgl-github",
        ),
        TaskSpec(
            dataset="patent_in",
            task_name="patent_in_edge_org_codeIII_transductive",
            task_type="edge_transductive",
            relation_target="org_codeIII",
            data_path=patent_root / "patent_in_edge_org_codeIII_transductive.pt",
            sthn_dataset_slot="thgl-github",
        ),
        TaskSpec(
            dataset="patent_us",
            task_name="patent_us_edge_org_codeIII_inductive",
            task_type="edge_inductive",
            relation_target="org_codeIII",
            data_path=patent_root / "patent_us_edge_org_codeIII_inductive.pt",
            sthn_dataset_slot="thgl-github",
        ),
        TaskSpec(
            dataset="patent_us",
            task_name="patent_us_edge_org_codeIII_transductive",
            task_type="edge_transductive",
            relation_target="org_codeIII",
            data_path=patent_root / "patent_us_edge_org_codeIII_transductive.pt",
            sthn_dataset_slot="thgl-github",
        ),
        TaskSpec(
            dataset="thgl_github",
            task_name="thgl_github_edge_inductive",
            task_type="edge_inductive",
            relation_target="dominant_relation",
            data_path=thgl_root / "thgl_github_edge_inductive.pt",
            sthn_dataset_slot="thgl-github",
        ),
        TaskSpec(
            dataset="thgl_github",
            task_name="thgl_github_edge_transductive",
            task_type="edge_transductive",
            relation_target="dominant_relation",
            data_path=thgl_root / "thgl_github_edge_transductive.pt",
            sthn_dataset_slot="thgl-github",
        ),
        TaskSpec(
            dataset="thgl_software",
            task_name="thgl_software_edge_inductive",
            task_type="edge_inductive",
            relation_target="dominant_relation",
            data_path=thgl_root / "thgl_software_edge_inductive.pt",
            sthn_dataset_slot="thgl-software",
        ),
        TaskSpec(
            dataset="thgl_software",
            task_name="thgl_software_edge_transductive",
            task_type="edge_transductive",
            relation_target="dominant_relation",
            data_path=thgl_root / "thgl_software_edge_transductive.pt",
            sthn_dataset_slot="thgl-software",
        ),
    ]
    out: List[TaskSpec] = []
    for spec in raw_specs:
        if spec.data_path.exists():
            out.append(
                TaskSpec(
                    dataset=spec.dataset,
                    task_name=spec.task_name,
                    task_type=spec.task_type,
                    relation_target=relation_target_name(spec.relation_target, spec.data_path),
                    data_path=spec.data_path,
                    sthn_dataset_slot=spec.sthn_dataset_slot,
                )
            )
    return out


def dump_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
