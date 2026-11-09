#!/usr/bin/env python3
"""Lightweight raw-data statistics for the Patent-US dataset.

This script intentionally reads only the raw Patent-US CSV files and does not
depend on any processed .pt artifacts.
"""

import argparse
import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set


DEFAULT_DATASET_DIR = Path("/home/xuxj/2026/datasets/patent_us")
MAIN_CSV = "all-us.csv"
IPC_CSV = "IPCSubClass_US.csv"
CLS_CSV = "clsIII-reInd.csv"


def choose_first(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def split_multi_value(value: object) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            out: List[str] = []
            for item in parsed:
                out.extend(split_multi_value(item))
            return out

    text = text.replace("|", ";")
    parts = [part.strip().strip("'").strip('"') for part in text.split(";")]
    return [part for part in parts if part and part.lower() not in {"nan", "none", "null"}]


def first_year(value: object) -> Optional[int]:
    if value is None:
        return None
    match = re.search(r"(19|20)\d{2}", str(value))
    if not match:
        return None
    year = int(match.group(0))
    if 1900 <= year <= 2100:
        return year
    return None


def read_valid_code_reids(path: Path) -> Set[str]:
    valid: Set[str] = set()
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return valid
        reid_col = choose_first(reader.fieldnames, ["codeIIIIdreId"])
        if not reid_col:
            return valid
        for row in reader:
            reid = str(row.get(reid_col, "")).strip()
            if reid:
                valid.add(reid)
    return valid


def summarize_file(path: Path) -> Dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "line_count": count_lines(path) if path.exists() else 0,
    }


def sorted_counter(counter: Counter, limit: int) -> List[Dict[str, object]]:
    return [{"value": key, "count": count} for key, count in counter.most_common(limit)]


def summarize_patent_us(dataset_dir: Path, top_k: int) -> Dict[str, object]:
    main_path = dataset_dir / MAIN_CSV
    cls_path = dataset_dir / CLS_CSV
    ipc_path = dataset_dir / IPC_CSV
    for required in [main_path, cls_path, ipc_path]:
        if not required.exists():
            raise FileNotFoundError(f"missing Patent-US raw file: {required}")

    valid_code_reids = read_valid_code_reids(cls_path)

    records = 0
    unique_patents: Set[str] = set()
    unique_orgs: Set[str] = set()
    unique_code_reids: Set[str] = set()
    valid_unique_code_reids: Set[str] = set()
    unique_applicant_names: Set[str] = set()
    field_nonempty = Counter()
    year_counts = Counter()
    row_stats = Counter()
    code_counts = Counter()
    org_counts = Counter()

    with main_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{main_path} has no CSV header")

        patent_col = choose_first(reader.fieldnames, ["patentId"])
        org_col = choose_first(reader.fieldnames, ["orgID"])
        code_col = choose_first(reader.fieldnames, ["codeIIIId"])
        date_col = choose_first(reader.fieldnames, ["publicationDate"])
        org_name_col = choose_first(reader.fieldnames, ["orgname", "applicantName"])
        title_col = choose_first(reader.fieldnames, ["title"])
        summary_col = choose_first(reader.fieldnames, ["summary"])
        required_cols = [("patentId", patent_col), ("orgID", org_col), ("codeIIIId", code_col)]
        missing = [name for name, col in required_cols if col is None]
        if missing:
            raise ValueError(f"{main_path} missing required columns: {', '.join(missing)}")

        for row in reader:
            records += 1
            patent_id = str(row.get(patent_col, "")).strip()
            org_ids = split_multi_value(row.get(org_col))
            code_ids = split_multi_value(row.get(code_col))
            valid_code_ids = [code for code in code_ids if code in valid_code_reids]

            if patent_id:
                unique_patents.add(patent_id)
                field_nonempty["patentId"] += 1
            if title_col and str(row.get(title_col, "")).strip():
                field_nonempty["title"] += 1
            if summary_col and str(row.get(summary_col, "")).strip():
                field_nonempty["summary"] += 1
            if date_col and str(row.get(date_col, "")).strip():
                field_nonempty["publicationDate"] += 1
            if org_col and org_ids:
                field_nonempty["orgID"] += 1
            if code_col and code_ids:
                field_nonempty["codeIIIId"] += 1

            for org_id in org_ids:
                unique_orgs.add(org_id)
                org_counts[org_id] += 1
            for code_id in code_ids:
                unique_code_reids.add(code_id)
                code_counts[code_id] += 1
            for code_id in valid_code_ids:
                valid_unique_code_reids.add(code_id)
            if org_name_col:
                for name in split_multi_value(row.get(org_name_col)):
                    unique_applicant_names.add(name)

            year = first_year(row.get(date_col)) if date_col else None
            if year is None:
                row_stats["rows_missing_or_unparsed_year"] += 1
            else:
                year_counts[year] += 1

            unique_org_ids = list(dict.fromkeys([org_id for org_id in org_ids if org_id]))
            unique_valid_codes = list(dict.fromkeys([code for code in valid_code_ids if code]))
            if not unique_org_ids or not unique_valid_codes:
                row_stats["rows_skipped_missing_org_or_code"] += 1
                continue
            row_stats["rows_processable_like_preprocessing"] += 1
            row_stats["patent_org_edges"] += len(unique_org_ids)
            row_stats["patent_codeIII_edges"] += len(unique_valid_codes)
            row_stats["org_codeIII_edges"] += len(unique_org_ids) * len(unique_valid_codes)
            row_stats["org_coorg_edges"] += max(0, len(unique_org_ids) * (len(unique_org_ids) - 1))

    years = sorted(year_counts)
    with main_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        header_columns = list(csv.DictReader(handle).fieldnames or [])

    relation_counts = {
        "patent_org": row_stats["patent_org_edges"],
        "patent_codeIII": row_stats["patent_codeIII_edges"],
        "org_codeIII": row_stats["org_codeIII_edges"],
        "org_coorg": row_stats["org_coorg_edges"],
    }
    return {
        "dataset": "patent_us",
        "dataset_dir": str(dataset_dir),
        "files": {
            MAIN_CSV: summarize_file(main_path),
            CLS_CSV: summarize_file(cls_path),
            IPC_CSV: summarize_file(ipc_path),
        },
        "header_columns": {
            "main_csv": header_columns,
        },
        "records": records,
        "line_count_main_csv": records + 1,
        "year_range": {"min": years[0] if years else None, "max": years[-1] if years else None},
        "year_counts": {str(year): year_counts[year] for year in years},
        "core_field_nonempty_counts": dict(field_nonempty),
        "entity_counts": {
            "patent": len(unique_patents),
            "org": len(unique_orgs),
            "applicant_name": len(unique_applicant_names),
            "codeIII_raw_reindexed": len(unique_code_reids),
            "codeIII_valid_in_clsIII_reInd": len(valid_unique_code_reids),
            "codeIII_reference_rows": len(valid_code_reids),
        },
        "relation_edge_counts_like_preprocessing": relation_counts,
        "rows": dict(row_stats),
        "top_codeIII_reindexed": sorted_counter(code_counts, top_k),
        "top_orgID": sorted_counter(org_counts, top_k),
    }


def print_text(summary: Dict[str, object]) -> None:
    print("Patent-US raw dataset statistics")
    print(f"dataset_dir: {summary['dataset_dir']}")
    print("files:")
    for name, info in summary["files"].items():
        print(f"  {name}: size_bytes={info['size_bytes']} line_count={info['line_count']} path={info['path']}")
    print(f"records: {summary['records']}")
    yr = summary["year_range"]
    print(f"year_range: {yr['min']}..{yr['max']}")
    print(f"entity_counts: {json.dumps(summary['entity_counts'], ensure_ascii=False, sort_keys=True)}")
    print(f"relation_edge_counts_like_preprocessing: {json.dumps(summary['relation_edge_counts_like_preprocessing'], ensure_ascii=False, sort_keys=True)}")
    print(f"rows: {json.dumps(summary['rows'], ensure_ascii=False, sort_keys=True)}")
    print(f"core_field_nonempty_counts: {json.dumps(summary['core_field_nonempty_counts'], ensure_ascii=False, sort_keys=True)}")
    print(f"top_codeIII_reindexed: {json.dumps(summary['top_codeIII_reindexed'], ensure_ascii=False)}")
    print(f"top_orgID: {json.dumps(summary['top_orgID'], ensure_ascii=False)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Lightweight raw statistics for Patent-US")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="Print the full summary as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_patent_us(args.dataset_dir.resolve(), max(0, args.top_k))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_text(summary)


if __name__ == "__main__":
    main()
