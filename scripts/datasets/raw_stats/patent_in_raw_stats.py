#!/usr/bin/env python3
"""Stream raw Patent-IN CSV files and report lightweight dataset statistics."""

import argparse
import ast
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_DATASET_DIR = Path("/home/xuxj/2026/datasets/patent_in")


def split_multi_value(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            out.extend(split_multi_value(item))
        return out

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            return split_multi_value(ast.literal_eval(text))
        except Exception:
            pass

    text = text.replace("|", ";")
    parts = [p.strip().strip("'").strip('"') for p in text.split(";")]
    return [p for p in parts if p and p.lower() not in {"nan", "none", "null"}]


def choose_first(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def parse_date(value: str) -> Tuple[Optional[datetime], str]:
    raw = str(value or "").strip()
    if not raw:
        return None, ""

    candidates = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y%m%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
    ]
    iso_raw = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        return datetime.fromisoformat(iso_raw), raw
    except ValueError:
        pass
    for fmt in candidates:
        try:
            return datetime.strptime(raw, fmt), raw
        except ValueError:
            continue
    if raw.isdigit() and len(raw) == 4:
        try:
            return datetime(int(raw), 1, 1), raw
        except ValueError:
            pass
    return None, raw


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(v for v in values if v))


def load_code_meta(cls_path: Path) -> Tuple[Dict[str, Dict[str, str]], Counter]:
    meta: Dict[str, Dict[str, str]] = {}
    stats = Counter()
    with cls_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["rows"] += 1
            reid = str(row.get("codeIIIIdreId", "")).strip()
            code_id = str(row.get("codeIIIId", "")).strip()
            code = str(row.get("codeIII", "")).strip()
            if not reid:
                stats["missing_reindexed_id"] += 1
                continue
            meta[reid] = {"codeIIIId": code_id, "codeIIIIdreId": reid, "codeIII": code}
    stats["unique_reindexed_ids"] = len(meta)
    return meta, stats


def count_ipc_rows(ipc_path: Path) -> Counter:
    stats = Counter()
    with ipc_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        stats["columns"] = len(header or [])
        for row in reader:
            if any(str(cell).strip() for cell in row):
                stats["rows"] += 1
    return stats


def stream_patent_stats(dataset_dir: Path, top_n: int) -> Dict:
    all_csv = dataset_dir / "all-in.csv"
    cls_csv = dataset_dir / "clsIII-reInd.csv"
    ipc_csv = dataset_dir / "IPCSubClass_US.csv"
    for path in [all_csv, cls_csv, ipc_csv]:
        if not path.exists():
            raise FileNotFoundError(path)

    code_meta, cls_stats = load_code_meta(cls_csv)
    valid_code_reids = set(code_meta)
    ipc_stats = count_ipc_rows(ipc_csv)

    csv.field_size_limit(sys.maxsize)
    with all_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        org_name_col = choose_first(columns, ["orgname"])
        title_col = choose_first(columns, ["title"])
        summary_col = choose_first(columns, ["summary"])
        date_col = choose_first(columns, ["publicationDate", "date_time", "date_parsed"])

        patent_ids = set()
        org_ids_all = set()
        code_reids_all = set()
        code_reids_valid = set()
        processed_patent_ids = set()
        processed_org_ids = set()
        processed_code_reids = set()
        unique_org_code_pairs = set()
        unique_org_org_pairs = set()
        org_freq = Counter()
        code_freq = Counter()
        year_freq = Counter()
        stats = Counter()
        org_count_sum = 0
        code_count_sum = 0
        max_orgs_per_processed_row = 0
        max_codes_per_processed_row = 0
        min_dt: Optional[datetime] = None
        max_dt: Optional[datetime] = None

        for row in reader:
            stats["rows_total"] += 1
            patent_id = str(row.get("patentId", "")).strip()
            if patent_id:
                patent_ids.add(patent_id)
            else:
                stats["rows_missing_patent_id"] += 1

            raw_org_ids = split_multi_value(row.get("orgID"))
            raw_code_reids = split_multi_value(row.get("codeIIIId"))
            org_ids = unique_preserve_order(raw_org_ids)
            code_reids = unique_preserve_order(raw_code_reids)
            valid_row_code_reids = [cid for cid in code_reids if cid in valid_code_reids]

            org_ids_all.update(org_ids)
            code_reids_all.update(code_reids)
            code_reids_valid.update(valid_row_code_reids)
            stats["raw_org_values"] += len(raw_org_ids)
            stats["raw_codeIIIId_values"] += len(raw_code_reids)
            stats["dedup_org_values"] += len(org_ids)
            stats["dedup_codeIIIId_values"] += len(code_reids)
            stats["invalid_or_unmapped_codeIIIId_values"] += len(code_reids) - len(valid_row_code_reids)
            stats["rows_with_org"] += int(bool(org_ids))
            stats["rows_with_codeIIIId"] += int(bool(code_reids))
            stats["rows_with_valid_codeIIIId"] += int(bool(valid_row_code_reids))
            stats["rows_with_orgname"] += int(bool(split_multi_value(row.get(org_name_col))) if org_name_col else False)
            stats["rows_with_title"] += int(bool(str(row.get(title_col, "")).strip()) if title_col else False)
            stats["rows_with_summary"] += int(bool(str(row.get(summary_col, "")).strip()) if summary_col else False)

            if date_col:
                dt, raw_date = parse_date(row.get(date_col, ""))
                stats["rows_with_date"] += int(bool(raw_date))
                if dt is None and raw_date:
                    stats["rows_with_unparsed_date"] += 1
                elif dt is not None:
                    year_freq[str(dt.year)] += 1
                    min_dt = dt if min_dt is None or dt < min_dt else min_dt
                    max_dt = dt if max_dt is None or dt > max_dt else max_dt

            if not org_ids or not valid_row_code_reids:
                stats["rows_skipped_missing_org_or_code"] += 1
                continue

            stats["rows_processed"] += 1
            processed_patent_ids.add(patent_id)
            processed_org_ids.update(org_ids)
            processed_code_reids.update(valid_row_code_reids)
            org_count_sum += len(org_ids)
            code_count_sum += len(valid_row_code_reids)
            max_orgs_per_processed_row = max(max_orgs_per_processed_row, len(org_ids))
            max_codes_per_processed_row = max(max_codes_per_processed_row, len(valid_row_code_reids))

            stats["num_edges_patent_org"] += len(org_ids)
            stats["num_edges_patent_codeIII"] += len(valid_row_code_reids)
            stats["num_edges_org_codeIII"] += len(org_ids) * len(valid_row_code_reids)
            stats["num_edges_org_coorg"] += len(org_ids) * (len(org_ids) - 1)

            for oid in org_ids:
                org_freq[oid] += 1
            for cid in valid_row_code_reids:
                code_freq[cid] += 1
            for oid in org_ids:
                for cid in valid_row_code_reids:
                    unique_org_code_pairs.add((oid, cid))
            for i in range(len(org_ids)):
                for j in range(i + 1, len(org_ids)):
                    unique_org_org_pairs.add(tuple(sorted((org_ids[i], org_ids[j]))))

    processed_rows = stats["rows_processed"]
    result = {
        "dataset": "patent_in",
        "raw_data_dir": str(dataset_dir),
        "raw_files": {
            "patent_csv": str(all_csv),
            "cls_csv": str(cls_csv),
            "ipc_csv": str(ipc_csv),
        },
        "file_sizes_bytes": {
            "patent_csv": all_csv.stat().st_size,
            "cls_csv": cls_csv.stat().st_size,
            "ipc_csv": ipc_csv.stat().st_size,
        },
        "columns": columns,
        "selected_columns": {
            "org_name": org_name_col,
            "title": title_col,
            "summary": summary_col,
            "date": date_col,
        },
        "raw_counts": dict(stats),
        "node_like_counts": {
            "num_patent_ids_raw": len(patent_ids),
            "num_org_ids_raw": len(org_ids_all),
            "num_codeIII_reindexed_ids_raw": len(code_reids_all),
            "num_codeIII_reindexed_ids_valid": len(code_reids_valid),
            "num_patent_nodes_processed_rule": len(processed_patent_ids),
            "num_org_nodes_processed_rule": len(processed_org_ids),
            "num_codeIII_nodes_processed_rule": len(processed_code_reids),
        },
        "edge_like_counts_processed_rule": {
            "num_edges_patent_org": stats["num_edges_patent_org"],
            "num_edges_patent_codeIII": stats["num_edges_patent_codeIII"],
            "num_edges_org_codeIII": stats["num_edges_org_codeIII"],
            "num_edges_org_coorg": stats["num_edges_org_coorg"],
            "num_unique_org_code_pairs": len(unique_org_code_pairs),
            "num_unique_org_org_pairs": len(unique_org_org_pairs),
        },
        "per_processed_row": {
            "mean_orgs": org_count_sum / processed_rows if processed_rows else 0.0,
            "mean_valid_codeIII": code_count_sum / processed_rows if processed_rows else 0.0,
            "max_orgs": max_orgs_per_processed_row,
            "max_valid_codeIII": max_codes_per_processed_row,
        },
        "date_range": {
            "min": min_dt.strftime("%Y-%m-%d") if min_dt else "",
            "max": max_dt.strftime("%Y-%m-%d") if max_dt else "",
            "num_years": len(year_freq),
        },
        "reference_counts": {
            "clsIII_reInd": dict(cls_stats),
            "IPCSubClass_US": dict(ipc_stats),
        },
        "top_org_ids": org_freq.most_common(top_n),
        "top_codeIII_reindexed_ids": [
            {"codeIIIIdreId": cid, "count": count, **code_meta.get(cid, {})}
            for cid, count in code_freq.most_common(top_n)
        ],
        "year_counts_head": year_freq.most_common(top_n),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    stats = stream_patent_stats(args.dataset_dir, args.top_n)
    text = json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
