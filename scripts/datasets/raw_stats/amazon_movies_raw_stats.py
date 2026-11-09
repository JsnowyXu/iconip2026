#!/usr/bin/env python3
"""Compute raw-input statistics for the Amazon Movies dataset."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Optional


DEFAULT_ARCHIVE = Path("/home/xuxj/2026/datasets/Amazon_movies.zip")
EDGE_MEMBER = "edge_list.csv"
ENTITY_TEXT_MEMBER = "entity_text.csv"
RELATION_TEXT_MEMBER = "relation_text.csv"


def file_size(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    return {
        "path": str(path),
        "size_bytes": size,
        "size_mib": round(size / (1024 * 1024), 3),
    }


def count_member_lines(zf: zipfile.ZipFile, member: str) -> Optional[int]:
    if member not in zf.namelist():
        return None
    with zf.open(member) as f:
        return sum(1 for _ in f)


def as_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def update_min_max(current: tuple[Optional[int], Optional[int]], value: Optional[int]) -> tuple[Optional[int], Optional[int]]:
    if value is None:
        return current
    min_value, max_value = current
    if min_value is None or value < min_value:
        min_value = value
    if max_value is None or value > max_value:
        max_value = value
    return min_value, max_value


def compute_edge_stats(zf: zipfile.ZipFile, max_rows: Optional[int]) -> dict[str, Any]:
    users = set()
    items = set()
    raw_relations = set()
    labels = Counter()
    ts_range: tuple[Optional[int], Optional[int]] = (None, None)
    edge_idx_range: tuple[Optional[int], Optional[int]] = (None, None)
    row_count = 0
    bad_rows = 0

    with zf.open(EDGE_MEMBER) as f:
        reader = csv.DictReader((line.decode("utf-8", errors="replace") for line in f))
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            if max_rows is not None and row_count >= max_rows:
                break
            row_count += 1

            user = as_int(row.get("u", ""))
            item = as_int(row.get("i", ""))
            relation = as_int(row.get("r", ""))
            ts = as_int(row.get("ts", ""))
            label = row.get("label")
            edge_idx = as_int(row.get("idx", ""))

            if user is None or item is None or relation is None or ts is None:
                bad_rows += 1
                continue
            users.add(user)
            items.add(item)
            raw_relations.add(relation)
            labels[str(label)] += 1
            ts_range = update_min_max(ts_range, ts)
            edge_idx_range = update_min_max(edge_idx_range, edge_idx)

    return {
        "member": EDGE_MEMBER,
        "columns": fieldnames,
        "rows_scanned": row_count,
        "scan_limited": max_rows is not None,
        "bad_rows": bad_rows,
        "unique_users": len(users),
        "unique_items": len(items),
        "unique_nodes_user_item_union": len(users | items),
        "unique_raw_relation_ids": len(raw_relations),
        "rating_label_counts": dict(sorted(labels.items())),
        "timestamp_min": ts_range[0],
        "timestamp_max": ts_range[1],
        "edge_idx_min": edge_idx_range[0],
        "edge_idx_max": edge_idx_range[1],
    }


def compute_stats(archive: Path, max_rows: Optional[int], count_text_lines: bool) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as zf:
        members = [
            {
                "name": info.filename,
                "compressed_size": info.compress_size,
                "file_size": info.file_size,
            }
            for info in zf.infolist()
            if not info.filename.startswith("__MACOSX/")
        ]
        stats = {
            "dataset": "amazon_movies",
            "archive": file_size(archive),
            "zip_members": members,
            "edge_list": compute_edge_stats(zf, max_rows=max_rows),
        }
        if count_text_lines:
            stats["entity_text_rows_including_header"] = count_member_lines(zf, ENTITY_TEXT_MEMBER)
            stats["relation_text_rows_including_header"] = count_member_lines(zf, RELATION_TEXT_MEMBER)
        return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--max-rows", type=int, default=None, help="Optionally scan only the first N edge rows.")
    parser.add_argument(
        "--count-text-lines",
        action="store_true",
        help="Also count entity_text.csv and relation_text.csv rows; this scans the large text files.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    stats = compute_stats(args.archive, max_rows=args.max_rows, count_text_lines=args.count_text_lines)
    text = json.dumps(stats, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
