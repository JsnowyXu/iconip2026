#!/usr/bin/env python3
"""Stream raw TGB2/THGL software CSV files and report dataset statistics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional


DEFAULT_RAW_DIR = Path("/home/xuxj/2026/datasets/tgb2/thgl-software")
DEFAULT_EDGELIST = DEFAULT_RAW_DIR / "thgl-software_edgelist.csv"
DEFAULT_NODETYPE = DEFAULT_RAW_DIR / "thgl-software_nodetype.csv"
DEFAULT_EDGEMAPPING = DEFAULT_RAW_DIR / "thgl-software_edgemapping.csv"
DEFAULT_NODEMAPPING = DEFAULT_RAW_DIR / "thgl-software_nodemapping.csv"


def require_columns(path: Path, fieldnames: Optional[Iterable[str]], required: set[str]) -> None:
    available = set(fieldnames or [])
    missing = required - available
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def parse_int(value: str, column: str, path: Path, line_no: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_no}: invalid integer in {column!r}: {value!r}") from exc


def iso_utc(ts: Optional[int]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def load_mapping(path: Path, key_col: str, value_col: str) -> Dict[str, str]:
    if not path.exists():
        return {}
    mapping: Dict[str, str] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        require_columns(path, reader.fieldnames, {key_col, value_col})
        for row in reader:
            mapping[str(row[value_col])] = str(row[key_col])
    return mapping


def scan_node_types(path: Path) -> dict:
    node_ids = set()
    type_counts: Counter[str] = Counter()
    duplicate_node_rows = 0

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        require_columns(path, reader.fieldnames, {"node_id", "type"})
        for line_no, row in enumerate(reader, start=2):
            node_id = row["node_id"]
            node_type = row["type"]
            if node_id in node_ids:
                duplicate_node_rows += 1
            node_ids.add(node_id)
            type_counts[node_type] += 1
            parse_int(node_type, "type", path, line_no)

    return {
        "node_ids": node_ids,
        "num_nodetype_rows": sum(type_counts.values()),
        "num_unique_nodetype_nodes": len(node_ids),
        "duplicate_node_rows": duplicate_node_rows,
        "node_type_counts": dict(sorted(type_counts.items(), key=lambda kv: int(kv[0]))),
    }


def scan_edges(path: Path) -> dict:
    endpoints = set()
    relation_counts: Counter[str] = Counter()
    source_node_counts: Counter[str] = Counter()
    destination_node_counts: Counter[str] = Counter()
    min_ts: Optional[int] = None
    max_ts: Optional[int] = None
    edge_count = 0

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        require_columns(path, reader.fieldnames, {"timestamp", "head", "tail", "relation_type"})
        for line_no, row in enumerate(reader, start=2):
            timestamp = parse_int(row["timestamp"], "timestamp", path, line_no)
            relation = row["relation_type"]
            parse_int(relation, "relation_type", path, line_no)

            head = row["head"]
            tail = row["tail"]
            endpoints.add(head)
            endpoints.add(tail)
            relation_counts[relation] += 1
            source_node_counts[head] += 1
            destination_node_counts[tail] += 1
            edge_count += 1

            min_ts = timestamp if min_ts is None else min(min_ts, timestamp)
            max_ts = timestamp if max_ts is None else max(max_ts, timestamp)

    return {
        "endpoint_nodes": endpoints,
        "num_edges": edge_count,
        "num_endpoint_nodes": len(endpoints),
        "num_source_nodes": len(source_node_counts),
        "num_destination_nodes": len(destination_node_counts),
        "num_edge_types": len(relation_counts),
        "relation_type_counts": dict(sorted(relation_counts.items(), key=lambda kv: int(kv[0]))),
        "min_timestamp": min_ts,
        "max_timestamp": max_ts,
        "min_timestamp_utc": iso_utc(min_ts),
        "max_timestamp_utc": iso_utc(max_ts),
    }


def attach_names(counts: Dict[str, int], id_to_name: Dict[str, str]) -> list[dict]:
    rows = []
    for raw_id, count in counts.items():
        rows.append(
            {
                "id": int(raw_id),
                "name": id_to_name.get(raw_id),
                "count": int(count),
            }
        )
    return rows


def compute_stats(args: argparse.Namespace) -> dict:
    edgelist = Path(args.edgelist_csv).expanduser().resolve()
    nodetype = Path(args.nodetype_csv).expanduser().resolve()
    edgemapping = Path(args.edgemapping_csv).expanduser().resolve()
    nodemapping = Path(args.nodemapping_csv).expanduser().resolve()

    edge_stats = scan_edges(edgelist)
    node_stats = scan_node_types(nodetype)
    edge_id_to_name = load_mapping(edgemapping, key_col="edge_id", value_col="type")
    node_type_id_to_name = load_mapping(nodemapping, key_col="node_type_id", value_col="type")

    endpoint_nodes = edge_stats.pop("endpoint_nodes")
    nodetype_nodes = node_stats.pop("node_ids")
    relation_type_counts = edge_stats.pop("relation_type_counts")
    node_type_counts = node_stats.pop("node_type_counts")

    missing_nodetype_nodes = endpoint_nodes - nodetype_nodes
    extra_nodetype_nodes = nodetype_nodes - endpoint_nodes

    return {
        "dataset": "thgl_software",
        "source": "TGB2/THGL raw CSV",
        "raw_paths": {
            "edgelist_csv": str(edgelist),
            "nodetype_csv": str(nodetype),
            "edgemapping_csv": str(edgemapping),
            "nodemapping_csv": str(nodemapping),
        },
        **edge_stats,
        **node_stats,
        "num_endpoint_nodes_missing_nodetype": len(missing_nodetype_nodes),
        "num_nodetype_nodes_not_in_edges": len(extra_nodetype_nodes),
        "edge_types": attach_names(relation_type_counts, edge_id_to_name),
        "node_types": attach_names(node_type_counts, node_type_id_to_name),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edgelist-csv", default=str(DEFAULT_EDGELIST))
    ap.add_argument("--nodetype-csv", default=str(DEFAULT_NODETYPE))
    ap.add_argument("--edgemapping-csv", default=str(DEFAULT_EDGEMAPPING))
    ap.add_argument("--nodemapping-csv", default=str(DEFAULT_NODEMAPPING))
    ap.add_argument("--out-json", default="", help="Optional path to write the JSON summary.")
    args = ap.parse_args()

    stats = compute_stats(args)
    text = json.dumps(stats, ensure_ascii=False, indent=2)
    print(text)

    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
