#!/usr/bin/env python3
"""Stream raw DBLP-Citation-network-V18 JSONL statistics without processed .pt files."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Iterator, Optional, TextIO


DEFAULT_INPUT = Path("/home/xuxj/2026/datasets/DBLP-Citation-network-V18.zip")
DEFAULT_ZIP_MEMBER = "DBLP-Citation-network-V18.jsonl"
KEY_FIELDS = [
    "id",
    "title",
    "abstract",
    "keywords",
    "year",
    "authors",
    "references",
    "venue",
    "doc_type",
    "doi",
    "n_citation",
    "lang",
]


def nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def open_raw_text(path: Path, zip_member: str) -> tuple[TextIO, Optional[zipfile.ZipFile]]:
    if path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path)
        member = zip_member
        if member not in zf.namelist():
            jsonl_members = [n for n in zf.namelist() if n.lower().endswith((".jsonl", ".json"))]
            if len(jsonl_members) != 1:
                zf.close()
                raise FileNotFoundError(
                    f"{member!r} not found in {path}; candidates={jsonl_members[:10]}"
                )
            member = jsonl_members[0]
        return zf.open(member, "r"), zf
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore"), None
    return path.open("r", encoding="utf-8", errors="ignore"), None


def iter_json_lines(handle: TextIO | BinaryIO) -> Iterator[Dict[str, Any]]:
    for line_no, line in enumerate(handle, start=1):
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="ignore")
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            yield {"__parse_error__": f"line={line_no} error={exc}"}
            continue
        if isinstance(obj, dict):
            yield obj
        else:
            yield {"__parse_error__": f"line={line_no} non_object={type(obj).__name__}"}


def pct(numer: int, denom: int) -> float:
    return round((100.0 * numer / denom), 4) if denom else 0.0


def top(counter: Counter, n: int) -> list[dict[str, Any]]:
    return [{"value": k, "count": int(v)} for k, v in counter.most_common(n)]


def capped_add(values: set[str], value: str, cap: int) -> bool:
    if cap <= 0:
        return True
    if len(values) >= cap and value not in values:
        return False
    values.add(value)
    return True


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(path)

    records = 0
    parse_errors = 0
    fields_present: Counter[str] = Counter()
    fields_nonempty: Counter[str] = Counter()
    all_fields: Counter[str] = Counter()
    year_counts: Counter[int] = Counter()
    doc_type_counts: Counter[str] = Counter()
    venue_counts: Counter[str] = Counter()
    lang_counts: Counter[str] = Counter()
    author_count_dist: Counter[int] = Counter()
    reference_count_dist: Counter[int] = Counter()

    author_entries = 0
    usable_author_edges = 0
    author_missing_id = 0
    citation_edges = 0
    citation_self_refs = 0
    reference_empty = 0
    records_with_authors = 0
    records_with_references = 0

    unique_paper_ids: set[str] = set()
    unique_author_ids: set[str] = set()
    unique_reference_ids: set[str] = set()
    paper_ids_capped = False
    author_ids_capped = False
    reference_ids_capped = False

    year_min = None
    year_max = None
    positive_year_min = None
    positive_year_max = None
    nonpositive_years = 0
    sample_records: list[dict[str, Any]] = []

    handle, closer = open_raw_text(path, args.zip_member)
    try:
        for obj in iter_json_lines(handle):
            if "__parse_error__" in obj:
                parse_errors += 1
                if args.max_parse_errors and parse_errors >= args.max_parse_errors:
                    break
                continue

            records += 1
            if args.max_records and records > args.max_records:
                records -= 1
                break

            for field in obj:
                all_fields[field] += 1
            for field in KEY_FIELDS:
                if field in obj:
                    fields_present[field] += 1
                if nonempty(obj.get(field)):
                    fields_nonempty[field] += 1

            pid = str(obj.get("id", "")).strip()
            if pid and args.unique_cap > 0 and not capped_add(unique_paper_ids, pid, args.unique_cap):
                paper_ids_capped = True

            year_raw = obj.get("year")
            try:
                year = int(year_raw)
            except (TypeError, ValueError):
                year = None
            if year is not None:
                year_counts[year] += 1
                year_min = year if year_min is None else min(year_min, year)
                year_max = year if year_max is None else max(year_max, year)
                if year > 0:
                    positive_year_min = year if positive_year_min is None else min(positive_year_min, year)
                    positive_year_max = year if positive_year_max is None else max(positive_year_max, year)
                else:
                    nonpositive_years += 1

            doc_type = str(obj.get("doc_type", "")).strip() or "<empty>"
            venue = str(obj.get("venue", "")).strip() or "<empty>"
            lang = str(obj.get("lang", "")).strip() or "<empty>"
            doc_type_counts[doc_type] += 1
            venue_counts[venue] += 1
            lang_counts[lang] += 1

            authors = obj.get("authors") or []
            if not isinstance(authors, list):
                authors = []
            refs = obj.get("references") or []
            if not isinstance(refs, list):
                refs = []
            author_count_dist[min(len(authors), args.dist_cap)] += 1
            reference_count_dist[min(len(refs), args.dist_cap)] += 1
            if authors:
                records_with_authors += 1
            if refs:
                records_with_references += 1

            for author in authors:
                author_entries += 1
                if isinstance(author, dict):
                    aid = str(author.get("id", "")).strip()
                else:
                    aid = str(author).strip()
                if aid:
                    usable_author_edges += 1
                    if args.unique_cap > 0 and not capped_add(unique_author_ids, aid, args.unique_cap):
                        author_ids_capped = True
                else:
                    author_missing_id += 1

            for ref in refs:
                rid = str(ref).strip()
                if rid:
                    citation_edges += 1
                    if pid and rid == pid:
                        citation_self_refs += 1
                    if args.unique_cap > 0 and not capped_add(unique_reference_ids, rid, args.unique_cap):
                        reference_ids_capped = True
                else:
                    reference_empty += 1

            if len(sample_records) < args.samples:
                sample_records.append(
                    {
                        "id": pid,
                        "year": year,
                        "title": str(obj.get("title", ""))[:120],
                        "authors": len(authors),
                        "references": len(refs),
                        "venue": venue,
                        "doc_type": doc_type,
                    }
                )
    finally:
        handle.close()
        if closer is not None:
            closer.close()

    return {
        "input": str(path),
        "input_size_bytes": path.stat().st_size,
        "zip_member": args.zip_member if path.suffix.lower() == ".zip" else None,
        "max_records": args.max_records,
        "records_scanned": records,
        "parse_errors": parse_errors,
        "year_range": {"min": year_min, "max": year_max},
        "positive_year_range": {"min": positive_year_min, "max": positive_year_max},
        "nonpositive_years": nonpositive_years,
        "field_coverage": {
            field: {
                "present": int(fields_present[field]),
                "present_pct": pct(fields_present[field], records),
                "nonempty": int(fields_nonempty[field]),
                "nonempty_pct": pct(fields_nonempty[field], records),
            }
            for field in KEY_FIELDS
        },
        "all_fields_top": top(all_fields, args.top_n),
        "edge_candidates": {
            "author_write_paper": {
                "candidate_edges_with_author_id": usable_author_edges,
                "raw_author_entries": author_entries,
                "author_entries_missing_id": author_missing_id,
                "records_with_authors": records_with_authors,
            },
            "paper_cite_paper": {
                "candidate_reference_edges": citation_edges,
                "empty_reference_entries": reference_empty,
                "self_references": citation_self_refs,
                "records_with_references": records_with_references,
            },
            "total_prepare_dataset_style_edges": usable_author_edges + citation_edges,
        },
        "unique_counts": {
            "paper_ids_seen": len(unique_paper_ids),
            "paper_ids_capped": paper_ids_capped,
            "author_ids_seen": len(unique_author_ids),
            "author_ids_capped": author_ids_capped,
            "reference_ids_seen": len(unique_reference_ids),
            "reference_ids_capped": reference_ids_capped,
            "cap": args.unique_cap,
        },
        "top_years": top(year_counts, args.top_n),
        "top_doc_types": top(doc_type_counts, args.top_n),
        "top_langs": top(lang_counts, args.top_n),
        "top_venues": top(venue_counts, args.top_n),
        "author_count_distribution_capped": dict(sorted(author_count_dist.items())),
        "reference_count_distribution_capped": dict(sorted(reference_count_dist.items())),
        "samples": sample_records,
    }


def print_text(summary: dict[str, Any]) -> None:
    print(f"input: {summary['input']}")
    if summary.get("zip_member"):
        print(f"zip_member: {summary['zip_member']}")
    print(f"records_scanned: {summary['records_scanned']}")
    print(f"parse_errors: {summary['parse_errors']}")
    yr = summary["year_range"]
    print(f"year_range: {yr['min']}..{yr['max']}")
    pyr = summary["positive_year_range"]
    print(f"positive_year_range: {pyr['min']}..{pyr['max']}")
    print(f"nonpositive_years: {summary['nonpositive_years']}")
    edges = summary["edge_candidates"]
    print("edge_candidates:")
    print(f"  author_write_paper: {edges['author_write_paper']['candidate_edges_with_author_id']}")
    print(f"  paper_cite_paper: {edges['paper_cite_paper']['candidate_reference_edges']}")
    print(f"  total_prepare_dataset_style_edges: {edges['total_prepare_dataset_style_edges']}")
    print("unique_counts:")
    for key, value in summary["unique_counts"].items():
        print(f"  {key}: {value}")
    print("field_coverage:")
    for field, stats in summary["field_coverage"].items():
        print(
            f"  {field}: present={stats['present']} ({stats['present_pct']}%), "
            f"nonempty={stats['nonempty']} ({stats['nonempty_pct']}%)"
        )
    print("top_doc_types:", summary["top_doc_types"])
    print("top_langs:", summary["top_langs"])
    print("top_years:", summary["top_years"])


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Raw DBLP zip/jsonl/jsonl.gz path.")
    parser.add_argument("--zip-member", default=DEFAULT_ZIP_MEMBER, help="JSONL member inside the zip.")
    parser.add_argument("--max-records", type=int, default=0, help="Stop after N valid records; 0 scans all.")
    parser.add_argument("--unique-cap", type=int, default=2_000_000, help="Cap per exact unique-id set.")
    parser.add_argument("--dist-cap", type=int, default=50, help="Cap author/reference count buckets at this value.")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-parse-errors", type=int, default=100)
    parser.add_argument("--json", action="store_true", help="Print JSON summary instead of text.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    summary = summarize(args)
    if args.json:
        json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        print_text(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
