#!/usr/bin/env python3
"""Compute raw-input statistics for the AMiner paper dataset."""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_ROOT = Path("/home/xuxj/2026/datasets/Aminer")
DEFAULT_PAPER_ARCHIVE = DEFAULT_ROOT / "AMiner-Paper.rar"
DEFAULT_AUTHOR_ARCHIVE = DEFAULT_ROOT / "AMiner-Author.zip"
DEFAULT_AUTHOR2PAPER_ARCHIVE = DEFAULT_ROOT / "AMiner-Author2Paper.zip"
DEFAULT_COAUTHOR_ARCHIVE = DEFAULT_ROOT / "AMiner-Coauthor.zip"


def file_size(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    return {
        "path": str(path),
        "size_bytes": size,
        "size_mib": round(size / (1024 * 1024), 3),
    }


def zip_members(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        return [
            {
                "name": info.filename,
                "compressed_size": info.compress_size,
                "file_size": info.file_size,
            }
            for info in zf.infolist()
            if not info.filename.startswith("__MACOSX/")
        ]


def update_min_max(
    current: tuple[Optional[int], Optional[int]], value: Optional[int]
) -> tuple[Optional[int], Optional[int]]:
    if value is None:
        return current
    min_value, max_value = current
    if min_value is None or value < min_value:
        min_value = value
    if max_value is None or value > max_value:
        max_value = value
    return min_value, max_value


def as_int(value: object) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def first_zip_text_lines(path: Path) -> Iterable[str]:
    with zipfile.ZipFile(path) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        if not names:
            return
        with zf.open(names[0]) as f:
            for raw_line in io.TextIOWrapper(f, encoding="utf-8", errors="ignore"):
                yield raw_line.rstrip("\r\n")


def iter_paper_records(path: Path) -> Iterable[dict[str, Any]]:
    if not shutil.which("unrar"):
        raise RuntimeError("unrar is required to stream AMiner-Paper.rar")

    proc = subprocess.Popen(
        ["unrar", "p", "-inul", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
        bufsize=1,
    )
    if proc.stdout is None:
        raise RuntimeError("failed to open unrar pipe for AMiner paper data")

    cur: Optional[dict[str, Any]] = None

    def emit_current() -> Optional[dict[str, Any]]:
        if cur is None:
            return None
        pid = str(cur.get("id", "")).strip()
        if not pid:
            return None
        return {
            "id": pid,
            "title": str(cur.get("title", "")).strip(),
            "authors": str(cur.get("authors", "")).strip(),
            "orgs": str(cur.get("orgs", "")).strip(),
            "year": str(cur.get("year", "")).strip(),
            "venue": str(cur.get("venue", "")).strip(),
            "abstract": str(cur.get("abstract", "")).strip(),
            "references": [str(x).strip() for x in cur.get("references", []) if str(x).strip()],
        }

    early_stop = False
    try:
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("#index"):
                item = emit_current()
                if item is not None:
                    yield item
                cur = {"id": line[len("#index") :].strip(), "references": []}
                continue
            if cur is None:
                continue
            if line.startswith("#*"):
                cur["title"] = line[2:].strip()
            elif line.startswith("#@"):
                cur["authors"] = line[2:].strip()
            elif line.startswith("#o"):
                cur["orgs"] = line[2:].strip()
            elif line.startswith("#t"):
                cur["year"] = line[2:].strip()
            elif line.startswith("#c"):
                cur["venue"] = line[2:].strip()
            elif line.startswith("#!"):
                old = str(cur.get("abstract", "")).strip()
                cur["abstract"] = (old + " " + line[2:].strip()).strip() if old else line[2:].strip()
            elif line.startswith("#%"):
                cur.setdefault("references", []).append(line[2:].strip())

        item = emit_current()
        if item is not None:
            yield item
    except GeneratorExit:
        early_stop = True
        raise
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        stderr = ""
        if proc.stderr is not None:
            stderr = proc.stderr.read()
            proc.stderr.close()
        ret = proc.wait()
        if ret != 0 and not early_stop:
            raise RuntimeError(f"unrar failed for {path}: {stderr.strip()}")


def iter_author_records(path: Path) -> Iterable[dict[str, str]]:
    cur: Optional[dict[str, str]] = None

    def emit_current() -> Optional[dict[str, str]]:
        if cur is None:
            return None
        aid = str(cur.get("id", "")).strip()
        if not aid:
            return None
        return {
            "id": aid,
            "name": str(cur.get("name", "")).strip(),
            "affiliation": str(cur.get("affiliation", "")).strip(),
            "paper_count": str(cur.get("paper_count", "")).strip(),
            "citation_count": str(cur.get("citation_count", "")).strip(),
            "h_index": str(cur.get("h_index", "")).strip(),
            "pi": str(cur.get("pi", "")).strip(),
            "upi": str(cur.get("upi", "")).strip(),
            "topics": str(cur.get("topics", "")).strip(),
        }

    for line in first_zip_text_lines(path):
        if not line:
            continue
        if line.startswith("#index"):
            item = emit_current()
            if item is not None:
                yield item
            cur = {"id": line[len("#index") :].strip()}
            continue
        if cur is None:
            continue
        if line.startswith("#n"):
            cur["name"] = line[2:].strip()
        elif line.startswith("#a"):
            cur["affiliation"] = line[2:].strip()
        elif line.startswith("#pc"):
            cur["paper_count"] = line[3:].strip()
        elif line.startswith("#cn"):
            cur["citation_count"] = line[3:].strip()
        elif line.startswith("#hi"):
            cur["h_index"] = line[3:].strip()
        elif line.startswith("#pi"):
            cur["pi"] = line[3:].strip()
        elif line.startswith("#upi"):
            cur["upi"] = line[4:].strip()
        elif line.startswith("#t"):
            cur["topics"] = line[2:].strip()

    item = emit_current()
    if item is not None:
        yield item


def coverage(counts: Counter[str], total: int, fields: list[str]) -> dict[str, dict[str, Any]]:
    return {
        field: {
            "non_empty": int(counts.get(field, 0)),
            "coverage": round(float(counts.get(field, 0)) / total, 6) if total else 0.0,
        }
        for field in fields
    }


def compute_paper_stats(path: Path, max_records: Optional[int]) -> dict[str, Any]:
    fields = ["id", "title", "authors", "orgs", "year", "year_parseable", "venue", "abstract", "references"]
    counts: Counter[str] = Counter()
    years: Counter[int] = Counter()
    year_range: tuple[Optional[int], Optional[int]] = (None, None)
    total = 0
    total_reference_lines = 0
    unique_referenced_papers = set()
    malformed_years = 0

    for record in iter_paper_records(path):
        if max_records is not None and total >= max_records:
            break
        total += 1
        for field in fields:
            if field in {"year_parseable", "references"}:
                continue
            value = record.get(field)
            if value:
                counts[field] += 1
        refs = record.get("references") or []
        if refs:
            counts["references"] += 1
            total_reference_lines += len(refs)
            unique_referenced_papers.update(refs)
        year_match = re.search(r"\d{4}", str(record.get("year", "")))
        if year_match is None:
            if record.get("year"):
                malformed_years += 1
            continue
        year = int(year_match.group(0))
        counts["year_parseable"] += 1
        years[year] += 1
        year_range = update_min_max(year_range, year)

    return {
        "archive": file_size(path),
        "records_scanned": total,
        "scan_limited": max_records is not None,
        "field_coverage": coverage(counts, total, fields),
        "year_min": year_range[0],
        "year_max": year_range[1],
        "malformed_nonempty_years": malformed_years,
        "papers_with_references": int(counts.get("references", 0)),
        "reference_lines": total_reference_lines,
        "unique_referenced_paper_ids": len(unique_referenced_papers),
        "top_years_by_record_count": dict(years.most_common(20)),
    }


def compute_author_stats(path: Path, max_records: Optional[int]) -> dict[str, Any]:
    fields = ["id", "name", "affiliation", "paper_count", "citation_count", "h_index", "pi", "upi", "topics"]
    counts: Counter[str] = Counter()
    total = 0
    numeric_ranges: dict[str, tuple[Optional[int], Optional[int]]] = {
        "paper_count": (None, None),
        "citation_count": (None, None),
        "h_index": (None, None),
    }

    for record in iter_author_records(path):
        if max_records is not None and total >= max_records:
            break
        total += 1
        for field in fields:
            if record.get(field):
                counts[field] += 1
        for field in numeric_ranges:
            numeric_ranges[field] = update_min_max(numeric_ranges[field], as_int(record.get(field)))

    return {
        "archive": file_size(path),
        "zip_members": zip_members(path),
        "records_scanned": total,
        "scan_limited": max_records is not None,
        "field_coverage": coverage(counts, total, fields),
        "numeric_ranges": {
            field: {"min": value_range[0], "max": value_range[1]}
            for field, value_range in numeric_ranges.items()
        },
    }


def compute_author2paper_stats(path: Path, max_records: Optional[int]) -> dict[str, Any]:
    total = 0
    malformed = 0
    unique_authors = set()
    unique_papers = set()
    row_id_range: tuple[Optional[int], Optional[int]] = (None, None)
    order_range: tuple[Optional[int], Optional[int]] = (None, None)
    order_counts: Counter[int] = Counter()

    for line in first_zip_text_lines(path):
        if max_records is not None and total >= max_records:
            break
        if not line.strip():
            continue
        total += 1
        parts = re.split(r"\s+", line.strip())
        if len(parts) < 4:
            malformed += 1
            continue
        row_id = as_int(parts[0])
        author_id = as_int(parts[1])
        paper_id = as_int(parts[2])
        author_order = as_int(parts[3])
        if author_id is None or paper_id is None or author_order is None:
            malformed += 1
            continue
        row_id_range = update_min_max(row_id_range, row_id)
        order_range = update_min_max(order_range, author_order)
        order_counts[author_order] += 1
        unique_authors.add(author_id)
        unique_papers.add(paper_id)

    return {
        "archive": file_size(path),
        "zip_members": zip_members(path),
        "rows_scanned": total,
        "scan_limited": max_records is not None,
        "malformed_rows": malformed,
        "unique_author_ids": len(unique_authors),
        "unique_paper_ids": len(unique_papers),
        "row_id_min": row_id_range[0],
        "row_id_max": row_id_range[1],
        "author_order_min": order_range[0],
        "author_order_max": order_range[1],
        "top_author_orders": dict(order_counts.most_common(20)),
    }


def compute_coauthor_stats(path: Path, max_records: Optional[int]) -> dict[str, Any]:
    total = 0
    malformed = 0
    unique_authors = set()
    unique_pairs = set()
    weight_range: tuple[Optional[int], Optional[int]] = (None, None)
    weight_counts: Counter[int] = Counter()
    self_loops = 0

    for line in first_zip_text_lines(path):
        if max_records is not None and total >= max_records:
            break
        if not line.strip():
            continue
        total += 1
        parts = re.split(r"\s+", line.strip().lstrip("#"))
        if len(parts) < 3:
            malformed += 1
            continue
        src = as_int(parts[0])
        dst = as_int(parts[1])
        weight = as_int(parts[2])
        if src is None or dst is None or weight is None:
            malformed += 1
            continue
        if src == dst:
            self_loops += 1
        unique_authors.update([src, dst])
        unique_pairs.add(tuple(sorted((src, dst))))
        weight_range = update_min_max(weight_range, weight)
        weight_counts[weight] += 1

    return {
        "archive": file_size(path),
        "zip_members": zip_members(path),
        "rows_scanned": total,
        "scan_limited": max_records is not None,
        "malformed_rows": malformed,
        "unique_author_ids": len(unique_authors),
        "unique_undirected_author_pairs": len(unique_pairs),
        "self_loop_rows": self_loops,
        "weight_min": weight_range[0],
        "weight_max": weight_range[1],
        "top_weights": dict(weight_counts.most_common(20)),
    }


def compute_stats(
    paper_archive: Path,
    author_archive: Path,
    author2paper_archive: Path,
    coauthor_archive: Path,
    max_records: Optional[int],
) -> dict[str, Any]:
    return {
        "dataset": "aminer_paper",
        "raw_inputs": {
            "paper": str(paper_archive),
            "author": str(author_archive),
            "author2paper": str(author2paper_archive),
            "coauthor": str(coauthor_archive),
        },
        "paper": compute_paper_stats(paper_archive, max_records=max_records),
        "author": compute_author_stats(author_archive, max_records=max_records),
        "author2paper": compute_author2paper_stats(author2paper_archive, max_records=max_records),
        "coauthor": compute_coauthor_stats(coauthor_archive, max_records=max_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-archive", type=Path, default=DEFAULT_PAPER_ARCHIVE)
    parser.add_argument("--author-archive", type=Path, default=DEFAULT_AUTHOR_ARCHIVE)
    parser.add_argument("--author2paper-archive", type=Path, default=DEFAULT_AUTHOR2PAPER_ARCHIVE)
    parser.add_argument("--coauthor-archive", type=Path, default=DEFAULT_COAUTHOR_ARCHIVE)
    parser.add_argument("--max-records", type=int, default=None, help="Optionally scan only the first N records/rows per file.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    stats = compute_stats(
        paper_archive=args.paper_archive,
        author_archive=args.author_archive,
        author2paper_archive=args.author2paper_archive,
        coauthor_archive=args.coauthor_archive,
        max_records=args.max_records,
    )
    text = json.dumps(stats, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
