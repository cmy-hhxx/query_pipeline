#!/usr/bin/env python3
"""Flatten a complex-queries output JSONL into a CSV-schema flat JSONL.

Reads the full pipeline output (``complex_queries<dataset>.jsonl``) and writes
one JSONL record per complex query whose top-level keys are the columns of the
legacy CSV export (see ``outputs/aime/complex_queries_0806.csv``)::

    trace_id, user_id, category, input.text, context, text_answer,
    request_time_ms, run_id, meta.translation

Nested values keep their JSON types — ``context`` stays an array; ``input.text``
and ``meta.translation`` are lifted to dotted top-level keys.

Usage::

    python scripts/export_flat.py outputs/aime/complex_queries_0806.jsonl \
        -o outputs/aime/complex_queries_0806_flat.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Top-level keys of the exported record, in CSV column order.
FIELDS = [
    "trace_id",
    "user_id",
    "category",
    "input.text",
    "context",
    "text_answer",
    "request_time_ms",
    "run_id",
    "meta.translation",
]


def flatten(row: dict) -> dict:
    """Project a pipeline row onto the CSV-schema fields."""
    return {
        "trace_id": row.get("trace_id", ""),
        "user_id": row.get("user_id", ""),
        "category": row.get("category", ""),
        "input.text": (row.get("input") or {}).get("text", ""),
        "context": row.get("context", []),
        "text_answer": row.get("text_answer", ""),
        "request_time_ms": row.get("request_time_ms"),
        "run_id": row.get("run_id"),
        "meta.translation": (row.get("meta") or {}).get("translation", ""),
    }


def export(input_path: Path, output_path: Path) -> int:
    count = 0
    with input_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            dst.write(json.dumps(flatten(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="full pipeline output JSONL (complex_queries*.jsonl)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output flat JSONL path")
    args = parser.parse_args()
    n = export(args.input, args.output)
    print(f"exported {n} rows -> {args.output}")


if __name__ == "__main__":
    main()
