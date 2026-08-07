from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record.setdefault("_line_number", line_number)
            yield record


def read_jsonl_skipping(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read input records, skipping lines that fail to parse.

    The upstream exporter may write truncated/garbage lines (a strict
    read would crash the whole run); callers get (records, skipped_count)
    and should surface the skipped count in stats/logs. errors="replace"
    also tolerates a byte-torn trailing line (invalid UTF-8 at EOF) — it
    decodes to a replacement char and the line fails JSON parsing, so it
    lands in the skipped count instead of crashing the read.
    """
    records: list[dict[str, Any]] = []
    skipped = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            record.setdefault("_line_number", line_number)
            records.append(record)
    return records, skipped


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            out = {k: v for k, v in record.items() if not k.startswith("_")}
            handle.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {k: v for k, v in record.items() if not k.startswith("_")}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
