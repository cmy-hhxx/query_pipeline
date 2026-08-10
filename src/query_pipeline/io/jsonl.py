from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# U+2028 Line Separator / U+2029 Paragraph Separator inside model answers or
# original data trip editors' "unusual line terminator" warnings and break
# naive line-based consumers. They are invisible separators — collapse to a
# space when writing (keeps JSONL strictly one line per record).
_UNUSUAL_LINE_TERMINATORS = str.maketrans({"\u2028": " ", "\u2029": " "})


def _dumps(record: dict[str, Any]) -> str:
    out = {k: v for k, v in record.items() if not k.startswith("_")}
    return json.dumps(out, ensure_ascii=False, separators=(",", ":")).translate(
        _UNUSUAL_LINE_TERMINATORS
    )



def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record.setdefault("_line_number", line_number)
            yield record


def read_jsonl_with_bad_lines(path: Path, bad_path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read records; unparseable lines go to bad_path as JSON objects (single
    format shared with adapt-failed records) and are skipped."""
    records: list[dict[str, Any]] = []
    skipped = 0
    bad_records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.rstrip("\n")
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                skipped += 1
                bad_records.append({"reason": "invalid_json", "line_number": line_number, "raw": raw})
                continue
            if not isinstance(record, dict):
                skipped += 1
                bad_records.append({"reason": "not_an_object", "line_number": line_number, "raw": raw})
                continue
            record.setdefault("_line_number", line_number)
            records.append(record)
    if bad_records:
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        with bad_path.open("w", encoding="utf-8") as handle:
            for record in bad_records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
    return records, skipped


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records atomically (tmp + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_dumps(record) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_dumps(record) + "\n")
