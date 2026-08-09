"""Input format detection (file-level) and record pre-cleaning.

Format rules:
- ``thread_id`` + ``context`` (list)  -> session
- ``judge_data`` (object)             -> chat

A file is session or chat only if every recognizable line agrees; mixed or
unrecognizable files are an error (dirty data must not be silently handled).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SESSION = "session"
CHAT = "chat"

_SAMPLE_LINES = 5


def classify_record(record: dict[str, Any]) -> str | None:
    has_thread = isinstance(record.get("thread_id"), str) and bool(record.get("thread_id"))
    has_context = isinstance(record.get("context"), list)
    has_judge = isinstance(record.get("judge_data"), dict)
    if has_thread and has_context:
        return SESSION
    if has_judge:
        return CHAT
    if "thread_id" in record or "context" in record or "judge_data" in record:
        raise ValueError(
            "record has partial or malformed format markers "
            "(thread_id/context/judge_data), cannot classify"
        )
    return None


def sniff_format(path: Path, *, sample_lines: int = _SAMPLE_LINES) -> str:
    """Detect the input dialect from the first non-empty lines of the file."""
    seen: set[str] = set()
    scanned = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            scanned += 1
            if scanned > sample_lines:
                break
            try:
                import json

                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue  # bad lines are handled by the pre-clean stage
            if not isinstance(record, dict):
                continue
            fmt = classify_record(record)
            if fmt is not None:
                seen.add(fmt)
            if len(seen) > 1:
                raise ValueError(f"mixed input format: {sorted(seen)}")
    if len(seen) == 1:
        return next(iter(seen))
    raise ValueError("cannot detect input format: no recognizable lines (need thread_id+context or judge_data)")


def preclean_records(
    records: list[dict[str, Any]], fmt: str
) -> tuple[list[dict[str, Any]], int, int]:
    """Drop duplicate input rows (session by thread_id, chat by case_id/trace_id)
    and empty-context sessions. Returns (records, duplicates, empties)."""
    key_fn = (lambda r: str(r.get("thread_id") or "")) if fmt == SESSION else (
        lambda r: str(((r.get("judge_data") or {}).get("case_id")) or r.get("trace_id") or "")
    )
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    duplicates = 0
    for record in records:
        key = key_fn(record)
        if not key:
            kept.append(record)
            continue
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        kept.append(record)

    empties = 0
    if fmt == SESSION:
        filtered: list[dict[str, Any]] = []
        for record in kept:
            if not record.get("context"):
                empties += 1
                continue
            filtered.append(record)
        kept = filtered
    return kept, duplicates, empties
