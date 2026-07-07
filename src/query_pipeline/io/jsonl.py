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
