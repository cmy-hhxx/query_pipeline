"""Append-only, resumable business logs for final pipeline rows."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, BinaryIO, Literal, TextIO

from query_pipeline.io.jsonl import dumps_jsonl

logger = logging.getLogger(__name__)

_STREAMS = ("cleaned", "complex", "normal")


class BusinessLogWriter:
    def __init__(self, log_dir: str | Path, batch_id: str) -> None:
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.batch_id = batch_id
        self.paths = {
            stream: self.log_dir / "business" / stream / f"{batch_id}.log"
            for stream in _STREAMS
        }
        self._handles: dict[str, TextIO] = {}
        self._seen: dict[str, set[str]] = {stream: set() for stream in _STREAMS}
        self._lock = threading.Lock()

    def __enter__(self) -> BusinessLogWriter:
        try:
            for stream in _STREAMS:
                path = self.paths[stream]
                path.parent.mkdir(parents=True, exist_ok=True)
                self._seen[stream] = _prepare_existing(path)
                self._handles[stream] = path.open("a", encoding="utf-8")
        except Exception:
            self.close()
            raise
        return self

    def __exit__(
        self, exc_type: object, exc: object, traceback: object
    ) -> Literal[False]:
        self.close()
        return False

    def close(self) -> None:
        close_error: Exception | None = None
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception as exc:
                close_error = close_error or exc
        self._handles.clear()
        if close_error is not None:
            raise close_error

    def write(self, record: dict[str, Any]) -> None:
        line = dumps_jsonl(record)
        digest = _digest(line)
        difficulty = record.get("difficulty_level")
        streams = ["cleaned"]
        if difficulty == "hard":
            streams.append("complex")
        elif difficulty == "normal":
            streams.append("normal")
        else:
            logger.warning(
                "business log row has unknown difficulty_level=%r; writing cleaned only",
                difficulty,
            )

        with self._lock:
            for stream in streams:
                if digest in self._seen[stream]:
                    continue
                handle = self._handles[stream]
                handle.write(line + "\n")
                handle.flush()
                self._seen[stream].add(digest)

    def write_many(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            self.write(record)


def _digest(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def _prepare_existing(path: Path) -> set[str]:
    """Validate existing JSON lines and repair only an incomplete final line."""
    path.touch(exist_ok=True)
    with path.open("r+b") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        if size == 0:
            return set()
        handle.seek(-1, 2)
        complete = handle.read(1) == b"\n"
        if not complete:
            tail_start = _last_line_start(handle, size)
            handle.seek(tail_start)
            tail = handle.read(size - tail_start)
            try:
                decoded = json.loads(tail.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                handle.truncate(tail_start)
                handle.flush()
                logger.warning("repaired incomplete trailing business log line: %s", path)
            else:
                if not isinstance(decoded, dict):
                    raise ValueError(f"business log trailing line is not an object in {path}")
                handle.seek(0, 2)
                handle.write(b"\n")
                handle.flush()

    seen: set[str] = set()
    with path.open("rb") as handle:
        for index, raw_line in enumerate(handle, start=1):
            if not raw_line.endswith(b"\n"):
                raise ValueError(f"incomplete business log line {index} in {path}")
            raw_line = raw_line[:-1]
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid business log line {index} in {path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"business log line {index} is not an object in {path}")
            seen.add(_digest(dumps_jsonl(record)))
    return seen


def _last_line_start(handle: BinaryIO, size: int) -> int:
    position = size
    chunk_size = 64 * 1024
    while position > 0:
        start = max(0, position - chunk_size)
        handle.seek(start)
        chunk = handle.read(position - start)
        newline = chunk.rfind(b"\n")
        if newline >= 0:
            return start + newline + 1
        position = start
    return 0
