from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from query_pipeline.config.models import PipelineConfig
from query_pipeline.prompts import resolve_prompt

logger = logging.getLogger(__name__)

# Every prompt that influences stage outputs; a prompt edit must invalidate
# checkpoints the same way it invalidates the LLM cache.
_PROMPT_IDS = ("segment", "complex_judge", "verify_complex", "translate")


def pipeline_fingerprint(cfg: PipelineConfig) -> str:
    """Hash of everything that can change stage outputs (config + resolved
    prompts; the input file is deliberately excluded — the session checkpoint
    guards input identity via its own size/mtime meta, and verify/translate
    results are pure functions of (text, prompt, model) so their checkpoints
    may be safely shared across configs). Any change invalidates existing
    checkpoints."""
    material = json.dumps(
        {"config": cfg.model_dump(mode="json"), "prompts": {pid: resolve_prompt(pid) for pid in _PROMPT_IDS}},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def stage_meta(cfg: PipelineConfig, stage: str) -> dict[str, Any]:
    """Meta that ties a stage's checkpoint to the run that produced it."""
    meta: dict[str, Any] = {"pipeline_hash": pipeline_fingerprint(cfg)}
    if stage == "session":
        stat = cfg.input.path.stat()
        meta.update(
            {"input_path": str(cfg.input.path), "input_size": stat.st_size, "input_mtime_ns": stat.st_mtime_ns}
        )
    return meta


def stage_checkpoint(cfg: PipelineConfig, stage: str) -> "Checkpoint":
    """Load (or seed) the checkpoint for a stage; no-op when disabled."""
    path = cfg.checkpoint.dir / f"{stage}.jsonl"
    if not cfg.checkpoint.enabled:
        return Checkpoint(path=path, enabled=False)
    return Checkpoint.load(path, expected_meta=stage_meta(cfg, stage))


def content_key(*parts: str) -> str:
    """Content-addressed key for row-level checkpoints: two rows with the same
    content share a result, and rows whose content changed self-heal instead
    of reusing stale records."""
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return f"c:{digest}"


@dataclass
class Checkpoint:
    """Append-only JSONL checkpoint of completed units.

    One line per completed unit: {"key": ..., **record}. A leading "meta"
    line ties the file to the input/config it was produced from; when it no
    longer matches, the whole file is ignored and re-seeded. Torn trailing
    lines from a hard-killed run are dropped on load (the unit re-runs; its
    LLM calls are cache hits), matching llm/cache.py.
    """

    path: Path
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    meta: dict[str, Any] | None = None
    enabled: bool = True
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @classmethod
    def disabled(cls) -> "Checkpoint":
        """No-op checkpoint: nothing is ever read or written."""
        return cls(path=Path(), enabled=False)

    @classmethod
    def load(cls, path: Path, *, expected_meta: dict[str, Any] | None = None) -> "Checkpoint":
        cp = cls(path=path)
        if path.exists():
            cp._read()
            if expected_meta is not None and cp.meta != expected_meta:
                logger.warning("checkpoint %s does not match input/config, starting fresh", path)
                cp = cls(path=path)
                cp._seed(expected_meta)
        elif expected_meta is not None:
            cp._seed(expected_meta)
        return cp

    def _seed(self, meta: dict[str, Any]) -> None:
        self.meta = meta
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"type": "meta", **meta}, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
        )

    def _read(self) -> None:
        skipped = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if row.get("type") == "meta":
                    if self.meta is None:
                        self.meta = {k: v for k, v in row.items() if k != "type"}
                    continue
                key = row.get("key")
                if not isinstance(key, str):
                    raise ValueError(f"invalid checkpoint row {self.path}:{line_number}")
                self.records[key] = row
        if skipped:
            logger.warning("checkpoint: dropped %d unparseable line(s) in %s", skipped, self.path)

    def get(self, key: str) -> dict[str, Any] | None:
        return self.records.get(key)

    async def mark(self, key: str, **record: Any) -> None:
        if not self.enabled:
            return
        self.records[key] = {"key": key, **record}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(self.records[key], ensure_ascii=False, separators=(",", ":")) + "\n"
        async with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
