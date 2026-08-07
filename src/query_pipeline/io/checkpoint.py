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


def _hash_material(material: dict[str, Any]) -> str:
    blob = json.dumps(material, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def stage_fingerprint(cfg: PipelineConfig, stage: str) -> str:
    """Hash of only the knobs that can change this stage's outputs.

    Unrelated stages (and pure orchestration knobs like concurrency) are
    excluded so e.g. editing the translate prompt does not wipe the session
    checkpoint.
    """
    llm = {
        "model": cfg.llm_stage.model,
        "enabled": cfg.llm_stage.enabled,
        "response_format": cfg.llm_stage.response_format,
    }
    if stage == "session":
        return _hash_material(
            {
                "input_format": cfg.input.format,
                "session_stage": cfg.session_stage.model_dump(mode="json"),
                "llm": llm,
                "prompts": {
                    "segment": resolve_prompt("segment"),
                    cfg.session_stage.step2.prompt_id: resolve_prompt(cfg.session_stage.step2.prompt_id),
                },
            }
        )
    if stage == "verify":
        return _hash_material(
            {
                "verify_stage": cfg.verify_stage.model_dump(mode="json"),
                "llm": llm,
                "prompts": {cfg.verify_stage.prompt_id: resolve_prompt(cfg.verify_stage.prompt_id)},
            }
        )
    if stage == "translate":
        return _hash_material(
            {
                "translate": cfg.post_stage.translate.model_dump(mode="json"),
                "llm": llm,
                "prompts": {"translate": resolve_prompt("translate")},
            }
        )
    raise ValueError(f"unknown checkpoint stage: {stage!r}")


def stage_meta(cfg: PipelineConfig, stage: str) -> dict[str, Any]:
    """Meta that ties a stage's checkpoint to the run that produced it."""
    meta: dict[str, Any] = {"stage_hash": stage_fingerprint(cfg, stage)}
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
                    skipped += 1
                    continue
                self.records[key] = row
        if skipped:
            logger.warning("checkpoint: dropped %d unparseable line(s) in %s", skipped, self.path)

    def get(self, key: str) -> dict[str, Any] | None:
        return self.records.get(key)

    async def mark(self, key: str, **record: Any) -> None:
        if not self.enabled:
            return
        row = {"key": key, **record}
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with self._lock:
            self.records[key] = row
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
