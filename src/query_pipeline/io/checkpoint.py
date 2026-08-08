from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from query_pipeline.config.models import PipelineConfig
from query_pipeline.prompts import resolve_prompt

logger = logging.getLogger(__name__)


def _hash_material(material: dict[str, Any]) -> str:
    blob = json.dumps(material, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _src_hash() -> str:
    """Hash of all pipeline source code; behavior fixes must invalidate checkpoints.

    Hashes the whole src/query_pipeline tree (not a per-stage list) so the fingerprint
    can never silently forget the modules a stage depends on.
    """
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(path.read_bytes())
    return h.hexdigest()


def stage_fingerprint(cfg: PipelineConfig, stage: str) -> str:
    """Hash of knobs that can change this stage's outputs (config, prompts, source)."""
    llm = {
        "model": cfg.llm.model,
        "enabled": cfg.llm.enabled,
        "response_format": cfg.llm.response_format,
    }
    if stage == "discover":
        return _hash_material(
            {
                "input_format": cfg.input.format,
                "segmentation": cfg.segmentation.model_dump(mode="json"),
                "step1": cfg.step1.model_dump(mode="json"),
                "step2": cfg.step2.model_dump(mode="json"),
                "llm": llm,
                "src": _src_hash(),
                "prompts": {
                    "segment": resolve_prompt("segment"),
                    cfg.step2.prompt_id: resolve_prompt(cfg.step2.prompt_id),
                },
            }
        )
    if stage == "verify":
        prompts = {cfg.verify.prompt_id: resolve_prompt(cfg.verify.prompt_id)}
        if cfg.verify.max_rounds > 1:
            prompts["verify_recheck"] = resolve_prompt("verify_recheck")
        return _hash_material(
            {
                "verify": cfg.verify.model_dump(mode="json"),
                "llm": llm,
                "src": _src_hash(),
                "prompts": prompts,
            }
        )
    if stage == "translate":
        return _hash_material(
            {
                "translate": cfg.post.translate.model_dump(mode="json"),
                "llm": llm,
                "src": _src_hash(),
                "prompts": {"translate": resolve_prompt("translate")},
            }
        )
    raise ValueError(f"unknown checkpoint stage: {stage!r}")


def stage_meta(cfg: PipelineConfig, stage: str) -> dict[str, Any]:
    meta: dict[str, Any] = {"stage_hash": stage_fingerprint(cfg, stage)}
    if stage == "discover":
        stat = cfg.input.path.stat()
        meta.update(
            {"input_path": str(cfg.input.path), "input_size": stat.st_size, "input_mtime_ns": stat.st_mtime_ns}
        )
    return meta


def stage_checkpoint(cfg: PipelineConfig, stage: str) -> "Checkpoint":
    path = cfg.checkpoint.dir / f"{stage}.jsonl"
    if not cfg.checkpoint.enabled:
        return Checkpoint(path=path, enabled=False)
    return Checkpoint.load(path, expected_meta=stage_meta(cfg, stage))


def content_key(*parts: str) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return f"c:{digest}"


@dataclass
class Checkpoint:
    """Append-only JSONL checkpoint of completed units (content-addressed keys)."""

    path: Path
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    meta: dict[str, Any] | None = None
    enabled: bool = True
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @classmethod
    def disabled(cls) -> "Checkpoint":
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
            for line in handle:
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
