from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from query_pipeline.config.models import PipelineConfig
from query_pipeline.llm import cache as llm_cache
from query_pipeline.prompts import resolve_prompt

logger = logging.getLogger(__name__)


def _hash_material(material: dict[str, Any]) -> str:
    blob = json.dumps(material, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def stage_fingerprint(cfg: PipelineConfig, stage: str) -> str:
    """Hash of knobs that can change this stage's outputs (config, prompts, source)."""
    llm = {
        "model": cfg.llm.model,
        "enabled": cfg.llm.enabled,
        "response_format": cfg.llm.response_format,
    }
    if stage == "judge":
        return _hash_material(
            {
                "input_format": cfg.input.format,
                "segmentation": cfg.segmentation.model_dump(mode="json"),
                "rule_gate": cfg.rule_gate.model_dump(mode="json"),
                "judge": cfg.judge.model_dump(mode="json"),
                "llm": llm,
                "src": llm_cache.src_hash(),
                "prompts": {
                    "segment": resolve_prompt("segment"),
                    cfg.judge.value_prompt: resolve_prompt(cfg.judge.value_prompt),
                    cfg.judge.complexity_prompt: resolve_prompt(cfg.judge.complexity_prompt),
                    cfg.judge.classify_complex_prompt: resolve_prompt(cfg.judge.classify_complex_prompt),
                    cfg.judge.classify_normal_prompt: resolve_prompt(cfg.judge.classify_normal_prompt),
                },
            }
        )
    if stage == "verify":
        prompts = {
            cfg.verify.prompt_id: resolve_prompt(cfg.verify.prompt_id),
            "template_family": resolve_prompt("template_family"),
            "dedup_pair": resolve_prompt("dedup_pair"),
        }
        if cfg.verify.max_rounds_hard > 1:
            prompts["verify_recheck"] = resolve_prompt("verify_recheck")
        return _hash_material(
            {
                "verify": cfg.verify.model_dump(mode="json"),
                "dedup": cfg.post.dedup.model_dump(mode="json"),
                "llm": llm,
                "src": llm_cache.src_hash(),
                "prompts": prompts,
            }
        )
    if stage == "translate":
        return _hash_material(
            {
                "translate": cfg.post.translate.model_dump(mode="json"),
                "llm": llm,
                "src": llm_cache.src_hash(),
                "prompts": {"translate": resolve_prompt("translate")},
            }
        )
    raise ValueError(f"unknown checkpoint stage: {stage!r}")


def stage_meta(cfg: PipelineConfig, stage: str) -> dict[str, Any]:
    meta: dict[str, Any] = {"stage_hash": stage_fingerprint(cfg, stage)}
    # 输入文件变化必须让所有"从输入行推导内容"的阶段 checkpoint 失效
    # （judge 行重建 → verify 前文/难度可能变 → translate 文本可能变）。
    if stage in {"judge", "verify", "translate"}:
        stat = cfg.input.path.stat()
        meta.update(
            {"input_path": str(cfg.input.path), "input_size": stat.st_size, "input_mtime_ns": stat.st_mtime_ns}
        )
    return meta


def stage_checkpoint(cfg: PipelineConfig, stage: str) -> "Checkpoint":
    path = cfg.checkpoint_dir / f"{stage}.jsonl"
    if not cfg.checkpoint.enabled:
        return Checkpoint(path=path, enabled=False)
    cp = Checkpoint.load(path, expected_meta=stage_meta(cfg, stage))
    if stage == "judge":
        _migrate_judge_checkpoint(cp)
    return cp


def _migrate_judge_checkpoint(cp: "Checkpoint") -> None:
    """旧格式 judge checkpoint 迁移：剥离 rows/judged 大字段（MB 级 chain）。

    第三轮修复前 judge checkpoint 每会话存 rows/judged + stats（500 会话实测
    91MB，且每次 run 全量解析进内存）；修复后只存 stats，rows/judged 由
    llm_cache 确定性重建。存量旧文件不会自愈——每次 run 仍全量解析大对象，
    这里在加载时检测并一次性剥离重写为 stats-only。
    """
    if not cp.enabled or not cp.path.exists():
        return
    heavy = [k for k, rec in cp.records.items() if "rows" in rec or "judged" in rec]
    if not heavy:
        return
    logger.warning(
        "judge checkpoint %s: %d 条旧格式记录（含 rows/judged）已迁移为 stats-only",
        cp.path, len(heavy),
    )
    for key in heavy:
        cp.records[key] = {k: v for k, v in cp.records[key].items() if k not in ("rows", "judged")}
    # 原子重写：meta 行 + 全部记录（stats-only）。tmp 名带 pid+随机后缀：
    # 两个进程并发迁移时不共用同一 tmp（与 llm_cache rewrite 同一竞态防护）。
    tmp = cp.path.with_name(f"{cp.path.name}.migrate.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            if cp.meta is not None:
                handle.write(
                    json.dumps({"type": "meta", **cp.meta}, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
            for record in cp.records.values():
                handle.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        os.replace(tmp, cp.path)
    finally:
        if tmp.exists():
            tmp.unlink()


def content_key(*parts: str) -> str:
    """Content-addressed checkpoint key.

    Parts are length-prefixed before hashing: a ``"\n"`` (or any byte) inside one
    part must never collide with a different boundary split (e.g. parts
    ``("a\nb", "c")`` vs ``("a", "b\nc")`` previously hashed identically and
    could silently reuse a wrong verdict/checkpoint entry across rows).
    """
    material = "".join(f"{len(part)}:{part}" for part in parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
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
