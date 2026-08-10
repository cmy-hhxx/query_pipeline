from __future__ import annotations

import asyncio
from typing import Any

from query_pipeline.io.checkpoint import stage_checkpoint
from query_pipeline.io.jsonl import write_jsonl
from query_pipeline.llm.client import LLMClient
from query_pipeline.pipeline.context import PipelineContext
from query_pipeline.post.dedup import dedup_rows
from query_pipeline.post.translate import translate_rows


async def run_post_stage(
    ctx: PipelineContext,
    client: LLMClient | None,
    cache: dict[str, dict[str, Any]],
    cache_lock: asyncio.Lock,
) -> PipelineContext:
    cfg = ctx.config
    ctx.prune_debug_artifacts("deduped.jsonl")
    if not cfg.post.enabled:
        return ctx

    if cfg.post.dedup.enabled:
        ctx.rows, dropped = dedup_rows(ctx.rows, cfg.post.dedup)
        ctx.stats["dedup_removed"] = len(dropped)
        if dropped and cfg.debug.dump_intermediates:
            ctx.work_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(ctx.path("deduped.jsonl"), dropped)

    if cfg.post.translate.enabled and client is not None and ctx.rows:
        checkpoint = stage_checkpoint(cfg, "translate")
        counts = await translate_rows(
            ctx.rows,
            client=client,
            llm_cfg=cfg.llm,
            cache=cache,
            cache_path=cfg.cache_path,
            checkpoint=checkpoint,
            cache_lock=cache_lock,
            on_complete=(
                ctx.business_writer.write
                if ctx.stream_business_rows and ctx.business_writer is not None
                else None
            ),
        )
    else:
        counts = {"translated": 0, "translate_skipped": 0, "translate_failed": 0}
        if ctx.stream_business_rows and ctx.business_writer is not None:
            ctx.business_writer.write_many(ctx.rows)
    ctx.stats.update(counts)
    return ctx
