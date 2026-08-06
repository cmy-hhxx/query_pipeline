from __future__ import annotations

import asyncio
from typing import Any

from query_pipeline.io.jsonl import write_jsonl
from query_pipeline.llm.cache import load_cache
from query_pipeline.llm.client import LLMClient
from query_pipeline.pipeline.context import PipelineContext
from query_pipeline.post.dedup import dedup_rows
from query_pipeline.post.translate import translate_rows


def run_post_stage(ctx: PipelineContext) -> PipelineContext:
    """Post-processing of the assembled complex-query rows: MinHash rule dedup
    first (fewer rows -> fewer LLM calls), then translation of input.text into
    meta.translation. Both modules are independently toggleable.
    """
    cfg = ctx.config
    if not cfg.post_stage.enabled:
        return ctx

    if cfg.post_stage.dedup.enabled:
        ctx.rows, dropped = dedup_rows(ctx.rows, cfg.post_stage.dedup)
        ctx.stats["dedup_removed"] = len(dropped)
        if dropped:
            ctx.work_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(ctx.path("deduped.jsonl"), dropped)

    if cfg.post_stage.translate.enabled and cfg.llm_stage.enabled and ctx.rows:
        client = LLMClient(cfg.llm_stage)
        cache = load_cache(cfg.llm_stage.cache)
        counts = asyncio.run(_translate_then_close(client, ctx, cache))
    else:
        counts = {"translated": 0, "translate_skipped": 0, "translate_failed": 0}
    ctx.stats.update(counts)
    return ctx


async def _translate_then_close(
    client: LLMClient, ctx: PipelineContext, cache: dict[str, dict[str, Any]]
) -> dict[str, int]:
    """Run translation and close the client in the SAME event loop.

    Closing an httpx connection pool in a different asyncio.run() than the one
    that opened it raises "Event loop is closed".
    """
    cfg = ctx.config
    try:
        return await translate_rows(
            ctx.rows,
            client=client,
            llm_cfg=cfg.llm_stage,
            translate_cfg=cfg.post_stage.translate,
            cache=cache,
            cache_path=cfg.llm_stage.cache,
        )
    finally:
        await client.close()
