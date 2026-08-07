from __future__ import annotations

import asyncio
import json

from query_pipeline.config.models import PipelineConfig
from query_pipeline.io.jsonl import write_jsonl
from query_pipeline.llm.cache import load_cache
from query_pipeline.llm.client import LLMClient
from query_pipeline.pipeline.context import PipelineContext, RunSummary, merge_stats
from query_pipeline.steps.discover_stage import run_discover_stage
from query_pipeline.steps.post_stage import run_post_stage
from query_pipeline.steps.verify_stage import run_verify_stage


def run_pipeline(config: PipelineConfig) -> RunSummary:
    return asyncio.run(run_pipeline_async(config))


async def run_pipeline_async(config: PipelineConfig) -> RunSummary:
    ctx = PipelineContext(config=config)
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)

    client: LLMClient | None = None
    cache: dict = {}
    cache_lock = asyncio.Lock()
    if config.llm.enabled:
        client = LLMClient(config.llm)
        cache = load_cache(config.llm.cache)

    try:
        ctx = await run_discover_stage(ctx, client, cache, cache_lock)
        ctx = await run_verify_stage(ctx, client, cache, cache_lock)
        ctx = await run_post_stage(ctx, client, cache, cache_lock)
    finally:
        if client is not None:
            await client.close()

    complex_path = ctx.output_dir / config.output.complex_queries
    summary_path = ctx.output_dir / config.output.summary
    write_jsonl(complex_path, ctx.rows)

    stats = merge_stats(ctx)
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return RunSummary(
        success=True,
        name=config.name,
        stats=stats,
        output_files={
            "complex_queries": str(complex_path),
            "summary": str(summary_path),
        },
    )
