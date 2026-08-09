from __future__ import annotations

import asyncio
import json
from typing import Any

from query_pipeline.config.models import PipelineConfig
from query_pipeline.io.jsonl import write_jsonl
from query_pipeline.llm.cache import load_cache
from query_pipeline.llm.client import LLMClient
from query_pipeline.pipeline.context import PipelineContext, RunSummary, merge_stats
from query_pipeline.pipeline.stages import get_stage, stage_names


def _run_success(stats: dict[str, Any]) -> bool:
    """A run fails when discover-level work errored or nothing was adapted.

    verify_failed / translate_failed are fail-open by design (kept, retried next run)
    and empty complex_rows on a clean run (e.g. llm.enabled=false) is legitimate —
    neither makes the run a failure.
    """
    if stats.get("session_errors", 0) > 0:
        return False
    if stats.get("llm_failed", 0) > 0:
        return False
    if stats.get("total_sessions", 0) == 0:
        return False
    if stats.get("input_bad_lines", 0) == stats.get("total_sessions", 0):
        return False
    return True


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
        for name in stage_names(ctx.config.stages):
            stage = get_stage(name)
            ctx = await stage(ctx, client, cache, cache_lock)
    finally:
        if client is not None:
            await client.close()

    complex_path = ctx.output_dir / config.output.complex_queries
    summary_path = ctx.output_dir / config.output.summary

    stats = merge_stats(ctx)
    success = _run_success(stats)
    # Write output unless the run failed AND produced nothing — avoid clobbering a
    # previous good output with an empty file. Partial rows are still inspectable;
    # the exit code and summary flag the failure.
    if success or ctx.rows:
        write_jsonl(complex_path, ctx.rows)
    summary_path.write_text(
        json.dumps({**stats, "success": success}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return RunSummary(
        success=success,
        name=config.name,
        stats=stats,
        output_files={
            "complex_queries": str(complex_path),
            "summary": str(summary_path),
        },
    )
