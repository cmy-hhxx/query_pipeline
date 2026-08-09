from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

from query_pipeline.config.models import PipelineConfig
from query_pipeline.io.jsonl import write_jsonl
from query_pipeline.llm.cache import load_cache
from query_pipeline.llm.client import LLMClient
import query_pipeline.steps  # noqa: F401  (stage registration side effect)
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
    # llm_failed is counted but not fatal: a deterministic LLM parse failure on a
    # single candidate must not block delivery of the rest (fail-closed drop).
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
        cache = load_cache(config.cache_path)

    try:
        for name in stage_names(ctx.config.stages):
            stage = get_stage(name)
            rows_before = len(ctx.rows)
            started = time.monotonic()
            ctx = await stage(ctx, client, cache, cache_lock)
            elapsed = time.monotonic() - started
            logger.info(
                "[stage] %-12s rows %d -> %d  (%.1fs)",
                name, rows_before, len(ctx.rows), elapsed,
            )
    finally:
        if client is not None:
            await client.close()

    cleaned_path = ctx.output_dir / config.output.cleaned_queries
    complex_path = ctx.output_dir / config.output.complex_queries
    normal_path = ctx.output_dir / config.output.normal_queries
    summary_path = ctx.output_dir / config.output.summary

    stats = merge_stats(ctx)
    success = _run_success(stats)
    # Write output unless the run failed AND produced nothing — avoid clobbering a
    # previous good output with an empty file. Partial rows are still inspectable;
    # the exit code and summary flag the failure.
    if success or ctx.rows:
        write_jsonl(cleaned_path, ctx.rows)
        write_jsonl(complex_path, [r for r in ctx.rows if r.get("difficulty_level") == "hard"])
        write_jsonl(normal_path, [r for r in ctx.rows if r.get("difficulty_level") == "normal"])
    summary_path.write_text(
        json.dumps({**stats, "success": success}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return RunSummary(
        success=success,
        name=config.name,
        stats=stats,
        output_files={
            "cleaned_queries": str(cleaned_path),
            "complex_queries": str(complex_path),
            "normal_queries": str(normal_path),
            "summary": str(summary_path),
        },
    )
