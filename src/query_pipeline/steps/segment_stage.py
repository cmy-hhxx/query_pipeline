from __future__ import annotations

import asyncio
from typing import Any

from query_pipeline.llm.runner import run_concurrent
from query_pipeline.models.session import Segment
from query_pipeline.pipeline.context import PipelineContext
from query_pipeline.session.segment import segment_session


async def run_segment_stage(
    ctx: PipelineContext,
    client: Any,
    cache: dict[str, dict[str, Any]],
    cache_lock: asyncio.Lock,
) -> PipelineContext:
    """Topic-split every session (LLM); chat and short sessions keep whole_session."""
    cfg = ctx.config
    if not ctx.sessions:
        ctx.stats["segments"] = 0
        return ctx

    llm_ok = cfg.segmentation.enabled and client is not None

    async def worker(session) -> tuple[str, list[Segment]]:
        turns = session.turns
        if not llm_ok or session.candidate_mode == "last_only" or len(turns) <= 1:
            segments = [Segment(0, len(turns) - 1, "whole_session")] if turns else []
        else:
            assert client is not None  # llm_ok implies client
            segments = await segment_session(
                client=client,
                turns=turns,
                llm_cfg=cfg.llm,
                cache=cache,
                cache_path=cfg.cache_path,
                cache_lock=cache_lock,
            )
        return session.thread_id, segments

    results = await run_concurrent(
        ctx.sessions, worker, concurrency=cfg.llm.concurrency, description="LLM segment"
    )
    ctx.segments = {thread_id: segments for thread_id, segments in results}
    ctx.stats["segments"] = sum(len(v) for v in ctx.segments.values())
    return ctx
