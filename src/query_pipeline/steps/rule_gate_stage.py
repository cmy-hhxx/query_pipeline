from __future__ import annotations

import asyncio
from typing import Any

from query_pipeline.pipeline.context import PipelineContext
from query_pipeline.session.candidates import is_eligible, select_candidates, select_last_only


async def run_rule_gate_stage(
    ctx: PipelineContext,
    client: Any,
    cache: dict[str, dict[str, Any]],
    cache_lock: asyncio.Lock,
) -> PipelineContext:
    """Rule value gate: reject rules + chain/tool AND-gates, active for both formats.

    chat (last_only) candidates are gated identically — chat records always
    carry judge_data.chain, so the tool gates are a meaningful coarse filter.
    """
    cfg = ctx.config
    candidates: dict[str, list[int]] = {}
    total = 0
    for session in ctx.sessions:
        if session.candidate_mode == "last_only":
            selected = select_last_only(session.turns, cfg.rule_gate) if cfg.rule_gate.enabled else (
                select_last_only(session.turns)
            )
        else:
            selected = select_candidates(session.turns, cfg.rule_gate) if cfg.rule_gate.enabled else [
                i for i, t in enumerate(session.turns) if is_eligible(t)
            ]
        candidates[session.thread_id] = selected
        total += len(selected)
    ctx.candidates = candidates
    ctx.stats["candidates"] = total
    return ctx
