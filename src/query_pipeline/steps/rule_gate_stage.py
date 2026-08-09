from __future__ import annotations

import asyncio
from typing import Any

from query_pipeline.pipeline.context import PipelineContext
from query_pipeline.session.candidates import effective_gate, is_eligible, select_candidates, select_last_only


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
    gate = effective_gate(cfg.rule_gate, ctx.stats.get("input_format", "session"))
    candidates: dict[str, list[int]] = {}
    total = 0
    for session in ctx.sessions:
        if session.candidate_mode == "last_only":
            selected = select_last_only(session.turns, gate) if gate.enabled else (
                select_last_only(session.turns)
            )
        else:
            selected = select_candidates(session.turns, gate) if gate.enabled else [
                i for i, t in enumerate(session.turns) if is_eligible(t)
            ]
        candidates[session.thread_id] = selected
        total += len(selected)
    ctx.candidates = candidates
    ctx.stats["candidates"] = total
    if ctx.sessions and total == 0:
        import logging

        logging.getLogger(__name__).warning(
            "rule_gate 过滤后候选为 0（%d 个会话全部被过滤）：当前门槛可能过严。"
            "可运行 `query-pipeline suggest <输入>` 查看推荐参数组合。",
            len(ctx.sessions),
        )
    return ctx
