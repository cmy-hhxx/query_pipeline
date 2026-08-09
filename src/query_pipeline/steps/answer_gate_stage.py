"""Answer quality gate: structural + content signals, rules only.

Structural: session rows must carry last_event_type=runFinished (cancelled/
interrupted/failed/expired turns are incomplete answers; feedbackUpsert turns
are feedback records, not answers). Chat rows have no event field — content
signals apply to every row: refusal phrases, dangling-punctuation truncation,
answers below a length floor. Drops are counted per reason for the summary.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

from query_pipeline.pipeline.context import PipelineContext

MIN_ANSWER_LEN = 50
_RUN_FINISHED = "runFinished"

_REFUSAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"抱歉[,，]?(?:我)?无法(?:回答|提供|满足)", re.I),
    re.compile(r"(?:我)?(?:不能|无法|拒绝)(?:回答|提供|满足|执行)", re.I),
    re.compile(r"我不(?:能|会)(?:回答|说)", re.I),
    re.compile(r"I['\u2019]?m sorry", re.I),
    re.compile(r"(?:I )?(?:cannot|can't|won't|will not) (?:answer|help|provide|comply)", re.I),
    re.compile(r"refus(?:e|ed) to (?:answer|provide|respond)", re.I),
    re.compile(r"as an AI (?:assistant|language model)", re.I),
    re.compile(r"(?:not|unable) (?:able|allowed|able to) to (?:answer|provide|respond)", re.I),
)

# Comma/colon/dash endings are strong truncation signals (sentence-enders are fine).
_DANGLING_END = ",，:：-—–"


def answer_gate_reason(row: dict[str, Any]) -> str | None:
    """Return the rejection reason, or None when the row passes the gate."""
    meta = row.get("meta") or {}
    event = meta.get("last_event_type") if isinstance(meta, dict) else None
    if event is not None and event != _RUN_FINISHED:
        return f"last_event_type={event}"

    inp = row.get("input")
    question = str(inp.get("text") or "") if isinstance(inp, dict) else ""
    text = str(row.get("text_answer") or "")
    if not text.strip():
        return "empty_answer"
    for pattern in _REFUSAL_PATTERNS:
        if pattern.search(text):
            return "refusal"
    if len(text.strip()) < MIN_ANSWER_LEN:
        return f"answer_too_short({len(text.strip())}<{MIN_ANSWER_LEN})"
    if question and text.rstrip()[-1:] in _DANGLING_END:
        return "truncated_dangling_punctuation"
    return None


async def run_answer_gate_stage(
    ctx: PipelineContext,
    client: Any,
    cache: dict[str, dict[str, Any]],
    cache_lock: asyncio.Lock,
) -> PipelineContext:
    if not ctx.rows:
        ctx.stats["answer_gate_rejected"] = 0
        return ctx
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in ctx.rows:
        reason = answer_gate_reason(row)
        if reason is None:
            kept.append(row)
        else:
            inp = row.get("input")
            question = str(inp.get("text") or "") if isinstance(inp, dict) else ""
            rejected.append(
                {
                    "trace_id": row.get("trace_id", ""),
                    "source_case_id": row.get("source_case_id", ""),
                    "difficulty": row.get("difficulty_level", ""),
                    "category": row.get("category", ""),
                    "question": question[:200],
                    "answer": str(row.get("text_answer") or "")[:200],
                    "reason": reason,
                }
            )
    ctx.rows = kept
    ctx.stats["answer_gate_rejected"] = len(rejected)
    if rejected:
        by_reason = Counter(r["reason"].split("(")[0] for r in rejected)
        top = ", ".join(f"{reason}={count}" for reason, count in by_reason.most_common(8))
        logger.info("[answer_gate] rejected %d row(s): %s", len(rejected), top)
    if rejected and ctx.config.debug.dump_intermediates:
        from query_pipeline.io.jsonl import write_jsonl

        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(ctx.path("answer_gate_rejected.jsonl"), rejected)
    return ctx
