from __future__ import annotations

import asyncio
import logging
from typing import Any

from query_pipeline.adapters import adapt_record
from query_pipeline.io.jsonl import append_jsonl, read_jsonl_with_bad_lines
from query_pipeline.io.sniff import preclean_records, sniff_format
from query_pipeline.models.turn import Session
from query_pipeline.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


async def run_preclean_stage(
    ctx: PipelineContext,
    client: Any,
    cache: dict[str, dict[str, Any]],
    cache_lock: asyncio.Lock,
) -> PipelineContext:
    """Read input, detect format, pre-clean, and adapt into unified Sessions."""
    cfg = ctx.config
    fmt = cfg.input.format
    if fmt == "auto":
        fmt = sniff_format(cfg.input.path)
        logger.info("input format auto-detected: %s", fmt)

    raw_records, bad_count = read_jsonl_with_bad_lines(cfg.input.path, ctx.path("bad_lines.jsonl"))
    if bad_count:
        logger.warning("input: skipped %d bad line(s) → %s", bad_count, ctx.path("bad_lines.jsonl"))

    raw_records, dup_count, empty_count = preclean_records(raw_records, fmt)
    if dup_count or empty_count:
        logger.info("input pre-clean: %d duplicate(s), %d empty-context session(s)", dup_count, empty_count)

    sessions: list[Session] = []
    adapt_skipped = 0
    bad_path = ctx.path("bad_lines.jsonl")
    for record in raw_records:
        try:
            sessions.append(adapt_record(record, fmt))
        except ValueError as exc:
            adapt_skipped += 1
            logger.warning("adapt: skipped record line %s: %s", record.get("_line_number", "?"), str(exc)[:200])
            append_jsonl(
                bad_path,
                {
                    "reason": "adapt_failed",
                    "detail": str(exc)[:200],
                    "line_number": record.get("_line_number"),
                    "record": {k: v for k, v in record.items() if not k.startswith("_")},
                },
            )

    ctx.sessions = sessions
    ctx.stats["total_sessions"] = len(sessions)
    ctx.stats["input_bad_lines"] = bad_count + adapt_skipped
    ctx.stats["input_duplicates"] = dup_count
    ctx.stats["input_empty_sessions"] = empty_count
    ctx.stats["input_format"] = fmt

    if not sessions:
        ctx.stats.update(
            {
                "segments": 0,
                "candidates": 0,
                "complex_rows": 0,
                "llm_failed": 0,
                "category_counts": {},
            }
        )
    return ctx
