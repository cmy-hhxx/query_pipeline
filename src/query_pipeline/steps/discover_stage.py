from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from typing import Any

from rich.progress import Progress

from query_pipeline.adapters import adapt_record
from query_pipeline.io.checkpoint import content_key, stage_checkpoint
from query_pipeline.io.jsonl import append_jsonl, read_jsonl_with_bad_lines, write_jsonl
from query_pipeline.llm.client import LLMClient
from query_pipeline.models.session import Segment
from query_pipeline.models.turn import Session
from query_pipeline.pipeline.context import PipelineContext
from query_pipeline.session.assemble import assemble_row
from query_pipeline.session.candidates import is_eligible, select_candidates, select_last_only
from query_pipeline.session.judge import is_complex_result, judge_candidates, segment_of
from query_pipeline.session.segment import segment_session


def session_content_key(session: Session) -> str:
    parts = [session.thread_id, session.candidate_mode]
    for t in session.turns:
        # status/outcome drive is_eligible; tool_count drives chain_tool_calls/
        # chain_steps when chain is absent; tool_names/chain drive the step1
        # AND-gates — the key must change when any of them change, or stale
        # discover results replay.
        parts.append(
            f"{t.trace_id}|{t.question}|{t.answer}|{t.request_time}|{t.status}|{t.outcome}|{t.tool_names}|{t.tool_count}|"
            + json.dumps(t.chain, ensure_ascii=False, sort_keys=True, default=str)
        )
    return content_key(*parts)


async def run_discover_stage(
    ctx: PipelineContext,
    client: LLMClient | None,
    cache: dict[str, dict[str, Any]],
    cache_lock: asyncio.Lock,
) -> PipelineContext:
    """Adapt input → segment/step1/step2 → assemble OutputRows."""
    cfg = ctx.config
    ctx.prune_debug_artifacts("segments.jsonl", "candidates.jsonl", "judged.jsonl")
    raw_records, bad_count = read_jsonl_with_bad_lines(cfg.input.path, ctx.path("bad_lines.jsonl"))
    if bad_count:
        logging.getLogger(__name__).warning(
            "input: skipped %d bad line(s) → %s", bad_count, ctx.path("bad_lines.jsonl")
        )

    sessions: list[Session] = []
    adapt_skipped = 0
    logger = logging.getLogger(__name__)
    bad_path = ctx.path("bad_lines.jsonl")
    for record in raw_records:
        try:
            sessions.append(adapt_record(record, cfg.input.format))
        except ValueError as exc:
            adapt_skipped += 1
            logger.warning(
                "adapt: skipped record line %s: %s", record.get("_line_number", "?"), str(exc)[:200]
            )
            append_jsonl(
                bad_path,
                {
                    "reason": "adapt_failed",
                    "detail": str(exc)[:200],
                    "line_number": record.get("_line_number"),
                    "record": {k: v for k, v in record.items() if not k.startswith("_")},
                },
            )
    skipped = bad_count + adapt_skipped
    ctx.stats["total_sessions"] = len(sessions)
    ctx.stats["input_bad_lines"] = skipped

    if not sessions:
        _zero_stats(ctx.stats)
        ctx.rows = []
        return ctx

    rows, stats, debug = await _run_all(ctx, sessions, client, cache, cache_lock, skipped)
    ctx.rows = rows
    ctx.stats.update(stats)
    if cfg.debug.dump_intermediates:
        _write_debug_files(ctx, debug)
    return ctx


async def _run_all(
    ctx: PipelineContext,
    sessions: list[Session],
    client: LLMClient | None,
    cache: dict[str, dict[str, Any]],
    cache_lock: asyncio.Lock,
    input_bad_lines: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], tuple[list, list, list]]:
    cfg = ctx.config
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    debug_segments: list[dict[str, Any]] = []
    debug_candidates: list[dict[str, Any]] = []
    debug_judged: list[dict[str, Any]] = []
    checkpoint = stage_checkpoint(cfg, "discover")

    if client is None and cfg.step2.enabled:
        logging.getLogger(__name__).warning(
            "llm.enabled=false: step2 judge is skipped, complex_queries will be empty"
        )

    results: dict[int, tuple] = {}

    def absorb(sess_rows, sess_stats, seg, cand, judged) -> None:
        rows.extend(sess_rows)
        counters.update({k: v for k, v in sess_stats.items() if k not in ("categories", "error")})
        if "error" in sess_stats:
            counters["session_errors"] += 1
        category_counts.update(sess_stats.get("categories") or {})
        debug_segments.extend(seg)
        debug_candidates.extend(cand)
        debug_judged.extend(judged)

    def is_clean(stats: dict[str, Any]) -> bool:
        return stats.get("llm_failed", 0) == 0 and "error" not in stats

    with Progress() as progress:
        task_id = progress.add_task("Sessions", total=len(sessions))

        async def process(index: int, session: Session) -> None:
            key = session_content_key(session)
            record = checkpoint.get(key)
            if record is not None and {"rows", "stats", "segments", "candidates", "judged"} <= set(record):
                results[index] = (
                    record["rows"],
                    record["stats"],
                    record["segments"],
                    record["candidates"],
                    record["judged"],
                )
                return
            try:
                sess_rows, sess_stats, seg, cand, judged = await _process_session(
                    ctx, client, cache, cache_lock, session, progress
                )
            except Exception as exc:  # noqa: BLE001
                sess_rows, sess_stats = [], {"session_error": 1, "error": str(exc)[:200]}
                seg, cand, judged = [], [], []
            if is_clean(sess_stats):
                await checkpoint.mark(
                    key, rows=sess_rows, stats=sess_stats, segments=seg, candidates=cand, judged=judged
                )
            results[index] = (sess_rows, sess_stats, seg, cand, judged)

        async def tracked(index: int, session: Session) -> None:
            await process(index, session)
            progress.advance(task_id)

        await asyncio.gather(*(tracked(i, s) for i, s in enumerate(sessions)))

    for index in range(len(sessions)):
        if index not in results:
            continue
        absorb(*results[index])

    stats: dict[str, Any] = {
        "total_sessions": len(sessions),
        "input_bad_lines": input_bad_lines,
        "segments": counters.get("segments", 0),
        "candidates": counters.get("candidates", 0),
        "complex_rows": len(rows),
        "non_complex": counters.get("non_complex", 0),
        "llm_failed": counters.get("llm_failed", 0),
        "empty_sessions": counters.get("empty_sessions", 0),
        "session_errors": counters.get("session_errors", 0),
        "category_counts": {cid: category_counts[cid] for cid in sorted(category_counts)},
    }
    return rows, stats, (debug_segments, debug_candidates, debug_judged)


async def _process_session(
    ctx: PipelineContext,
    client: LLMClient | None,
    cache: dict[str, dict[str, Any]],
    cache_lock: asyncio.Lock,
    session: Session,
    progress: Progress | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list, list, list]:
    cfg = ctx.config
    turns = session.turns
    if not turns:
        return [], {"empty_sessions": 1}, [], [], []

    if session.candidate_mode == "last_only":
        # Chat semantics: the trailing turn is the candidate by definition, gated only on
        # eligibility. Step1's chain/tool AND-gates do not apply (prior turns carry no chain).
        segments = [Segment(0, len(turns) - 1, "whole_session")]
        candidates = select_last_only(turns)
    else:
        if cfg.segmentation.enabled and client is not None:
            segments = await segment_session(
                client=client,
                turns=turns,
                llm_cfg=cfg.llm,
                cache=cache,
                cache_path=cfg.llm.cache,
                cache_lock=cache_lock,
            )
        else:
            segments = [Segment(0, len(turns) - 1, "whole_session")]

        if cfg.step1.enabled:
            candidates = select_candidates(turns, cfg.step1)
        else:
            candidates = [i for i, turn in enumerate(turns) if is_eligible(turn)]

    judged: list[dict[str, Any]] = []
    if cfg.step2.enabled and client is not None and candidates:
        judged = await judge_candidates(
            client=client,
            turns=turns,
            segments=segments,
            candidates=candidates,
            llm_cfg=cfg.llm,
            step2_cfg=cfg.step2,
            cache=cache,
            cache_path=cfg.llm.cache,
            cache_lock=cache_lock,
            progress=progress,
        )

    rows: list[dict[str, Any]] = []
    non_complex = 0
    llm_failed = 0
    category_counts: Counter[str] = Counter()
    for judged_row in judged:
        if judged_row.get("error"):
            llm_failed += 1
        elif is_complex_result(judged_row):
            idx = judged_row["idx"]
            segment = segment_of(segments, idx)
            rows.append(assemble_row(session, segment, idx, judged_row["category_id"], judged_row.get("reason")))
            category_counts[judged_row["category_id"]] += 1
        else:
            non_complex += 1

    seg_debug = [
        {
            "thread_id": session.thread_id,
            "seg_idx": i,
            "start": seg.start,
            "end": seg.end,
            "n_turns": seg.end - seg.start + 1,
            "topic": seg.topic,
        }
        for i, seg in enumerate(segments)
    ]
    cand_debug = [
        {"thread_id": session.thread_id, "idx": i, "question": turns[i].question[:200]} for i in candidates
    ]
    judged_debug = [
        {
            "thread_id": session.thread_id,
            "idx": j.get("idx"),
            "is_complex": j.get("is_complex"),
            "category_id": j.get("category_id"),
            "reason": j.get("reason"),
            "error": j.get("error"),
        }
        for j in judged
    ]

    stats = {
        "segments": len(segments),
        "candidates": len(candidates),
        "complex_rows": len(rows),
        "non_complex": non_complex,
        "llm_failed": llm_failed,
        "categories": dict(category_counts),
    }
    return rows, stats, seg_debug, cand_debug, judged_debug


def _write_debug_files(ctx: PipelineContext, debug: tuple[list, list, list]) -> None:
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    segments, candidates, judged = debug
    for name, records in (
        ("segments.jsonl", segments),
        ("candidates.jsonl", candidates),
        ("judged.jsonl", judged),
    ):
        if records:
            write_jsonl(ctx.path(name), records)


def _zero_stats(stats: dict[str, Any]) -> None:
    stats.update(
        {
            "segments": 0,
            "candidates": 0,
            "complex_rows": 0,
            "non_complex": 0,
            "llm_failed": 0,
            "category_counts": {},
        }
    )
