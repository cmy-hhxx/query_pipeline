from __future__ import annotations

import asyncio
import logging
from collections import Counter
from typing import Any

from rich.progress import Progress

from query_pipeline.config.models import PipelineConfig
from query_pipeline.io.jsonl import read_jsonl_skipping, write_jsonl
from query_pipeline.llm.cache import load_cache
from query_pipeline.llm.client import LLMClient
from query_pipeline.models.session import Segment
from query_pipeline.pipeline.context import PipelineContext
from query_pipeline.session.assemble import assemble_row
from query_pipeline.session.candidates import is_eligible, select_candidates
from query_pipeline.session.judge import is_complex_result, judge_candidates, segment_of
from query_pipeline.session.segment import segment_session


def run_session_stage(ctx: PipelineContext) -> PipelineContext:
    cfg = ctx.config
    sessions, skipped = read_jsonl_skipping(cfg.input.path)
    if skipped:
        logging.getLogger(__name__).warning("input: skipped %d unparseable line(s) in %s", skipped, cfg.input.path)
    ctx.stats["total_sessions"] = len(sessions)
    ctx.stats["input_bad_lines"] = skipped

    if not cfg.session_stage.enabled:
        ctx.rows = []
        _zero_stats(ctx.stats)
        return ctx

    ctx.rows, ctx.stats, debug = asyncio.run(_run_all(ctx, sessions, skipped))
    _write_debug_files(ctx, debug)
    return ctx


async def _run_all(
    ctx: PipelineContext, sessions: list[dict[str, Any]], input_bad_lines: int
) -> tuple[list[dict[str, Any]], dict[str, Any], tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]]:
    cfg = ctx.config
    client: LLMClient | None = None
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    debug_segments: list[dict[str, Any]] = []
    debug_candidates: list[dict[str, Any]] = []
    debug_judged: list[dict[str, Any]] = []

    try:
        if cfg.llm_stage.enabled:
            client = LLMClient(cfg.llm_stage)
            cache = load_cache(cfg.llm_stage.cache)
        else:
            cache = {}

        with Progress() as progress:
            task_id = progress.add_task("Sessions", total=len(sessions))
            for session in sessions:
                try:
                    sess_rows, sess_stats, seg, cand, judged = await _process_session(cfg, client, cache, session)
                except Exception as exc:  # noqa: BLE001 - one bad session must not kill the run
                    sess_rows, sess_stats = [], {"session_error": 1, "error": str(exc)[:200]}
                    seg, cand, judged = [], [], []
                rows.extend(sess_rows)
                counters.update({k: v for k, v in sess_stats.items() if k not in ("categories", "error")})
                if "error" in sess_stats:
                    counters["session_errors"] += 1
                category_counts.update(sess_stats.get("categories") or {})
                debug_segments.extend(seg)
                debug_candidates.extend(cand)
                debug_judged.extend(judged)
                progress.advance(task_id)
    finally:
        if client is not None:
            await client.close()

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
    cfg: PipelineConfig,
    client: LLMClient | None,
    cache: dict[str, dict[str, Any]],
    session: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    s = cfg.session_stage
    turns = session.get("context") or []
    if not isinstance(turns, list) or not turns:
        return [], {"empty_sessions": 1}, [], [], []

    if s.segmentation.enabled and client is not None:
        segments = await segment_session(
            client=client, turns=turns, llm_cfg=cfg.llm_stage, cache=cache, cache_path=cfg.llm_stage.cache
        )
    else:
        segments = [Segment(0, len(turns) - 1, "whole_session")]

    if s.step1.enabled:
        candidates = select_candidates(turns, s.step1)
    else:
        candidates = [i for i, turn in enumerate(turns) if is_eligible(turn)]

    judged: list[dict[str, Any]] = []
    if s.step2.enabled and client is not None and candidates:
        judged = await judge_candidates(
            client=client,
            turns=turns,
            segments=segments,
            candidates=candidates,
            llm_cfg=cfg.llm_stage,
            step2_cfg=s.step2,
            cache=cache,
            cache_path=cfg.llm_stage.cache,
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
            rows.append(
                assemble_row(session, turns, segment, idx, judged_row["category_id"], judged_row.get("reason"))
            )
            category_counts[judged_row["category_id"]] += 1
        else:
            non_complex += 1

    seg_debug = [
        {
            "thread_id": session.get("thread_id", ""),
            "seg_idx": i,
            "start": seg.start,
            "end": seg.end,
            "n_turns": seg.end - seg.start + 1,
            "topic": seg.topic,
        }
        for i, seg in enumerate(segments)
    ]
    cand_debug = [
        {"thread_id": session.get("thread_id", ""), "idx": i, "question": turns[i].get("question", "")[:200]}
        for i in candidates
    ]
    judged_debug = [
        {
            "thread_id": session.get("thread_id", ""),
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


def _write_debug_files(
    ctx: PipelineContext,
    debug: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
) -> None:
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
