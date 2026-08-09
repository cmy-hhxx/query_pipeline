from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from typing import Any

from query_pipeline.io.checkpoint import content_key, stage_checkpoint
from query_pipeline.io.jsonl import write_jsonl
from query_pipeline.llm.runner import run_concurrent
from query_pipeline.models.session import Segment
from query_pipeline.pipeline.context import PipelineContext
from query_pipeline.session.assemble import assemble_row
from query_pipeline.session.funnel import funnel_candidate
from query_pipeline.session.judge import segment_of

logger = logging.getLogger(__name__)


def session_content_key(session) -> str:
    parts = [session.thread_id, session.candidate_mode]
    for t in session.turns:
        parts.append(
            f"{t.trace_id}|{t.question}|{t.answer}|{t.request_time}|{t.status}|{t.outcome}|{t.tool_names}|{t.tool_count}|"
            + json.dumps(t.chain, ensure_ascii=False, sort_keys=True, default=str)
        )
    return content_key(*parts)


async def run_judge_stage(
    ctx: PipelineContext,
    client: Any,
    cache: dict[str, dict[str, Any]],
    cache_lock: asyncio.Lock,
) -> PipelineContext:
    """Semantic gates + classification per candidate: value -> complexity -> classify.

    LLM failures drop the candidate (fail-closed). Session-level checkpointing
    resumes completed sessions without re-calling the LLM.
    """
    cfg = ctx.config
    ctx.prune_debug_artifacts("segments.jsonl", "candidates.jsonl", "judged.jsonl")
    if not ctx.sessions:
        ctx.rows = []
        ctx.stats.update(
            {
                "segments": 0,
                "candidates": 0,
                "complex_rows": 0,
                "normal_rows": 0,
                "value_rejected": 0,
                "non_complex": 0,
                "llm_failed": 0,
                "category_counts": {},
                "category_counts_normal": {},
            }
        )
        return ctx

    if client is None or not cfg.judge.enabled:
        logger.warning("llm disabled: semantic gates skipped, output will be empty")
        ctx.rows = []
        ctx.stats.update(
            {
                "segments": sum(len(v) for v in ctx.segments.values()),
                "candidates": sum(len(v) for v in ctx.candidates.values()),
                "complex_rows": 0,
                "normal_rows": 0,
                "value_rejected": 0,
                "non_complex": 0,
                "llm_failed": 0,
                "category_counts": {},
                "category_counts_normal": {},
            }
        )
        return ctx

    checkpoint = stage_checkpoint(cfg, "judge")
    logger.info("[judge] judging %d session(s), concurrency=%d", len(ctx.sessions), cfg.llm.concurrency)
    counters: Counter[str] = Counter()
    complex_categories: Counter[str] = Counter()
    normal_categories: Counter[str] = Counter()
    debug_judged: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    async def process(session) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
        key = session_content_key(session)
        record = checkpoint.get(key)
        if record is not None and {"rows", "stats", "judged"} <= set(record):
            return record["rows"], record["stats"], record["judged"]
        try:
            sess_rows, sess_stats, judged = await _process_session(ctx, client, cache, cache_lock, session)
            if sess_stats.get("llm_failed", 0) == 0 and "error" not in sess_stats:
                await checkpoint.mark(key, rows=sess_rows, stats=sess_stats, judged=judged)
        except Exception as exc:  # noqa: BLE001 checkpoint 磁盘异常等也按会话错误兜底
            sess_rows, sess_stats, judged = [], {"session_error": 1, "error": str(exc)[:200]}, []
        return sess_rows, sess_stats, judged

    # 会话层与其余阶段一致走 run_concurrent：纯任务编排 + 单行异常兜底（返回 None）。
    results_list = await run_concurrent(ctx.sessions, process, description="LLM judge sessions")
    for index, result in enumerate(results_list):
        if result is None:  # run_concurrent 兜底网捕获的意外异常
            counters["session_errors"] += 1
            continue
        sess_rows, sess_stats, judged = result
        rows.extend(sess_rows)
        counters.update({k: v for k, v in sess_stats.items() if k not in ("categories", "categories_normal", "error")})
        if "error" in sess_stats:
            counters["session_errors"] += 1
        complex_categories.update(sess_stats.get("categories") or {})
        normal_categories.update(sess_stats.get("categories_normal") or {})
        debug_judged.extend(judged)

    ctx.rows = rows
    candidates = counters.get("candidates", 0)
    value_rejected = counters.get("value_rejected", 0)
    llm_failed = counters.get("llm_failed", 0)
    ctx.stats.update(
        {
            "segments": sum(len(v) for v in ctx.segments.values()),
            "candidates": candidates,
            "complex_rows": counters.get("complex_rows", 0),
            "normal_rows": counters.get("normal_rows", 0),
            "value_rejected": value_rejected,
            "non_complex": counters.get("non_complex", 0),
            "llm_failed": llm_failed,
            "session_errors": counters.get("session_errors", 0),
            "category_counts": {cid: complex_categories[cid] for cid in sorted(complex_categories)},
            "category_counts_normal": {cid: normal_categories[cid] for cid in sorted(normal_categories)},
        }
    )
    logger.info(
        "[judge] candidates=%d valuable=%d complex=%d normal=%d value_rejected=%d llm_failed=%d",
        candidates,
        candidates - value_rejected - llm_failed,
        counters.get("complex_rows", 0),
        counters.get("normal_rows", 0),
        value_rejected,
        llm_failed,
    )
    if cfg.debug.dump_intermediates:
        _write_debug_files(ctx, debug_judged)
    return ctx


async def _process_session(
    ctx: PipelineContext,
    client: Any,
    cache: dict[str, dict[str, Any]],
    cache_lock: asyncio.Lock,
    session,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    turns = session.turns
    if not turns:
        return [], {"empty_sessions": 1}, []
    segments = ctx.segments.get(session.thread_id)
    if segments is None:
        segments = [Segment(0, len(turns) - 1, "whole_session")] if turns else []
    candidates = ctx.candidates.get(session.thread_id, [])

    judged = await run_concurrent(
        candidates,
        lambda idx: funnel_candidate(
            client=client,
            turns=turns,
            segments=segments,
            idx=idx,
            llm_cfg=ctx.config.llm,
            cache=cache,
            cache_path=ctx.config.cache_path,
            cache_lock=cache_lock,
        ),
        description="LLM funnel",
    )

    rows: list[dict[str, Any]] = []
    llm_failed = 0
    value_rejected = 0
    non_complex = 0
    complex_count = 0
    normal_count = 0
    complex_categories: Counter[str] = Counter()
    normal_categories: Counter[str] = Counter()
    for j in judged:
        if j is None:  # run_concurrent 兜底网捕获的意外异常（如 cache 磁盘 OSError）
            llm_failed += 1
            continue
        if j.get("error"):
            llm_failed += 1
            continue
        if j.get("dropped") == "value":
            value_rejected += 1
            continue
        idx = j["idx"]
        if j["complexity"].is_complex:
            complex_count += 1
            complex_categories[j["category_id"]] += 1
        else:
            non_complex += 1
            normal_count += 1
            normal_categories[j["category_id"]] += 1
        rows.append(
            assemble_row(
                session,
                segment_of(segments, idx),
                idx,
                j["category_id"],
                j.get("reason"),
                j["difficulty"],
            )
        )

    debug = [
        {
            "thread_id": session.thread_id,
            "idx": j.get("idx"),
            "question": turns[j["idx"]].question[:200] if j.get("idx") is not None else "",
            "is_valuable": bool(getattr(j.get("value"), "is_valuable", None)),
            "is_complex": bool(getattr(j.get("complexity"), "is_complex", None)),
            "difficulty": j.get("difficulty"),
            "category_id": j.get("category_id"),
            "reason": j.get("reason"),
            "error": j.get("error"),
        }
        for j in judged
    ]
    stats = {
        "candidates": len(candidates),
        "complex_rows": complex_count,
        "normal_rows": normal_count,
        "value_rejected": value_rejected,
        "non_complex": non_complex,
        "llm_failed": llm_failed,
        "categories": dict(complex_categories),
        "categories_normal": dict(normal_categories),
    }
    return rows, stats, debug


def _write_debug_files(ctx: PipelineContext, judged: list[dict[str, Any]]) -> None:
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    if judged:
        write_jsonl(ctx.path("judged.jsonl"), judged)
    if ctx.segments:
        seg_records = [
            {"thread_id": tid, "seg_idx": i, "start": seg.start, "end": seg.end, "topic": seg.topic}
            for tid, segs in ctx.segments.items()
            for i, seg in enumerate(segs)
        ]
        write_jsonl(ctx.path("segments.jsonl"), seg_records)
    if ctx.candidates:
        turns_by_id = {s.thread_id: s.turns for s in ctx.sessions}
        cand_records = [
            {"thread_id": tid, "idx": i, "question": turns_by_id[tid][i].question[:200]}
            for tid, idxs in ctx.candidates.items()
            for i in idxs
            if tid in turns_by_id and i < len(turns_by_id[tid])
        ]
        write_jsonl(ctx.path("candidates.jsonl"), cand_records)
