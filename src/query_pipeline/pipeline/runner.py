from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from typing import Any

logger = logging.getLogger(__name__)

from query_pipeline.config.models import PipelineConfig
from query_pipeline.io.business_log import BusinessLogWriter
from query_pipeline.io.jsonl import write_jsonl
from query_pipeline.llm.cache import load_cache
from query_pipeline.llm.client import LLMClient
from query_pipeline.logging_setup import LoggingSession, logging_session
import query_pipeline.steps  # noqa: F401  (stage registration side effect)
from query_pipeline.pipeline.context import PipelineContext, RunSummary, merge_stats
from query_pipeline.pipeline.stages import DEFAULT_STAGES, get_stage, stage_names


def _run_success(stats: dict[str, Any]) -> bool:
    """A run fails when discover-level work errored or nothing was adapted.

    verify_failed / translate_failed are fail-open by design (kept, retried next run)
    and empty complex_rows on a clean run (e.g. llm.enabled=false) is legitimate —
    neither makes the run a failure. bad input lines / adapt failures don't fail a
    run that still adapted sessions (they surface in the summary + bad_lines.jsonl).
    """
    if stats.get("session_errors", 0) > 0:
        return False
    # llm_failed is counted but not fatal: a deterministic LLM parse failure on a
    # single candidate must not block delivery of the rest (fail-closed drop).
    if stats.get("total_sessions", 0) == 0:
        return False
    return True


def run_pipeline(config: PipelineConfig) -> RunSummary:
    return asyncio.run(run_pipeline_async(config))


async def run_pipeline_async(config: PipelineConfig) -> RunSummary:
    with logging_session(
        config.log_dir,
        command="run",
        batch_id=config.logging.batch_id,
        verbose=config.logging.level == "DEBUG",
    ) as log_session:
        with BusinessLogWriter(log_session.log_dir, log_session.batch_id) as business_writer:
            return await _execute_pipeline(config, log_session, business_writer)


async def _execute_pipeline(
    config: PipelineConfig,
    log_session: LoggingSession,
    business_writer: BusinessLogWriter,
) -> RunSummary:
    ctx = PipelineContext(config=config)
    names = stage_names(ctx.config.stages)
    ctx.business_writer = business_writer
    ctx.stream_business_rows = names == list(DEFAULT_STAGES)
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    if not config.input.path.exists():
        raise FileNotFoundError(f"input file not found: {config.input.path}")
    # 本轮 run 开始时清空 diagnostics：上次 run 的中间产物（bad_lines / rejected /
    # segments / verified 等）若本次无内容不会重写，残留会冒充本次结果。
    diag_dir = ctx.work_dir / "runtime" / "diagnostics"
    if diag_dir.exists():
        shutil.rmtree(diag_dir)

    client: LLMClient | None = None
    cache: dict = {}
    cache_lock = asyncio.Lock()

    try:
        for name in names:
            # The leading precheck must be able to reject bad input before LLM
            # credentials, client construction, or cache loading are required.
            if name != "precheck" and config.llm.enabled and client is None:
                client = LLMClient(config.llm)
                cache = load_cache(config.cache_path)
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
    # Standard runs stream rows from the terminal post stage. This final pass is
    # both the fallback for custom stage orders and a completeness check; the
    # writer's per-stream fingerprints make it idempotent.
    business_writer.write_many(ctx.rows)
    logs = {
        "batch_id": log_session.batch_id,
        "ordinary": str(log_session.ordinary_path),
        "business": {name: str(path) for name, path in business_writer.paths.items()},
    }
    # Write output unless the run failed AND produced nothing — avoid clobbering a
    # previous good output with an empty file. Partial rows are still inspectable;
    # the exit code and summary flag the failure.
    if ctx.rows:
        write_jsonl(cleaned_path, ctx.rows)
        write_jsonl(complex_path, [r for r in ctx.rows if r.get("difficulty_level") == "hard"])
        write_jsonl(normal_path, [r for r in ctx.rows if r.get("difficulty_level") == "normal"])
    elif success:
        # 成功但零输出（全部候选被 value 拒 / 门槛调严 / --no-llm 等）：已存在上次产物时
        # 保留旧文件并告警，不得用空文件覆盖；首次运行（无历史产物）仍写空文件，
        # 保证 --no-llm 首跑与 output_files 契约不变。
        targets = (cleaned_path, complex_path, normal_path)
        existing = [p for p in targets if p.exists()]
        if existing:
            logger.warning(
                "run 成功但零输出：保留上次产物（%s），未用空文件覆盖。如确需清空请先删除产物。",
                ", ".join(str(p) for p in existing),
            )
            stats["output_preserved_previous"] = True
        else:
            write_jsonl(cleaned_path, [])
            write_jsonl(complex_path, [])
            write_jsonl(normal_path, [])
    summary_path.write_text(
        json.dumps({**stats, "success": success, "logs": logs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
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
        logs=logs,
    )
