from __future__ import annotations

import asyncio
import logging
from typing import Any

from query_pipeline.io.sniff import sniff_format
from query_pipeline.pipeline.context import PipelineContext
from query_pipeline.precheck import precheck, render

logger = logging.getLogger(__name__)


async def run_precheck_stage(
    ctx: PipelineContext,
    client: Any,
    cache: dict[str, dict[str, Any]],
    cache_lock: asyncio.Lock,
) -> PipelineContext:
    """数据预检：在 preclean / 任何 LLM 阶段之前扫描原始输入。

    坏行占比、整体缺 chain、零合格 turn 等结构性问题立即中止运行
    （fail fast，避免浪费 LLM 资源）；warning 级别问题只报告不拦截。
    纯规则、单遍流式扫描，不调 LLM。
    """
    cfg = ctx.config.precheck
    if not cfg.enabled:
        logger.info("precheck disabled — skip")
        return ctx

    fmt = ctx.config.input.format
    if fmt == "auto":
        fmt = sniff_format(ctx.config.input.path)
    report = precheck(
        ctx.config.input.path,
        format=fmt,
        min_chain_coverage=cfg.min_chain_coverage,
        max_bad_line_ratio=cfg.max_bad_line_ratio,
    )
    ctx.stats["precheck"] = report.as_dict()
    for issue in report.issues:
        (logger.warning if issue.severity == "warning" else logger.error)(
            "precheck [%s] %s: %s", issue.severity, issue.code, issue.message
        )
    if not report.ok:
        critical = sum(1 for i in report.issues if i.severity == "critical")
        raise ValueError(
            f"precheck 未通过（{critical} 个严重问题），中止运行以避免浪费 LLM 资源：\n{render(report)}"
        )
    logger.info(
        "precheck PASS: lines=%d eligible_turns=%d chain_coverage=%.1f%%",
        report.lines, report.eligible_turns, report.chain_coverage * 100,
    )
    return ctx
