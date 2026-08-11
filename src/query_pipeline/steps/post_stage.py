from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

from query_pipeline.io.checkpoint import stage_checkpoint
from query_pipeline.io.jsonl import write_jsonl
from query_pipeline.llm.client import LLMClient
from query_pipeline.pipeline.context import PipelineContext
from query_pipeline.post.dedup import dedup_rows, semantic_dedup_rows
from query_pipeline.post.translate import translate_rows


async def run_post_stage(
    ctx: PipelineContext,
    client: LLMClient | None,
    cache: dict[str, dict[str, Any]],
    cache_lock: asyncio.Lock,
) -> PipelineContext:
    cfg = ctx.config
    ctx.prune_debug_artifacts("deduped.jsonl")
    if any(
        ctx.stats.get(field, 0) > 0
        for field in (
            "llm_failed",
            "session_errors",
            "verify_failed",
            "template_family_failed",
            "semantic_dedup_failed",
        )
    ):
        return ctx
    if not cfg.post.enabled:
        return ctx

    if cfg.post.dedup.enabled and not ctx.stats.get("verify_corpus_review_done"):
        if cfg.post.dedup.mode == "semantic" and client is not None:
            ctx.rows, dropped, semantic_stats = await semantic_dedup_rows(
                ctx.rows,
                cfg.post.dedup,
                client=client,
                llm_cfg=cfg.llm,
                cache=cache,
                cache_path=cfg.cache_path,
                cache_lock=cache_lock,
            )
            ctx.stats.update(semantic_stats)
        else:
            ctx.rows, dropped = dedup_rows(ctx.rows, cfg.post.dedup)
            ctx.stats.update(
                {
                    "semantic_dedup_candidates": 0,
                    "semantic_dedup_removed": 0,
                    "semantic_dedup_failed": 0,
                }
            )
        ctx.stats["dedup_removed"] = len(dropped)
        if dropped and cfg.debug.dump_intermediates:
            ctx.work_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(ctx.path("deduped.jsonl"), dropped)

    if cfg.post.translate.enabled and client is not None and ctx.rows:
        checkpoint = stage_checkpoint(cfg, "translate")
        counts = await translate_rows(
            ctx.rows,
            client=client,
            llm_cfg=cfg.llm,
            cache=cache,
            cache_path=cfg.cache_path,
            checkpoint=checkpoint,
            cache_lock=cache_lock,
            on_complete=(
                ctx.business_writer.write
                if ctx.stream_business_rows and ctx.business_writer is not None
                else None
            ),
        )
    else:
        counts = {"translated": 0, "translate_skipped": 0, "translate_failed": 0}
        if ctx.stream_business_rows and ctx.business_writer is not None:
            ctx.business_writer.write_many(ctx.rows)
    ctx.stats.update(counts)
    # Final means publishable rows after verify, answer gates and dedup. The
    # legacy complex_rows/category_counts remain initial-judge statistics.
    final_complex_features: Counter[str] = Counter()
    final_complex_categories: Counter[str] = Counter()
    final_normal_categories: Counter[str] = Counter()
    for row in ctx.rows:
        meta = row.get("meta")
        profile = meta.get("complexity_profile") if isinstance(meta, dict) else None
        if isinstance(profile, dict) and isinstance(profile.get("complex_features"), list):
            final_complex_features.update(str(item) for item in profile["complex_features"])
        category = str(row.get("category") or "")
        if row.get("difficulty_level") == "hard":
            final_complex_categories[category] += 1
        else:
            final_normal_categories[category] += 1
    ctx.stats.update(
        {
            "final_complex_rows": sum(row.get("difficulty_level") == "hard" for row in ctx.rows),
            "final_normal_rows": sum(row.get("difficulty_level") == "normal" for row in ctx.rows),
            "complex_feature_counts_final": dict(sorted(final_complex_features.items())),
            "category_counts_final": dict(sorted(final_complex_categories.items())),
            "category_counts_normal_final": dict(sorted(final_normal_categories.items())),
        }
    )
    return ctx
