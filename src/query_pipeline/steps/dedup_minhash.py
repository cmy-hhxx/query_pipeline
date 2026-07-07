from __future__ import annotations

import json

from query_pipeline.dedup.minhash import minhash_dedup
from query_pipeline.io.jsonl import write_jsonl
from query_pipeline.pipeline.context import PipelineContext


def run_dedup_minhash_step(ctx: PipelineContext) -> PipelineContext:
    cfg = ctx.config.dedup.minhash
    if not cfg.enabled:
        return ctx

    kept, removed, report = minhash_dedup(
        ctx.records,
        text_field=ctx.text_field,
        num_perm=cfg.num_perm,
        threshold=cfg.threshold,
        normalization=cfg.normalization,
        method=cfg.method,
    )
    ctx.records = kept
    ctx.rejected.extend(removed)
    ctx.stats["minhash_report"] = report
    write_jsonl(ctx.path("dedup_minhash.jsonl"), kept)
    report_path = ctx.path("minhash_dedup_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return ctx
