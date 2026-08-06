from __future__ import annotations

import json

from query_pipeline.config.models import PipelineConfig
from query_pipeline.io.jsonl import write_jsonl
from query_pipeline.pipeline.context import PipelineContext, RunSummary, merge_stats
from query_pipeline.steps.session_stage import run_session_stage


def run_pipeline(config: PipelineConfig) -> RunSummary:
    ctx = PipelineContext(config=config)
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)

    ctx = run_session_stage(ctx)

    complex_path = ctx.output_dir / config.output.complex_queries
    summary_path = ctx.output_dir / config.output.summary

    write_jsonl(complex_path, ctx.rows)

    stats = merge_stats(ctx)
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return RunSummary(
        success=True,
        name=config.name,
        stats=stats,
        output_files={
            "complex_queries": str(complex_path),
            "summary": str(summary_path),
        },
    )
