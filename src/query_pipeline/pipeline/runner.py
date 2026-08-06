from __future__ import annotations

import json

from query_pipeline.config.models import PipelineConfig
from query_pipeline.io.jsonl import write_jsonl
from query_pipeline.pipeline.context import PipelineContext, RunSummary, merge_stats
from query_pipeline.steps.llm_label import run_llm_label_stage
from query_pipeline.steps.rules_stage import run_rules_stage


def run_pipeline(config: PipelineConfig) -> RunSummary:
    ctx = PipelineContext(config=config)
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)

    ctx = run_rules_stage(ctx)
    ctx = run_llm_label_stage(ctx)

    accepted_path = ctx.output_dir / config.output.accepted
    non_complex_path = ctx.output_dir / config.output.non_complex
    rejected_path = ctx.output_dir / config.output.rejected
    skipped_path = ctx.output_dir / config.output.skipped
    summary_path = ctx.output_dir / config.output.summary

    write_jsonl(accepted_path, ctx.records)
    write_jsonl(non_complex_path, ctx.non_complex)
    write_jsonl(rejected_path, ctx.rejected)
    write_jsonl(skipped_path, ctx.skipped)

    stats = merge_stats(ctx)
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return RunSummary(
        success=True,
        name=config.name,
        stats=stats,
        output_files={
            "accepted": str(accepted_path),
            "non_complex": str(non_complex_path),
            "rejected": str(rejected_path),
            "skipped": str(skipped_path),
            "summary": str(summary_path),
        },
    )
