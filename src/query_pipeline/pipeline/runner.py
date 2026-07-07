from __future__ import annotations

import json
from collections.abc import Callable

from query_pipeline.io.jsonl import write_jsonl
from query_pipeline.pipeline.context import PipelineContext, RunSummary, merge_stats
from query_pipeline.steps.clean import run_clean_step
from query_pipeline.steps.dedup_exact import run_dedup_exact_step
from query_pipeline.steps.dedup_minhash import run_dedup_minhash_step
from query_pipeline.steps.llm_classify import run_llm_classify_step
from query_pipeline.steps.llm_difficulty import run_llm_difficulty_step

STEP_REGISTRY: dict[str, Callable[[PipelineContext], PipelineContext]] = {
    "clean": run_clean_step,
    "dedup_exact": run_dedup_exact_step,
    "dedup_minhash": run_dedup_minhash_step,
    "llm_classify": run_llm_classify_step,
    "llm_difficulty": run_llm_difficulty_step,
}


def run_pipeline(config) -> RunSummary:
    ctx = PipelineContext(config=config)
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)

    for step_name in config.steps:
        step_fn = STEP_REGISTRY.get(step_name)
        if step_fn is None:
            raise ValueError(f"unknown step: {step_name}")
        ctx = step_fn(ctx)

    labeled_path = ctx.output_dir / config.output.labeled
    rejected_path = ctx.output_dir / config.output.rejected
    skipped_path = ctx.output_dir / config.output.skipped
    summary_path = ctx.output_dir / config.output.summary

    write_jsonl(labeled_path, ctx.records)
    write_jsonl(rejected_path, ctx.rejected)
    write_jsonl(skipped_path, ctx.skipped)

    stats = merge_stats(ctx)
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return RunSummary(
        success=True,
        name=config.name,
        pipeline_version=config.pipeline_version,
        stats=stats,
        output_files={
            "labeled": str(labeled_path),
            "rejected": str(rejected_path),
            "skipped": str(skipped_path),
            "summary": str(summary_path),
        },
    )
