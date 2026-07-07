from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from query_pipeline.io.jsonl import read_jsonl, write_jsonl
from query_pipeline.pipeline.context import PipelineContext
from query_pipeline.rules.complexity import complexity_score
from query_pipeline.rules.finance import finance_reject_reason
from query_pipeline.rules.normalize import normalize_question
from query_pipeline.rules.reject import CONTEXT_MARKER_RE, generic_reject_reason


def load_input_records(ctx: PipelineContext) -> list[dict[str, Any]]:
    source = ctx.config.input.path.stem
    records: list[dict[str, Any]] = []
    for record in read_jsonl(ctx.config.input.path):
        text_field = ctx.text_field
        raw_question = record.get(text_field, "")
        question = normalize_question(raw_question)
        records.append(
            {
                **record,
                "question": question,
                "source": record.get("source", source),
                "line_number": record.get("_line_number", record.get("line_number")),
            }
        )
    return records


def run_clean_step(ctx: PipelineContext) -> PipelineContext:
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    input_records = load_input_records(ctx)
    passed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for record in input_records:
        question = record["question"]
        reason = generic_reject_reason(question, min_length=ctx.config.rules.min_length)
        if not reason and ctx.config.rules.finance_semantic:
            reason = finance_reject_reason(question)
        if reason:
            rejected.append({**record, "reject_reason": reason, "pipeline_version": ctx.config.pipeline_version})
            continue
        score, reasons = complexity_score(question)
        passed.append(
            {
                **record,
                "complexity_score": score,
                "complexity_reasons": reasons,
                "context_risk": bool(CONTEXT_MARKER_RE.search(question)),
                "cleaning_version": ctx.config.rules.cleaning_version,
                "pipeline_version": ctx.config.pipeline_version,
            }
        )

    ctx.records = passed
    ctx.rejected.extend(rejected)
    ctx.stats["clean_input_rows"] = len(input_records)
    ctx.stats["clean_passed_rows"] = len(passed)
    ctx.stats["clean_rejected_rows"] = len(rejected)
    write_jsonl(ctx.path("clean_passed.jsonl"), passed)
    write_jsonl(ctx.path("rejected_clean.jsonl"), rejected)
    return ctx
