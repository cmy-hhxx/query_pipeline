from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from query_pipeline.config.models import PipelineConfig


@dataclass
class PipelineContext:
    config: PipelineConfig
    records: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def work_dir(self) -> Path:
        return self.config.work_dir

    @property
    def output_dir(self) -> Path:
        return self.config.output.dir

    @property
    def text_field(self) -> str:
        return self.config.input.text_field

    def path(self, name: str) -> Path:
        return self.work_dir / name


@dataclass
class RunSummary:
    success: bool
    name: str
    pipeline_version: str
    stats: dict[str, Any]
    output_files: dict[str, str]

    def model_dump_json(self, *, indent: int | None = None, ensure_ascii: bool = False) -> str:
        import json

        return json.dumps(
            {
                "success": self.success,
                "name": self.name,
                "pipeline_version": self.pipeline_version,
                "stats": self.stats,
                "output_files": self.output_files,
            },
            ensure_ascii=ensure_ascii,
            indent=indent,
        )


def merge_stats(ctx: PipelineContext) -> dict[str, Any]:
    reject_reasons = Counter(record.get("reject_reason", "unknown") for record in ctx.rejected)
    category_counts = Counter(
        record.get("category_id")
        for record in ctx.records
        if record.get("category_id")
    )
    difficulty_scores = [
        record.get("difficulty_score")
        for record in ctx.records
        if record.get("difficulty_score") is not None
    ]
    return {
        **ctx.stats,
        "final_rows": len(ctx.records),
        "rejected_rows": len(ctx.rejected),
        "skipped_rows": len(ctx.skipped),
        "reject_reasons": dict(reject_reasons),
        "category_counts": dict(category_counts),
        "difficulty_avg": round(sum(difficulty_scores) / len(difficulty_scores), 2) if difficulty_scores else None,
    }
