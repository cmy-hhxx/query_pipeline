from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from query_pipeline.config.models import PipelineConfig

if TYPE_CHECKING:
    from query_pipeline.io.business_log import BusinessLogWriter


@dataclass
class PipelineContext:
    config: PipelineConfig
    rows: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    sessions: list[Any] = field(default_factory=list)
    segments: dict[str, list[Any]] = field(default_factory=dict)
    candidates: dict[str, list[int]] = field(default_factory=dict)
    business_writer: BusinessLogWriter | None = None
    stream_business_rows: bool = False

    @property
    def work_dir(self) -> Path:
        return self.config.work_dir or self.config.output.dir

    @property
    def output_dir(self) -> Path:
        return self.config.output.dir

    def path(self, name: str) -> Path:
        return self.work_dir / "runtime" / "diagnostics" / name

    def prune_debug_artifacts(self, *names: str) -> None:
        # When intermediates aren't dumped, don't let a prior run's debug artifact
        # masquerade as this run's.
        if self.config.debug.dump_intermediates:
            return
        for name in names:
            self.path(name).unlink(missing_ok=True)


@dataclass
class RunSummary:
    success: bool
    name: str
    stats: dict[str, Any]
    output_files: dict[str, str]
    logs: dict[str, Any]

    def model_dump_json(self, *, indent: int | None = None, ensure_ascii: bool = False) -> str:
        import json

        return json.dumps(
            {
                "success": self.success,
                "name": self.name,
                "stats": self.stats,
                "output_files": self.output_files,
                "logs": self.logs,
            },
            ensure_ascii=ensure_ascii,
            indent=indent,
        )


_STATS_DEFAULTS: dict[str, Any] = {
    "total_sessions": 0,
    "input_bad_lines": 0,
    "input_duplicates": 0,
    "input_empty_sessions": 0,
    "segments": 0,
    "candidates": 0,
    "complex_rows": 0,
    "normal_rows": 0,
    "value_rejected": 0,
    "llm_failed": 0,
    "empty_sessions": 0,
    "session_errors": 0,
    "category_counts": {},
    "category_counts_normal": {},
    "verify_kept": 0,
    "verify_rejected": 0,
    "verify_failed": 0,
    "answer_gate_rejected": 0,
    "simple_gate_rejected": 0,
    "dedup_removed": 0,
    "translated": 0,
    "translate_skipped": 0,
    "translate_failed": 0,
}


def merge_stats(ctx: PipelineContext) -> dict[str, Any]:
    # Stage subsets (custom stage lists) must not produce summary dicts with
    # missing keys — downstream consumers read the same fields regardless.
    return {**_STATS_DEFAULTS, **ctx.stats, "output_rows": len(ctx.rows)}
