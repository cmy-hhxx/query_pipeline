from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from query_pipeline.config.models import PipelineConfig


@dataclass
class PipelineContext:
    config: PipelineConfig
    rows: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def work_dir(self) -> Path:
        return self.config.work_dir

    @property
    def output_dir(self) -> Path:
        return self.config.output.dir

    def path(self, name: str) -> Path:
        return self.work_dir / name


@dataclass
class RunSummary:
    success: bool
    name: str
    stats: dict[str, Any]
    output_files: dict[str, str]

    def model_dump_json(self, *, indent: int | None = None, ensure_ascii: bool = False) -> str:
        import json

        return json.dumps(
            {
                "success": self.success,
                "name": self.name,
                "stats": self.stats,
                "output_files": self.output_files,
            },
            ensure_ascii=ensure_ascii,
            indent=indent,
        )


def merge_stats(ctx: PipelineContext) -> dict[str, Any]:
    return {**ctx.stats, "complex_rows": len(ctx.rows)}
