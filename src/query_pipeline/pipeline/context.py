from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
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
    "value_rejected_no_task": 0,
    "value_rejected_not_self_contained": 0,
    "value_rejected_template": 0,
    "value_rejected_other": 0,
    "llm_failed": 0,
    "empty_sessions": 0,
    "session_errors": 0,
    "category_counts": {},
    "category_counts_normal": {},
    "complexity_rejected": 0,
    "complex_feature_counts_initial": {},
    "verify_complex_kept": 0,
    "verify_to_normal": 0,
    "verify_rejected_template": 0,
    "template_family_rejected": 0,
    "template_family_rejected_rows": 0,
    "template_family_candidates": 0,
    "template_family_duplicates": 0,
    "template_family_failed": 0,
    "duplicate_removed": 0,
    "verify_uncertain": 0,
    "verify_failed": 0,
    "final_complex_rows": 0,
    "final_normal_rows": 0,
    "complex_feature_counts_final": {},
    "category_counts_final": {},
    "category_counts_normal_final": {},
    "answer_gate_rejected": 0,
    "dedup_removed": 0,
    "semantic_dedup_candidates": 0,
    "semantic_dedup_removed": 0,
    "semantic_dedup_failed": 0,
    "translated": 0,
    "translate_skipped": 0,
    "translate_failed": 0,
}


def merge_stats(ctx: PipelineContext) -> dict[str, Any]:
    # Stage subsets (custom stage lists) must not produce summary dicts with
    # missing keys — downstream consumers read the same fields regardless.
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
    final_stats = {
        "final_complex_rows": sum(row.get("difficulty_level") == "hard" for row in ctx.rows),
        "final_normal_rows": sum(row.get("difficulty_level") == "normal" for row in ctx.rows),
        "complex_feature_counts_final": dict(sorted(final_complex_features.items())),
        "category_counts_final": dict(sorted(final_complex_categories.items())),
        "category_counts_normal_final": dict(sorted(final_normal_categories.items())),
    }
    return {
        **_STATS_DEFAULTS,
        **ctx.stats,
        **final_stats,
        "output_rows": len(ctx.rows),
    }
