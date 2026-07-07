from __future__ import annotations

from collections import defaultdict
from typing import Any

from query_pipeline.dedup.exact import dedup_key, sha1_12
from query_pipeline.io.jsonl import write_jsonl
from query_pipeline.pipeline.context import PipelineContext


def run_dedup_exact_step(ctx: PipelineContext) -> PipelineContext:
    if not ctx.config.dedup.exact:
        return ctx

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected_blank: list[dict[str, Any]] = []

    for record in ctx.records:
        key = dedup_key(record["question"])
        if not key:
            rejected_blank.append({**record, "reject_reason": "blank_dedup_key"})
            continue
        groups[key].append(record)

    kept: list[dict[str, Any]] = []
    for key, group in groups.items():
        canonical = group[0]
        key_hash = sha1_12(key)
        kept_record = {
            **canonical,
            "id": canonical.get("id") or f"q_{key_hash}",
            "dedup_key_hash": key_hash,
            "duplicate_count": len(group),
        }
        kept.append(kept_record)
        if len(group) > 1:
            for duplicate_rank, duplicate in enumerate(group[1:], start=2):
                ctx.rejected.append(
                    {
                        **duplicate,
                        "reject_reason": "duplicate_exact",
                        "duplicate_of_id": kept_record["id"],
                        "duplicate_rank": duplicate_rank,
                        "dedup_key_hash": key_hash,
                    }
                )

    ctx.rejected.extend(rejected_blank)
    ctx.records = kept
    ctx.stats["dedup_exact_rows"] = len(kept)
    ctx.stats["dedup_exact_rejected_rows"] = len(groups) - len(kept) + len(rejected_blank)
    write_jsonl(ctx.path("dedup_exact.jsonl"), kept)
    return ctx
