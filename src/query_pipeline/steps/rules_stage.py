from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from datasketch import MinHashLSH

from query_pipeline.constants import MINHASH_METHOD, MINHASH_NUM_PERM
from query_pipeline.dedup.exact import dedup_key, sha1_12
from query_pipeline.dedup.minhash import build_minhash
from query_pipeline.io.jsonl import read_jsonl, write_jsonl
from query_pipeline.pipeline.context import PipelineContext
from query_pipeline.pipeline.records import (
    OUTPUT_FIELD,
    is_text_scalar,
    normalized_text,
    output_value,
    resolve_dot_path,
    set_pipeline_output,
    source_line_number,
    update_rule_signals,
)
from query_pipeline.rules.complexity import complexity_score
from query_pipeline.rules.finance import finance_reject_reason
from query_pipeline.rules.normalize import normalize_question, question_length_without_punctuation
from query_pipeline.rules.reject import CONTEXT_MARKER_RE, generic_reject_reason

logger = logging.getLogger(__name__)


def run_rules_stage(ctx: PipelineContext) -> PipelineContext:
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    input_records = _load_input_records(ctx)
    candidates = input_records

    if ctx.config.rules_stage.enabled:
        candidates = _apply_clean_rules(ctx, candidates)
        candidates = _apply_exact_dedup(ctx, candidates)
        candidates = _apply_minhash_dedup(ctx, candidates)
        candidates = _apply_complexity_gate(ctx, candidates)
    else:
        logger.warning(
            "rules_stage.enabled=false: skipping clean, dedup, and complexity gate; "
            "only input load/normalize/basic reject will run"
        )

    ctx.records = candidates
    ctx.stats["rules_candidate_rows"] = len(ctx.records)
    write_jsonl(ctx.path("rules_candidates.jsonl"), ctx.records)
    write_jsonl(ctx.path("rules_rejected.jsonl"), ctx.rejected)
    write_jsonl(ctx.path("rules_skipped.jsonl"), ctx.skipped)
    return ctx


def _load_input_records(ctx: PipelineContext) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    source_name = ctx.config.input.path.stem

    for record in read_jsonl(ctx.config.input.path):
        if OUTPUT_FIELD in record:
            line_number = source_line_number(record)
            suffix = f" at line {line_number}" if line_number is not None else ""
            raise ValueError(f"input record{suffix} already contains reserved field {OUTPUT_FIELD!r}")
        base_output = {
            "source_name": record.get("source", source_name),
            "source_text_path": ctx.text_path,
            "source_line_number": source_line_number(record),
        }
        selection = resolve_dot_path(record, ctx.text_path)
        if not selection.found:
            rejected.append(
                set_pipeline_output(
                    record,
                    **base_output,
                    status="rejected",
                    reject_reason="missing_text_path",
                    missing_text_path=ctx.text_path,
                )
            )
            continue
        if not is_text_scalar(selection.value):
            rejected.append(
                set_pipeline_output(
                    record,
                    **base_output,
                    status="rejected",
                    reject_reason="invalid_text_value",
                )
            )
            continue

        source_text = str(selection.value)
        normalized = normalize_question(source_text)
        if not normalized:
            rejected.append(
                set_pipeline_output(
                    record,
                    **base_output,
                    status="rejected",
                    source_text=source_text,
                    normalized_text=normalized,
                    reject_reason="blank",
                )
            )
            continue

        loaded.append(
            set_pipeline_output(
                record,
                **base_output,
                status="pending",
                source_text=source_text,
                normalized_text=normalized,
            )
        )

    ctx.rejected.extend(rejected)
    ctx.stats["input_rows"] = len(loaded) + len(rejected)
    ctx.stats["input_valid_text_rows"] = len(loaded)
    ctx.stats["input_rejected_rows"] = len(rejected)
    return loaded


def _apply_clean_rules(ctx: PipelineContext, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cfg = ctx.config.rules_stage.clean
    passed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for record in records:
        text = normalized_text(record)
        reason = ""
        if cfg.enabled:
            reason = generic_reject_reason(text, min_length=cfg.min_length)
            if not reason and cfg.finance_semantic:
                reason = finance_reject_reason(text)

        score, reasons = complexity_score(text)
        with_signals = update_rule_signals(
            record,
            complexity_score=score,
            complexity_reasons=reasons,
            context_risk=bool(CONTEXT_MARKER_RE.search(text)),
        )
        if reason:
            rejected.append(set_pipeline_output(with_signals, status="rejected", reject_reason=reason))
            continue
        passed.append(with_signals)

    ctx.rejected.extend(rejected)
    ctx.stats["clean_passed_rows"] = len(passed)
    ctx.stats["clean_rejected_rows"] = len(rejected)
    write_jsonl(ctx.path("clean_passed.jsonl"), passed)
    return passed


def _apply_exact_dedup(ctx: PipelineContext, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not ctx.config.rules_stage.exact_dedup.enabled:
        return records

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected_blank: list[dict[str, Any]] = []

    for record in records:
        key = dedup_key(normalized_text(record))
        if not key:
            rejected_blank.append(set_pipeline_output(record, status="rejected", reject_reason="blank_dedup_key"))
            continue
        groups[key].append(record)

    kept: list[dict[str, Any]] = []
    duplicate_rejected = 0
    for key, group in groups.items():
        canonical = group[0]
        key_hash = sha1_12(key)
        canonical_id = str(canonical.get("id") or f"q_{key_hash}")
        kept_record = update_rule_signals(
            canonical,
            exact_dedup_key_hash=key_hash,
            duplicate_count=len(group),
            canonical_id=canonical_id,
        )
        kept.append(kept_record)
        for duplicate_rank, duplicate in enumerate(group[1:], start=2):
            ctx.rejected.append(
                set_pipeline_output(
                    duplicate,
                    status="rejected",
                    reject_reason="duplicate_exact",
                    duplicate_of=canonical_id,
                    duplicate_rank=duplicate_rank,
                    dedup_method="exact",
                    dedup_key_hash=key_hash,
                )
            )
            duplicate_rejected += 1

    ctx.rejected.extend(rejected_blank)
    ctx.stats["exact_dedup_kept_rows"] = len(kept)
    ctx.stats["exact_dedup_rejected_rows"] = duplicate_rejected + len(rejected_blank)
    write_jsonl(ctx.path("dedup_exact.jsonl"), kept)
    return kept


def _apply_minhash_dedup(ctx: PipelineContext, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cfg = ctx.config.rules_stage.minhash
    if not cfg.enabled:
        return records

    lsh = MinHashLSH(threshold=cfg.threshold, num_perm=MINHASH_NUM_PERM)
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    lsh_id_to_business_id: dict[str, str] = {}

    for index, record in enumerate(records):
        text = normalized_text(record)
        business_id = str(
            record.get("id") or output_value(record, "rule_signals", {}).get("canonical_id") or f"row_{index}"
        )
        lsh_key = f"row_{index}"
        minhash = build_minhash(text, num_perm=MINHASH_NUM_PERM)
        candidates = lsh.query(minhash)
        if candidates:
            duplicate_of = lsh_id_to_business_id[str(candidates[0])]
            removed.append(
                set_pipeline_output(
                    record,
                    status="rejected",
                    reject_reason="duplicate_minhash",
                    duplicate_of=duplicate_of,
                    dedup_method="minhash",
                )
            )
            continue
        lsh.insert(lsh_key, minhash)
        lsh_id_to_business_id[lsh_key] = business_id
        kept.append(update_rule_signals(record, minhash_id=business_id))

    report = {
        "method": MINHASH_METHOD,
        "num_perm": MINHASH_NUM_PERM,
        "threshold": cfg.threshold,
        "source_rows": len(records),
        "dedup_rows": len(kept),
        "removed_rows": len(removed),
    }
    ctx.rejected.extend(removed)
    ctx.stats["minhash_report"] = report
    write_jsonl(ctx.path("dedup_minhash.jsonl"), kept)
    ctx.path("minhash_dedup_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return kept


def _apply_complexity_gate(ctx: PipelineContext, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cfg = ctx.config.rules_stage.complexity_gate
    if not cfg.enabled:
        return records

    passed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for record in records:
        text = normalized_text(record)
        rule_signals = output_value(record, "rule_signals", {})
        score = int(rule_signals.get("complexity_score", 0))
        text_length = question_length_without_punctuation(text)
        if score < cfg.min_score or text_length < cfg.min_text_length:
            skipped.append(
                set_pipeline_output(
                    update_rule_signals(record, text_length=text_length),
                    status="skipped",
                    skip_reason="low_complexity_score_or_short",
                )
            )
            continue
        passed.append(update_rule_signals(record, text_length=text_length))

    ctx.skipped.extend(skipped)
    ctx.stats["complexity_gate_passed_rows"] = len(passed)
    ctx.stats["complexity_gate_skipped_rows"] = len(skipped)
    return passed
