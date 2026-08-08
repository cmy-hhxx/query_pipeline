from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from query_pipeline.quality.rules import PER_RECORD_RULES

_STATUSES = ("pass", "fail", "needs_review")


def record_key(row: dict[str, Any]) -> str:
    """Stable per-record identity: source_case_id|trace_id when trace_id present,
    else source line number. Composite key so duplicate trace_ids across sessions
    or reimports fold onto their own rows instead of overwriting one another."""
    trace_id = row.get("trace_id")
    if trace_id:
        return f"{row.get('source_case_id') or ''}|{trace_id}"
    return f"line_{row.get('_line_number', '?')}"


def _judge_payload(judge: dict[str, Any] | None) -> dict[str, Any] | None:
    if judge is None:
        return None
    return {
        "question_quality": judge.get("question_quality"),
        "label_ok": judge.get("label_ok"),
        "reason": judge.get("reason", ""),
        "error": judge.get("error"),
    }


def build_results(
    records: list[dict[str, Any]],
    per_record: dict[str, list[dict[str, Any]]],
    sample_set: set[str],
    judge_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Per-record QC results. Rule failure -> fail (deterministic); LLM flag ->
    needs_review (probabilistic, kept out of the hard-fail bucket)."""
    results: list[dict[str, Any]] = []
    for row in records:
        key = record_key(row)
        rules = per_record.get(key, [])
        failed = [rule for rule in rules if not rule["ok"]]
        judge = judge_results.get(key)
        sampled = key in sample_set
        if failed:
            status = "fail"
        elif sampled and judge is not None and (
            judge.get("question_quality") == "low"
            or judge.get("label_ok") is False
            or judge.get("error")
        ):
            status = "needs_review"
        else:
            status = "pass"
        inp = row.get("input")
        question = inp.get("text") if isinstance(inp, dict) else ""
        results.append(
            {
                # display the real trace_id (join keys are composite in record_key)
                "trace_id": str(row.get("trace_id") or "") or key,
                "source_case_id": str(row.get("source_case_id") or ""),
                "category": str(row.get("category") or ""),
                "question": str(question)[:80],
                "status": status,
                "sampled": sampled,
                "rules": rules,
                "judge": _judge_payload(judge) if sampled else None,
            }
        )
    return results


def build_overview(
    records: list[dict[str, Any]],
    results: list[dict[str, Any]],
    dataset_rules: list[dict[str, Any]],
    *,
    dataset: str,
    date: str,
    source: str,
    ratio: float,
    seed: int,
    bad_lines: int = 0,
    flagged_limit: int = 100,
) -> dict[str, Any]:
    status_counts = {status: 0 for status in _STATUSES}
    for result in results:
        status_counts[result["status"]] += 1

    rule_hits = {rule.name: {"pass": 0, "fail": 0} for rule in PER_RECORD_RULES}
    for result in results:
        for item in result["rules"]:
            if item["rule"] in rule_hits:
                rule_hits[item["rule"]]["pass" if item["ok"] else "fail"] += 1

    category_distribution: dict[str, int] = {}
    for row in records:
        category = str(row.get("category") or "")
        category_distribution[category] = category_distribution.get(category, 0) + 1

    sampled = [result for result in results if result["sampled"]]
    sample = {
        "count": len(sampled),
        "ratio": ratio,
        "seed": seed,
        "question_quality_high": sum(
            1 for result in sampled if (result["judge"] or {}).get("question_quality") == "high"
        ),
        "question_quality_low": sum(
            1 for result in sampled if (result["judge"] or {}).get("question_quality") == "low"
        ),
        "label_ok": sum(1 for result in sampled if (result["judge"] or {}).get("label_ok") is True),
        "label_not_ok": sum(1 for result in sampled if (result["judge"] or {}).get("label_ok") is False),
        "judge_errors": sum(1 for result in sampled if (result["judge"] or {}).get("error")),
    }

    flagged: list[dict[str, Any]] = []
    for result in results:
        if result["status"] == "pass":
            continue
        if result["status"] == "fail":
            failed = [item for item in result["rules"] if not item["ok"]]
            reason = failed[0]["detail"] if failed else ""
        else:
            reason = (result["judge"] or {}).get("error") or (result["judge"] or {}).get("reason", "")
        flagged.append(
            {
                "trace_id": result["trace_id"],
                "source_case_id": result["source_case_id"],
                "category": result["category"],
                "question": result["question"],
                "status": result["status"],
                "reason": str(reason)[:200],
            }
        )
        if len(flagged) >= flagged_limit:
            break

    return {
        "dataset": dataset,
        "date": date,
        "source": source,
        "total": len(results),
        "skipped_bad_lines": bad_lines,
        "status_counts": status_counts,
        "rule_hits": rule_hits,
        "category_distribution": category_distribution,
        "sample": sample,
        "dataset_rules": dataset_rules,
        "flagged": flagged,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
