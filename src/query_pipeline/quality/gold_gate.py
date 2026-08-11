from __future__ import annotations

import json
from pathlib import Path
from typing import Any


GOLD_DIR = Path(__file__).resolve().parents[1] / "templates" / "gold"
SUPPORTED_DATASETS = {"aime", "iwencai"}


def load_complex_policy_gold(dataset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized = dataset.strip().lower()
    if normalized not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(f"complex policy gold 不支持 dataset={dataset!r}；可选：{supported}")
    return (
        _read_jsonl(GOLD_DIR / f"{normalized}_positive.jsonl"),
        _read_jsonl(GOLD_DIR / f"{normalized}_negative.jsonl"),
    )


def evaluate_complex_policy_gold(
    records: list[dict[str, Any]], dataset: str
) -> dict[str, Any]:
    """Evaluate the exact release contract against a replay's cleaned rows.

    Positives must be present and hard. Negatives may be normal or absent, but
    none may be hard. Identity is trace_id; text mismatches are also failures so
    a reused identifier cannot accidentally satisfy the gate.
    """
    positives, negatives = load_complex_policy_gold(dataset)
    output_by_id = {
        str(row.get("trace_id") or ""): row
        for row in records
        if str(row.get("trace_id") or "")
    }

    missed_positive_ids: list[str] = []
    positive_text_mismatch_ids: list[str] = []
    for gold in positives:
        trace_id = str(gold["trace_id"])
        output = output_by_id.get(trace_id)
        if output is None or output.get("difficulty_level") != "hard":
            missed_positive_ids.append(trace_id)
            continue
        if _question(output) != _question(gold):
            positive_text_mismatch_ids.append(trace_id)

    false_accept_ids: list[str] = []
    negative_text_mismatch_ids: list[str] = []
    for gold in negatives:
        trace_id = str(gold["trace_id"])
        output = output_by_id.get(trace_id)
        if output is None or output.get("difficulty_level") != "hard":
            continue
        false_accept_ids.append(trace_id)
        if _question(output) != _question(gold):
            negative_text_mismatch_ids.append(trace_id)

    positive_total = len(positives)
    positive_accepted = positive_total - len(missed_positive_ids) - len(
        positive_text_mismatch_ids
    )
    positive_recall = positive_accepted / positive_total if positive_total else 1.0
    passed = (
        positive_recall == 1.0
        and not false_accept_ids
        and not positive_text_mismatch_ids
        and not negative_text_mismatch_ids
    )
    return {
        "passed": passed,
        "positive_total": positive_total,
        "positive_accepted": positive_accepted,
        "positive_recall": positive_recall,
        "negative_total": len(negatives),
        "negative_false_accepts": len(false_accept_ids),
        "missed_positive_ids": missed_positive_ids[:100],
        "positive_text_mismatch_ids": positive_text_mismatch_ids[:100],
        "negative_false_accept_ids": false_accept_ids[:100],
        "negative_text_mismatch_ids": negative_text_mismatch_ids[:100],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"gold row must be an object: {path}")
            rows.append(row)
    return rows


def _question(row: dict[str, Any]) -> str:
    inp = row.get("input")
    return str(inp.get("text") or "") if isinstance(inp, dict) else ""
