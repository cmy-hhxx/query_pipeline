"""QC rules for answer quality (refusal / event type) mirror the pipeline gate."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_pipeline.quality.rules import check_record

LONG = "这是一段足够长的正常回答。" * 20


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "capture_mode": "full_link",
        "user_cohort": "regular",
        "source_case_id": "a",
        "trace_id": "t1",
        "category": "complex-topic/03-analysis-research",
        "input": {"text": "分析一下某股票的估值并给出买卖建议", "image": "", "file": ""},
        "context": [],
        "chain": [{"plan": "p", "tools": [{"name": "web_search", "input": {}, "output": "o"}]}],
        "tools": ["web_search"],
        "raw_answer": LONG,
        "text_answer": LONG,
        "multimodal": [],
        "translation": None,
        "user_id": "u",
        "difficulty_level": "hard",
        "first_token_time_ms": 100,
        "finish_answer_time_ms": 200,
        "input_tokens": 10,
        "output_tokens": 10,
        "request_time_ms": 1785854845000,
        "meta": {"reason": "r", "request_time": "2026-08-04 10:47:25", "last_event_type": "runFinished"},
    }
    row.update(overrides)
    return row


def _rules(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["rule"]: item for item in check_record(row)}


class QcAnswerRuleTest(unittest.TestCase):
    def test_refusal_rule(self) -> None:
        row = _row(text_answer="抱歉，我无法回答这个问题。", raw_answer="抱歉，我无法回答这个问题。")
        self.assertFalse(_rules(row)["refusal"]["ok"])

    def test_event_type_rule(self) -> None:
        row = _row(meta={"reason": "r", "last_event_type": "runCancelled"})
        self.assertFalse(_rules(row)["event_type"]["ok"])

    def test_event_type_absent_ok(self) -> None:
        row = _row(meta={"reason": "r"})  # chat-style rows have no event field
        self.assertTrue(_rules(row)["event_type"]["ok"])

    def test_clean_row_passes_answer_rules(self) -> None:
        rules = _rules(_row())
        self.assertTrue(rules["refusal"]["ok"])
        self.assertTrue(rules["event_type"]["ok"])
        self.assertTrue(rules["truncation"]["ok"])


if __name__ == "__main__":
    unittest.main()
