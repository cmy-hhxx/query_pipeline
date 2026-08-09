"""QC rule checks: per-record rules, dataset-level rules, answer-gate mirror."""

from __future__ import annotations

import unittest
from typing import Any

from query_pipeline.quality.rules import check_record, run_dataset_rules

def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "capture_mode": "full_link",
        "user_cohort": "regular",
        "source_case_id": "thread_a",
        "answer_key": "",
        "trace_id": "trace_001",
        "category": "complex-topic/03-analysis-research",
        "input": {
            "text": "今天8月3日，分析金安国际这只股票，从k线、市盈率、换手率等指标分析明天是否可以重仓？",
            "image": "",
            "file": "",
        },
        "context": [],
        "chain": [
            {
                "plan": "先识别股票代码",
                "tools": [
                    {"name": "stock_ner_parse", "input": {"query": "金安国际"}, "output": '{"code":"600318"}'}
                ],
            }
        ],
        "tools": ["stock_ner_parse"],
        "raw_answer": "金安国际今日走势稳健，市盈率处于合理区间，换手率适中，但主力筹码集中度仍需观察。"
        "建议明日轻仓试仓，不宜重仓。",
        "text_answer": "金安国际今日走势稳健，市盈率处于合理区间，换手率适中，但主力筹码集中度仍需观察。"
        "建议明日轻仓试仓，不宜重仓。",
        "multimodal": [],
        "model_version": "",
        "release_id": "",
        "agent_mode": "",
        "translation": None,
        "user_id": "u1",
        "difficulty_level": "hard",
        "first_token_time_ms": 1000,
        "finish_answer_time_ms": 2000,
        "input_tokens": 100,
        "output_tokens": 50,
        "request_time_ms": 1785854845000,
        "meta": {
            "reason": "需要多指标综合分析",
            "request_time": "2026-08-04 10:47:25",
        },
    }
    row.update(overrides)
    return row

def _rules_by_name(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["rule"]: item for item in check_record(row)}

class RuleTest(unittest.TestCase):
    def test_valid_row_passes_all(self) -> None:
        for item in check_record(_row()):
            self.assertTrue(item["ok"], f"{item['rule']} 应通过: {item['detail']}")

    def test_category_invalid_slug(self) -> None:
        rules = _rules_by_name(_row(category="03-wrong-slug"))
        self.assertFalse(rules["category"]["ok"])

    def test_category_unknown_id(self) -> None:
        rules = _rules_by_name(_row(category="99-action-output"))
        self.assertFalse(rules["category"]["ok"])

    def test_question_too_short(self) -> None:
        rules = _rules_by_name(_row(input={"text": "好的"}))
        self.assertFalse(rules["question"]["ok"])

    def test_question_too_long(self) -> None:
        rules = _rules_by_name(_row(input={"text": "长" * 3000}))
        self.assertFalse(rules["question"]["ok"])

    def test_chain_empty(self) -> None:
        rules = _rules_by_name(_row(chain=[]))
        self.assertFalse(rules["chain"]["ok"])

    def test_chain_malformed_hop(self) -> None:
        rules = _rules_by_name(_row(chain=[{"plan": "", "tools": "oops"}]))
        self.assertFalse(rules["chain"]["ok"])

    def test_answer_empty(self) -> None:
        rules = _rules_by_name(_row(text_answer="", raw_answer=""))
        self.assertFalse(rules["answer"]["ok"])
        self.assertFalse(rules["truncation"]["ok"])

    def test_answer_too_short(self) -> None:
        rules = _rules_by_name(_row(text_answer="太短", raw_answer="太短"))
        self.assertFalse(rules["answer"]["ok"])

    def test_truncation_dangling_punctuation(self) -> None:
        rules = _rules_by_name(_row(text_answer="分析了很多指标，结论是，", raw_answer="分析了很多指标，结论是，"))
        self.assertFalse(rules["truncation"]["ok"])

    def test_timing_inverted(self) -> None:
        rules = _rules_by_name(_row(first_token_time_ms=5000, finish_answer_time_ms=1000))
        self.assertFalse(rules["timing"]["ok"])

    def test_negative_tokens(self) -> None:
        rules = _rules_by_name(_row(input_tokens=-3))
        self.assertFalse(rules["timing"]["ok"])

    def test_meta_missing_reason(self) -> None:
        rules = _rules_by_name(_row(meta={"request_time": "t"}))
        self.assertFalse(rules["meta"]["ok"])

    def test_zh_question_translation_must_be_null(self) -> None:
        # 中文问句不需要翻译：translation 应为 null，回填原文/译文都算错
        rules = _rules_by_name(_row(translation="今天8月3日，分析金安国际这只股票…"))
        self.assertFalse(rules["meta"]["ok"])

    def test_english_question_requires_translation(self) -> None:
        row = _row(input={"text": "What drives GPIQ performance?"})
        row["translation"] = None
        rules = _rules_by_name(row)
        self.assertFalse(rules["meta"]["ok"])

    def test_english_question_with_translation_ok(self) -> None:
        row = _row(input={"text": "What drives GPIQ performance?"})
        row["translation"] = "是什么驱动了 GPIQ 的表现？"
        rules = _rules_by_name(row)
        self.assertTrue(rules["meta"]["ok"])

class DatasetRuleTest(unittest.TestCase):
    def test_constant_field_detected(self) -> None:
        rows = [_row(trace_id="t1"), _row(trace_id="t2", category="06-investment-decision")]
        rows[1]["text_answer"] = rows[0]["text_answer"]
        rows[1]["raw_answer"] = rows[0]["raw_answer"]
        rules = {r["rule"]: r for r in run_dataset_rules(rows)}
        self.assertFalse(rules["constant_field"]["ok"])

    def test_near_duplicate_detected(self) -> None:
        rows = [
            _row(trace_id="t1", category="03-analysis-research"),
            _row(trace_id="t2", category="06-investment-decision"),
        ]
        # identical question text -> token-Jaccard 1.0
        rows[1]["input"] = dict(rows[0]["input"])
        rules = {r["rule"]: r for r in run_dataset_rules(rows)}
        self.assertFalse(rules["near_duplicate"]["ok"])

    def test_unknown_fields_ignore_underscore(self) -> None:
        rows = [_row(), dict(_row(), **{"_line_number": 7, "legacy_run_id": "x"})]
        rules = {r["rule"]: r for r in run_dataset_rules(rows)}
        self.assertTrue(rules["unknown_fields"]["ok"])
        self.assertEqual(rules["unknown_fields"]["evidence"], ["legacy_run_id"])

