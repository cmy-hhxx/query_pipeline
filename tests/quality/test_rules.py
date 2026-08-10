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

    def test_chain_end2end_skips_chain_requirement(self) -> None:
        # end2end 行合法无 chain：只要求 tools 非空
        row = _row(capture_mode="end2end", chain=[], tools=["web_search"])
        rules = _rules_by_name(row)
        self.assertTrue(rules["chain"]["ok"])

    def test_chain_end2end_requires_tools(self) -> None:
        row = _row(capture_mode="end2end", chain=[], tools=[])
        rules = _rules_by_name(row)
        self.assertFalse(rules["chain"]["ok"])

    def test_chain_malformed_hop(self) -> None:
        rules = _rules_by_name(_row(chain=[{"plan": "", "tools": "oops"}]))
        self.assertFalse(rules["chain"]["ok"])

    def test_answer_empty(self) -> None:
        # 空回答由 answer 规则负责；truncation 与 answer_gate 同一判定（单源），
        # 不再对同一缺陷重复报两条。
        rules = _rules_by_name(_row(text_answer="", raw_answer=""))
        self.assertFalse(rules["answer"]["ok"])
        self.assertTrue(rules["truncation"]["ok"])

    def test_truncation_no_question_passes(self) -> None:
        # 第四轮 #7：无 input.text 的行以未完结标点结尾 → gate 放行（无问句无法
        # 判断回答相对什么不完整），QC 必须一致放行（旧实现 QC 无条件判 fail）。
        row = _row(text_answer="分析了很多指标，结论是，", raw_answer="分析了很多指标，结论是，")
        row["input"] = None
        rules = _rules_by_name(row)
        self.assertTrue(rules["truncation"]["ok"])

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

    def test_english_question_translate_failed_fail_open(self) -> None:
        # 翻译失败是故意 fail-open（translate.py 落 meta.translate_failed 标记）：
        # null + 失败标记可接受；"从未翻译"（null 且无标记）仍判 fail。
        row = _row(input={"text": "What drives GPIQ performance?"})
        row["translation"] = None
        row["meta"] = {"reason": "r", "translate_failed": True}
        rules = _rules_by_name(row)
        self.assertTrue(rules["meta"]["ok"])

    def test_english_question_never_translated_fails(self) -> None:
        # 无失败标记的非中文行 translation=null：必须判 fail（管线没跑 translate 或漏翻）
        row = _row(input={"text": "What drives GPIQ performance?"})
        row["translation"] = None
        row["meta"] = {"reason": "r"}
        rules = _rules_by_name(row)
        self.assertFalse(rules["meta"]["ok"])

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

    def test_category_skew_complex_id_not_falsely_zero(self) -> None:
        # 仅 complex 09 有记录：09 不能出现在"零记录类别"里（旧实现 split("-")[0]
        # 得到 "complex"，误报 09；normal 01 的 id 还会"覆盖"complex 01）。
        rows = [
            _row(trace_id="t1", category="complex-topic/09-action-output"),
            _row(trace_id="t2", category="complex-topic/09-action-output", difficulty_level="hard"),
        ]
        rules = {r["rule"]: r for r in run_dataset_rules(rows)}
        skew = rules["category_skew"]
        self.assertNotIn("09", skew["detail"])
        # complex 01-08 仍然如实报零
        self.assertIn("零记录类别", skew["detail"])
        self.assertIn("01", skew["detail"])

    def test_category_skew_normal_id_does_not_cover_complex(self) -> None:
        # 仅 normal 01 有记录：complex 01 仍是零记录类别（id 碰撞不得串扰）。
        rows = [
            _row(trace_id="t1", category="01-event-and-concept-stock-selection", difficulty_level="normal"),
            _row(trace_id="t2", category="01-event-and-concept-stock-selection", difficulty_level="normal"),
        ]
        rules = {r["rule"]: r for r in run_dataset_rules(rows)}
        skew = rules["category_skew"]
        self.assertIn("零记录类别", skew["detail"])
        self.assertIn("01", skew["detail"])

