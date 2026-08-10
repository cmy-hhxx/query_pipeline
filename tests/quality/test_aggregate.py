"""Aggregate report: status mapping, record keys, overview counts."""

from __future__ import annotations

import unittest
from typing import Any

from query_pipeline.quality import aggregate
from query_pipeline.quality.api import overview
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

class AggregateTest(unittest.TestCase):
    def test_status_mapping(self) -> None:
        records = [
            _row(trace_id="pass"),
            _row(trace_id="fail", category="99-bad"),
            _row(trace_id="low"),
            _row(trace_id="err"),
            _row(trace_id="ok"),
        ]
        k = {r["trace_id"]: aggregate.record_key(r) for r in records}
        per_record = {aggregate.record_key(r): check_record(r) for r in records}
        sample_set = {k[t] for t in ("low", "err", "ok")}
        judge_results = {
            k["low"]: {"trace_id": k["low"], "question_quality": "low", "label_ok": False, "reason": "低质", "error": None},
            k["err"]: {"trace_id": k["err"], "question_quality": None, "label_ok": None, "reason": "", "error": "boom"},
            k["ok"]: {"trace_id": k["ok"], "question_quality": "high", "label_ok": True, "reason": "好", "error": None},
        }
        results = aggregate.build_results(records, per_record, sample_set, judge_results)
        by_status = {r["trace_id"]: r["status"] for r in results}
        self.assertEqual(by_status["pass"], "pass")
        self.assertEqual(by_status["fail"], "fail")       # rule fail dominates
        self.assertEqual(by_status["low"], "needs_review")  # LLM flag
        self.assertEqual(by_status["err"], "needs_review")  # judge error
        self.assertEqual(by_status["ok"], "pass")            # LLM clean

    def test_record_key_includes_source_case_id(self) -> None:
        a = _row(trace_id="t1", source_case_id="case_a")
        b = _row(trace_id="t1", source_case_id="case_b")
        self.assertNotEqual(aggregate.record_key(a), aggregate.record_key(b))
        self.assertEqual(aggregate.record_key(a), "case_a|t1")

    def test_record_key_disambiguates_duplicate_trace_ids(self) -> None:
        # 第四轮 #2：同 (source_case_id|trace_id) 的重复行必须键唯一——
        # per_record / sample_set / judge_results 都是 dict，键塌缩会让第二行
        # 覆盖第一行的规则结果、judge 判定与 sampled 标志。
        a = _row(trace_id="t1", source_case_id="case_a")
        b = _row(trace_id="t1", source_case_id="case_a")
        a["_line_number"] = 3
        b["_line_number"] = 7
        self.assertNotEqual(aggregate.record_key(a), aggregate.record_key(b))
        # 无 _line_number 的手造行保持旧键格式
        c = _row(trace_id="t1", source_case_id="case_a")
        self.assertEqual(aggregate.record_key(c), "case_a|t1")

    def test_duplicate_trace_ids_keep_independent_results(self) -> None:
        # 重复行各自判定：r1 规则失败（fail）、r2 规则通过（pass），互不覆盖。
        r1 = _row(trace_id="t1", category="99-bad")
        r2 = _row(trace_id="t1")
        r1["_line_number"] = 1
        r2["_line_number"] = 2
        records = [r1, r2]
        per_record = {aggregate.record_key(r): check_record(r) for r in records}
        results = aggregate.build_results(records, per_record, set(), {})
        self.assertEqual([r["status"] for r in results], ["fail", "pass"])

    def test_overview_counts_and_flagged(self) -> None:
        records = [_row(trace_id="t1"), _row(trace_id="t2", category="99-bad")]
        per_record = {aggregate.record_key(r): check_record(r) for r in records}
        results = aggregate.build_results(records, per_record, set(), {})
        dataset_rules = run_dataset_rules(records)
        overview = aggregate.build_overview(
            records, results, dataset_rules,
            dataset="aime", date="0807", source="s", ratio=0.0, seed=42,
        )
        self.assertEqual(overview["total"], 2)
        self.assertEqual(overview["status_counts"]["fail"], 1)
        self.assertEqual(overview["status_counts"]["pass"], 1)
        self.assertEqual(len(overview["flagged"]), 1)
        self.assertEqual(overview["flagged"][0]["trace_id"], "t2")

