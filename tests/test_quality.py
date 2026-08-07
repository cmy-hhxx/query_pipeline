from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_pipeline.quality import aggregate, judge as judge_mod
from query_pipeline.quality.api import overview, record_detail
from query_pipeline.quality.cli import main as cli_main
from query_pipeline.quality.prompts import build_judge_payload
from query_pipeline.quality.rules import check_record, run_dataset_rules


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "capture_mode": "full_link",
        "user_cohort": "regular",
        "source_case_id": "thread_a",
        "answer_key": "",
        "trace_id": "trace_001",
        "category": "03-analysis-research",
        "input": {
            "text": "今天8月3日，分析金安国际这只股票，从k线、市盈率、换手率等指标分析明天是否可以重仓？",
            "image": "",
            "file": "",
        },
        "session_round": 1,
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
        "translation": "",
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
            "translation": "今天8月3日，分析金安国际这只股票…",
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
        rules = _rules_by_name(_row(input={"text": "好的", "image": "", "file": ""}))
        self.assertFalse(rules["question"]["ok"])

    def test_question_too_long(self) -> None:
        rules = _rules_by_name(_row(input={"text": "长" * 3000, "image": "", "file": ""}))
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
        rules = _rules_by_name(_row(meta={"translation": "x", "request_time": "t"}))
        self.assertFalse(rules["meta"]["ok"])

    def test_zh_question_requires_translation(self) -> None:
        rules = _rules_by_name(_row(meta={"reason": "r", "request_time": "t", "translation": ""}))
        self.assertFalse(rules["meta"]["ok"])

    def test_english_question_no_translation_ok(self) -> None:
        row = _row(input={"text": "What drives GPIQ performance?", "image": "", "file": ""})
        row["meta"] = {"reason": "r", "request_time": "t", "translation": ""}
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
        # identical question text -> MinHash Jaccard 1.0
        rows[1]["input"] = dict(rows[0]["input"])
        rules = {r["rule"]: r for r in run_dataset_rules(rows)}
        self.assertFalse(rules["near_duplicate"]["ok"])

    def test_unknown_fields_ignore_underscore(self) -> None:
        rows = [_row(), dict(_row(), **{"_line_number": 7, "legacy_run_id": "x"})]
        rules = {r["rule"]: r for r in run_dataset_rules(rows)}
        self.assertTrue(rules["unknown_fields"]["ok"])
        self.assertEqual(rules["unknown_fields"]["evidence"], ["legacy_run_id"])


class FakeJudgeClient:
    def __init__(self, config: object) -> None:
        self.config = config
        self.calls = 0

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        payload = json.loads(user_prompt.split("\n", 1)[1])
        if payload["question"].startswith("今天"):
            return json.dumps(
                {"question_quality": "high", "label_ok": True, "reason": "清晰完整"}
            )
        return json.dumps(
            {"question_quality": "low", "label_ok": False, "reason": "低质"}
        )


class JudgeTest(unittest.TestCase):
    def _client(self) -> FakeJudgeClient:
        return FakeJudgeClient(SimpleNamespace(model="gpt-5.4-mini"))

    def test_sample_selection_deterministic(self) -> None:
        records = [_row(trace_id=f"t{i}") for i in range(10)]
        first = judge_mod.select_sample(records, ratio=0.3, seed=42)
        second = judge_mod.select_sample(records, ratio=0.3, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_ratio_zero_no_sample(self) -> None:
        records = [_row(trace_id="t1")]
        self.assertEqual(judge_mod.select_sample(records, ratio=0, seed=42), [])

    def test_judge_one_writes_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "llm_cache.jsonl"
            client = self._client()
            verdict = asyncio.run(
                judge_mod.judge_one(
                    _row(),
                    client,
                    {},
                    asyncio.Lock(),
                    cache_path,
                    system_prompt="sys",
                    model="gpt-5.4-mini",
                )
            )
            self.assertEqual(verdict["question_quality"], "high")
            self.assertTrue(verdict["label_ok"])
            self.assertIsNone(verdict["error"])
            lines = cache_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            entry = json.loads(lines[0])
            self.assertEqual(entry["label"]["question_quality"], "high")

    def test_judge_one_cache_hit_avoids_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "llm_cache.jsonl"
            cache: dict[str, dict[str, Any]] = {}
            lock = asyncio.Lock()
            client = self._client()
            row = _row()
            for _ in range(2):
                asyncio.run(
                    judge_mod.judge_one(
                        row, client, cache, lock, cache_path,
                        system_prompt="sys", model="gpt-5.4-mini",
                    )
                )
            self.assertEqual(client.calls, 1)  # second run served from cache

    def test_judge_one_degrades_on_parse_error(self) -> None:
        class GarbageClient(FakeJudgeClient):
            async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
                return "not json"

        with tempfile.TemporaryDirectory() as tmp:
            verdict = asyncio.run(
                judge_mod.judge_one(
                    _row(),
                    GarbageClient(SimpleNamespace(model="m")),
                    {},
                    asyncio.Lock(),
                    Path(tmp) / "c.jsonl",
                    system_prompt="sys",
                    model="m",
                )
            )
            self.assertIsNotNone(verdict["error"])
            self.assertIsNone(verdict["question_quality"])

    def test_run_llm_judge_samples_and_judges(self) -> None:
        records = [
            _row(
                trace_id=f"t{i}",
                input={"text": f"问题{i}：分析标的最优持仓结构并给出配置比例建议", "image": "", "file": ""},
            )
            for i in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "llm_cache.jsonl"
            client = self._client()
            indices, verdicts = asyncio.run(
                judge_mod.run_llm_judge(
                    records, client, {}, asyncio.Lock(), cache_path,
                    ratio=0.5, seed=1, concurrency=4,
                )
            )
            self.assertEqual(len(indices), 4)
            self.assertEqual(len(verdicts), 4)
            self.assertEqual(client.calls, 4)  # 4 distinct questions -> 4 calls
            for verdict in verdicts:
                self.assertIsNone(verdict["error"])


class AggregateTest(unittest.TestCase):
    def test_status_mapping(self) -> None:
        records = [
            _row(trace_id="pass"),
            _row(trace_id="fail", category="99-bad"),
            _row(trace_id="low"),
            _row(trace_id="err"),
            _row(trace_id="ok"),
        ]
        per_record = {r["trace_id"]: check_record(r) for r in records}
        sample_set = {"low", "err", "ok"}
        judge_results = {
            "low": {"trace_id": "low", "question_quality": "low", "label_ok": False, "reason": "低质", "error": None},
            "err": {"trace_id": "err", "question_quality": None, "label_ok": None, "reason": "", "error": "boom"},
            "ok": {"trace_id": "ok", "question_quality": "high", "label_ok": True, "reason": "好", "error": None},
        }
        results = aggregate.build_results(records, per_record, sample_set, judge_results)
        by_status = {r["trace_id"]: r["status"] for r in results}
        self.assertEqual(by_status["pass"], "pass")
        self.assertEqual(by_status["fail"], "fail")       # rule fail dominates
        self.assertEqual(by_status["low"], "needs_review")  # LLM flag
        self.assertEqual(by_status["err"], "needs_review")  # judge error
        self.assertEqual(by_status["ok"], "pass")            # LLM clean

    def test_overview_counts_and_flagged(self) -> None:
        records = [_row(trace_id="t1"), _row(trace_id="t2", category="99-bad")]
        per_record = {r["trace_id"]: check_record(r) for r in records}
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


class ApiTest(unittest.TestCase):
    def _write_run(self, root: Path) -> Path:
        source = root / "outputs" / "aime" / "complex_queries_0807.jsonl"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps(_row(), ensure_ascii=False) + "\n", encoding="utf-8")

        qc = root / "work" / "aime" / "0807" / "qc"
        qc.mkdir(parents=True, exist_ok=True)
        overview_data = {
            "dataset": "aime",
            "date": "0807",
            "source": str(source),
            "total": 1,
            "skipped_bad_lines": 0,
            "status_counts": {"pass": 1, "fail": 0, "needs_review": 0},
        }
        (qc / "overview.json").write_text(json.dumps(overview_data), encoding="utf-8")
        results = [{"trace_id": "trace_001", "status": "pass", "sampled": False, "rules": [], "judge": None}]
        (qc / "results.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in results), encoding="utf-8"
        )
        return qc

    def test_overview_and_record_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root)
            data = overview("aime", "0807", root=root)
            self.assertEqual(data["total"], 1)
            detail = record_detail("aime", "0807", "trace_001", root=root)
            self.assertEqual(detail["record"]["trace_id"], "trace_001")
            self.assertEqual(detail["qc"]["status"], "pass")
            with self.assertRaises(KeyError):
                record_detail("aime", "0807", "nope", root=root)


class CliE2ETest(unittest.TestCase):
    def test_run_no_llm_produces_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "complex_queries.jsonl"
            source.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in [_row(), _row(trace_id="t2", category="99-bad")])
                + "\n",
                encoding="utf-8",
            )
            qc = tmp / "qc"
            rc = cli_main(
                [
                    "run",
                    "--dataset", "aime",
                    "--date", "9999",
                    "--input", str(source),
                    "--qc-dir", str(qc),
                    "--no-llm",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((qc / "overview.json").exists())
            self.assertTrue((qc / "results.jsonl").exists())
            self.assertTrue((qc / "report.md").exists())
            overview_data = json.loads((qc / "overview.json").read_text(encoding="utf-8"))
            self.assertEqual(overview_data["status_counts"]["fail"], 1)
            self.assertIn("## 状态概览", (qc / "report.md").read_text(encoding="utf-8"))

    def test_missing_input_returns_nonzero(self) -> None:
        rc = cli_main(
            ["run", "--dataset", "aime", "--date", "9999", "--input", "/nonexistent/x.jsonl", "--no-llm"]
        )
        self.assertEqual(rc, 1)


class PromptTest(unittest.TestCase):
    def test_payload_has_question_and_label(self) -> None:
        payload = build_judge_payload(_row())
        self.assertIn("question", payload)
        self.assertIn("label", payload)
        self.assertEqual(payload["label_definition"], "03 分析研究（analysis-research）")
        self.assertEqual(payload["chain_hops"], 1)


if __name__ == "__main__":
    unittest.main()
