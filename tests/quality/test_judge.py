"""LLM judge: deterministic sampling, cache reuse, parse-error degradation."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from query_pipeline.quality import judge as judge_mod

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
                    ratio=0.5, seed=1,
                )
            )
            self.assertEqual(len(indices), 4)
            self.assertEqual(len(verdicts), 4)
            self.assertEqual(client.calls, 4)  # 4 distinct questions -> 4 calls
            for verdict in verdicts:
                self.assertIsNone(verdict["error"])

