"""precheck stage：默认顺序最前、critical 中止、可跳过、可放宽 chain。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from query_pipeline import run as api_run
from query_pipeline.pipeline.stages import DEFAULT_STAGES


def _session_lines(n_turns: int = 1, *, with_chain: bool = True) -> str:
    chain = [{"plan": "p", "tools": [{"name": "t", "input": {}, "output": "o"}]}]
    return json.dumps(
        {
            "thread_id": "t1",
            "context": [
                {
                    "question": "帮我分析茅台的估值",
                    "answer": "分析如下，" + "数据" * 30,
                    "trace_id": "tr1",
                    "status": "completed",
                    "outcome": "success",
                    "last_event_type": "runFinished",
                    "chain": chain if with_chain else [],
                    "tool_count": 3,
                }
                for _ in range(n_turns)
            ],
        },
        ensure_ascii=False,
    )


class FakeClient:
    def __init__(self, config: object) -> None:
        self.config = config
        self.calls = 0

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        payload = json.loads(user_prompt.split("\n", 1)[1])
        if "questions" in payload:
            return json.dumps({"segments": [{"start": 0, "end": 0, "topic": "t"}]})
        if "current_question" in payload:
            return json.dumps({"is_valuable": True})
        return json.dumps({"is_complex": True, "reason": "复杂"})

    async def close(self) -> None:
        return None


class PrecheckStageTest(unittest.TestCase):
    def test_precheck_is_first_default_stage(self) -> None:
        self.assertEqual(DEFAULT_STAGES[0], "precheck")

    def test_missing_chain_aborts_before_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "input.jsonl"
            src.write_text(_session_lines(with_chain=False), encoding="utf-8")
            with patch(
                "query_pipeline.pipeline.runner.LLMClient",
                side_effect=AssertionError("LLM client initialized before precheck"),
            ):
                with self.assertRaisesRegex(ValueError, "missing_chain"):
                    api_run(str(src), output_dir=str(tmp_path / "out"), llm_enabled=True)
            # 中止在 precheck：没有 cleaned 输出
            self.assertFalse((tmp_path / "out" / "cleaned_queries.jsonl").exists())

    def test_skip_precheck_runs_through(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "input.jsonl"
            src.write_text(_session_lines(with_chain=False), encoding="utf-8")
            with patch("query_pipeline.pipeline.runner.LLMClient", FakeClient):
                summary = api_run(
                    str(src),
                    output_dir=str(tmp_path / "out"),
                    llm_enabled=True,
                    precheck_enabled=False,
                )
            self.assertTrue(summary["success"])
            self.assertNotIn("precheck", summary["stats"])

    def test_allow_no_chain_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "input.jsonl"
            src.write_text(_session_lines(with_chain=False), encoding="utf-8")
            with patch("query_pipeline.pipeline.runner.LLMClient", FakeClient):
                summary = api_run(
                    str(src),
                    output_dir=str(tmp_path / "out"),
                    llm_enabled=True,
                    precheck_min_chain_coverage=0.0,
                )
            self.assertTrue(summary["success"])
            self.assertEqual(summary["stats"]["precheck"]["chain_coverage"], 0.0)
            self.assertTrue(summary["stats"]["precheck"]["ok"])

    def test_ok_input_records_precheck_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "input.jsonl"
            src.write_text(_session_lines(with_chain=True), encoding="utf-8")
            with patch("query_pipeline.pipeline.runner.LLMClient", FakeClient):
                summary = api_run(str(src), output_dir=str(tmp_path / "out"), llm_enabled=True)
            self.assertTrue(summary["success"])
            pc = summary["stats"]["precheck"]
            self.assertTrue(pc["ok"])
            self.assertEqual(pc["chain_coverage"], 1.0)
            self.assertEqual(pc["issues"], [])


if __name__ == "__main__":
    unittest.main()
