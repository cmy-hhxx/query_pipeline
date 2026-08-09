"""suggest_gates: rule-gate threshold scan, ordered by candidate count."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from query_pipeline.suggest import suggest_gates


def _session_line(n_turns: int = 1, tool_calls: int = 8) -> str:
    names = ("web", "fin", "calc")
    chain = [{"plan": "p", "tools": [{"name": names[n % 3], "input": {}, "output": "o"} for n in range(tool_calls)]}]
    return json.dumps(
        {
            "thread_id": "t1",
            "context": [
                {
                    "question": f"帮我分析一下第{i}只股票的走势",
                    "answer": "答" * 60,
                    "trace_id": f"tr{i}",
                    "status": "completed",
                    "outcome": "success",
                    "chain": chain,
                    "tool_count": tool_calls,
                }
                for i in range(n_turns)
            ],
        }
    )


def _chat_line(tool_calls: int = 4) -> str:
    names = ("web", "fin", "calc")
    chain = [{"plan": "p", "tools": [{"name": names[n % 3], "input": {}, "output": "o"} for n in range(tool_calls)]}]
    return json.dumps(
        {
            "trace_id": "x",
            "judge_data": {
                "case_id": "c1",
                "input": {"text": "帮我分析一下某股票的走势并给出操作建议"},
                "context": [{"question": "前文问题", "answer": "答" * 60}],
                "chain": chain,
                "text_answer": "答" * 60,
                "meta": {},
            },
        }
    )


class SuggestGatesTest(unittest.TestCase):
    def test_session_scan_sorted_and_marked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "in.jsonl"
            path.write_text(_session_line(n_turns=10, tool_calls=8) + "\n", encoding="utf-8")
            items = suggest_gates(path, top=10)
            self.assertLessEqual(len(items), 11)  # 10 sampled + maybe default
            counts = [s["candidates"] for s in items]
            self.assertEqual(counts, sorted(counts))  # 按候选数从低到高
            default = next(s for s in items if s["is_default"])
            self.assertEqual(
                (default["min_chain_tool_calls"], default["min_unique_tools"], default["reject_rules"]),
                (7, 2, True),
            )
            # 最严组合候选数最少；最松端有候选
            self.assertLess(items[0]["candidates"], items[-1]["candidates"])
            self.assertGreater(items[-1]["ratio"], 0)

    def test_chat_base_is_record_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "in.jsonl"
            path.write_text(_chat_line(tool_calls=4) + "\n" * 1, encoding="utf-8")
            items = suggest_gates(path, format="chat", top=5)
            self.assertEqual(items[0]["total_turns"], 1)  # 1 条记录 = 1 个候选位
            default = next(s for s in items if s["is_default"])
            self.assertEqual(default["min_chain_tool_calls"], 3)  # chat 默认 3/1/2
            self.assertEqual(default["candidates"], 1)  # 4 次调用 >= 3 且 3 种工具 >= 2 ✓

    def test_empty_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "in.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                suggest_gates(path)


if __name__ == "__main__":
    unittest.main()
