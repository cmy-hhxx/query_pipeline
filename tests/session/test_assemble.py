"""Session-row assembly: context fallback and output field mapping."""

from __future__ import annotations

import unittest
from typing import Any

from query_pipeline.adapters.session import adapt_session
from query_pipeline.models.session import Segment
from query_pipeline.session.assemble import assemble_row

def _make_turn(
    idx: int,
    question: str,
    *,
    answer: str | None = None,
    tool_names: str = "",
    tool_count: int = 0,
    chain: list[dict[str, Any]] | None = None,
    status: str | None = "completed",
    outcome: str | None = "success",
) -> dict[str, Any]:
    return {
        "question": question,
        "answer": answer if answer is not None else f"answer{idx} " + "x" * 60,
        "run_id": f"r{idx}",
        "trace_id": f"trace{idx}",
        "request_time": f"2026-08-05 04:{idx:02d}:00",
        "user_id": f"u{idx}",
        "status": status,
        "outcome": outcome,
        "tool_names": tool_names,
        "tool_count": tool_count,
        "first_token_ms": idx * 100,
        "total_duration_ms": idx * 100 + 200,
        "chain": chain if chain is not None else [],
    }

def _chain_with_tool_calls(n: int, name: str = "t") -> list[dict[str, Any]]:
    return [{"plan": "", "tools": [{"name": name, "input": {}, "output": "x"} for _ in range(n)]}]

def _chain_with_steps(n: int, names: tuple[str, ...] = ("t",)) -> list[dict[str, Any]]:
    return [
        {"plan": "", "tools": [{"name": names[i % len(names)], "input": {}, "output": "x"}]} for i in range(n)
    ]

def _sample_turns() -> list[dict[str, Any]]:
    names = ("web_search", "finquery", "compute")
    return [
        _make_turn(0, "Q1 简单查询", tool_names="web_search", tool_count=1, chain=_chain_with_tool_calls(1)),
        _make_turn(1, "Q2 复杂取数", tool_names="web_search,finquery,compute", tool_count=8, chain=_chain_with_steps(8, names)),
        _make_turn(2, "Q3 复杂预测", tool_names="web_search,finquery,compute", tool_count=8, chain=_chain_with_steps(8, names)),
        _make_turn(3, "好的", tool_names="", tool_count=0),
    ]

class AssembleRowTest(unittest.TestCase):
    def test_assemble_row_context_fallback(self) -> None:
        turns = _sample_turns()
        # Segment-leading turn (idx == segment.start), not session-first:
        # context falls back to every earlier session turn.
        segment = Segment(start=2, end=3, topic="topic")
        row = assemble_row(adapt_session({"thread_id": "t1", "context": turns}), segment, idx=2, category_id="03")
        self.assertEqual(
            row["context"],
            [{"question": "Q1 简单查询", "answer": "answer0 " + "x" * 60}, {"question": "Q2 复杂取数", "answer": "answer1 " + "x" * 60}],
        )
        # Session-first turn: context stays empty (nothing precedes it).
        first = assemble_row(adapt_session({"thread_id": "t1", "context": turns}), Segment(start=0, end=3, topic="topic"), idx=0, category_id="03")
        self.assertEqual(first["context"], [])

    def test_assemble_row_field_mapping(self) -> None:
        turns = _sample_turns()
        segment = Segment(start=0, end=3, topic="topic")
        row = assemble_row(adapt_session({"thread_id": "t1", "context": turns}), segment, idx=2, category_id="03", reason="需要多步分析")

        self.assertEqual(row["capture_mode"], "full_link")  # turn 带 chain
        self.assertEqual(row["user_cohort"], "regular")
        self.assertEqual(row["source_case_id"], "t1")
        self.assertEqual(row["trace_id"], "trace2")  # original input turn's trace_id
        self.assertEqual(row["category"], "complex-topic/03-analysis-research")
        self.assertEqual(row["input"]["text"], "Q3 复杂预测")
        self.assertEqual(row["context"], [{"question": "Q1 简单查询", "answer": "answer0 " + "x" * 60}, {"question": "Q2 复杂取数", "answer": "answer1 " + "x" * 60}])
        self.assertEqual(row["tools"], ["web_search", "finquery", "compute"])
        self.assertEqual(row["raw_answer"], "answer2 " + "x" * 60)
        self.assertEqual(row["text_answer"], "answer2 " + "x" * 60)
        self.assertEqual(row["user_id"], "u2")
        self.assertEqual(row["difficulty_level"], "hard")
        self.assertEqual(
            row["meta"],
            {"reason": "需要多步分析", "request_time": "2026-08-05 04:02:00", "run_id": "r2", "last_event_type": None},
        )
        self.assertIsNone(row["translation"])  # 中文原文 → null
        self.assertEqual(row["first_token_time_ms"], 200)
        self.assertEqual(row["finish_answer_time_ms"], 400)
        self.assertFalse(any(k in row["context"][0] for k in ("chain", "tools", "run_id")))

        # 无 chain 的 turn → capture_mode=end2end
        row_no_chain = assemble_row(adapt_session({"thread_id": "t1", "context": turns}), segment, idx=3, category_id="01", reason="r")
        self.assertEqual(row_no_chain["capture_mode"], "end2end")

