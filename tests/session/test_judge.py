"""Judge payload construction: prior-question context per segment boundary."""

from __future__ import annotations

import unittest
from typing import Any

from query_pipeline.adapters.session import adapt_turn
from query_pipeline.models.session import Segment
from query_pipeline.session.judge import build_judge_payload

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

class JudgePayloadTest(unittest.TestCase):
    def test_judge_payload_context_fallback(self) -> None:
        turns = _sample_turns()
        payload = build_judge_payload([adapt_turn(t) for t in turns], Segment(start=2, end=3, topic="t"), 2)
        self.assertEqual(payload["prior_questions"], ["Q1 简单查询", "Q2 复杂取数"])
        first = build_judge_payload([adapt_turn(t) for t in turns], Segment(start=0, end=3, topic="t"), 0)
        self.assertEqual(first["prior_questions"], [])
        # non-boundary turn keeps same-segment prior only
        same = build_judge_payload([adapt_turn(t) for t in turns], Segment(start=0, end=3, topic="t"), 1)
        self.assertEqual(same["prior_questions"], ["Q1 简单查询"])

