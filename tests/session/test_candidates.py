"""Step-1 candidate selection: chain/tool AND-gates and eligibility filters."""

from __future__ import annotations

import unittest
from typing import Any

from query_pipeline.adapters.session import adapt_turn
from query_pipeline.config.models import RuleGateConfig
from query_pipeline.session.candidates import chain_steps, chain_tool_calls, select_candidates

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

def _chain(*steps: tuple[str, ...]) -> list[dict[str, Any]]:
    """Chain with explicit tool-name list per step."""
    return [{"plan": "", "tools": [{"name": name, "input": {}, "output": "x"} for name in names]} for names in steps]

def _sample_turns() -> list[dict[str, Any]]:
    names = ("web_search", "finquery", "compute")
    return [
        _make_turn(0, "Q1 简单查询", tool_names="web_search", tool_count=1, chain=_chain_with_tool_calls(1)),
        _make_turn(1, "Q2 复杂取数", tool_names="web_search,finquery,compute", tool_count=8, chain=_chain_with_steps(8, names)),
        _make_turn(2, "Q3 复杂预测", tool_names="web_search,finquery,compute", tool_count=8, chain=_chain_with_steps(8, names)),
        _make_turn(3, "好的", tool_names="", tool_count=0),
    ]

class CandidateSelectionTest(unittest.TestCase):
    def test_chainless_turn_passes_via_tool_count(self) -> None:
        cfg = RuleGateConfig()  # defaults: min_chain_tool_calls=7, min_chain_steps=1, min_unique_tools=2
        turns = [
            adapt_turn(
                _make_turn(0, "复杂多步取数计算预测", tool_count=8, tool_names="web_search,finquery,compute")
            )
        ]
        # chain-less but tool_count/tool_names are present: fallback keeps the AND-gates satisfiable.
        self.assertEqual(chain_tool_calls(turns[0]), 8)
        self.assertEqual(chain_steps(turns[0]), 1)
        self.assertIn(0, select_candidates(turns, cfg))

    def test_step1_select_candidates(self) -> None:
        cfg = RuleGateConfig()
        candidates = select_candidates([adapt_turn(t) for t in _sample_turns()], cfg)

        # turn1 and turn2 clear all three AND thresholds (8 tool calls / 8 chain
        # steps / 3 unique tools); turn0 and turn3 do not.
        self.assertEqual(candidates, [1, 2])

    def test_step1_funnel_requires_all_signals(self) -> None:
        cfg = RuleGateConfig()  # AND: tool_calls>=7 AND steps>=1 AND unique>=2
        turns = [
            _make_turn(0, "八次调用四种工具", chain=_chain(("a", "b"), ("a", "b"), ("c", "d"), ("e", "f"))),
            _make_turn(1, "四次调用两种工具", chain=_chain(("a", "b"), ("a", "b"))),
            _make_turn(2, "八次调用一种工具", chain=_chain(("a", "a", "a", "a"), ("a", "a", "a", "a"))),
            _make_turn(3, "没有推理链", chain=[]),
        ]
        # turn1: tool_calls fail; turn2: unique fail; turn3: no chain.
        self.assertEqual(select_candidates([adapt_turn(t) for t in turns], cfg), [0])

    def test_step1_skips_ineligible_turns(self) -> None:
        names = ("web_search", "finquery", "compute")
        turns = [
            _make_turn(0, "没有回答的复杂问题", tool_names="web_search,finquery,compute", tool_count=8, answer="", chain=_chain_with_steps(8, names)),
            _make_turn(1, "失败状态", tool_names="web_search,finquery,compute", tool_count=8, status="failed", chain=_chain_with_steps(8, names)),
            _make_turn(2, "正常复杂问题", tool_names="web_search,finquery,compute", tool_count=8, chain=_chain_with_steps(8, names)),
        ]
        self.assertEqual(select_candidates([adapt_turn(t) for t in turns], RuleGateConfig()), [2])

