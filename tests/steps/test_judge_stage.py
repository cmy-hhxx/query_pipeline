"""Judge stage: checkpoint content key and run-success predicate."""

from __future__ import annotations

import unittest
from typing import Any

from query_pipeline.adapters.session import adapt_session
from query_pipeline.steps.judge_stage import session_content_key

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

class JudgeStageContractTest(unittest.TestCase):
    def test_session_content_key_sensitive_to_chain_status_tool_count(self) -> None:
        base = {
            "thread_id": "t1",
            "context": [_make_turn(0, "Q1 复杂查询", tool_names="web_search", tool_count=1)],
        }
        s1 = adapt_session(base)
        # Same Q/A/time but different status must produce a different checkpoint key.
        failed_status = dict(base)
        failed_status["context"] = [
            {**_make_turn(0, "Q1 复杂查询", tool_names="web_search", tool_count=1), "status": "failed"}
        ]
        s2 = adapt_session(failed_status)
        self.assertNotEqual(session_content_key(s1), session_content_key(s2))
        # Chain change alone also changes the key (drives step1 gates).
        chained = dict(base)
        chained["context"] = [
            _make_turn(0, "Q1 复杂查询", tool_names="web_search", tool_count=3, chain=_chain_with_tool_calls(3))
        ]
        s3 = adapt_session(chained)
        self.assertNotEqual(session_content_key(s1), session_content_key(s3))
        # Tool-count change alone (chain absent → step1 falls back to tool_count) also changes the key.
        retooled = dict(base)
        retooled["context"] = [_make_turn(0, "Q1 复杂查询", tool_names="web_search", tool_count=2)]
        s4 = adapt_session(retooled)
        self.assertNotEqual(session_content_key(s1), session_content_key(s4))
        # Identical input replays the same key.
        self.assertEqual(session_content_key(s1), session_content_key(adapt_session(base)))

    def test_run_success_predicate(self) -> None:
        from query_pipeline.pipeline.runner import _run_success

        clean = {
            "total_sessions": 10,
            "input_bad_lines": 0,
            "llm_failed": 0,
            "session_errors": 0,
            "complex_rows": 0,
        }
        self.assertTrue(_run_success(clean))
        # llm_failed is counted, not fatal (deterministic single-candidate failures
        # must not block delivery); session-level errors still fail the run.
        self.assertTrue(_run_success({**clean, "llm_failed": 1}))
        self.assertFalse(_run_success({**clean, "session_errors": 1}))
        self.assertFalse(_run_success({**clean, "total_sessions": 0}))
        # Bad input lines never fail a run that still adapted sessions — the old
        # `input_bad_lines == total_sessions` check failed exactly-half-bad input
        # while passing 80%-bad input (non-monotonic, arbitrary). 全坏行由
        # total_sessions == 0 覆盖；部分坏行走 fail-open（summary/bad_lines 可审计）。
        self.assertTrue(_run_success({**clean, "input_bad_lines": 10}))
        self.assertTrue(_run_success({**clean, "total_sessions": 5, "input_bad_lines": 5}))
        # fail-open stages and empty-but-clean output do not fail the run.
        self.assertTrue(_run_success({**clean, "verify_failed": 1, "translate_failed": 1}))

