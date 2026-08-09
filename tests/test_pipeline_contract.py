from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_pipeline.config.loader import load_pipeline_config
from query_pipeline.config.models import RuleGateConfig, VerifyConfig
from query_pipeline.llm.cache import make_cache_key
from query_pipeline.models.session import Segment, parse_segment_response
from query_pipeline.pipeline.runner import run_pipeline
from query_pipeline.prompts import resolve_prompt
from query_pipeline.prompts import resolve_prompt as _resolve
from query_pipeline.session.assemble import assemble_row
from query_pipeline.session.candidates import chain_steps, chain_tool_calls, select_candidates
from query_pipeline.adapters.chat import adapt_chat
from query_pipeline.adapters.session import adapt_turn, adapt_session
from query_pipeline.session.judge import build_judge_payload
from query_pipeline.session.segment import _segments_from_cache
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
        "answer": answer if answer is not None else f"answer{idx}",
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


class SessionPipelineContractTest(unittest.TestCase):
    def test_default_config_loads(self) -> None:
        cfg = load_pipeline_config(ROOT / "configs/aime/config.yaml")

        self.assertEqual(cfg.name, "session_pipeline")
        self.assertEqual(cfg.input.path, (ROOT / "data/aime/0807.jsonl").resolve())
        self.assertEqual(cfg.input.format, "auto")
        self.assertEqual(cfg.output.complex_queries, "complex_queries_0807.jsonl")
        self.assertEqual(cfg.llm.base_url_env, "OPENAI_BASE_URL")
        self.assertEqual(cfg.llm.api_key_env, "OPENAI_API_KEY")
        self.assertEqual(cfg.rule_gate.min_chain_tool_calls, 7)
        self.assertEqual(cfg.rule_gate.min_chain_steps, 1)
        self.assertEqual(cfg.rule_gate.min_unique_tools, 2)
        self.assertEqual(cfg.judge.complexity_prompt, "complexity_gate")

    def test_input_format_validation(self) -> None:
        from query_pipeline.config.models import InputConfig

        self.assertEqual(InputConfig(path=Path("x.jsonl")).format, "auto")
        self.assertEqual(InputConfig(path=Path("x.jsonl"), format="chat").format, "chat")
        self.assertEqual(InputConfig(path=Path("x.jsonl"), format="auto").format, "auto")
        with self.assertRaises(ValueError):
            InputConfig(path=Path("x.jsonl"), format="bogus")

    def test_segment_cache_partial_coverage_rejected(self) -> None:
        # A locally-contiguous but partial cache must be rejected as a miss (re-call LLM)
        # rather than pass and later IndexError inside segment_of.
        partial = {"segments": [{"start": 0, "end": 2, "topic": "t"}]}
        with self.assertRaises(ValueError):
            _segments_from_cache(partial, num_turns=5)
        # Full coverage still accepted.
        full = {"segments": [{"start": 0, "end": 2, "topic": "a"}, {"start": 3, "end": 4, "topic": "b"}]}
        self.assertEqual(len(_segments_from_cache(full, num_turns=5)), 2)

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
        self.assertFalse(_run_success({**clean, "llm_failed": 1}))
        self.assertFalse(_run_success({**clean, "session_errors": 1}))
        self.assertFalse(_run_success({**clean, "total_sessions": 0}))
        self.assertFalse(_run_success({**clean, "input_bad_lines": 10}))
        # fail-open stages and empty-but-clean output do not fail the run.
        self.assertTrue(_run_success({**clean, "verify_failed": 1, "translate_failed": 1}))

    def test_adapt_chat(self) -> None:
        record = {
            "trace_id": "t1",
            "question": "当前问句",
            "judge_data": {
                "case_id": "c1",
                "trace_id": "t1",
                "input": {"text": "当前问句", "image": None, "file": None},
                "context": [{"question": "前文1", "answer": "a1"}, {"question": "前文2", "answer": "a2"}],
                "chain": [{"plan": "", "tools": [{"name": "Search", "input": {}, "output": "x"}]}],
                "raw_answer": "raw",
                "text_answer": "text",
                "meta": {"session_round": 3, "request_time": "2026-08-05 04:02:00", "first_token_time_cost": 10},
            },
        }
        session = adapt_chat(record)
        self.assertEqual(session.thread_id, "c1")
        self.assertEqual(len(session.turns), 3)
        self.assertEqual(session.turns[0].question, "前文1")
        self.assertEqual(session.turns[0].answer, "a1")
        current = session.turns[2]
        self.assertEqual(current.question, "当前问句")
        self.assertEqual(current.answer, "text")  # text_answer preferred over raw_answer
        self.assertEqual(current.answer_full, "raw")  # raw_answer 独立保留
        self.assertEqual(current.trace_id, "t1")
        self.assertEqual(current.first_token_ms, 10)
        self.assertEqual(current.request_time, "2026-08-05 04:02:00")
        self.assertEqual(session.candidate_mode, "last_only")

    def test_adapt_chat_missing_wrapper(self) -> None:
        with self.assertRaises(ValueError):
            adapt_chat({"question": "x"})

    def test_prompt_contracts(self) -> None:
        segment_prompt = resolve_prompt("segment")
        self.assertIn("segments", segment_prompt)
        self.assertIn("start", segment_prompt)
        self.assertIn("end", segment_prompt)
        self.assertIn("topic", segment_prompt)
        self.assertIn("同一个主题不能再次出现", segment_prompt)
        self.assertIn("宏观", segment_prompt)

        judge_prompt = resolve_prompt("complex_judge")
        for category_id, name in {
            "01": "复杂取数计算",
            "05": "资产配置",
            "07": "策略触发任务类",
            "09": "动作类",
        }.items():
            self.assertIn(f"{category_id} {name}", judge_prompt)
        self.assertIn("is_complex", judge_prompt)
        self.assertIn("category_id", judge_prompt)
        self.assertIn("reason", judge_prompt)
        # category definitions + priority rules embedded (guards 08/09 boundary collapse)
        self.assertIn("长期帮我盯着并迭代", judge_prompt)
        self.assertIn("→ 优先 09", judge_prompt)
        # few_shot.md examples fused in (07 remapped to trigger/setup semantics;
        # backtest-audit questions fall under 03 now)
        self.assertIn("每类典型示例", judge_prompt)
        self.assertIn("回测一个基于5周均线的短线择时策略", judge_prompt)
        self.assertIn("审计一个多因子策略", judge_prompt)
        self.assertIn("07/08/09 边界", judge_prompt)
        # screening caliber: pure filters are non-complex; validation+trend-point tasks stay 01
        self.assertIn("仅按显式条件过滤", judge_prompt)
        self.assertIn("validate the BAR columns", judge_prompt)

    def test_cache_key_versioned_by_prompt(self) -> None:
        q = "question"
        base = make_cache_key(q, step="s", model="m")
        self.assertEqual(base, make_cache_key(q, step="s", model="m"))
        # a prompt change must invalidate the cached label, or prompt edits never take effect
        self.assertNotEqual(base, make_cache_key(q, step="s", model="m", prompt="v1"))
        self.assertNotEqual(
            make_cache_key(q, step="s", model="m", prompt="v1"),
            make_cache_key(q, step="s", model="m", prompt="v2"),
        )

    def test_segment_parser_merges_recurring_topics(self) -> None:
        raw = json.dumps(
            {
                "segments": [
                    {"start": 0, "end": 1, "topic": "A"},
                    {"start": 2, "end": 3, "topic": "B"},
                    {"start": 4, "end": 4, "topic": "A"},
                ]
            },
            ensure_ascii=False,
        )

        segments = parse_segment_response(raw, num_turns=5)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].start, 0)
        self.assertEqual(segments[0].end, 4)
        self.assertEqual(segments[0].topic, "A")

    def test_segment_parser_keeps_distinct_topics(self) -> None:
        raw = json.dumps(
            {"segments": [{"start": 0, "end": 2, "topic": "A"}, {"start": 3, "end": 4, "topic": "B"}]},
            ensure_ascii=False,
        )

        segments = parse_segment_response(raw, num_turns=5)

        self.assertEqual([(s.start, s.end, s.topic) for s in segments], [(0, 2, "A"), (3, 4, "B")])

    def test_segment_parser_repairs_small_boundary_slips(self) -> None:
        # LLM dropped index 35 (gap) and ended at 54 instead of 55: both are
        # off-by-one slips that must be snapped into a valid covering, not
        # thrown away (this was silently degrading 56-turn sessions to 1).
        raw = json.dumps(
            {
                "segments": [
                    {"start": 0, "end": 3, "topic": "A"},
                    {"start": 4, "end": 18, "topic": "B"},
                    {"start": 19, "end": 34, "topic": "C"},
                    {"start": 36, "end": 46, "topic": "D"},
                    {"start": 47, "end": 54, "topic": "E"},
                ]
            }
        )
        segments = parse_segment_response(raw, num_turns=56)
        self.assertEqual(
            [(s.start, s.end, s.topic) for s in segments],
            [(0, 3, "A"), (4, 18, "B"), (19, 34, "C"), (35, 46, "D"), (47, 55, "E")],
        )

    def test_segment_parser_rejects_malformed(self) -> None:
        with self.assertRaises(ValueError):
            parse_segment_response(json.dumps({"segments": []}), num_turns=5)
        with self.assertRaises(ValueError):
            parse_segment_response(json.dumps({"segments": [{"start": 2, "end": 3, "topic": "A"}]}), num_turns=5)
        with self.assertRaises(ValueError):
            parse_segment_response(
                json.dumps({"segments": [{"start": 0, "end": 1, "topic": "A"}, {"start": 1, "end": 2, "topic": "B"}]}),
                num_turns=5,
            )

    def test_funnel_parsers(self) -> None:
        from query_pipeline.session.funnel import (
            parse_classify_response,
            parse_complexity_response,
            parse_value_response,
        )

        valuable = parse_value_response(json.dumps({"is_valuable": True, "reason": "金融相关"}))
        self.assertTrue(valuable.is_valuable)

        complex_result = parse_complexity_response(
            json.dumps({"is_complex": True, "reason": "需要多步分析"}, ensure_ascii=False)
        )
        self.assertTrue(complex_result.is_complex)

        classified = parse_classify_response(json.dumps({"category_id": "03", "reason": "需要多步分析"}, ensure_ascii=False))
        self.assertEqual(classified.category_id, "03")

        with self.assertRaises(ValueError):
            parse_classify_response(json.dumps({"category_id": "99", "reason": "x"}))

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

    def test_assemble_row_context_fallback(self) -> None:
        turns = _sample_turns()
        # Segment-leading turn (idx == segment.start), not session-first:
        # context falls back to every earlier session turn.
        segment = Segment(start=2, end=3, topic="topic")
        row = assemble_row(adapt_session({"thread_id": "t1", "context": turns}), segment, idx=2, category_id="03")
        self.assertEqual(
            row["context"],
            [{"question": "Q1 简单查询", "answer": "answer0"}, {"question": "Q2 复杂取数", "answer": "answer1"}],
        )
        # Session-first turn: context stays empty (nothing precedes it).
        first = assemble_row(adapt_session({"thread_id": "t1", "context": turns}), Segment(start=0, end=3, topic="topic"), idx=0, category_id="03")
        self.assertEqual(first["context"], [])

    def test_judge_payload_context_fallback(self) -> None:
        turns = _sample_turns()
        payload = build_judge_payload([adapt_turn(t) for t in turns], Segment(start=2, end=3, topic="t"), 2)
        self.assertEqual(payload["prior_questions"], ["Q1 简单查询", "Q2 复杂取数"])
        first = build_judge_payload([adapt_turn(t) for t in turns], Segment(start=0, end=3, topic="t"), 0)
        self.assertEqual(first["prior_questions"], [])
        # non-boundary turn keeps same-segment prior only
        same = build_judge_payload([adapt_turn(t) for t in turns], Segment(start=0, end=3, topic="t"), 1)
        self.assertEqual(same["prior_questions"], ["Q1 简单查询"])

    def test_step1_skips_ineligible_turns(self) -> None:
        names = ("web_search", "finquery", "compute")
        turns = [
            _make_turn(0, "没有回答的复杂问题", tool_names="web_search,finquery,compute", tool_count=8, answer="", chain=_chain_with_steps(8, names)),
            _make_turn(1, "失败状态", tool_names="web_search,finquery,compute", tool_count=8, status="failed", chain=_chain_with_steps(8, names)),
            _make_turn(2, "正常复杂问题", tool_names="web_search,finquery,compute", tool_count=8, chain=_chain_with_steps(8, names)),
        ]
        self.assertEqual(select_candidates([adapt_turn(t) for t in turns], RuleGateConfig()), [2])

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
        self.assertEqual(row["context"], [{"question": "Q1 简单查询", "answer": "answer0"}, {"question": "Q2 复杂取数", "answer": "answer1"}])
        self.assertEqual(row["tools"], ["web_search", "finquery", "compute"])
        self.assertEqual(row["raw_answer"], "answer2")
        self.assertEqual(row["text_answer"], "answer2")
        self.assertEqual(row["user_id"], "u2")
        self.assertEqual(row["difficulty_level"], "hard")
        self.assertEqual(
            row["meta"],
            {"reason": "需要多步分析", "request_time": "2026-08-05 04:02:00", "run_id": "r2"},
        )
        self.assertIsNone(row["translation"])  # 中文原文 → null
        self.assertEqual(row["first_token_time_ms"], 200)
        self.assertEqual(row["finish_answer_time_ms"], 400)
        self.assertFalse(any(k in row["context"][0] for k in ("chain", "tools", "run_id")))

        # 无 chain 的 turn → capture_mode=end2end
        row_no_chain = assemble_row(adapt_session({"thread_id": "t1", "context": turns}), segment, idx=3, category_id="01", reason="r")
        self.assertEqual(row_no_chain["capture_mode"], "end2end")

    def test_end_to_end_produces_complex_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session = {"thread_id": "t1", "context": _sample_turns()}
            _write_jsonl(tmp_path / "input.jsonl", [session])
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=True))

            with patch("query_pipeline.pipeline.runner.LLMClient", FakeSessionLLMClient):
                summary = run_pipeline(cfg)

            self.assertTrue(summary.success)
            self.assertEqual(summary.stats["total_sessions"], 1)
            self.assertEqual(summary.stats["segments"], 2)
            self.assertEqual(summary.stats["candidates"], 2)
            self.assertEqual(summary.stats["complex_rows"], 1)
            self.assertEqual(summary.stats["normal_rows"], 1)
            self.assertEqual(summary.stats["non_complex"], 1)
            self.assertEqual(summary.stats["llm_failed"], 0)
            self.assertEqual(summary.stats["verify_kept"], 2)  # hard keep + normal keep
            self.assertEqual(summary.stats["verify_rejected"], 0)
            self.assertEqual(summary.stats["verify_failed"], 0)
            self.assertEqual(summary.stats["category_counts"], {"01": 1})
            self.assertEqual(summary.stats["category_counts_normal"], {"03": 1})

            rows = _read_jsonl(Path(summary.output_files["complex_queries"]))
            self.assertEqual(len(rows), 2)
            row = rows[0]
            self.assertEqual(row["source_case_id"], "t1")
            self.assertEqual(row["trace_id"], "trace1")
            self.assertEqual(row["category"], "complex-topic/01-data-metrics-calculation")
            self.assertEqual(row["input"]["text"], "Q2 复杂取数")
            self.assertEqual(row["context"], [{"question": "Q1 简单查询", "answer": "answer0"}])
            self.assertEqual(row["difficulty_level"], "hard")
            self.assertEqual(row["meta"], {"reason": "多步工具调用取数", "request_time": "2026-08-05 04:01:00", "run_id": "r1"})
            self.assertIsNone(row["translation"])  # 中文原文 → null
            normal = rows[1]
            self.assertEqual(normal["trace_id"], "trace2")
            self.assertEqual(normal["category"], "03-stock-diagnosis-and-data-lookup")
            self.assertEqual(normal["difficulty_level"], "normal")

    def test_end_to_end_llm_failure_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session = {"thread_id": "t1", "context": _sample_turns()}
            _write_jsonl(tmp_path / "input.jsonl", [session])
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=True))

            with patch("query_pipeline.pipeline.runner.LLMClient", FakeFailingSessionLLMClient):
                summary = run_pipeline(cfg)

            self.assertFalse(summary.success)  # llm_failed=1 makes the run a failure (exit 1)
            # segmentation failure -> whole session is one segment; judge failure on turn1 dropped.
            self.assertEqual(summary.stats["segments"], 1)
            self.assertEqual(summary.stats["complex_rows"], 1)
            self.assertEqual(summary.stats["llm_failed"], 1)
            # verify failure is fail-closed: the surviving row is dropped.
            self.assertEqual(summary.stats["verify_failed"], 1)
            self.assertEqual(summary.stats["verify_kept"], 0)
            # run failed and produced nothing -> no output file written
            self.assertFalse(Path(summary.output_files["complex_queries"]).exists())

    def test_end_to_end_value_gate_rejects_context_only_followups(self) -> None:
        # The value gate (first semantic layer) rejects the context-only
        # follow-up "Q3 再看下"; only Q2 reaches complexity/classify/verify.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            names = ("web_search", "finquery", "compute")
            turns = [
                _make_turn(0, "Q1 简单查询", tool_names="web_search", tool_count=1, chain=_chain_with_tool_calls(1)),
                _make_turn(1, "Q2 复杂取数", tool_names="web_search,finquery,compute", tool_count=8, chain=_chain_with_steps(8, names)),
                _make_turn(2, "Q3 再看下", tool_names="web_search,finquery,compute", tool_count=8, chain=_chain_with_steps(8, names)),
            ]
            session = {"thread_id": "t1", "context": turns}
            _write_jsonl(tmp_path / "input.jsonl", [session])
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=True))

            with patch("query_pipeline.pipeline.runner.LLMClient", FakeJudgeThenVerifyClient):
                summary = run_pipeline(cfg)

            self.assertEqual(summary.stats["candidates"], 2)
            self.assertEqual(summary.stats["value_rejected"], 1)
            self.assertEqual(summary.stats["verify_kept"], 1)
            self.assertEqual(summary.stats["verify_rejected"], 0)
            self.assertEqual(summary.stats["verify_failed"], 0)
            self.assertEqual(summary.stats["complex_rows"], 1)

            rows = _read_jsonl(Path(summary.output_files["complex_queries"]))
            self.assertEqual([r["trace_id"] for r in rows], ["trace1"])

            verified = _read_jsonl(tmp_path / "work/verified.jsonl")
            self.assertEqual([v["trace_id"] for v in verified], ["trace1"])
            self.assertEqual(verified[0]["is_complex"], True)

    def test_verify_config_rounds(self) -> None:
        self.assertEqual(VerifyConfig().max_rounds_hard, 5)
        self.assertEqual(VerifyConfig().max_rounds_normal, 2)
        self.assertEqual(VerifyConfig(max_rounds_hard=1).max_rounds_hard, 1)
        with self.assertRaises(ValueError):
            VerifyConfig(max_rounds_hard=0)

    def test_cache_key_verify_rounds_distinct(self) -> None:
        q = "question"
        base = make_cache_key(q, step="verify:verify_complex", model="m", prompt=_resolve("verify_complex"))
        r2 = make_cache_key(q, step="verify:verify_recheck", model="m", prompt=_resolve("verify_recheck").format(round_no=2))
        r3 = make_cache_key(q, step="verify:verify_recheck", model="m", prompt=_resolve("verify_recheck").format(round_no=3))
        self.assertNotEqual(base, r2)
        self.assertNotEqual(r2, r3)
        self.assertNotEqual(base, r3)

    def test_verify_multi_round_cascade(self) -> None:
        # Cascade filter: "reject r1" drops in round 1, "reject r2" survives
        # round 1 but drops in round 2, "keep" survives all 3 rounds. Round 2/3
        # are distinct LLM calls (not cache replays) — proven by the recorded
        # per-question round sequence.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            names = ("web_search", "finquery", "compute")
            turns = [
                _make_turn(0, "帮我分析贵州茅台的估值并给出买卖建议", tool_names="web_search,finquery,compute", tool_count=8, chain=_chain_with_steps(8, names)),
                _make_turn(1, "分析一下宁德时代的三季度业绩并预测走势", tool_names="web_search,finquery,compute", tool_count=8, chain=_chain_with_steps(8, names)),
                _make_turn(2, "计算比亚迪过去五年的平均市盈率", tool_names="web_search,finquery,compute", tool_count=8, chain=_chain_with_steps(8, names)),
            ]
            session = {"thread_id": "t1", "context": turns}
            _write_jsonl(tmp_path / "input.jsonl", [session])
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=True))

            clients: list[FakeVerifyCascadeLLMClient] = []

            def factory(config: object) -> FakeVerifyCascadeLLMClient:
                client = FakeVerifyCascadeLLMClient(config)
                clients.append(client)
                return client

            with patch("query_pipeline.pipeline.runner.LLMClient", factory):
                summary = run_pipeline(cfg)
            client = clients[0]

            self.assertEqual(summary.stats["verify_kept"], 1)
            self.assertEqual(summary.stats["verify_rejected"], 2)
            self.assertEqual(summary.stats["verify_failed"], 0)
            self.assertEqual(
                client.rounds_called,
                {
                    "帮我分析贵州茅台的估值并给出买卖建议": [1, 2, 3],
                    "分析一下宁德时代的三季度业绩并预测走势": [1],
                    "计算比亚迪过去五年的平均市盈率": [1, 2],
                },
            )

            rows = _read_jsonl(Path(summary.output_files["complex_queries"]))
            self.assertEqual([r["input"]["text"] for r in rows], ["帮我分析贵州茅台的估值并给出买卖建议"])

            verified = _read_jsonl(tmp_path / "work/verified.jsonl")
            self.assertEqual([v["trace_id"] for v in verified], ["trace0", "trace1", "trace2"])
            by_q = {v["question"]: v for v in verified}
            self.assertTrue(by_q["帮我分析贵州茅台的估值并给出买卖建议"]["is_complex"])
            self.assertEqual([r["round"] for r in by_q["帮我分析贵州茅台的估值并给出买卖建议"]["rounds"]], [1, 2, 3])
            self.assertFalse(by_q["分析一下宁德时代的三季度业绩并预测走势"]["is_complex"])
            self.assertEqual([r["round"] for r in by_q["分析一下宁德时代的三季度业绩并预测走势"]["rounds"]], [1])
            self.assertFalse(by_q["计算比亚迪过去五年的平均市盈率"]["is_complex"])
            self.assertEqual(
                [(r["round"], r["is_complex"]) for r in by_q["计算比亚迪过去五年的平均市盈率"]["rounds"]],
                [(1, True), (2, False)],
            )

    def test_verify_multi_round_error_mid_round_fail_closed(self) -> None:
        # Round 2 of the "flaky" question raises: fail-closed drops the row
        # (verify_failed=1) and checkpoints the failure, so a re-run replays
        # both rows from the checkpoint with zero LLM calls.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            names = ("web_search", "finquery", "compute")
            turns = [
                _make_turn(0, "帮我构建一个沪深300的增强策略并回测", tool_names="web_search,finquery,compute", tool_count=8, chain=_chain_with_steps(8, names)),
                _make_turn(1, "分析一下中概股的估值并给出配置建议", tool_names="web_search,finquery,compute", tool_count=8, chain=_chain_with_steps(8, names)),
            ]
            session = {"thread_id": "t1", "context": turns}
            _write_jsonl(tmp_path / "input.jsonl", [session])
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=True))

            clients: list[FakeVerifyMidRoundErrorLLMClient] = []

            def factory(config: object) -> FakeVerifyMidRoundErrorLLMClient:
                client = FakeVerifyMidRoundErrorLLMClient(config)
                clients.append(client)
                return client

            with patch("query_pipeline.pipeline.runner.LLMClient", factory):
                summary = run_pipeline(cfg)
            client = clients[0]

            self.assertEqual(summary.stats["verify_kept"], 1)
            self.assertEqual(summary.stats["verify_failed"], 1)
            self.assertEqual(summary.stats["verify_rejected"], 0)
            self.assertEqual(
                client.rounds_called,
                {
                    "帮我构建一个沪深300的增强策略并回测": [1, 2, 3],
                    "分析一下中概股的估值并给出配置建议": [1, 2],
                },
            )

            clients2: list[FakeVerifyMidRoundErrorLLMClient] = []

            def factory2(config: object) -> FakeVerifyMidRoundErrorLLMClient:
                client = FakeVerifyMidRoundErrorLLMClient(config)
                clients2.append(client)
                return client

            with patch("query_pipeline.pipeline.runner.LLMClient", factory2):
                summary2 = run_pipeline(cfg)
            self.assertEqual(summary2.stats["verify_kept"], 1)
            self.assertEqual(summary2.stats["verify_failed"], 1)
            # both rows replay from the checkpoint (errored rows are sticky fail-closed)
            self.assertEqual(len(clients2[0].calls), 0)

            rows = _read_jsonl(Path(summary.output_files["complex_queries"]))
            self.assertEqual(
                {r["input"]["text"] for r in rows},
                {"帮我构建一个沪深300的增强策略并回测"},
            )

    def test_end_to_end_post_stage_dedup_and_translate(self) -> None:
        # Two candidate turns with identical English text: verify keeps both,
        # dedup drops the second, translate fills top-level translation on the
        # survivor, and the client closes inside the same event loop.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            names = ("web_search", "finquery", "compute")
            turns = [
                _make_turn(0, "simple lookup", tool_names="web_search", tool_count=1, chain=_chain_with_tool_calls(1)),
                _make_turn(1, "complex calc A", tool_names="web_search,finquery,compute", tool_count=8, chain=_chain_with_steps(8, names)),
                _make_turn(2, "complex calc A", tool_names="web_search,finquery,compute", tool_count=8, chain=_chain_with_steps(8, names)),
            ]
            session = {"thread_id": "t1", "context": turns}
            _write_jsonl(tmp_path / "input.jsonl", [session])
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=True, post_enabled=True))

            with patch("query_pipeline.pipeline.runner.LLMClient", FakePostStageLLMClient):
                summary = run_pipeline(cfg)

            self.assertEqual(summary.stats["verify_kept"], 2)
            self.assertEqual(summary.stats["dedup_removed"], 1)
            self.assertEqual(summary.stats["translated"], 1)
            self.assertEqual(summary.stats["translate_skipped"], 0)
            self.assertEqual(summary.stats["translate_failed"], 0)
            self.assertEqual(summary.stats["complex_rows"], 2)  # both candidates judged hard
            self.assertEqual(summary.stats["output_rows"], 1)  # after dedup

            rows = _read_jsonl(Path(summary.output_files["complex_queries"]))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["trace_id"], "trace1")
            self.assertEqual(rows[0]["translation"], "翻译：complex calc A")

    def test_end_to_end_chat_format(self) -> None:
        # input.format=chat: each line is a single-case question with a
        # pre-assembled prior context. No segmentation or step1 heuristics — the
        # trailing turn is judged directly and its context is all prior turns.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            case = {
                "trace_id": "t1",
                "question": "Q2 复杂取数",
                "judge_data": {
                    "case_id": "c1",
                    "trace_id": "t1",
                    "input": {"text": "Q2 复杂取数", "image": None, "file": None},
                    "context": [{"question": "Q1 简单查询", "answer": "answer0"}],
                    "chain": [{"plan": "", "tools": [{"name": "web_search", "input": {}, "output": "x"}]}],
                    "raw_answer": "raw_answer",
                    "text_answer": "text_answer",
                    "meta": {"session_round": 2, "request_time": "2026-08-05 04:01:00"},
                },
            }
            _write_jsonl(tmp_path / "input.jsonl", [case])
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=True, input_format="chat"))

            with patch("query_pipeline.pipeline.runner.LLMClient", FakeSessionLLMClient):
                summary = run_pipeline(cfg)

            self.assertTrue(summary.success)
            self.assertEqual(summary.stats["total_sessions"], 1)
            self.assertEqual(summary.stats["segments"], 1)  # single whole-session segment, no segmentation call
            self.assertEqual(summary.stats["candidates"], 1)
            self.assertEqual(summary.stats["complex_rows"], 1)
            self.assertEqual(summary.stats["verify_kept"], 1)

            rows = _read_jsonl(Path(summary.output_files["complex_queries"]))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["source_case_id"], "c1")
            self.assertEqual(row["trace_id"], "t1")
            self.assertEqual(row["category"], "complex-topic/01-data-metrics-calculation")
            self.assertEqual(row["input"]["text"], "Q2 复杂取数")
            self.assertEqual(row["context"], [{"question": "Q1 简单查询", "answer": "answer0"}])
            self.assertEqual(row["tools"], ["web_search"])
            self.assertEqual(row["raw_answer"], "raw_answer")
            self.assertEqual(row["text_answer"], "text_answer")
            self.assertEqual(row["meta"], {"reason": "多步工具调用取数", "request_time": "2026-08-05 04:01:00", "run_id": ""})

    def test_end_to_end_chat_respects_tool_gates(self) -> None:
        # Chat records always carry judge_data.chain, so the chain/tool AND-gates
        # apply to chat too: a single-tool-call trailing turn is filtered.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            names = ("web_search", "finquery", "compute")
            single = {
                "trace_id": "t1",
                "question": "Q2 复杂取数",
                "judge_data": {
                    "case_id": "c1",
                    "trace_id": "t1",
                    "input": {"text": "Q2 复杂取数", "image": None, "file": None},
                    "context": [{"question": "Q1 简单查询", "answer": "answer0"}],
                    "chain": [{"plan": "", "tools": [{"name": "web_search", "input": {}, "output": "x"}]}],
                    "raw_answer": "raw_answer",
                    "text_answer": "text_answer",
                    "meta": {"session_round": 2, "request_time": "2026-08-05 04:01:00"},
                },
            }
            toolful = {
                "trace_id": "t2",
                "question": "Q2 复杂取数",
                "judge_data": {
                    "case_id": "c2",
                    "trace_id": "t2",
                    "input": {"text": "Q2 复杂取数", "image": None, "file": None},
                    "context": [{"question": "Q1 简单查询", "answer": "answer0"}],
                    "chain": _chain_with_steps(8, names),
                    "raw_answer": "raw_answer",
                    "text_answer": "text_answer",
                    "meta": {"session_round": 2, "request_time": "2026-08-05 04:01:00"},
                },
            }
            _write_jsonl(tmp_path / "input.jsonl", [single, toolful])
            cfg = load_pipeline_config(
                _write_config(tmp_path, llm_enabled=True, input_format="chat", step1_enabled=True)
            )

            with patch("query_pipeline.pipeline.runner.LLMClient", FakeSessionLLMClient):
                summary = run_pipeline(cfg)

            self.assertEqual(summary.stats["candidates"], 1)  # only the toolful case
            self.assertEqual(summary.stats["complex_rows"], 1)
            self.assertEqual(summary.stats["verify_kept"], 1)

    def test_end_to_end_chat_empty_answer_no_row(self) -> None:
        # Empty trailing answer (text_answer/raw_answer both empty) is ineligible: no row.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            case = {
                "trace_id": "t1",
                "question": "Q2 复杂取数",
                "judge_data": {
                    "case_id": "c1",
                    "trace_id": "t1",
                    "input": {"text": "Q2 复杂取数", "image": None, "file": None},
                    "context": [{"question": "Q1 简单查询", "answer": "answer0"}],
                    "chain": [{"plan": "", "tools": [{"name": "web_search", "input": {}, "output": "x"}]}],
                    "raw_answer": "",
                    "text_answer": "",
                    "meta": {"session_round": 2, "request_time": "2026-08-05 04:01:00"},
                },
            }
            _write_jsonl(tmp_path / "input.jsonl", [case])
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=True, input_format="chat"))

            with patch("query_pipeline.pipeline.runner.LLMClient", FakeSessionLLMClient):
                summary = run_pipeline(cfg)

            self.assertEqual(summary.stats["candidates"], 0)
            self.assertEqual(summary.stats["complex_rows"], 0)

    def test_llm_disabled_no_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session = {"thread_id": "t1", "context": _sample_turns()}
            _write_jsonl(tmp_path / "input.jsonl", [session])
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=False))

            summary = run_pipeline(cfg)

            self.assertTrue(summary.success)
            self.assertEqual(summary.stats["candidates"], 2)
            self.assertEqual(summary.stats["complex_rows"], 0)
            self.assertEqual(len(_read_jsonl(Path(summary.output_files["complex_queries"]))), 0)

    def test_adapt_failure_lands_in_bad_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            records = [
                {"thread_id": "good", "context": _sample_turns()},
                {"thread_id": "bad", "context": "not-a-list"},  # adapt_session raises ValueError
            ]
            _write_jsonl(tmp_path / "input.jsonl", records)
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=False))

            summary = run_pipeline(cfg)

            self.assertEqual(summary.stats["total_sessions"], 1)
            self.assertEqual(summary.stats["input_bad_lines"], 1)
            bad_lines = _read_jsonl(tmp_path / "work" / "bad_lines.jsonl")
            self.assertTrue(any(r.get("reason") == "adapt_failed" for r in bad_lines))


def _funnel_response(system_prompt: str, payload: dict[str, Any]) -> str | None:
    """Shared fake for the decoupled funnel; returns None when not a funnel call."""
    if "current_question" not in payload:
        return None  # segment / verify / translate payloads handled elsewhere
    if "价值判官" in system_prompt:
        return json.dumps({"is_valuable": True, "reason": "金融相关"}, ensure_ascii=False)
    if "已判定为复杂金融问句" in system_prompt:
        q = payload["current_question"]
        cid = "01" if q == "Q2 复杂取数" else "02"
        reason = "多步工具调用取数" if q == "Q2 复杂取数" else "复杂归类"
        return json.dumps({"category_id": cid, "reason": reason}, ensure_ascii=False)
    if "有价值但非复杂" in system_prompt:
        return json.dumps({"category_id": "03", "reason": "普通归类"}, ensure_ascii=False)
    q = payload["current_question"]
    return json.dumps(
        {"is_complex": q == "Q2 复杂取数", "reason": "判定"}, ensure_ascii=False
    )


class FakeSessionLLMClient:
    def __init__(self, config: object) -> None:
        self.config = config

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        assert system_prompt
        payload = json.loads(user_prompt.split("\n", 1)[1])
        if "questions" in payload:
            n = len(payload["questions"])
            if n <= 2:
                return json.dumps({"segments": [{"start": 0, "end": n - 1, "topic": "topic"}]}, ensure_ascii=False)
            return json.dumps(
                {
                    "segments": [
                        {"start": 0, "end": 1, "topic": "topic_a"},
                        {"start": 2, "end": n - 1, "topic": "topic_b"},
                    ]
                },
                ensure_ascii=False,
            )
        funnel = _funnel_response(system_prompt, payload)
        if funnel is not None:
            return funnel
        # verify (standalone question): keep Q2 complex, others non-complex
        if payload["question"] == "Q2 复杂取数":
            return json.dumps({"is_complex": True, "reason": "自身复杂"}, ensure_ascii=False)
        return json.dumps({"is_complex": False, "reason": "单独看不复杂"}, ensure_ascii=False)

    async def close(self) -> None:
        return None


class FakePostStageLLMClient:
    """Handles segment/judge/verify/translate payloads.

    close() asserts it runs in the same event loop that served complete() —
    a regression guard for the double-asyncio.run client-close bug that
    raised "Event loop is closed" once post_stage was enabled.
    """

    def __init__(self, config: object) -> None:
        self.config = config

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        import asyncio

        self.used_loop = asyncio.get_running_loop()
        payload = json.loads(user_prompt.split("\n", 1)[1])
        if "questions" in payload:
            n = len(payload["questions"])
            return json.dumps({"segments": [{"start": 0, "end": n - 1, "topic": "t"}]}, ensure_ascii=False)
        if "价值判官" in system_prompt:
            return json.dumps({"is_valuable": True, "reason": "金融相关"}, ensure_ascii=False)
        if "已判定为复杂金融问句" in system_prompt:
            return json.dumps({"category_id": "03", "reason": "复杂归类"}, ensure_ascii=False)
        if "有价值但非复杂" in system_prompt:
            return json.dumps({"category_id": "03", "reason": "普通归类"}, ensure_ascii=False)
        if "current_question" in payload:
            return json.dumps({"is_complex": True, "reason": "上下文复杂"}, ensure_ascii=False)
        if "question" in payload:  # verify
            return json.dumps({"is_complex": True, "reason": "自身复杂"}, ensure_ascii=False)
        if "text" in payload:  # translate
            return json.dumps({"translation": "翻译：" + payload["text"]}, ensure_ascii=False)
        raise AssertionError(f"unexpected payload keys: {sorted(payload)}")

    async def close(self) -> None:
        import asyncio

        assert asyncio.get_running_loop() is self.used_loop, "client closed in a different event loop"


class FakeJudgeThenVerifyClient:
    """Value gate rejects the context-only follow-up (Q3); Q2 survives to hard."""

    def __init__(self, config: object) -> None:
        self.config = config

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt.split("\n", 1)[1])
        if "questions" in payload:
            n = len(payload["questions"])
            return json.dumps({"segments": [{"start": 0, "end": n - 1, "topic": "t"}]}, ensure_ascii=False)
        if "价值判官" in system_prompt:  # value gate: reject "Q3 再看下"
            q = payload["current_question"]
            return json.dumps(
                {"is_valuable": not q.startswith("Q3"), "reason": "承接短指令无独立任务" if q.startswith("Q3") else "金融相关"},
                ensure_ascii=False,
            )
        if "已判定为复杂金融问句" in system_prompt:
            return json.dumps({"category_id": "03", "reason": "复杂归类"}, ensure_ascii=False)
        if "有价值但非复杂" in system_prompt:
            return json.dumps({"category_id": "03", "reason": "普通归类"}, ensure_ascii=False)
        if "current_question" in payload:  # complexity gate
            q = payload["current_question"]
            return json.dumps({"is_complex": q == "Q2 复杂取数", "reason": "判定"}, ensure_ascii=False)
        if payload["question"] == "Q2 复杂取数":
            return json.dumps({"is_complex": True, "reason": "自身复杂"}, ensure_ascii=False)
        return json.dumps({"is_complex": False, "reason": "单独看是承接句"}, ensure_ascii=False)

    async def close(self) -> None:
        return None


class FakeFailingSessionLLMClient:
    def __init__(self, config: object) -> None:
        self.config = config

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt.split("\n", 1)[1])
        if "questions" in payload:
            raise RuntimeError("simulated segmentation failure")
        if "价值判官" in system_prompt:
            return json.dumps({"is_valuable": True, "reason": "金融相关"}, ensure_ascii=False)
        if "已判定为复杂金融问句" in system_prompt:
            return json.dumps({"category_id": "02", "reason": "需要预测"}, ensure_ascii=False)
        if "有价值但非复杂" in system_prompt:
            return json.dumps({"category_id": "03", "reason": "普通归类"}, ensure_ascii=False)
        if "current_question" in payload:  # complexity gate
            if payload["current_question"] == "Q2 复杂取数":
                raise RuntimeError("simulated judge failure")
            return json.dumps({"is_complex": True, "reason": "需要预测"}, ensure_ascii=False)
        raise RuntimeError("simulated verify failure")

    async def close(self) -> None:
        return None


_CASCADE_VERDICTS: dict[str, dict[int, bool]] = {
    "帮我分析贵州茅台的估值并给出买卖建议": {1: True, 2: True, 3: True},
    "分析一下宁德时代的三季度业绩并预测走势": {1: False},
    "计算比亚迪过去五年的平均市盈率": {1: True, 2: False},
}


class FakeVerifyCascadeLLMClient:
    """Judge marks every candidate complex; verify flips per question per round.

    Rounds are told apart by the system prompt (round 1 = VERIFY_COMPLEX,
    round >= 2 = formatted VERIFY_RECHECK). The recorded round sequence proves
    rounds 2/3 are genuinely fresh calls, not cache replays.
    """

    def __init__(self, config: object) -> None:
        self.config = config
        self.rounds_called: dict[str, list[int]] = {}

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt.split("\n", 1)[1])
        if "questions" in payload:  # segmentation
            n = len(payload["questions"])
            return json.dumps({"segments": [{"start": 0, "end": n - 1, "topic": "t"}]}, ensure_ascii=False)
        if "价值判官" in system_prompt:
            return json.dumps({"is_valuable": True, "reason": "金融相关"}, ensure_ascii=False)
        if "已判定为复杂金融问句" in system_prompt:
            return json.dumps({"category_id": "03", "reason": "复杂归类"}, ensure_ascii=False)
        if "有价值但非复杂" in system_prompt:
            return json.dumps({"category_id": "03", "reason": "普通归类"}, ensure_ascii=False)
        if "current_question" in payload:
            return json.dumps({"is_complex": True, "reason": "上下文复杂"}, ensure_ascii=False)
        question = payload["question"]
        if system_prompt == _resolve("verify_complex"):
            round_no = 1
        elif system_prompt == _resolve("verify_recheck").format(round_no=2):
            round_no = 2
        elif system_prompt == _resolve("verify_recheck").format(round_no=3):
            round_no = 3
        else:
            raise AssertionError(f"unexpected verify system prompt: {system_prompt[:60]!r}")
        self.rounds_called.setdefault(question, []).append(round_no)
        is_complex = _CASCADE_VERDICTS[question][round_no]
        return json.dumps({"is_complex": is_complex, "reason": f"第{round_no}轮"}, ensure_ascii=False)

    async def close(self) -> None:
        return None


class FakeVerifyMidRoundErrorLLMClient:
    """Round 2 of the 'flaky' question raises (fail-open keeps the row)."""

    def __init__(self, config: object) -> None:
        self.config = config
        self.calls: list[dict[str, Any]] = []
        self.rounds_called: dict[str, list[int]] = {}

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt.split("\n", 1)[1])
        self.calls.append(payload)
        if "questions" in payload:  # segmentation
            n = len(payload["questions"])
            return json.dumps({"segments": [{"start": 0, "end": n - 1, "topic": "t"}]}, ensure_ascii=False)
        if "价值判官" in system_prompt:
            return json.dumps({"is_valuable": True, "reason": "金融相关"}, ensure_ascii=False)
        if "已判定为复杂金融问句" in system_prompt:
            return json.dumps({"category_id": "03", "reason": "复杂归类"}, ensure_ascii=False)
        if "有价值但非复杂" in system_prompt:
            return json.dumps({"category_id": "03", "reason": "普通归类"}, ensure_ascii=False)
        if "current_question" in payload:
            return json.dumps({"is_complex": True, "reason": "上下文复杂"}, ensure_ascii=False)
        question = payload["question"]
        if system_prompt == _resolve("verify_complex"):
            round_no = 1
        elif system_prompt == _resolve("verify_recheck").format(round_no=2):
            round_no = 2
        else:
            round_no = 3
        self.rounds_called.setdefault(question, []).append(round_no)
        if question == "分析一下中概股的估值并给出配置建议" and round_no == 2:
            raise RuntimeError("simulated verify failure")
        return json.dumps({"is_complex": True, "reason": "复杂"}, ensure_ascii=False)

    async def close(self) -> None:
        return None


def _write_config(
    tmp_path: Path,
    *,
    llm_enabled: bool,
    post_enabled: bool = False,
    input_format: str = "session",
    step1_enabled: bool | None = None,
) -> Path:
    config_path = tmp_path / "config.yaml"
    is_chat = input_format == "chat"
    seg_on = "false" if is_chat else "true"
    step1_on = "false" if is_chat else "true"
    if step1_enabled is not None:
        step1_on = "true" if step1_enabled else "false"
    post_block = ""
    if post_enabled:
        post_block = """
            post:
              enabled: true
              dedup:
                enabled: true
                threshold: 0.80
              translate:
                enabled: true
"""
    config_path.write_text(
        textwrap.dedent(
            f"""
            name: test_pipeline
            input:
              path: input.jsonl
              format: {input_format}
            output:
              dir: out
              complex_queries: complex_queries.jsonl
              summary: summary.json
            work_dir: work
            segmentation:
              enabled: {seg_on}
            rule_gate:
              enabled: {step1_on}
              reject_rules: true
              min_chain_tool_calls: 7
              min_chain_steps: 1
              min_unique_tools: 2
            judge:
              enabled: true
            verify:
              enabled: true
              prompt_id: verify_complex
              max_rounds_hard: 3
              max_rounds_normal: 2
            {post_block}
            llm:
              enabled: {str(llm_enabled).lower()}
              base_url_env: OPENAI_BASE_URL
              model: fake-model
              api_key_env: FAKE_API_KEY
              concurrency: 2
              max_retries: 1
              timeout_seconds: 1
              response_format: json_object
              cache: work/llm_cache.jsonl
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
