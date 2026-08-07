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
from query_pipeline.config.models import Step1Config
from query_pipeline.llm.cache import make_cache_key
from query_pipeline.models.records import ENGLISH_CATEGORIES
from query_pipeline.models.session import Segment, parse_segment_response, parse_step2_response
from query_pipeline.pipeline.runner import run_pipeline
from query_pipeline.prompts import resolve_prompt
from query_pipeline.session.assemble import assemble_row
from query_pipeline.session.candidates import select_candidates
from query_pipeline.session.cases import normalize_judge_data_record
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
        self.assertEqual(cfg.input.format, "session")
        self.assertEqual(cfg.output.complex_queries, "complex_queries_0807.jsonl")
        self.assertEqual(cfg.llm_stage.base_url_env, "OPENAI_BASE_URL")
        self.assertEqual(cfg.llm_stage.api_key_env, "OPENAI_API_KEY")
        self.assertEqual(cfg.session_stage.step1.min_chain_tool_calls, 7)
        self.assertEqual(cfg.session_stage.step1.min_chain_steps, 1)
        self.assertEqual(cfg.session_stage.step1.min_unique_tools, 2)
        self.assertEqual(cfg.session_stage.step2.prompt_id, "complex_judge")

    def test_input_format_validation(self) -> None:
        from query_pipeline.config.models import InputConfig

        self.assertEqual(InputConfig(path=Path("x.jsonl")).format, "session")
        self.assertEqual(InputConfig(path=Path("x.jsonl"), format="judge_data").format, "judge_data")
        with self.assertRaises(ValueError):
            InputConfig(path=Path("x.jsonl"), format="bogus")

    def test_normalize_judge_data_record(self) -> None:
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
        session = normalize_judge_data_record(record)
        self.assertEqual(session["thread_id"], "c1")
        self.assertEqual(len(session["context"]), 3)
        self.assertEqual(session["context"][0], {"question": "前文1", "answer": "a1"})
        current = session["context"][2]
        self.assertEqual(current["question"], "当前问句")
        self.assertEqual(current["answer"], "text")  # text_answer preferred over raw_answer
        self.assertEqual(current["trace_id"], "t1")
        self.assertEqual(current["first_token_ms"], 10)
        self.assertEqual(current["request_time"], "2026-08-05 04:02:00")

    def test_normalize_judge_data_record_missing_wrapper(self) -> None:
        with self.assertRaises(ValueError):
            normalize_judge_data_record({"question": "x"})

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
            "01": "数据与指标计算",
            "05": "资产配置",
            "07": "策略触发与设置",
            "09": "动作输出",
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

    def test_step2_parser(self) -> None:
        complex_result = parse_step2_response(
            json.dumps({"is_complex": True, "category_id": "03", "reason": "需要多步分析"}, ensure_ascii=False)
        )
        self.assertTrue(complex_result.is_complex)
        self.assertEqual(complex_result.category_id, "03")

        non_complex = parse_step2_response(
            json.dumps({"is_complex": False, "category_id": "03", "reason": "简单查询"}, ensure_ascii=False)
        )
        self.assertFalse(non_complex.is_complex)
        self.assertIsNone(non_complex.category_id)

        with self.assertRaises(ValueError):
            parse_step2_response(json.dumps({"is_complex": True, "category_id": "99", "reason": "x"}))

    def test_step1_select_candidates(self) -> None:
        cfg = Step1Config()
        candidates = select_candidates(_sample_turns(), cfg)

        # turn1 and turn2 clear all three AND thresholds (8 tool calls / 8 chain
        # steps / 3 unique tools); turn0 and turn3 do not.
        self.assertEqual(candidates, [1, 2])

    def test_step1_funnel_requires_all_signals(self) -> None:
        cfg = Step1Config()  # AND: tool_calls>=7 AND steps>=1 AND unique>=2
        turns = [
            _make_turn(0, "八次调用四种工具", chain=_chain(("a", "b"), ("a", "b"), ("c", "d"), ("e", "f"))),
            _make_turn(1, "四次调用两种工具", chain=_chain(("a", "b"), ("a", "b"))),
            _make_turn(2, "八次调用一种工具", chain=_chain(("a", "a", "a", "a"), ("a", "a", "a", "a"))),
            _make_turn(3, "没有推理链", chain=[]),
        ]
        # turn1: tool_calls fail; turn2: unique fail; turn3: no chain.
        self.assertEqual(select_candidates(turns, cfg), [0])

    def test_assemble_row_context_fallback(self) -> None:
        turns = _sample_turns()
        # Segment-leading turn (idx == segment.start), not session-first:
        # context falls back to every earlier session turn.
        segment = Segment(start=2, end=3, topic="topic")
        row = assemble_row({"thread_id": "t1"}, turns, segment, idx=2, category_id="03")
        self.assertEqual(
            row["context"],
            [{"question": "Q1 简单查询", "answer": "answer0"}, {"question": "Q2 复杂取数", "answer": "answer1"}],
        )
        # Session-first turn: context stays empty (nothing precedes it).
        first = assemble_row({"thread_id": "t1"}, turns, Segment(start=0, end=3, topic="topic"), idx=0, category_id="03")
        self.assertEqual(first["context"], [])

    def test_judge_payload_context_fallback(self) -> None:
        turns = _sample_turns()
        payload = build_judge_payload(turns, Segment(start=2, end=3, topic="t"), 2)
        self.assertEqual(payload["prior_questions"], ["Q1 简单查询", "Q2 复杂取数"])
        first = build_judge_payload(turns, Segment(start=0, end=3, topic="t"), 0)
        self.assertEqual(first["prior_questions"], [])
        # non-boundary turn keeps same-segment prior only
        same = build_judge_payload(turns, Segment(start=0, end=3, topic="t"), 1)
        self.assertEqual(same["prior_questions"], ["Q1 简单查询"])

    def test_step1_skips_ineligible_turns(self) -> None:
        names = ("web_search", "finquery", "compute")
        turns = [
            _make_turn(0, "没有回答的复杂问题", tool_names="web_search,finquery,compute", tool_count=8, answer="", chain=_chain_with_steps(8, names)),
            _make_turn(1, "失败状态", tool_names="web_search,finquery,compute", tool_count=8, status="failed", chain=_chain_with_steps(8, names)),
            _make_turn(2, "正常复杂问题", tool_names="web_search,finquery,compute", tool_count=8, chain=_chain_with_steps(8, names)),
        ]
        self.assertEqual(select_candidates(turns, Step1Config()), [2])

    def test_assemble_row_field_mapping(self) -> None:
        turns = _sample_turns()
        segment = Segment(start=0, end=3, topic="topic")
        row = assemble_row({"thread_id": "t1"}, turns, segment, idx=2, category_id="03", reason="需要多步分析")

        self.assertEqual(row["capture_mode"], "full_link")
        self.assertEqual(row["user_cohort"], "regular")
        self.assertEqual(row["source_case_id"], "t1")
        self.assertEqual(row["trace_id"], "trace2")  # original input turn's trace_id
        self.assertEqual(row["category"], "03-analysis-research")
        self.assertEqual(row["input"]["text"], "Q3 复杂预测")
        self.assertEqual(row["session_round"], 3)  # 1-based within segment
        self.assertEqual(row["context"], [{"question": "Q1 简单查询", "answer": "answer0"}, {"question": "Q2 复杂取数", "answer": "answer1"}])
        self.assertEqual(row["tools"], ["web_search", "finquery", "compute"])
        self.assertEqual(row["raw_answer"], "answer2")
        self.assertEqual(row["text_answer"], "answer2")
        self.assertEqual(row["user_id"], "u2")
        self.assertEqual(row["difficulty_level"], "hard")
        self.assertEqual(
            row["meta"], {"reason": "需要多步分析", "request_time": "2026-08-05 04:02:00"}
        )
        self.assertEqual(row["first_token_time_ms"], 200)
        self.assertEqual(row["finish_answer_time_ms"], 400)
        self.assertFalse(any(k in row["context"][0] for k in ("chain", "tools", "run_id")))

    def test_end_to_end_produces_complex_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session = {"thread_id": "t1", "context": _sample_turns()}
            _write_jsonl(tmp_path / "input.jsonl", [session])
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=True))

            with patch("query_pipeline.steps.session_stage.LLMClient", FakeSessionLLMClient), patch(
                "query_pipeline.steps.verify_stage.LLMClient", FakeSessionLLMClient
            ):
                summary = run_pipeline(cfg)

            self.assertTrue(summary.success)
            self.assertEqual(summary.stats["total_sessions"], 1)
            self.assertEqual(summary.stats["segments"], 2)
            self.assertEqual(summary.stats["candidates"], 2)
            self.assertEqual(summary.stats["complex_rows"], 1)
            self.assertEqual(summary.stats["non_complex"], 1)
            self.assertEqual(summary.stats["llm_failed"], 0)
            self.assertEqual(summary.stats["verify_kept"], 1)
            self.assertEqual(summary.stats["verify_rejected"], 0)
            self.assertEqual(summary.stats["verify_failed"], 0)
            self.assertEqual(summary.stats["category_counts"], {"01": 1})

            rows = _read_jsonl(Path(summary.output_files["complex_queries"]))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["source_case_id"], "t1")
            self.assertEqual(row["trace_id"], "trace1")
            self.assertEqual(row["category"], "01-data-metrics-calculation")
            self.assertEqual(row["input"]["text"], "Q2 复杂取数")
            self.assertEqual(row["session_round"], 2)
            self.assertEqual(row["context"], [{"question": "Q1 简单查询", "answer": "answer0"}])
            self.assertEqual(row["difficulty_level"], "hard")
            self.assertEqual(row["meta"], {"reason": "多步工具调用取数", "request_time": "2026-08-05 04:01:00"})

    def test_end_to_end_llm_failure_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session = {"thread_id": "t1", "context": _sample_turns()}
            _write_jsonl(tmp_path / "input.jsonl", [session])
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=True))

            with patch("query_pipeline.steps.session_stage.LLMClient", FakeFailingSessionLLMClient), patch(
                "query_pipeline.steps.verify_stage.LLMClient", FakeFailingSessionLLMClient
            ):
                summary = run_pipeline(cfg)

            self.assertTrue(summary.success)
            # segmentation failure -> whole session is one segment; judge failure on turn1 dropped.
            self.assertEqual(summary.stats["segments"], 1)
            self.assertEqual(summary.stats["complex_rows"], 1)
            self.assertEqual(summary.stats["llm_failed"], 1)
            # verify failure is fail-open: the row survives and is counted.
            self.assertEqual(summary.stats["verify_failed"], 1)
            self.assertEqual(summary.stats["verify_kept"], 0)

            rows = _read_jsonl(Path(summary.output_files["complex_queries"]))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["trace_id"], "trace2")
            self.assertEqual(rows[0]["category"], "02-forecasting-and-projection")
            self.assertEqual(rows[0]["session_round"], 3)
            self.assertEqual(len(rows[0]["context"]), 2)

    def test_end_to_end_verify_filters_context_only_rows(self) -> None:
        # Pass 1 (with context) judges Q3 complex; pass 2 (standalone) does
        # not, so the row must be dropped with verify_rejected=1.
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

            with patch("query_pipeline.steps.session_stage.LLMClient", FakeJudgeThenVerifyClient), patch(
                "query_pipeline.steps.verify_stage.LLMClient", FakeJudgeThenVerifyClient
            ):
                summary = run_pipeline(cfg)

            self.assertEqual(summary.stats["candidates"], 2)
            self.assertEqual(summary.stats["verify_kept"], 1)
            self.assertEqual(summary.stats["verify_rejected"], 1)
            self.assertEqual(summary.stats["verify_failed"], 0)
            self.assertEqual(summary.stats["complex_rows"], 1)

            rows = _read_jsonl(Path(summary.output_files["complex_queries"]))
            self.assertEqual([r["trace_id"] for r in rows], ["trace1"])

            verified = _read_jsonl(tmp_path / "work/verified.jsonl")
            self.assertEqual([v["trace_id"] for v in verified], ["trace1", "trace2"])
            self.assertEqual(verified[1]["is_complex"], False)

    def test_end_to_end_post_stage_dedup_and_translate(self) -> None:
        # Two candidate turns with identical English text: verify keeps both,
        # dedup drops the second, translate fills meta.translation on the
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

            with patch("query_pipeline.steps.session_stage.LLMClient", FakePostStageLLMClient), patch(
                "query_pipeline.steps.verify_stage.LLMClient", FakePostStageLLMClient
            ), patch("query_pipeline.steps.post_stage.LLMClient", FakePostStageLLMClient):
                summary = run_pipeline(cfg)

            self.assertEqual(summary.stats["verify_kept"], 2)
            self.assertEqual(summary.stats["dedup_removed"], 1)
            self.assertEqual(summary.stats["translated"], 1)
            self.assertEqual(summary.stats["translate_skipped"], 0)
            self.assertEqual(summary.stats["translate_failed"], 0)
            self.assertEqual(summary.stats["complex_rows"], 1)

            rows = _read_jsonl(Path(summary.output_files["complex_queries"]))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["trace_id"], "trace1")
            self.assertEqual(rows[0]["meta"]["translation"], "翻译：complex calc A")

    def test_end_to_end_judge_data_format(self) -> None:
        # input.format=judge_data: each line is a single-case question with a
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
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=True, input_format="judge_data"))

            with patch("query_pipeline.steps.session_stage.LLMClient", FakeSessionLLMClient), patch(
                "query_pipeline.steps.verify_stage.LLMClient", FakeSessionLLMClient
            ):
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
            self.assertEqual(row["category"], "01-data-metrics-calculation")
            self.assertEqual(row["input"]["text"], "Q2 复杂取数")
            self.assertEqual(row["session_round"], 2)  # context_len + 1 == judge_data.meta.session_round
            self.assertEqual(row["context"], [{"question": "Q1 简单查询", "answer": "answer0"}])
            self.assertEqual(row["tools"], ["web_search"])
            self.assertEqual(row["raw_answer"], "text_answer")
            self.assertEqual(row["meta"], {"reason": "多步工具调用取数", "request_time": "2026-08-05 04:01:00"})

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
        if "current_question" in payload:
            current = payload["current_question"]
            if current == "Q2 复杂取数":
                return json.dumps({"is_complex": True, "category_id": "01", "reason": "多步工具调用取数"}, ensure_ascii=False)
            return json.dumps({"is_complex": False, "category_id": None, "reason": "简单查询"}, ensure_ascii=False)
        # second-pass verify (standalone question): keep Q2 only
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
        if "current_question" in payload:
            q = payload["current_question"]
            if q.startswith("complex calc"):
                return json.dumps({"is_complex": True, "category_id": "03", "reason": "复杂"}, ensure_ascii=False)
            return json.dumps({"is_complex": False, "category_id": None, "reason": "简单"}, ensure_ascii=False)
        if "question" in payload:  # verify: standalone question
            return json.dumps({"is_complex": True, "reason": "自身复杂"}, ensure_ascii=False)
        if "text" in payload:  # translate
            return json.dumps({"translation": "翻译：" + payload["text"]}, ensure_ascii=False)
        raise AssertionError(f"unexpected payload keys: {sorted(payload)}")

    async def close(self) -> None:
        import asyncio

        assert asyncio.get_running_loop() is self.used_loop, "client closed in a different event loop"


class FakeJudgeThenVerifyClient:
    """Judge marks Q2/Q3 complex with context; verify keeps only Q2 standalone."""

    def __init__(self, config: object) -> None:
        self.config = config

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt.split("\n", 1)[1])
        if "questions" in payload:
            n = len(payload["questions"])
            return json.dumps({"segments": [{"start": 0, "end": n - 1, "topic": "t"}]}, ensure_ascii=False)
        if "current_question" in payload:
            q = payload["current_question"]
            if q.startswith("Q2") or q.startswith("Q3"):
                return json.dumps({"is_complex": True, "category_id": "03", "reason": "上下文看着复杂"}, ensure_ascii=False)
            return json.dumps({"is_complex": False, "category_id": None, "reason": "简单"}, ensure_ascii=False)
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
        if "current_question" in payload:
            if payload["current_question"] == "Q2 复杂取数":
                raise RuntimeError("simulated judge failure")
            return json.dumps({"is_complex": True, "category_id": "02", "reason": "需要预测"}, ensure_ascii=False)
        raise RuntimeError("simulated verify failure")

    async def close(self) -> None:
        return None


def _write_config(tmp_path: Path, *, llm_enabled: bool, post_enabled: bool = False, input_format: str = "session") -> Path:
    config_path = tmp_path / "config.yaml"
    post_block = ""
    if post_enabled:
        post_block = """
            post_stage:
              enabled: true
              dedup:
                enabled: true
                threshold: 0.85
              translate:
                enabled: true
                target: zh
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
            session_stage:
              enabled: true
              segmentation:
                enabled: true
              step1:
                enabled: true
                reject_rules: true
                min_chain_tool_calls: 7
                min_chain_steps: 1
                min_unique_tools: 2
              step2:
                enabled: true
                prompt_id: complex_judge
            {post_block}
            llm_stage:
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
