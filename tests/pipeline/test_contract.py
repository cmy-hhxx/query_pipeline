"""End-to-end pipeline contract: fake-LLM runs over the full stage stack.

Covers session/chat input formats, segmentation/judge/verify cascades,
post-stage dedup+translate, checkpoint interplay, and failure fallbacks.
"""

from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from query_pipeline.config.loader import load_pipeline_config
from query_pipeline.pipeline.runner import run_pipeline
from query_pipeline.prompts import resolve_prompt

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

class SessionPipelineContractTest(unittest.TestCase):
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

            rows = _read_jsonl(Path(summary.output_files["cleaned_queries"]))
            self.assertEqual(len(rows), 2)
            row = rows[0]
            self.assertEqual(row["source_case_id"], "t1")
            self.assertEqual(row["trace_id"], "trace1")
            self.assertEqual(row["category"], "complex-topic/01-data-metrics-calculation")
            self.assertEqual(row["input"]["text"], "Q2 复杂取数")
            self.assertEqual(row["context"], [{"question": "Q1 简单查询", "answer": "answer0 " + "x" * 60}])
            self.assertEqual(row["difficulty_level"], "hard")
            self.assertEqual(row["meta"], {"reason": "多步工具调用取数", "request_time": "2026-08-05 04:01:00", "run_id": "r1", "last_event_type": None})
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

            self.assertTrue(summary.success)  # llm_failed counted, not fatal
            # segmentation failure -> whole session is one segment; judge failure on turn1 dropped.
            self.assertEqual(summary.stats["segments"], 1)
            self.assertEqual(summary.stats["complex_rows"], 1)
            self.assertEqual(summary.stats["llm_failed"], 1)
            # verify failure is fail-closed: the surviving row is dropped.
            self.assertEqual(summary.stats["verify_failed"], 1)
            self.assertEqual(summary.stats["verify_kept"], 0)
            # run succeeds (llm_failed counted, not fatal); empty output is written
            out_path = Path(summary.output_files["cleaned_queries"])
            self.assertTrue(out_path.exists())
            self.assertEqual(out_path.read_text(encoding="utf-8").strip(), "")

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

            rows = _read_jsonl(Path(summary.output_files["cleaned_queries"]))
            self.assertEqual([r["trace_id"] for r in rows], ["trace1"])

            verified = _read_jsonl(tmp_path / "work" / "logs" / "verified.jsonl")
            self.assertEqual([v["trace_id"] for v in verified], ["trace1"])
            self.assertEqual(verified[0]["is_complex"], True)

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

            rows = _read_jsonl(Path(summary.output_files["cleaned_queries"]))
            self.assertEqual([r["input"]["text"] for r in rows], ["帮我分析贵州茅台的估值并给出买卖建议"])

            verified = _read_jsonl(tmp_path / "work" / "logs" / "verified.jsonl")
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

            rows = _read_jsonl(Path(summary.output_files["cleaned_queries"]))
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

            rows = _read_jsonl(Path(summary.output_files["cleaned_queries"]))
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
                    "raw_answer": "raw_answer " + "y" * 60,
                    "text_answer": "text_answer " + "y" * 60,
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

            rows = _read_jsonl(Path(summary.output_files["cleaned_queries"]))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["source_case_id"], "c1")
            self.assertEqual(row["trace_id"], "t1")
            self.assertEqual(row["category"], "complex-topic/01-data-metrics-calculation")
            self.assertEqual(row["input"]["text"], "Q2 复杂取数")
            self.assertEqual(row["context"], [{"question": "Q1 简单查询", "answer": "answer0"}])
            self.assertEqual(row["tools"], ["web_search"])
            self.assertEqual(row["raw_answer"], "raw_answer " + "y" * 60)
            self.assertEqual(row["text_answer"], "text_answer " + "y" * 60)
            self.assertEqual(row["meta"], {"reason": "多步工具调用取数", "request_time": "2026-08-05 04:01:00", "run_id": "", "last_event_type": None})

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
                    "raw_answer": "raw_answer " + "y" * 60,
                    "text_answer": "text_answer " + "y" * 60,
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
                    "raw_answer": "raw_answer " + "y" * 60,
                    "text_answer": "text_answer " + "y" * 60,
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

    def test_value_gate_string_false_rejects_fail_closed(self) -> None:
        # LLM 输出 "is_valuable": "false"（字符串）：严格解析后应为 False 并丢弃候选，
        # 不得因 bool("false") == True 静默放行。
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session = {"thread_id": "t1", "context": _sample_turns()}
            _write_jsonl(tmp_path / "input.jsonl", [session])
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=True))

            class StringBoolClient(FakeSessionLLMClient):
                async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
                    payload = json.loads(user_prompt.split("\n", 1)[1])
                    if "current_question" in payload and "价值判官" in system_prompt:
                        return json.dumps({"is_valuable": "false", "reason": "字符串布尔"}, ensure_ascii=False)
                    return await super().complete(system_prompt=system_prompt, user_prompt=user_prompt)

            with patch("query_pipeline.pipeline.runner.LLMClient", StringBoolClient):
                summary = run_pipeline(cfg)

            self.assertEqual(summary.stats["value_rejected"], 2)  # 全部候选被 value 门拒
            self.assertEqual(summary.stats["complex_rows"], 0)
            self.assertEqual(summary.stats["llm_failed"], 0)

    def test_empty_success_run_keeps_previous_output(self) -> None:
        # 成功但零输出的 run 不得用空文件覆盖上次良好产物。
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session = {"thread_id": "t1", "context": _sample_turns()}
            _write_jsonl(tmp_path / "input.jsonl", [session])
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=True))
            cfg_nollm = load_pipeline_config(_write_config(tmp_path, llm_enabled=False))

            with patch("query_pipeline.pipeline.runner.LLMClient", FakeSessionLLMClient):
                summary1 = run_pipeline(cfg)
            self.assertEqual(summary1.stats["complex_rows"], 1)
            out_path = Path(summary1.output_files["cleaned_queries"])
            before = out_path.read_text(encoding="utf-8")

            summary2 = run_pipeline(cfg_nollm)  # success=True 且零输出
            self.assertTrue(summary2.success)
            self.assertEqual(summary2.stats["complex_rows"], 0)
            self.assertTrue(summary2.stats.get("output_preserved_previous"))
            self.assertEqual(out_path.read_text(encoding="utf-8"), before)  # 未被空文件覆盖

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
            self.assertEqual(len(_read_jsonl(Path(summary.output_files["cleaned_queries"]))), 0)

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
            bad_lines = _read_jsonl(tmp_path / "work" / "logs" / "bad_lines.jsonl")
            self.assertTrue(any(r.get("reason") == "adapt_failed" for r in bad_lines))

    def test_chat_non_dict_judge_data_lands_in_bad_lines(self) -> None:
        # judge_data 为 truthy 非 dict：preclean 不再崩溃；行走 adapt 失败路径进 bad_lines。
        # 坏行放在第 6 行（前 5 行样本都是正常 chat），避免 sniff_format 的 partial-marker 报错。
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            def good(case_id: str) -> dict[str, Any]:
                return {
                    "trace_id": f"t{case_id}",
                    "question": "Q2 复杂取数",
                    "judge_data": {
                        "case_id": case_id,
                        "input": {"text": "Q2 复杂取数"},
                        "context": [{"question": "Q1", "answer": "a"}],
                        "chain": _chain_with_steps(8, ("web_search", "finquery", "compute")),
                        "raw_answer": "raw " + "y" * 60,
                        "text_answer": "text " + "y" * 60,
                    },
                }

            records = [good(f"c{i}") for i in range(1, 6)] + [{"trace_id": "bad", "judge_data": "not-a-dict"}]
            _write_jsonl(tmp_path / "input.jsonl", records)
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=False, input_format="chat"))

            summary = run_pipeline(cfg)

            self.assertEqual(summary.stats["total_sessions"], 5)
            self.assertEqual(summary.stats["input_bad_lines"], 1)
            bad_lines = _read_jsonl(tmp_path / "work" / "logs" / "bad_lines.jsonl")
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
        if "简单问句识别器" in system_prompt:  # simple_finder 视角
            return json.dumps({"is_simple": False, "reason": "不是简单问句"}, ensure_ascii=False)
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
        if "简单问句识别器" in system_prompt:  # simple_finder 视角
            return json.dumps({"is_simple": False, "reason": "不是简单问句"}, ensure_ascii=False)
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
        if "简单问句识别器" in system_prompt:  # simple_finder 视角
            return json.dumps({"is_simple": False, "reason": "不是简单问句"}, ensure_ascii=False)
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
        if system_prompt == resolve_prompt("verify_complex"):
            round_no = 1
        elif system_prompt == resolve_prompt("verify_recheck").format(round_no=2):
            round_no = 2
        elif system_prompt == resolve_prompt("verify_recheck").format(round_no=3):
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
        if system_prompt == resolve_prompt("verify_complex"):
            round_no = 1
        elif system_prompt == resolve_prompt("verify_recheck").format(round_no=2):
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
              cleaned_queries: cleaned_queries.jsonl
              complex_queries: complex_queries.jsonl
              normal_queries: normal_queries.jsonl
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
