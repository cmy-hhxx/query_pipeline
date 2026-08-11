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
from tests._profiles import complexity_label, verify_label

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
            self.assertEqual(summary.stats["llm_failed"], 0)
            self.assertEqual(summary.stats["verify_complex_kept"], 1)
            self.assertEqual(summary.stats["verify_to_normal"], 0)
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
            self.assertEqual(row["meta"]["reason"], "多步工具调用取数")
            self.assertIn("complexity_profile", row["meta"])
            self.assertIn("semantic_signature", row["meta"])
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

            self.assertTrue(summary.success)  # llm_failed / verify_failed 均 fail-open
            # segmentation failure -> whole session is one segment; judge failure on turn1 dropped.
            self.assertEqual(summary.stats["segments"], 1)
            self.assertEqual(summary.stats["complex_rows"], 1)
            self.assertEqual(summary.stats["llm_failed"], 1)
            # llm_failed fail-open：verify 不再跳过，照跑复核幸存行；该 stub 的
            # verify 抛错 → verify_failed=1 → 该行丢弃、批次照常成功（空输出）。
            self.assertEqual(summary.stats["verify_failed"], 1)
            self.assertEqual(summary.stats["verify_complex_kept"], 0)
            self.assertEqual(summary.stats["verify_to_normal"], 0)
            self.assertEqual(summary.stats["verify_uncertain"], 0)
            out_path = Path(summary.output_files["cleaned_queries"])
            self.assertTrue(out_path.exists())

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
            self.assertEqual(summary.stats["verify_complex_kept"], 1)
            self.assertEqual(summary.stats["verify_to_normal"], 0)
            self.assertEqual(summary.stats["verify_failed"], 0)
            self.assertEqual(summary.stats["complex_rows"], 1)

            rows = _read_jsonl(Path(summary.output_files["cleaned_queries"]))
            self.assertEqual([r["trace_id"] for r in rows], ["trace1"])

            verified = _read_jsonl(
                tmp_path / "work" / "runtime" / "diagnostics" / "verified.jsonl"
            )
            self.assertEqual([v["trace_id"] for v in verified], ["trace1"])
            self.assertEqual(verified[0]["route"], "complex")

    def test_verify_reject_is_absent_from_every_final_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session = {"thread_id": "t1", "context": _sample_turns()}
            _write_jsonl(tmp_path / "input.jsonl", [session])
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=True))

            with patch("query_pipeline.pipeline.runner.LLMClient", FakeRejectVerifyClient):
                summary = run_pipeline(cfg)

            self.assertTrue(summary.success)
            self.assertEqual(summary.stats["verify_rejected_template"], 1)
            self.assertEqual(summary.stats["verify_complex_kept"], 0)
            self.assertEqual(summary.stats["final_complex_rows"], 0)
            self.assertEqual(summary.stats["final_normal_rows"], 1)
            self.assertEqual(_read_jsonl(Path(summary.output_files["complex_queries"])), [])
            cleaned = _read_jsonl(Path(summary.output_files["cleaned_queries"]))
            normal = _read_jsonl(Path(summary.output_files["normal_queries"]))
            self.assertEqual([row["trace_id"] for row in cleaned], ["trace2"])
            self.assertEqual([row["trace_id"] for row in normal], ["trace2"])
            rejected = _read_jsonl(
                tmp_path / "work" / "runtime" / "diagnostics" / "complex_policy_rejected.jsonl"
            )
            self.assertEqual([row["trace_id"] for row in rejected], ["trace1"])
            self.assertEqual(rejected[0]["method"], "single_question_verify")

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

            self.assertEqual(summary.stats["verify_complex_kept"], 1)
            self.assertEqual(summary.stats["verify_to_normal"], 2)
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
            self.assertEqual(len(rows), 3)
            self.assertEqual(
                [r["difficulty_level"] for r in rows], ["hard", "normal", "normal"]
            )
            for downgraded in rows[1:]:
                profile = downgraded["meta"]["complexity_profile"]
                self.assertEqual(profile["route"], "normal")
                self.assertEqual(profile["complex_features"], [])
                self.assertTrue(profile["exclusion_reasons"])
                self.assertTrue(profile["evidence"])
                self.assertIn("verify_reason", downgraded["meta"])
                self.assertIn("normal_classification_reason", downgraded["meta"])

            verified = _read_jsonl(
                tmp_path / "work" / "runtime" / "diagnostics" / "verified.jsonl"
            )
            self.assertEqual([v["trace_id"] for v in verified], ["trace0", "trace1", "trace2"])
            by_q = {v["question"]: v for v in verified}
            self.assertEqual(by_q["帮我分析贵州茅台的估值并给出买卖建议"]["route"], "complex")
            self.assertEqual([r["round"] for r in by_q["帮我分析贵州茅台的估值并给出买卖建议"]["rounds"]], [1, 2, 3])
            self.assertEqual(by_q["分析一下宁德时代的三季度业绩并预测走势"]["route"], "normal")
            self.assertEqual([r["round"] for r in by_q["分析一下宁德时代的三季度业绩并预测走势"]["rounds"]], [1])
            self.assertEqual(by_q["计算比亚迪过去五年的平均市盈率"]["route"], "normal")
            self.assertEqual(
                [(r["round"], r["route"]) for r in by_q["计算比亚迪过去五年的平均市盈率"]["rounds"]],
                [(1, "complex"), (2, "normal")],
            )

    def test_verify_multi_round_error_mid_round_fail_closed(self) -> None:
        # Round 2 of the "flaky" question raises: hard admission fails and the
        # row is downgraded, but the failure is not checkpointed — a re-run
        # retries it (round 1 replays from llm_cache, round 2 calls the LLM
        # again). A transient outage must not permanently poison the output.
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

            self.assertEqual(summary.stats["verify_complex_kept"], 1)
            self.assertEqual(summary.stats["verify_failed"], 1)
            self.assertEqual(summary.stats["verify_to_normal"], 0)
            self.assertEqual(summary.stats["verify_uncertain"], 0)
            self.assertTrue(summary.success)  # verify_failed fail-open
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
            self.assertEqual(summary2.stats["verify_complex_kept"], 1)
            self.assertEqual(summary2.stats["verify_failed"], 1)
            self.assertEqual(summary2.stats["verify_uncertain"], 0)
            # the errored row is retried: exactly one LLM call (round 2 of the
            # flaky question; round 1 and the healthy row replay from llm_cache)
            self.assertEqual(len(clients2[0].calls), 1)

            self.assertTrue(Path(summary.output_files["cleaned_queries"]).exists())

    def _corrupt_cache_entry(self, cache_path: Path, step_prefix: str, label: dict[str, Any]) -> None:
        """把 cache 文件中指定 step 的第一个 label 替换为坏值。"""
        lines = cache_path.read_text(encoding="utf-8").splitlines()
        out: list[str] = []
        replaced = False
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("cache_key", "").startswith(step_prefix) and not replaced:
                row["label"] = label
                replaced = True
            out.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        assert replaced, f"no {step_prefix} entry in cache"
        cache_path.write_text("\n".join(out) + "\n", encoding="utf-8")

    def test_bad_funnel_cache_label_self_heals(self) -> None:
        # 坏 value_gate 缓存 label：驱逐并重调 LLM，候选保留；不得每次运行重复丢弃。
        # 删除 judge checkpoint 让 funnel 真正重跑（否则会话级 checkpoint 直接回放行）。
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session = {"thread_id": "t1", "context": _sample_turns()}
            _write_jsonl(tmp_path / "input.jsonl", [session])
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=True))

            with patch("query_pipeline.pipeline.runner.LLMClient", FakeSessionLLMClient):
                run_pipeline(cfg)
            cache_path = tmp_path / "work" / "llm_cache.jsonl"
            self._corrupt_cache_entry(cache_path, "value_gate:", {"is_valuable": "not-a-bool"})
            (tmp_path / "work" / "runtime" / "checkpoints" / "judge.jsonl").unlink()

            class RecordingFunnelClient(FakeSessionLLMClient):
                def __init__(self, config: object) -> None:
                    super().__init__(config)
                    self.calls: list[dict[str, Any]] = []

                async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
                    self.calls.append(json.loads(user_prompt.split("\n", 1)[1]))
                    return await super().complete(system_prompt=system_prompt, user_prompt=user_prompt)

            clients: list[RecordingFunnelClient] = []

            def factory(config: object) -> RecordingFunnelClient:
                client = RecordingFunnelClient(config)
                clients.append(client)
                return client

            with patch("query_pipeline.pipeline.runner.LLMClient", factory):
                summary = run_pipeline(cfg)
            client = clients[0]

            self.assertEqual(summary.stats["complex_rows"], 1)  # 候选未被重复丢弃
            self.assertEqual(summary.stats["value_rejected"], 0)
            funnel_calls = [c for c in client.calls if "current_question" in c]
            # 只有 value gate 重调（坏缓存驱逐）；complexity/classify 命中完好缓存
            self.assertEqual(len(funnel_calls), 1)

    def test_bad_verify_cache_label_self_heals(self) -> None:
        # 坏 verify 缓存 label：驱逐并重调，行保留；不得固化坏裁决。
        # 删除 judge/verify checkpoint 让 round 循环真正执行（否则 checkpoint 直接回放）。
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session = {"thread_id": "t1", "context": _sample_turns()}
            _write_jsonl(tmp_path / "input.jsonl", [session])
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=True))

            with patch("query_pipeline.pipeline.runner.LLMClient", FakeSessionLLMClient):
                run_pipeline(cfg)
            cache_path = tmp_path / "work" / "llm_cache.jsonl"
            self._corrupt_cache_entry(cache_path, "verify:", {"is_complex": "garbage"})
            for name in ("judge.jsonl", "verify.jsonl"):
                (tmp_path / "work" / "runtime" / "checkpoints" / name).unlink()

            class RecordingVerifyClient(FakeSessionLLMClient):
                def __init__(self, config: object) -> None:
                    super().__init__(config)
                    self.calls: list[dict[str, Any]] = []

                async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
                    self.calls.append(json.loads(user_prompt.split("\n", 1)[1]))
                    return await super().complete(system_prompt=system_prompt, user_prompt=user_prompt)

            clients: list[RecordingVerifyClient] = []

            def factory(config: object) -> RecordingVerifyClient:
                client = RecordingVerifyClient(config)
                clients.append(client)
                return client

            with patch("query_pipeline.pipeline.runner.LLMClient", factory):
                summary = run_pipeline(cfg)
            client = clients[0]

            self.assertEqual(summary.stats["verify_complex_kept"], 1)
            self.assertEqual(summary.stats["verify_failed"], 0)
            verify_calls = [
                c
                for c in client.calls
                if "question" in c and "current_question" not in c and "questions" not in c and "text" not in c
            ]
            self.assertGreaterEqual(len(verify_calls), 1)  # round 1 重调（坏缓存驱逐）

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

            self.assertEqual(summary.stats["verify_complex_kept"], 1)
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
            self.assertEqual(summary.stats["verify_complex_kept"], 1)

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
            self.assertEqual(row["meta"]["reason"], "多步工具调用取数")
            self.assertIn("complexity_profile", row["meta"])
            self.assertIn("semantic_signature", row["meta"])

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
            self.assertEqual(summary.stats["verify_complex_kept"], 1)

    def test_end_to_end_chat_empty_answer_fails_precheck(self) -> None:
        # An all-ineligible chat batch is rejected before any LLM runtime is initialized.
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

            with self.assertRaisesRegex(ValueError, "no_eligible_turns"):
                run_pipeline(cfg)

    def test_unexpected_candidate_error_fails_batch_without_session_crash(self) -> None:
        # 兜底网返回 None（如 cache 磁盘 OSError）：单候选按 llm_failed 计，
        # debug 推导不得再对 None 崩溃 → 会话不变成 session_error；llm_failed
        # fail-open，批次照常成功（失败候选丢弃、失败数留 summary）。
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session = {"thread_id": "t1", "context": _sample_turns()}
            _write_jsonl(tmp_path / "input.jsonl", [session])
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=True))

            with patch("query_pipeline.pipeline.runner.LLMClient", FakeSessionLLMClient), patch(
                "query_pipeline.session.funnel.put_cache", side_effect=OSError("disk full")
            ):
                summary = run_pipeline(cfg)

            self.assertTrue(summary.success)
            self.assertEqual(summary.stats["session_errors"], 0)
            self.assertGreater(summary.stats["llm_failed"], 0)

    def test_api_4xx_in_funnel_drops_candidate_not_session(self) -> None:
        # fix #9 让 4xx 立即抛出；funnel 必须捕获 APIStatusError 按候选失败处理
        # 并记录 llm_failed；会话不崩溃，llm_failed fail-open → 批次照常成功。
        import httpx
        from openai import BadRequestError

        class FourHundredClient(FakeSessionLLMClient):
            async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
                payload = json.loads(user_prompt.split("\n", 1)[1])
                if "current_question" in payload and "价值判官" in system_prompt:
                    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
                    raise BadRequestError("context too long", response=httpx.Response(400, request=req), body=None)
                return await super().complete(system_prompt=system_prompt, user_prompt=user_prompt)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session = {"thread_id": "t1", "context": _sample_turns()}
            _write_jsonl(tmp_path / "input.jsonl", [session])
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=True))

            with patch("query_pipeline.pipeline.runner.LLMClient", FourHundredClient):
                summary = run_pipeline(cfg)

            self.assertTrue(summary.success)
            self.assertEqual(summary.stats["session_errors"], 0)
            self.assertEqual(summary.stats["llm_failed"], 2)  # 两个候选均 400 → 丢弃
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

    def test_empty_sessions_counted_in_stats(self) -> None:
        # context=[非 dict] 可产出 0 turns 会话：empty_sessions 必须出现在 stats 里
        # （preclean 只滤空 context 列表，不滤非 dict 元素）。
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session = {"thread_id": "t1", "context": [42, "not-a-dict"]}
            _write_jsonl(tmp_path / "input.jsonl", [session])
            # 畸形 context 正是 precheck 要拦的场景；本测试只验证 empty_sessions 计数 → 显式关闭预检
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=True, precheck_enabled=False))

            with patch("query_pipeline.pipeline.runner.LLMClient", FakeSessionLLMClient):
                summary = run_pipeline(cfg)

            self.assertEqual(summary.stats["total_sessions"], 1)
            self.assertEqual(summary.stats["empty_sessions"], 1)
            self.assertEqual(summary.stats["candidates"], 0)

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
            bad_lines = _read_jsonl(
                tmp_path / "work" / "runtime" / "diagnostics" / "bad_lines.jsonl"
            )
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
            bad_lines = _read_jsonl(
                tmp_path / "work" / "runtime" / "diagnostics" / "bad_lines.jsonl"
            )
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
        complexity_label(
            q == "Q2 复杂取数", reason="判定", goal=q
        ),
        ensure_ascii=False,
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
            return json.dumps(
                verify_label(True, reason="自身复杂", evidence_quote=payload["question"]),
                ensure_ascii=False,
            )
        return json.dumps(
            verify_label(False, reason="单独看不复杂", evidence_quote=payload["question"]),
            ensure_ascii=False,
        )

    async def close(self) -> None:
        return None


class FakeRejectVerifyClient(FakeSessionLLMClient):
    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt.split("\n", 1)[1])
        if "question" in payload:
            return json.dumps(
                verify_label(
                    False,
                    route="reject",
                    exclusion_reasons=["eval_template"],
                    evidence_quote=payload["question"],
                    reason="严重 eval 模板",
                ),
                ensure_ascii=False,
            )
        return await super().complete(system_prompt=system_prompt, user_prompt=user_prompt)

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
            return json.dumps(
                complexity_label(True, reason="上下文复杂", goal=payload["current_question"]),
                ensure_ascii=False,
            )
        if "简单问句识别器" in system_prompt:  # simple_finder 视角
            return json.dumps({"is_simple": False, "reason": "不是简单问句"}, ensure_ascii=False)
        if "question" in payload:  # verify
            return json.dumps(
                verify_label(True, reason="自身复杂", evidence_quote=payload["question"]),
                ensure_ascii=False,
            )
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
            return json.dumps(
                complexity_label(q == "Q2 复杂取数", reason="判定", goal=q),
                ensure_ascii=False,
            )
        if "简单问句识别器" in system_prompt:  # simple_finder 视角
            return json.dumps({"is_simple": False, "reason": "不是简单问句"}, ensure_ascii=False)
        if payload["question"] == "Q2 复杂取数":
            return json.dumps(
                verify_label(True, reason="自身复杂", evidence_quote=payload["question"]),
                ensure_ascii=False,
            )
        return json.dumps(
            verify_label(False, reason="单独看是承接句", evidence_quote=payload["question"]),
            ensure_ascii=False,
        )

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
            return json.dumps(
                complexity_label(True, reason="需要预测", goal=payload["current_question"]),
                ensure_ascii=False,
            )
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
            return json.dumps(
                complexity_label(True, reason="上下文复杂", goal=payload["current_question"]),
                ensure_ascii=False,
            )
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
        return json.dumps(
            verify_label(
                is_complex,
                reason=f"第{round_no}轮",
                evidence_quote=payload["question"],
            ),
            ensure_ascii=False,
        )

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
            return json.dumps(
                complexity_label(True, reason="上下文复杂", goal=payload["current_question"]),
                ensure_ascii=False,
            )
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
        return json.dumps(
            verify_label(True, reason="复杂", evidence_quote=payload["question"]),
            ensure_ascii=False,
        )

    async def close(self) -> None:
        return None

def _write_config(
    tmp_path: Path,
    *,
    llm_enabled: bool,
    post_enabled: bool = False,
    input_format: str = "session",
    step1_enabled: bool | None = None,
    precheck_enabled: bool = True,
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
            precheck:
              enabled: {str(precheck_enabled).lower()}
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
              cache: llm_cache.jsonl
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
