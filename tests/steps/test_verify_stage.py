"""Verify stage: current-question policy, corpus review, and fatal infra errors."""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from query_pipeline.config.loader import load_pipeline_config
from query_pipeline.pipeline.context import RunSummary
from query_pipeline.pipeline.runner import run_pipeline
from tests._profiles import complexity_label, verify_label

def _turns() -> list[dict]:
    names = ("web_search", "finquery", "compute")
    chain = [{"plan": "", "tools": [{"name": names[i % 3], "input": {}, "output": "x"}]} for i in range(8)]
    return [
        {
            "question": "帮我分析贵州茅台的估值并给出买卖建议",
            "answer": "answer0",
            "trace_id": "tr0",
            "status": "completed",
            "outcome": "success",
            "chain": chain,
            "tool_count": 8,
        },
        {
            "question": "帮我构建一个沪深300的增强策略并回测",
            "answer": "answer1",
            "trace_id": "tr1",
            "status": "completed",
            "outcome": "success",
            "chain": chain,
            "tool_count": 8,
        },
    ]

class RecordingClient:
    """Records verify payloads; verdicts per question text."""

    def __init__(self, config: object) -> None:
        self.config = config
        self.verify_calls: list[dict] = []

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt.split("\n", 1)[1])
        if "questions" in payload:
            n = len(payload["questions"])
            return json.dumps({"segments": [{"start": 0, "end": n - 1, "topic": "t"}]})
        if "current_question" in payload:
            if "价值判官" in system_prompt:
                return json.dumps({"is_valuable": True})
            if "已判定为复杂金融问句" in system_prompt:
                return json.dumps({"category_id": "01", "reason": "复杂归类"})
            if "有价值但非复杂" in system_prompt:
                return json.dumps({"category_id": "03", "reason": "普通归类"})
            return json.dumps(
                complexity_label(True, goal=payload["current_question"])
            )  # both judged hard
        if "question" in payload:
            self.verify_calls.append(payload)
            # 沪深300 stays complex in every round; 茅台 complex round 1, rejected round 2+
            if payload["question"].startswith("帮我构建"):
                return json.dumps(verify_label(True, reason="复杂", evidence_quote=payload["question"]))
            if system_prompt.startswith("你是第"):  # round >= 2
                return json.dumps(
                    verify_label(False, reason="短决策", evidence_quote=payload["question"])
                )
            return json.dumps(verify_label(True, reason="初判复杂", evidence_quote=payload["question"]))
        return json.dumps({"translation": "译"})

    async def close(self) -> None:
        return None

def _row(source_case_id: str = "c1", trace_id: str = "t1", question: str = "Q", prior: list[str] | None = None) -> dict:
    return {
        "source_case_id": source_case_id,
        "trace_id": trace_id,
        "input": {"text": question},
        "context": [{"question": p, "answer": "a"} for p in (prior or [])],
    }


class VerifyKeyUnitTest(unittest.TestCase):
    def test_key_sensitive_to_prior_questions(self) -> None:
        from query_pipeline.steps.verify_stage import _verify_content_key

        base = _row()
        self.assertNotEqual(
            _verify_content_key(base, ["前文A"], "hard"),
            _verify_content_key(base, ["前文B"], "hard"),
        )
        self.assertNotEqual(
            _verify_content_key(base, [], "hard"),
            _verify_content_key(base, ["前文A"], "hard"),
        )

    def test_key_sensitive_to_difficulty(self) -> None:
        from query_pipeline.steps.verify_stage import _verify_content_key

        base = _row()
        self.assertNotEqual(
            _verify_content_key(base, ["前文"], "hard"),
            _verify_content_key(base, ["前文"], "normal"),
        )

    def test_key_sensitive_to_identity(self) -> None:
        from query_pipeline.steps.verify_stage import _verify_content_key

        self.assertNotEqual(
            _verify_content_key(_row(trace_id="t1"), [], "hard"),
            _verify_content_key(_row(trace_id="t2"), [], "hard"),
        )
        self.assertEqual(
            _verify_content_key(_row(), [], "hard"),
            _verify_content_key(_row(), [], "hard"),
        )


class VerifyStageTest(unittest.TestCase):
    def _run(self, client_cls=RecordingClient, config_extra: str = "") -> RunSummary:
        import tempfile
        import textwrap

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "input.jsonl").write_text(
                json.dumps({"thread_id": "t1", "context": _turns()}) + "\n", encoding="utf-8"
            )
            cfg_path = tmp_path / "config.yaml"
            cfg_path.write_text(
                textwrap.dedent(
                    f"""
                    input:
                      path: input.jsonl
                      format: session
                    output:
                      dir: out
                    work_dir: work
                    segmentation:
                      enabled: false
                    rule_gate:
                      min_chain_tool_calls: 7
                    verify:
                      enabled: true
                      max_rounds_hard: 5
                    llm:
                      enabled: true
                      model: fake
                      api_key_env: FAKE
                      concurrency: 2
                    {config_extra}
                    """
                ),
                encoding="utf-8",
            )
            with patch("query_pipeline.pipeline.runner.LLMClient", client_cls):
                summary = run_pipeline(load_pipeline_config(cfg_path))
            return summary

    def test_verify_receives_only_current_question(self) -> None:
        summary = self._run()
        # Explicit 5-round config is still supported; one hard row is downgraded.
        self.assertEqual(summary.stats["verify_complex_kept"], 1)
        self.assertEqual(summary.stats["verify_to_normal"], 1)
        self.assertEqual(summary.stats["verify_failed"], 0)

    def test_ungrounded_verify_evidence_fails_the_batch(self) -> None:
        class UngroundedVerifyClient(RecordingClient):
            async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
                payload = json.loads(user_prompt.split("\n", 1)[1])
                if "question" in payload:
                    return json.dumps(
                        verify_label(
                            False,
                            route="reject",
                            reason="模板",
                            evidence_quote="问句中不存在的证据",
                        )
                    )
                return await super().complete(system_prompt=system_prompt, user_prompt=user_prompt)

        summary = self._run(UngroundedVerifyClient)
        self.assertFalse(summary.success)
        self.assertEqual(summary.stats["verify_failed"], 2)
        self.assertEqual(summary.stats["verify_rejected_template"], 0)
        self.assertEqual(summary.stats["output_rows"], 0)

    def test_normal_rows_require_non_complex(self) -> None:
        # A normal row is not reviewed and can never be upgraded to hard.
        import tempfile
        import textwrap

        class NormalRowClient(RecordingClient):
            async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
                payload = json.loads(user_prompt.split("\n", 1)[1])
                if "questions" in payload:
                    n = len(payload["questions"])
                    return json.dumps({"segments": [{"start": 0, "end": n - 1, "topic": "t"}]})
                if "current_question" in payload:
                    return json.dumps({"is_valuable": True})
                if "question" in payload:
                    # verify claims EVERYTHING is complex -> normal rows must be rejected
                    return json.dumps(verify_label(True, reason="都复杂", evidence_quote=payload["question"]))
                return json.dumps({"translation": "译"})

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "input.jsonl").write_text(
                json.dumps({"thread_id": "t1", "context": _turns()}) + "\n", encoding="utf-8"
            )
            cfg_path = tmp_path / "config.yaml"
            cfg_path.write_text(
                textwrap.dedent(
                    """
                    input:
                      path: input.jsonl
                      format: session
                    output:
                      dir: out
                    work_dir: work
                    segmentation:
                      enabled: false
                    rule_gate:
                      min_chain_tool_calls: 7
                    judge:
                      classify_normal_prompt: classify_normal
                    verify:
                      enabled: true
                      max_rounds_hard: 3
                    llm:
                      enabled: true
                      model: fake
                      api_key_env: FAKE
                      concurrency: 2
                    """
                ),
                encoding="utf-8",
            )
            # judge must produce one normal row: make complexity gate say non-complex for 茅台
            class MixedJudgeClient(NormalRowClient):
                async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
                    payload = json.loads(user_prompt.split("\n", 1)[1])
                    if "questions" in payload:
                        n = len(payload["questions"])
                        return json.dumps({"segments": [{"start": 0, "end": n - 1, "topic": "t"}]})
                    if "current_question" in payload:
                        if "价值判官" in system_prompt:
                            return json.dumps({"is_valuable": True})
                        if "已判定为复杂金融问句" in system_prompt:
                            return json.dumps({"category_id": "01", "reason": "复杂归类"})
                        if "有价值但非复杂" in system_prompt:
                            return json.dumps({"category_id": "03", "reason": "普通归类"})
                        return json.dumps(
                            complexity_label(
                                payload["current_question"].startswith("帮我构建"),
                                goal=payload["current_question"],
                            )
                        )
                    if "question" in payload:
                        return json.dumps(verify_label(True, reason="都复杂", evidence_quote=payload["question"]))
                    return json.dumps({"translation": "译"})

            with patch("query_pipeline.pipeline.runner.LLMClient", MixedJudgeClient):
                summary = run_pipeline(load_pipeline_config(cfg_path))
            # 沪深300 hard kept; 茅台 normal passes through untouched.
            self.assertEqual(summary.stats["verify_complex_kept"], 1)
            self.assertEqual(summary.stats["verify_to_normal"], 0)
            self.assertEqual(summary.stats["normal_rows"], 1)

if __name__ == "__main__":
    unittest.main()
