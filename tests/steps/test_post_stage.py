"""Post stage: config wiring and end-to-end dedup+translate through run_pipeline."""

from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]  # repo root (tests/steps/ -> project root)

from query_pipeline.config.loader import load_pipeline_config
from query_pipeline.io.jsonl import read_jsonl, write_jsonl
from query_pipeline.pipeline.runner import run_pipeline

def _row(text: str, trace_id: str) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "source_case_id": "t1",
        "input": {"text": text, "image": "", "file": ""},
        "meta": {"reason": "r"},
    }

def _turns(prefix: str) -> list[dict[str, Any]]:
    names = ("web_search", "finquery", "compute")
    return [
        _make_turn(prefix, 0, "hello", tool_names="web_search", tool_count=1, chain=_chain_with_tool_calls(1)),
        _make_turn(
            prefix,
            1,
            "How do rising interest rates affect bond prices?",
            tool_names="web_search,finquery,compute",
            tool_count=8,
            chain=_chain_with_steps(8, names),
        ),
        _make_turn(prefix, 2, "thanks", tool_names="", tool_count=0),
    ]

def _make_turn(
    prefix: str,
    idx: int,
    question: str,
    *,
    tool_names: str = "",
    tool_count: int = 0,
    chain: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "question": question,
        "answer": f"answer{idx} " + "x" * 60,
        "run_id": f"{prefix}r{idx}",
        "trace_id": f"{prefix}trace{idx}",
        "request_time": f"2026-08-05 04:{idx:02d}:00",
        "user_id": f"{prefix}u{idx}",
        "status": "completed",
        "outcome": "success",
        "tool_names": tool_names,
        "tool_count": tool_count,
        "first_token_ms": idx * 100,
        "total_duration_ms": idx * 100 + 200,
        "chain": chain if chain is not None else [],
    }

def _chain_with_tool_calls(n: int, name: str = "t") -> list[dict[str, Any]]:
    return [{"plan": "", "tools": [{"name": name, "input": {}, "output": "x"} for _ in range(n)]}]

def _chain_with_steps(n: int, names: tuple[str, ...]) -> list[dict[str, Any]]:
    return [{"plan": "", "tools": [{"name": names[i % len(names)], "input": {}, "output": "x"}]} for i in range(n)]

class FakePipelineLLMClient:
    """Handles segment / judge / verify / translate payloads for the e2e test."""

    COMPLEX = "How do rising interest rates affect bond prices?"

    def __init__(self, config: object) -> None:
        self.config = config

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        assert system_prompt
        payload = json.loads(user_prompt.split("\n", 1)[1])
        if "questions" in payload:
            n = len(payload["questions"])
            return json.dumps({"segments": [{"start": 0, "end": n - 1, "topic": "topic"}]}, ensure_ascii=False)
        if "价值判官" in system_prompt:
            return json.dumps({"is_valuable": True, "reason": "金融相关"}, ensure_ascii=False)
        if "已判定为复杂金融问句" in system_prompt:
            return json.dumps({"category_id": "02", "reason": "需要预测"}, ensure_ascii=False)
        if "有价值但非复杂" in system_prompt:
            return json.dumps({"category_id": "03", "reason": "普通归类"}, ensure_ascii=False)
        if "current_question" in payload:  # complexity gate
            if payload["current_question"] == self.COMPLEX:
                return json.dumps({"is_complex": True, "reason": "需要预测"}, ensure_ascii=False)
            return json.dumps({"is_complex": False, "reason": "简单查询"}, ensure_ascii=False)
        if "简单问句识别器" in system_prompt:  # simple_finder 视角
            return json.dumps({"is_simple": False, "reason": "不是简单问句"}, ensure_ascii=False)
        if "question" in payload:  # pass-2 verify (standalone)
            if payload["question"] == self.COMPLEX:
                return json.dumps({"is_complex": True, "reason": "独立成立"}, ensure_ascii=False)
            return json.dumps({"is_complex": False, "reason": "独立不成立"}, ensure_ascii=False)
        return json.dumps({"translation": "利率上升如何影响债券价格？"}, ensure_ascii=False)

    async def close(self) -> None:
        return None

def _write_config(tmp_path: Path, *, post_enabled: bool = True) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            name: test_pipeline
            input:
              path: input.jsonl
              format: session
            output:
              dir: out
              cleaned_queries: cleaned_queries.jsonl
              complex_queries: complex_queries.jsonl
              normal_queries: normal_queries.jsonl
              summary: summary.json
            work_dir: work
            segmentation:
              enabled: true
            rule_gate:
              enabled: true
              reject_rules: true
              min_chain_tool_calls: 7
              min_chain_steps: 1
              min_unique_tools: 2
            judge:
              enabled: true
            verify:
              enabled: true
              prompt_id: verify_complex
            post:
              enabled: {str(post_enabled).lower()}
              dedup:
                enabled: true
                threshold: 0.80
                entity_slot: true
              translate:
                enabled: true
            llm:
              enabled: true
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

class PostStagePipelineTest(unittest.TestCase):
    def test_config_defaults_disabled(self) -> None:
        from query_pipeline.config.models import PostConfig

        post = PostConfig()
        self.assertFalse(post.enabled)
        self.assertTrue(post.dedup.enabled)
        self.assertEqual(post.dedup.threshold, 0.80)
        self.assertTrue(post.translate.enabled)

    def test_aime_0807_config_enables_post_stage(self) -> None:
        cfg = load_pipeline_config(ROOT / "configs/aime/0807.yaml")
        self.assertTrue(cfg.post.enabled)
        self.assertEqual(cfg.post.dedup.threshold, 0.80)
        self.assertTrue(cfg.post.translate.enabled)

    def test_end_to_end_dedup_and_translate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            turns = _turns("s1")
            write_jsonl(tmp_path / "input.jsonl", [
                {"thread_id": "t1", "context": turns},
                {"thread_id": "t2", "context": _turns("s2")},
            ])
            cfg = load_pipeline_config(_write_config(tmp_path))

            with patch("query_pipeline.pipeline.runner.LLMClient", FakePipelineLLMClient):
                summary = run_pipeline(cfg)

            self.assertEqual(summary.stats["complex_rows"], 2)  # judged hard rows
            self.assertEqual(summary.stats["output_rows"], 1)  # post-dedup
            self.assertEqual(summary.stats["verify_kept"], 2)
            self.assertEqual(summary.stats["dedup_removed"], 1)
            self.assertEqual(summary.stats["translated"], 1)
            self.assertEqual(summary.stats["translate_skipped"], 0)
            self.assertEqual(summary.stats["translate_failed"], 0)

            rows = list(read_jsonl(tmp_path / "out" / "cleaned_queries.jsonl"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_case_id"], "t1")  # first occurrence kept
            self.assertEqual(rows[0]["translation"], "利率上升如何影响债券价格？")
            self.assertEqual(rows[0]["meta"]["reason"], "需要预测")

            deduped = list(
                read_jsonl(tmp_path / "work" / "runtime" / "diagnostics" / "deduped.jsonl")
            )
            self.assertEqual(len(deduped), 1)
            self.assertEqual(deduped[0]["trace_id"], "s2trace1")
            self.assertEqual(deduped[0]["dedup_of_trace_id"], "s1trace1")
            self.assertEqual(deduped[0]["similarity"], 1.0)

    def test_post_stage_disabled_no_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            write_jsonl(tmp_path / "input.jsonl", [{"thread_id": "t1", "context": _turns("s1")}])
            cfg = load_pipeline_config(_write_config(tmp_path, post_enabled=False))

            with patch("query_pipeline.pipeline.runner.LLMClient", FakePipelineLLMClient):
                summary = run_pipeline(cfg)

            self.assertEqual(summary.stats["complex_rows"], 1)
            self.assertEqual(summary.stats["dedup_removed"], 0)  # post disabled
            rows = list(read_jsonl(tmp_path / "out" / "cleaned_queries.jsonl"))
            self.assertEqual(
                rows[0]["meta"], {"reason": "需要预测", "request_time": "2026-08-05 04:01:00", "run_id": "s1r1", "last_event_type": None}
            )  # post stage 不再触碰 meta
            self.assertIsNone(rows[0]["translation"])  # 中文原文 → null（post 关闭时亦然）

if __name__ == "__main__":
    unittest.main()
