from __future__ import annotations

import asyncio
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
from query_pipeline.config.models import DedupConfig, LLMConfig
from query_pipeline.io.jsonl import read_jsonl, write_jsonl
from query_pipeline.llm.cache import load_cache
from query_pipeline.llm.client import LLMClient
from query_pipeline.pipeline.runner import run_pipeline
from query_pipeline.post.dedup import dedup_rows, jaccard_estimate, minhash_signature
from query_pipeline.post.translate import needs_translation, translate_rows

_BASE = "how does the fed rate hike affect bond prices this year and what should investors watch for"
_NEAR = _BASE.replace("affect", "impact")
_DISTINCT = "compare vanguard and fidelity etf expense ratios"


def _row(text: str, trace_id: str) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "source_case_id": "t1",
        "input": {"text": text, "image": "", "file": ""},
        "meta": {"reason": "r"},
    }


def _llm_cfg(tmp: str) -> LLMConfig:
    return LLMConfig(model="fake-model", cache=Path(tmp) / "cache.jsonl", concurrency=2)


class FakeLLMClient(LLMClient):
    """Unit-test client whose handler may raise to simulate LLM failure."""

    def __init__(self, handler: Any) -> None:
        self._handler = handler

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        return await self._handler(system_prompt, user_prompt)

    async def close(self) -> None:
        return None


class DedupTest(unittest.TestCase):
    def test_exact_and_near_duplicates_dropped(self) -> None:
        rows = [
            _row(_BASE, "r1"),
            _row(_BASE, "r2"),  # exact duplicate
            _row(_NEAR, "r3"),  # near duplicate (word swap)
            _row(_DISTINCT, "r4"),
        ]
        kept, dropped = dedup_rows(rows, DedupConfig())

        self.assertEqual([r["trace_id"] for r in kept], ["r1", "r4"])
        self.assertEqual([d["trace_id"] for d in dropped], ["r2", "r3"])
        self.assertEqual(dropped[0]["dedup_of_trace_id"], "r1")
        self.assertEqual(dropped[1]["dedup_of_trace_id"], "r1")
        self.assertEqual(dropped[0]["similarity"], 1.0)
        self.assertGreaterEqual(dropped[1]["similarity"], 0.85)
        self.assertEqual(dropped[0]["method"], "minhash_threshold_0.85")

    def test_near_duplicate_survives_high_threshold(self) -> None:
        kept, dropped = dedup_rows([_row(_BASE, "r1"), _row(_NEAR, "r2")], DedupConfig(threshold=0.99))
        self.assertEqual([r["trace_id"] for r in kept], ["r1", "r2"])
        self.assertEqual(dropped, [])

    def test_distinct_queries_kept(self) -> None:
        rows = [_row(_DISTINCT, "r1"), _row(_BASE, "r2")]
        kept, dropped = dedup_rows(rows, DedupConfig())
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])

    def test_empty_text_never_dropped(self) -> None:
        rows = [_row("", "r1"), _row("", "r2"), _row("   ", "r3")]
        kept, dropped = dedup_rows(rows, DedupConfig())
        self.assertEqual(len(kept), 3)
        self.assertEqual(dropped, [])

    def test_signatures_deterministic(self) -> None:
        a = minhash_signature(_BASE, n_gram=3, num_perm=128)
        b = minhash_signature(_BASE, n_gram=3, num_perm=128)
        self.assertEqual(a, b)

    def test_threshold_range_validated(self) -> None:
        with self.assertRaises(ValueError):
            DedupConfig(threshold=1.5)
        with self.assertRaises(ValueError):
            DedupConfig(threshold=-0.1)

    def test_near_duplicate_similarity_above_default_threshold(self) -> None:
        sig_base = minhash_signature(_BASE, n_gram=3, num_perm=128)
        sig_near = minhash_signature(_NEAR, n_gram=3, num_perm=128)
        assert sig_base is not None and sig_near is not None
        sim = jaccard_estimate(sig_base, sig_near)
        self.assertGreaterEqual(sim, 0.85)


class TranslateTest(unittest.TestCase):
    def test_adds_translation_to_meta(self) -> None:
        rows = [_row("how to hedge against inflation", "r1")]

        async def handler(system_prompt: str, user_prompt: str) -> str:
            del system_prompt, user_prompt
            return json.dumps({"translation": "如何对冲通胀"}, ensure_ascii=False)

        with tempfile.TemporaryDirectory() as tmp:
            counters = asyncio.run(
                translate_rows(
                    rows,
                    client=FakeLLMClient(handler),
                    llm_cfg=_llm_cfg(tmp),
                    cache={},
                    cache_path=Path(tmp) / "cache.jsonl",
                )
            )

        self.assertEqual(rows[0]["meta"]["translation"], "如何对冲通胀")
        self.assertEqual(counters, {"translated": 1, "translate_skipped": 0, "translate_failed": 0})

    def test_chinese_text_skipped_no_llm_call(self) -> None:
        rows = [_row("如何对冲通胀风险", "r1")]

        async def handler(system_prompt: str, user_prompt: str) -> str:  # pragma: no cover - must not be called
            del system_prompt, user_prompt
            raise AssertionError("LLM must not be called for Chinese text")

        with tempfile.TemporaryDirectory() as tmp:
            counters = asyncio.run(
                translate_rows(
                    rows,
                    client=FakeLLMClient(handler),
                    llm_cfg=_llm_cfg(tmp),
                    cache={},
                    cache_path=Path(tmp) / "cache.jsonl",
                )
            )

        self.assertEqual(rows[0]["meta"]["translation"], "如何对冲通胀风险")
        self.assertEqual(counters, {"translated": 0, "translate_skipped": 1, "translate_failed": 0})

    def test_failure_falls_back_to_original(self) -> None:
        rows = [_row("how to hedge against inflation", "r1")]

        async def handler(system_prompt: str, user_prompt: str) -> str:
            del system_prompt, user_prompt
            raise RuntimeError("simulated LLM failure")

        with tempfile.TemporaryDirectory() as tmp:
            counters = asyncio.run(
                translate_rows(
                    rows,
                    client=FakeLLMClient(handler),
                    llm_cfg=_llm_cfg(tmp),
                    cache={},
                    cache_path=Path(tmp) / "cache.jsonl",
                )
            )

        self.assertEqual(rows[0]["meta"]["translation"], "how to hedge against inflation")
        self.assertEqual(counters, {"translated": 0, "translate_skipped": 0, "translate_failed": 1})

    def test_cache_round_trip_reuses_translation(self) -> None:
        rows = [_row("how to hedge against inflation", "r1")]

        async def handler(system_prompt: str, user_prompt: str) -> str:
            del system_prompt, user_prompt
            return json.dumps({"translation": "如何对冲通胀"}, ensure_ascii=False)

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.jsonl"
            cache: dict[str, Any] = {}
            asyncio.run(
                translate_rows(
                    rows,
                    client=FakeLLMClient(handler),
                    llm_cfg=_llm_cfg(tmp),
                    cache=cache,
                    cache_path=cache_path,
                )
            )
            # Second run over a fresh row: the cache file must serve the
            # translation without any LLM call.
            rows2 = [_row("how to hedge against inflation", "r2")]

            async def fail_if_called(system_prompt: str, user_prompt: str) -> str:
                del system_prompt, user_prompt
                raise AssertionError("cached translation must be reused, not re-requested")

            counters2 = asyncio.run(
                translate_rows(
                    rows2,
                    client=FakeLLMClient(fail_if_called),
                    llm_cfg=_llm_cfg(tmp),
                    cache=load_cache(cache_path),
                    cache_path=cache_path,
                )
            )

        self.assertEqual(rows2[0]["meta"]["translation"], "如何对冲通胀")
        self.assertEqual(counters2, {"translated": 1, "translate_skipped": 0, "translate_failed": 0})

    def test_needs_translation_heuristics(self) -> None:
        self.assertTrue(needs_translation("what is the pe ratio of tesla"))
        self.assertFalse(needs_translation("特斯拉的市盈率是多少"))
        self.assertFalse(needs_translation(""))
        self.assertFalse(needs_translation("   "))
        self.assertFalse(needs_translation("美股最近怎么样，Tesla 涨了吗"))


class PostStagePipelineTest(unittest.TestCase):
    def test_config_defaults_disabled(self) -> None:
        from query_pipeline.config.models import PostConfig

        post = PostConfig()
        self.assertFalse(post.enabled)
        self.assertTrue(post.dedup.enabled)
        self.assertEqual(post.dedup.threshold, 0.85)
        self.assertTrue(post.translate.enabled)

    def test_config_yaml_enables_post_stage(self) -> None:
        cfg = load_pipeline_config(ROOT / "configs/aime/config.yaml")
        self.assertTrue(cfg.post.enabled)
        self.assertEqual(cfg.post.dedup.threshold, 0.85)
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

            self.assertEqual(summary.stats["complex_rows"], 1)  # post-dedup
            self.assertEqual(summary.stats["verify_kept"], 2)
            self.assertEqual(summary.stats["dedup_removed"], 1)
            self.assertEqual(summary.stats["translated"], 1)
            self.assertEqual(summary.stats["translate_skipped"], 0)
            self.assertEqual(summary.stats["translate_failed"], 0)

            rows = list(read_jsonl(tmp_path / "out" / "complex_queries.jsonl"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_case_id"], "t1")  # first occurrence kept
            self.assertEqual(rows[0]["meta"]["translation"], "利率上升如何影响债券价格？")
            self.assertEqual(rows[0]["meta"]["reason"], "需要预测")

            deduped = list(read_jsonl(tmp_path / "work" / "deduped.jsonl"))
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
            self.assertNotIn("dedup_removed", summary.stats)
            rows = list(read_jsonl(tmp_path / "out" / "complex_queries.jsonl"))
            self.assertEqual(
                rows[0]["meta"], {"reason": "需要预测", "request_time": "2026-08-05 04:01:00", "translation": ""}
            )  # untouched by post stage


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
        "answer": f"answer{idx}",
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
        if "current_question" in payload:  # pass-1 judge (with context)
            if payload["current_question"] == self.COMPLEX:
                return json.dumps({"is_complex": True, "category_id": "02", "reason": "需要预测"}, ensure_ascii=False)
            return json.dumps({"is_complex": False, "category_id": None, "reason": "简单查询"}, ensure_ascii=False)
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
              complex_queries: complex_queries.jsonl
              summary: summary.json
            work_dir: work
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
            verify:
              enabled: true
              prompt_id: verify_complex
            post:
              enabled: {str(post_enabled).lower()}
              dedup:
                enabled: true
                threshold: 0.85
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


if __name__ == "__main__":
    unittest.main()
