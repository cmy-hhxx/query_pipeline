"""Translate: needs-translation heuristics, LLM fill, cache round-trip, failure tolerance."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from query_pipeline.config.models import LLMConfig
from query_pipeline.llm.cache import load_cache
from query_pipeline.llm.client import LLMClient
from query_pipeline.post.translate import needs_translation, translate_rows
from query_pipeline.rules.normalize import exact_token_jaccard, tokenize_question

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

class TranslateTest(unittest.TestCase):
    def test_adds_translation_to_row(self) -> None:
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

        self.assertEqual(rows[0]["translation"], "如何对冲通胀")
        self.assertEqual(counters, {"translated": 1, "translate_skipped": 0, "translate_failed": 0})

    def test_adds_translation_with_null_meta(self) -> None:
        rows = [
            {
                "trace_id": "r1",
                "source_case_id": "t1",
                "input": {"text": "how to hedge against inflation", "image": "", "file": ""},
                "meta": None,  # input rows may carry meta: null
            }
        ]

        async def handler(system_prompt: str, user_prompt: str) -> str:
            del system_prompt, user_prompt
            return json.dumps({"translation": "如何对冲通胀"}, ensure_ascii=False)

        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(
                translate_rows(
                    rows,
                    client=FakeLLMClient(handler),
                    llm_cfg=_llm_cfg(tmp),
                    cache={},
                    cache_path=Path(tmp) / "cache.jsonl",
                )
            )

        self.assertEqual(rows[0]["translation"], "如何对冲通胀")
        self.assertIsNone(rows[0]["meta"])  # 不再触碰 meta

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

        self.assertIsNone(rows[0]["translation"])  # 中文原文 → null
        self.assertEqual(counters, {"translated": 0, "translate_skipped": 1, "translate_failed": 0})

    def test_failure_leaves_null(self) -> None:
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

        self.assertIsNone(rows[0]["translation"])  # 翻译失败 → null（fail-open 保留行）
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

        self.assertEqual(rows2[0]["translation"], "如何对冲通胀")
        self.assertEqual(counters2, {"translated": 1, "translate_skipped": 0, "translate_failed": 0})

    def test_needs_translation_heuristics(self) -> None:
        self.assertTrue(needs_translation("what is the pe ratio of tesla"))
        self.assertFalse(needs_translation("特斯拉的市盈率是多少"))
        self.assertFalse(needs_translation(""))
        self.assertFalse(needs_translation("   "))
        self.assertFalse(needs_translation("美股最近怎么样，Tesla 涨了吗"))
