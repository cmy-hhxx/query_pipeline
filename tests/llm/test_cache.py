"""LLM cache-key versioning: prompt/step/model changes must invalidate entries."""

from __future__ import annotations

import unittest

from query_pipeline.llm.cache import make_cache_key
from query_pipeline.prompts import resolve_prompt

class CacheKeyTest(unittest.TestCase):
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

    def test_cache_key_versioned_by_source_fingerprint(self) -> None:
        # "改代码不改 prompt" 的修复也必须让旧缓存失效（与 checkpoint 策略一致）
        from unittest.mock import patch

        q = "question"
        with patch("query_pipeline.llm.cache.src_hash", return_value="aaaa"):
            key_a = make_cache_key(q, step="s", model="m", prompt="p")
        with patch("query_pipeline.llm.cache.src_hash", return_value="bbbb"):
            key_b = make_cache_key(q, step="s", model="m", prompt="p")
        self.assertNotEqual(key_a, key_b)

    def test_cache_key_verify_rounds_distinct(self) -> None:
        q = "question"
        base = make_cache_key(q, step="verify:verify_complex", model="m", prompt=resolve_prompt("verify_complex"))
        r2 = make_cache_key(q, step="verify:verify_recheck", model="m", prompt=resolve_prompt("verify_recheck").format(round_no=2))
        r3 = make_cache_key(q, step="verify:verify_recheck", model="m", prompt=resolve_prompt("verify_recheck").format(round_no=3))
        self.assertNotEqual(base, r2)
        self.assertNotEqual(r2, r3)
        self.assertNotEqual(base, r3)

