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

    def test_cache_key_verify_rounds_distinct(self) -> None:
        q = "question"
        base = make_cache_key(q, step="verify:verify_complex", model="m", prompt=resolve_prompt("verify_complex"))
        r2 = make_cache_key(q, step="verify:verify_recheck", model="m", prompt=resolve_prompt("verify_recheck").format(round_no=2))
        r3 = make_cache_key(q, step="verify:verify_recheck", model="m", prompt=resolve_prompt("verify_recheck").format(round_no=3))
        self.assertNotEqual(base, r2)
        self.assertNotEqual(r2, r3)
        self.assertNotEqual(base, r3)

