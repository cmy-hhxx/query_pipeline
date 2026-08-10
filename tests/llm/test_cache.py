"""LLM cache-key versioning: prompt/step/model changes must invalidate entries."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from query_pipeline.llm.cache import load_cache, make_cache_key
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

    def _write_entries(self, path: Path, src: str, n: int) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for i in range(n):
                row = {
                    "cache_key": f"step:m:{i}",
                    "label": {"v": i},
                    "src": src,
                    "step": "s",
                }
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    def test_orphaned_generation_compacted_on_load(self) -> None:
        # src_hash 进 key 后，代码变更会让全部旧 key 永久失效；缓存 append-only
        # 永不清理则文件只增不减。load_cache 必须检测孤儿代并一次性 rewrite。
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm_cache.jsonl"
            self._write_entries(path, "dead-generation", 3)
            self._write_entries(path, "dead-generation", 2)  # 模拟追加（旧代码）
            cache = load_cache(path)
            self.assertEqual(cache, {})  # 旧代条目全部不可用

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 0)  # 文件已被压缩清空

    def test_current_generation_kept_on_load(self) -> None:
        from query_pipeline.llm.cache import src_hash

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm_cache.jsonl"
            self._write_entries(path, src_hash(), 3)
            cache = load_cache(path)
            self.assertEqual(len(cache), 3)  # 当前代条目保留

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)  # 无孤儿 → 不重写

    def test_mixed_generation_keeps_only_current(self) -> None:
        from query_pipeline.llm.cache import src_hash

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm_cache.jsonl"
            self._write_entries(path, "dead-generation", 2)
            self._write_entries(path, src_hash(), 2)
            cache = load_cache(path)
            self.assertEqual(len(cache), 2)  # 只保留当前代
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue(all(json.loads(l)["src"] == src_hash() for l in lines))

    def test_rewrite_uses_unique_tmp_name(self) -> None:
        # 第四轮 #1：孤儿代 rewrite 的 tmp 名必须带 pid+随机后缀——固定 tmp 名
        # 会让双进程 rewrite 互相截断（实测 5000 条仅存 1959 条、33 行损坏）。
        import os

        from query_pipeline.llm.cache import _rewrite_cache

        real_replace = os.replace
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm_cache.jsonl"
            rows = [{"cache_key": f"k{i}", "label": {"v": i}, "src": "s"} for i in range(3)]
            seen_tmps: list[str] = []

            def _replace(src: str, dst: str) -> None:
                seen_tmps.append(src)
                real_replace(src, dst)

            with patch("query_pipeline.llm.cache.os.replace", side_effect=_replace):
                _rewrite_cache(path, rows)
                _rewrite_cache(path, rows)

            self.assertEqual(len(seen_tmps), 2)
            self.assertNotEqual(seen_tmps[0], seen_tmps[1])  # 两次 rewrite 不共用 tmp
            self.assertIn(".tmp.", str(seen_tmps[0]))
            # 重写后内容完整
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 3)

    def test_append_after_rewrite_keeps_entries(self) -> None:
        # rewrite（tmp+replace）与 append 都持跨进程锁：append 不得写进将被
        # replace 的旧 inode 而静默丢失。
        from query_pipeline.llm.cache import _rewrite_cache, append_cache

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm_cache.jsonl"
            path.write_text(
                json.dumps({"cache_key": "k1", "label": {"v": 1}, "src": "old-gen"}) + "\n",
                encoding="utf-8",
            )
            # load 触发孤儿代 rewrite（压缩为空）→ 之后 append 的新条目必须保留
            self.assertEqual(load_cache(path), {})
            append_cache(path, "k2", {"v": 2}, meta={"step": "s"})
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["cache_key"], "k2")

