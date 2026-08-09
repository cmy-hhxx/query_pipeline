"""JSONL write hygiene: LS/PS terminator stripping on write and append."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from query_pipeline.io.jsonl import append_jsonl, write_jsonl

class JsonlHygieneTest(unittest.TestCase):
    def test_ls_ps_stripped_on_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            rows = [{"trace_id": "t1", "input": {"text": "你好\u2028世界\u2029测试"}}]
            write_jsonl(path, rows)
            raw = path.read_text(encoding="utf-8")
            self.assertNotIn("\u2028", raw)
            self.assertNotIn("\u2029", raw)
            self.assertEqual(raw.count("\n"), 1)  # strictly one line per record
            loaded = json.loads(raw)
            self.assertEqual(loaded["input"]["text"], "你好 世界 测试")

    def test_append_jsonl_strips_too(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            append_jsonl(path, {"text": "a\u2028b"})
            append_jsonl(path, {"text": "c\u2029d"})
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertNotIn("\u2028", path.read_text(encoding="utf-8"))

