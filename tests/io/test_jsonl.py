"""JSONL write hygiene: LS/PS terminator stripping on write and append."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from query_pipeline.io.jsonl import append_jsonl, read_jsonl_with_bad_lines, write_jsonl

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

    def test_bad_lines_written_as_json_records(self) -> None:
        # bad_lines.jsonl 必须统一为 JSON 对象行（单一 reader 可解析），
        # 不得混写原始文本（下游 JSON 读取会崩）。
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "input.jsonl"
            src.write_text(
                "{\"trace_id\": \"t1\"}\nnot-json-line\n[1,2]\n", encoding="utf-8"
            )
            bad_path = tmp_path / "bad_lines.jsonl"
            records, skipped = read_jsonl_with_bad_lines(src, bad_path)

            self.assertEqual(skipped, 2)
            self.assertEqual([r["trace_id"] for r in records], ["t1"])
            lines = bad_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            parsed = [json.loads(line) for line in lines]  # 单一 JSON reader 可解析全部行
            self.assertEqual(parsed[0]["reason"], "invalid_json")
            self.assertEqual(parsed[0]["raw"], "not-json-line")
            self.assertEqual(parsed[1]["reason"], "not_an_object")
            self.assertEqual(parsed[1]["raw"], "[1,2]")

    def test_append_jsonl_strips_too(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            append_jsonl(path, {"text": "a\u2028b"})
            append_jsonl(path, {"text": "c\u2029d"})
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertNotIn("\u2028", path.read_text(encoding="utf-8"))

