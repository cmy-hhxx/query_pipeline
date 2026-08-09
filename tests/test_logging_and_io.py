"""Output hygiene: LS/PS terminator stripping, Beijing-time logs, logs/ layout."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_pipeline.io.jsonl import append_jsonl, write_jsonl
from query_pipeline.logging_setup import beijing_converter, setup_logging
from query_pipeline.pipeline.context import PipelineContext


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


class BeijingTimeTest(unittest.TestCase):
    def test_converter_renders_utc8(self) -> None:
        # 2026-08-09 06:00:00 UTC == 2026-08-09 14:00:00 Beijing
        utc = datetime(2026, 8, 9, 6, 0, 0, tzinfo=timezone.utc).timestamp()
        rendered = datetime(*beijing_converter(utc)[:6], tzinfo=timezone(timedelta(hours=8)))
        self.assertEqual(rendered.hour, 14)
        self.assertEqual(rendered.date().isoformat(), "2026-08-09")

    def test_setup_logging_writes_beijing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_file = Path(tmp) / "run.log"
            setup_logging(log_file, verbose=False)
            logging.getLogger("query_pipeline").info("时间戳检查")
            logging.getLogger("query_pipeline.steps.judge_stage").info("子 logger")
            for handler in list(logging.getLogger("query_pipeline").handlers):
                if isinstance(handler, logging.FileHandler):
                    handler.flush()
            content = log_file.read_text(encoding="utf-8")
            self.assertIn("时间戳检查", content)
            self.assertIn("子 logger", content)
            self.assertRegex(content, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

    def test_converter_is_global_formatter_setting(self) -> None:
        # setup_logging must switch every Formatter to Beijing time
        with tempfile.TemporaryDirectory() as tmp:
            setup_logging(Path(tmp) / "run.log", verbose=False)
            self.assertIs(logging.Formatter.converter, beijing_converter)


class LogsLayoutTest(unittest.TestCase):
    def test_ctx_path_under_logs(self) -> None:
        from types import SimpleNamespace

        ctx = PipelineContext(config=SimpleNamespace(work_dir=Path("/x/y")))
        self.assertEqual(ctx.path("bad_lines.jsonl"), Path("/x/y/logs/bad_lines.jsonl"))


if __name__ == "__main__":
    unittest.main()
