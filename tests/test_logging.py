"""Logging setup: UTC+8 timestamps, Beijing-time log files, global formatter."""

from __future__ import annotations

import logging
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from query_pipeline.logging_setup import beijing_converter, setup_logging

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

