"""Structured ordinary logging and command batch locking."""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from query_pipeline.logging_setup import generate_batch_id, logging_session, validate_batch_id


class LoggingSessionTest(unittest.TestCase):
    def test_generated_and_explicit_batch_ids(self) -> None:
        self.assertRegex(generate_batch_id(), r"^\d{8}T\d{6}\+0800_[0-9a-f]{8}$")
        self.assertEqual(validate_batch_id("upstream.batch-1"), "upstream.batch-1")
        for invalid in ("../escape", "a/b", "with space", ""):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_batch_id(invalid)

    def test_json_file_and_human_console(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            console = io.StringIO()
            with redirect_stderr(console):
                with logging_session(tmp, command="run", batch_id="batch-1") as session:
                    logging.getLogger("query_pipeline.steps.judge_stage").info("子 logger")
                    logging.getLogger("query_pipeline").debug("hidden debug")
                    logging.getLogger("unrelated").warning("outside namespace")

            rows = [
                json.loads(line)
                for line in session.ordinary_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[0]["message"].split()[0], "command_started")
            child = next(row for row in rows if row["message"] == "子 logger")
            self.assertEqual(child["level"], "INFO")
            self.assertEqual(child["logger"], "query_pipeline.steps.judge_stage")
            self.assertEqual(child["command"], "run")
            self.assertEqual(child["batch_id"], "batch-1")
            self.assertTrue(child["timestamp"].endswith("+08:00"))
            self.assertFalse(any(row["message"] == "hidden debug" for row in rows))
            self.assertFalse(any(row["message"] == "outside namespace" for row in rows))
            self.assertIn("INFO query_pipeline.steps.judge_stage: 子 logger", console.getvalue())
            self.assertNotIn('{"timestamp"', console.getvalue())

    def test_verbose_and_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stderr(io.StringIO()):
                try:
                    with logging_session(tmp, command="audit", batch_id="batch-2", verbose=True):
                        logging.getLogger("query_pipeline").debug("debug detail")
                        raise RuntimeError("boom")
                except RuntimeError:
                    pass
            path = Path(tmp) / "ordinary" / "audit" / "batch-2.log"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(row["message"] == "debug detail" for row in rows))
            failed = next(row for row in rows if row["message"].startswith("command_failed"))
            self.assertIn("RuntimeError: boom", failed["exception"])

    def test_same_batch_is_locked_across_processes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result: subprocess.CompletedProcess[str] | None = None
            code = """
from query_pipeline.logging_setup import logging_session
try:
    with logging_session(%r, command="run", batch_id="locked"):
        pass
except RuntimeError:
    raise SystemExit(0)
raise SystemExit(1)
""" % tmp
            with redirect_stderr(io.StringIO()):
                with logging_session(tmp, command="run", batch_id="locked"):
                    env = os.environ.copy()
                    root = Path(__file__).resolve().parents[1]
                    env["PYTHONPATH"] = str(root / "src")
                    result = subprocess.run(
                        [sys.executable, "-c", code],
                        env=env,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
            assert result is not None
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_package_imports_without_fcntl(self) -> None:
        code = """
import builtins

real_import = builtins.__import__

def without_fcntl(name, *args, **kwargs):
    if name == "fcntl":
        raise ImportError("fcntl is unavailable")
    return real_import(name, *args, **kwargs)

builtins.__import__ = without_fcntl
import query_pipeline
"""
        env = os.environ.copy()
        root = Path(__file__).resolve().parents[1]
        env["PYTHONPATH"] = str(root / "src")
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
