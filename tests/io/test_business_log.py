"""Append-only business log routing, recovery, and resume semantics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from query_pipeline.io.business_log import BusinessLogWriter


def _row(trace_id: str, difficulty: str) -> dict:
    return {
        "trace_id": trace_id,
        "difficulty_level": difficulty,
        "input": {"text": f"question-{trace_id}"},
    }


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class BusinessLogWriterTest(unittest.TestCase):
    def test_creates_empty_files_and_routes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with BusinessLogWriter(tmp, "batch") as writer:
                for path in writer.paths.values():
                    self.assertTrue(path.exists())
                    self.assertEqual(path.read_text(encoding="utf-8"), "")
                writer.write(_row("hard", "hard"))
                writer.write(_row("normal", "normal"))

            self.assertEqual([r["trace_id"] for r in _read(writer.paths["cleaned"])], ["hard", "normal"])
            self.assertEqual([r["trace_id"] for r in _read(writer.paths["complex"])], ["hard"])
            self.assertEqual([r["trace_id"] for r in _read(writer.paths["normal"])], ["normal"])

    def test_resume_skips_existing_and_repairs_missing_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hard = _row("hard", "hard")
            with BusinessLogWriter(tmp, "batch") as writer:
                writer.write(hard)
            writer.paths["complex"].write_text("", encoding="utf-8")

            with BusinessLogWriter(tmp, "batch") as resumed:
                resumed.write(hard)
                resumed.write(hard)

            self.assertEqual(len(_read(resumed.paths["cleaned"])), 1)
            self.assertEqual(len(_read(resumed.paths["complex"])), 1)

    def test_repairs_only_incomplete_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "business" / "cleaned" / "batch.log"
            path.parent.mkdir(parents=True)
            path.write_text('{"trace_id":"ok"}\n{"trace_id":', encoding="utf-8")
            with BusinessLogWriter(tmp, "batch") as writer:
                writer.write(_row("new", "normal"))
            self.assertEqual([r["trace_id"] for r in _read(path)], ["ok", "new"])

    def test_rejects_invalid_middle_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "business" / "cleaned" / "batch.log"
            path.parent.mkdir(parents=True)
            path.write_text('{"trace_id":"ok"}\nnot-json\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid business log line 2"):
                with BusinessLogWriter(tmp, "batch"):
                    pass

    def test_rejects_complete_non_object_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "business" / "cleaned" / "batch.log"
            path.parent.mkdir(parents=True)
            path.write_text('["not", "an", "object"]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trailing line is not an object"):
                with BusinessLogWriter(tmp, "batch"):
                    pass

    def test_write_failure_is_raised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with BusinessLogWriter(tmp, "batch") as writer:
                writer._handles["cleaned"].close()
                with self.assertRaises(ValueError):
                    writer.write(_row("hard", "hard"))


if __name__ == "__main__":
    unittest.main()
