"""Input format sniffing and record pre-cleaning."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from query_pipeline.adapters import CHAT, SESSION, match_adapter
from query_pipeline.io.sniff import preclean_records, sniff_format

def _session_line(thread_id: str = "t1", n_turns: int = 1) -> str:
    return json.dumps(
        {"thread_id": thread_id, "context": [{"question": "q", "answer": "a"}] * n_turns}
    )

def _chat_line(case_id: str = "c1", trace_id: str = "tr1") -> str:
    return json.dumps(
        {
            "trace_id": trace_id,
            "question": "q",
            "judge_data": {
                "case_id": case_id,
                "input": {"text": "q"},
                "context": [],
                "chain": [],
            },
        }
    )

def _write(lines: list[str]) -> Path:
    tmp = tempfile.mkdtemp()
    path = Path(tmp) / "input.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

class ClassifyRecordTest(unittest.TestCase):
    def test_session(self) -> None:
        self.assertEqual(match_adapter({"thread_id": "t", "context": []}), SESSION)

    def test_chat(self) -> None:
        self.assertEqual(match_adapter({"judge_data": {"input": {}}}), CHAT)

    def test_partial_markers_rejected(self) -> None:
        with self.assertRaises(ValueError):
            match_adapter({"thread_id": "t"})  # context missing
        with self.assertRaises(ValueError):
            match_adapter({"context": []})  # thread_id missing
        with self.assertRaises(ValueError):
            match_adapter({"judge_data": "not-a-dict"})  # wrong type

    def test_unrecognizable(self) -> None:
        self.assertIsNone(match_adapter({"foo": "bar"}))

class SniffFormatTest(unittest.TestCase):
    def test_session_file(self) -> None:
        self.assertEqual(sniff_format(_write([_session_line(), _session_line("t2")])), SESSION)

    def test_chat_file(self) -> None:
        self.assertEqual(sniff_format(_write([_chat_line(), _chat_line("c2")])), CHAT)

    def test_mixed_raises(self) -> None:
        with self.assertRaises(ValueError):
            sniff_format(_write([_session_line(), _chat_line()]))

    def test_unrecognizable_raises(self) -> None:
        with self.assertRaises(ValueError):
            sniff_format(_write([json.dumps({"foo": "bar"})]))

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            sniff_format(_write([]))

    def test_bad_lines_skipped(self) -> None:
        self.assertEqual(sniff_format(_write(["not json", _session_line()])), SESSION)

class PrecleanTest(unittest.TestCase):
    def test_session_dedup_and_empty_filter(self) -> None:
        records = [
            {"thread_id": "a", "context": [{"question": "q"}]},
            {"thread_id": "a", "context": [{"question": "q2"}]},  # duplicate
            {"thread_id": "b", "context": []},  # empty context
            {"thread_id": "c", "context": [{"question": "q3"}]},
        ]
        kept, dup, empty = preclean_records(records, SESSION)
        self.assertEqual([r["thread_id"] for r in kept], ["a", "c"])
        self.assertEqual(dup, 1)
        self.assertEqual(empty, 1)

    def test_chat_dedup_by_case_id_and_keeps_empty_context(self) -> None:
        records = [
            {"trace_id": "t1", "judge_data": {"case_id": "c1"}},
            {"trace_id": "t2", "judge_data": {"case_id": "c1"}},  # duplicate by case_id
            {"trace_id": "t3", "judge_data": {}},  # empty context is fine for chat
        ]
        kept, dup, empty = preclean_records(records, CHAT)
        self.assertEqual(len(kept), 2)
        self.assertEqual(dup, 1)
        self.assertEqual(empty, 0)

    def test_records_without_key_kept(self) -> None:
        records = [{"foo": "bar", "context": [{"question": "q"}]}, {"foo": "baz", "context": [{"question": "q"}]}]
        kept, dup, empty = preclean_records(records, SESSION)
        self.assertEqual(len(kept), 2)
        self.assertEqual(dup, 0)
        self.assertEqual(empty, 0)

if __name__ == "__main__":
    unittest.main()
