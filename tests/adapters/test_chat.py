"""Chat-record adaptation into the session model."""

from __future__ import annotations

import unittest

from query_pipeline.adapters.chat import adapt_chat

class ChatAdapterTest(unittest.TestCase):
    def test_adapt_chat(self) -> None:
        record = {
            "trace_id": "t1",
            "question": "当前问句",
            "judge_data": {
                "case_id": "c1",
                "trace_id": "t1",
                "input": {"text": "当前问句", "image": None, "file": None},
                "context": [{"question": "前文1", "answer": "a1"}, {"question": "前文2", "answer": "a2"}],
                "chain": [{"plan": "", "tools": [{"name": "Search", "input": {}, "output": "x"}]}],
                "raw_answer": "raw",
                "text_answer": "text",
                "meta": {"session_round": 3, "request_time": "2026-08-05 04:02:00", "first_token_time_cost": 10},
            },
        }
        session = adapt_chat(record)
        self.assertEqual(session.thread_id, "c1")
        self.assertEqual(len(session.turns), 3)
        self.assertEqual(session.turns[0].question, "前文1")
        self.assertEqual(session.turns[0].answer, "a1")
        current = session.turns[2]
        self.assertEqual(current.question, "当前问句")
        self.assertEqual(current.answer, "text")  # text_answer preferred over raw_answer
        self.assertEqual(current.answer_full, "raw")  # raw_answer 独立保留
        self.assertEqual(current.trace_id, "t1")
        self.assertEqual(current.first_token_ms, 10)
        self.assertEqual(current.request_time, "2026-08-05 04:02:00")
        self.assertEqual(session.candidate_mode, "last_only")

    def test_adapt_chat_missing_wrapper(self) -> None:
        with self.assertRaises(ValueError):
            adapt_chat({"question": "x"})

    def test_adapt_chat_non_dict_turn_raises(self) -> None:
        # 非 dict turn 静默过滤会让整行在 chat 门槛下无声落选：必须 fail-loud
        # （preclean 按 adapt_failed 进 bad_lines，可审计、可计数）。
        record = {
            "judge_data": {
                "case_id": "c1",
                "input": {"text": "当前问句"},
                "context": [{"question": "前文1", "answer": "a1"}, "not-a-dict", 42],
                "chain": [],
            }
        }
        with self.assertRaises(ValueError):
            adapt_chat(record)

    def test_adapt_chat_malformed_chain_warns_and_empties(self) -> None:
        # 畸形 chain 记 warning 并置空（chain 缺省对 chat 合法），不得崩溃。
        record = {
            "judge_data": {
                "case_id": "c1",
                "input": {"text": "当前问句"},
                "context": [],
                "chain": "not-a-list",
                "raw_answer": "raw",
                "text_answer": "text",
            }
        }
        session = adapt_chat(record)
        self.assertEqual(session.turns[-1].chain, [])

