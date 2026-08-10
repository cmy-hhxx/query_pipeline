"""Session-record adaptation into the session model."""

from __future__ import annotations

import unittest

from query_pipeline.adapters.session import adapt_session

class SessionAdapterTest(unittest.TestCase):
    def test_adapt_session(self) -> None:
        session = adapt_session(
            {
                "thread_id": "t1",
                "context": [
                    {"question": "q1", "answer": "a1", "trace_id": "tr1"},
                    {"question": "q2", "answer": "a2", "trace_id": "tr2"},
                ],
            }
        )
        self.assertEqual(session.thread_id, "t1")
        self.assertEqual(len(session.turns), 2)
        self.assertEqual(session.turns[1].trace_id, "tr2")
        self.assertEqual(session.candidate_mode, "all")

    def test_adapt_session_non_dict_turns_become_empty_session(self) -> None:
        # 第二轮契约：context=[非 dict] → 0 turns 会话（judge 阶段计入
        # empty_sessions 统计），不丢行。与 chat 适配器（fail-loud 进 bad_lines）
        # 语义不同：session 方言有 empty_sessions 审计口径。
        session = adapt_session({"thread_id": "t1", "context": [42, "not-a-dict"]})
        self.assertEqual(session.thread_id, "t1")
        self.assertEqual(len(session.turns), 0)

    def test_adapt_session_missing_context_raises(self) -> None:
        with self.assertRaises(ValueError):
            adapt_session({"thread_id": "t1"})


if __name__ == "__main__":
    unittest.main()
