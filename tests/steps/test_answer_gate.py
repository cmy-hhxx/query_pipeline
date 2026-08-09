"""Answer quality gate: structural + content signals, rules only."""

from __future__ import annotations

import unittest

from query_pipeline.steps.answer_gate_stage import answer_gate_reason

LONG = "这是一段足够长的正常回答。" * 20  # > 50 chars, no dangling end

def _row(**overrides) -> dict:
    row = {
        "trace_id": "t1",
        "difficulty_level": "hard",
        "input": {"text": "帮我分析一下某股票走势"},
        "text_answer": LONG,
        "raw_answer": LONG,
        "meta": {"last_event_type": "runFinished"},
    }
    row.update(overrides)
    return row

class AnswerGateTest(unittest.TestCase):
    def test_pass(self) -> None:
        self.assertIsNone(answer_gate_reason(_row()))

    def test_event_type_rejects(self) -> None:
        for event in ("runCancelled", "runInterrupted", "runFailed", "runExpired", "feedbackUpsert"):
            reason = answer_gate_reason(_row(meta={"last_event_type": event}))
            self.assertEqual(reason, f"last_event_type={event}")

    def test_no_event_field_ok(self) -> None:
        # chat rows carry no event field -> content signals only
        self.assertIsNone(answer_gate_reason(_row(meta={})))

    def test_refusal_phrases(self) -> None:
        cases = [
            "抱歉，我无法回答这个问题。",
            "我不能提供投资建议。",
            "作为AI助手，我无法执行该操作。",
            "I'm sorry, I can't answer that.",
            "As an AI assistant, I cannot provide financial advice.",
            "I refuse to answer.",
        ]
        for text in cases:
            self.assertEqual(answer_gate_reason(_row(text_answer=text, raw_answer=text)), "refusal", text)

    def test_dangling_punctuation_truncation(self) -> None:
        for end in (",", "，", ":", "：", "-", "—"):
            text = LONG + end
            self.assertEqual(answer_gate_reason(_row(text_answer=text)), "truncated_dangling_punctuation", end)
        # sentence-enders are fine
        for end in (".", "。", "!", "！", "…"):
            self.assertIsNone(answer_gate_reason(_row(text_answer=LONG + end)))

    def test_too_short(self) -> None:
        self.assertEqual(answer_gate_reason(_row(text_answer="好的")), "answer_too_short(2<50)")

    def test_empty_answer(self) -> None:
        self.assertEqual(answer_gate_reason(_row(text_answer="")), "empty_answer")

if __name__ == "__main__":
    unittest.main()
