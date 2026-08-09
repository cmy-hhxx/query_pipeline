"""Funnel response parsing: strict boolean validation (fail-closed on garbage)."""

from __future__ import annotations

import unittest

from query_pipeline.session.funnel import parse_complexity_response, parse_value_response


class FunnelParseTest(unittest.TestCase):
    def test_string_false_is_false(self) -> None:
        # bool("false") == True —— 手工转换会静默放行；pydantic 正确解析为 False
        result = parse_value_response({"is_valuable": "false"})
        self.assertFalse(result.is_valuable)
        complexity = parse_complexity_response({"is_complex": "no"})
        self.assertFalse(complexity.is_complex)

    def test_string_true_is_true(self) -> None:
        self.assertTrue(parse_value_response({"is_valuable": "true"}).is_valuable)

    def test_missing_field_fails_closed(self) -> None:
        # 缺字段 = ValidationError = ValueError 子类 → 候选丢弃（fail-closed）
        with self.assertRaises(ValueError):
            parse_value_response({"reason": "no verdict"})

    def test_garbage_value_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            parse_value_response({"is_valuable": "maybe"})
        with self.assertRaises(ValueError):
            parse_complexity_response({"is_complex": 42})

    def test_reason_optional(self) -> None:
        self.assertIsNone(parse_value_response({"is_valuable": False}).reason)
        self.assertEqual(
            parse_complexity_response({"is_complex": True, "reason": "r"}).reason, "r"
        )


if __name__ == "__main__":
    unittest.main()
