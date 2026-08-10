"""Audit precision gate: LLM failures must FAIL the audit, never pass silently."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from query_pipeline.audit import audit_rows, render


class FakeClient:
    """Raises for questions containing 'boom', else returns is_complex."""

    def __init__(self, config: object) -> None:
        self.config = config
        self.calls = 0

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        payload = json.loads(user_prompt)  # audit prompt 是单行 JSON
        question = payload.get("question", "")
        if "boom" in question:
            raise RuntimeError("simulated API outage")
        return json.dumps({"is_complex": True, "reason": "多步分析"})

    async def close(self) -> None:
        return None


def _row(trace_id: str, question: str) -> dict:
    return {"trace_id": trace_id, "category": "complex-topic/01-…", "input": {"text": question}}


class AuditTest(unittest.TestCase):
    def _audit(self, rows: list[dict]) -> list[dict]:
        with patch("query_pipeline.audit.LLMClient", FakeClient):
            return asyncio.run(audit_rows(rows))

    def test_systematic_failure_fails_audit(self) -> None:
        # API 全挂：所有行 3 票全 error → 错误率 100% → 必须 FAIL（旧实现 PASS + exit 0）
        rows = [_row(f"t{i}", f"boom {i}") for i in range(3)]
        results = self._audit(rows)
        self.assertTrue(all(r["audit_errors"] == 3 for r in results))
        text = render(results, max_ratio=0.05)
        self.assertIn("FAIL", text)
        self.assertIn("无法判定 3 行", text)

    def test_missing_field_is_unable_to_judge(self) -> None:
        class NoFieldClient(FakeClient):
            async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
                return json.dumps({"reason": "没有 is_complex 字段"})

        rows = [_row("t1", "问题")]
        with patch("query_pipeline.audit.LLMClient", NoFieldClient):
            results = asyncio.run(audit_rows(rows))
        self.assertEqual(results[0]["audit_errors"], 3)
        self.assertIn("无法判定", render(results, max_ratio=0.05))

    def test_healthy_audit_passes(self) -> None:
        rows = [_row("t1", "查一下茅台现价"), _row("t2", "帮我算个平均值")]
        results = self._audit(rows)
        self.assertEqual([r["audit_errors"] for r in results], [0, 0])
        self.assertIn("PASS", render(results, max_ratio=0.05))

    def test_partial_failure_raises_error_ratio(self) -> None:
        # 一半行无法判定：错误率 50% > 5% → FAIL，即使非复杂率很低
        rows = [_row("t1", "正常问题"), _row("t2", "boom 挂了")]
        results = self._audit(rows)
        self.assertIn("FAIL", render(results, max_ratio=0.05))


if __name__ == "__main__":
    unittest.main()
