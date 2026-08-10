"""Audit precision gate: LLM failures must FAIL the audit, never pass silently."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from query_pipeline.audit import _load_rows, audit_rows, render


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

    def test_load_rows_tolerates_bad_json(self) -> None:
        # 一行坏 JSON 不得崩溃整个 audit：坏行落盘 bad_lines 并跳过
        # （与管线 read_jsonl_with_bad_lines 同一套语义）。
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "input.jsonl"
            src.write_text(
                json.dumps(_row("t1", "正常问题")) + "\nnot-json-line\n[1,2]\n", encoding="utf-8"
            )
            rows = _load_rows(src)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["trace_id"], "t1")
            bad_path = tmp_path / "input.jsonl.bad_lines.jsonl"
            self.assertTrue(bad_path.exists())
            lines = bad_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)

    def test_non_dict_row_guarded_in_check(self) -> None:
        # check() 对非 dict 行兜底：按"无法判定"处理，不得 AttributeError 穿透。
        with patch("query_pipeline.audit.LLMClient", FakeClient):
            results = asyncio.run(audit_rows(["not-a-dict"]))
        self.assertEqual(results[0]["audit_errors"], 3)
        self.assertIn("FAIL", render(results, max_ratio=0.05))

    def test_error_ratio_uses_independent_threshold(self) -> None:
        # 第四轮 #9：错误率与非复杂率分用独立阈值。1/100 行无法判定（1% ≤ 非复杂
        # 阈值 5%）在旧实现会 PASS；默认 max_error_ratio=0 时任何一行判定失败即 FAIL。
        rows = [_row(f"t{i}", "正常问题") for i in range(99)] + [_row("t99", "boom 挂了")]
        results = self._audit(rows)
        text = render(results, max_ratio=0.05)
        self.assertIn("FAIL", text)
        self.assertIn("错误率 1.0% > 0%", text)

    def test_error_ratio_knob_relaxes_gate(self) -> None:
        rows = [_row(f"t{i}", "正常问题") for i in range(99)] + [_row("t99", "boom 挂了")]
        results = self._audit(rows)
        # 显式放宽错误率阈值后，1% 错误率可接受 → PASS（非复杂率 0%）
        text = render(results, max_ratio=0.05, max_error_ratio=0.05)
        self.assertIn("PASS", text)

    def test_conclusion_is_single_source_for_cli(self) -> None:
        # render 的 PASS/FAIL 与 cli 退出码必须共用 conclusion，禁止两处独立实现。
        from query_pipeline.audit import conclusion

        rows = [_row("t1", "正常问题"), _row("t2", "正常问题")]
        results = self._audit(rows)
        passed, ratio, error_ratio = conclusion(results, max_ratio=0.05, max_error_ratio=0.0)
        self.assertTrue(passed)
        self.assertEqual(ratio, 0.0)
        self.assertEqual(error_ratio, 0.0)
        self.assertIn("PASS", render(results, max_ratio=0.05))
        # 与 render 同一结论：单错误行 → conclusion FAIL 且 render FAIL
        bad = self._audit([_row("t1", "正常问题"), _row("t2", "boom 挂了")])
        passed2, ratio2, error_ratio2 = conclusion(bad, max_ratio=0.05, max_error_ratio=0.0)
        self.assertFalse(passed2)
        self.assertEqual(ratio2, 0.0)
        self.assertEqual(error_ratio2, 0.5)
        self.assertIn("FAIL", render(bad, max_ratio=0.05))


if __name__ == "__main__":
    unittest.main()
