"""precheck 纯规则扫描：坏行 / 缺 chain / 零合格 turn / 重复 / 空 context。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from query_pipeline.precheck import PrecheckIssue, precheck, render


def _turn(
    question: str = "帮我分析茅台估值",
    *,
    answer: str = "答" * 60,
    chain: list | None = None,
    status: str = "completed",
    outcome: str = "success",
) -> dict:
    return {
        "question": question,
        "answer": answer,
        "status": status,
        "outcome": outcome,
        "chain": chain if chain is not None else [{"plan": "", "tools": [{"name": "t", "input": {}, "output": "o"}]}],
    }


def _session(*turns: dict, thread_id: str = "t1") -> dict:
    return {"thread_id": thread_id, "context": list(turns)}


def _chat(
    *,
    text: str = "find me stocks",
    answer: str = "analysis result",
    chain: list | None = None,
    case_id: str = "c1",
    context: object | None = None,
) -> dict:
    return {
        "judge_data": {
            "case_id": case_id,
            "input": {"text": text},
            "context": [] if context is None else context,
            "chain": chain if chain is not None else [{"plan": "", "tools": [{"name": "t"}]}],
            "text_answer": answer,
        }
    }


def _write(tmp: Path, records: list[dict], *, raw_lines: list[str] | None = None) -> Path:
    p = tmp / "input.jsonl"
    lines = list(raw_lines) if raw_lines is not None else [json.dumps(r, ensure_ascii=False) for r in records]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


class PrecheckScanTest(unittest.TestCase):
    def test_session_ok(self) -> None:
        with self.subTest("auto format detection"):
            with tempfile.TemporaryDirectory() as tmp:
                path = _write(Path(tmp), [_session(_turn(), _turn()), _session(_turn(), thread_id="t2")])
                report = precheck(path)
                self.assertEqual(report.format, "session")
                self.assertTrue(report.ok)
                self.assertEqual(report.records, 2)
                self.assertEqual(report.turns, 3)
                self.assertEqual(report.eligible_turns, 3)
                self.assertEqual(report.turns_with_chain, 3)
                self.assertEqual(report.chain_coverage, 1.0)
                self.assertEqual(report.issues, [])
                self.assertIn("verdict: OK", render(report))
                d = report.as_dict()
                self.assertTrue(d["ok"])
                self.assertEqual(d["chain_coverage"], 1.0)

    def test_session_missing_chain_critical(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as tmp:
            path = _write(Path(tmp), [_session(_turn(chain=[]), _turn(chain=[]))])
            report = precheck(path)
            self.assertFalse(report.ok)
            codes = [i.code for i in report.issues]
            self.assertIn("missing_chain", codes)
            issue = next(i for i in report.issues if i.code == "missing_chain")
            self.assertEqual(issue.severity, "critical")
            self.assertEqual(report.chain_coverage, 0.0)
            self.assertIn("FAIL", render(report))

    def test_session_partial_chain_threshold(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as tmp:
            # 2/10 = 20% < 50% → critical
            turns = [_turn(chain=[]) for _ in range(8)] + [_turn() for _ in range(2)]
            path = _write(Path(tmp), [_session(*turns)])
            report = precheck(path)
            self.assertFalse(report.ok)
            self.assertAlmostEqual(report.chain_coverage, 0.2)
            # 8/10 = 80% ≥ 50% → ok
            turns = [_turn(chain=[]) for _ in range(2)] + [_turn() for _ in range(8)]
            path = _write(Path(tmp), [_session(*turns)])
            report = precheck(path)
            self.assertTrue(report.ok)
            # 阈值可调：0.9 时 80% 也失败
            report = precheck(path, min_chain_coverage=0.9)
            self.assertFalse(report.ok)
            # 0.0 完全放行
            report = precheck(path, min_chain_coverage=0.0)
            self.assertTrue(report.ok)

    def test_ratios_must_be_in_unit_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), [_session(_turn())])
            for value in (-0.1, 1.1):
                with self.subTest(name="min_chain_coverage", value=value), self.assertRaisesRegex(
                    ValueError, "min_chain_coverage"
                ):
                    precheck(path, min_chain_coverage=value)
                with self.subTest(name="max_bad_line_ratio", value=value), self.assertRaisesRegex(
                    ValueError, "max_bad_line_ratio"
                ):
                    precheck(path, max_bad_line_ratio=value)

    def test_ineligible_turns_not_counted_for_coverage(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as tmp:
            # 失败的 turn 无 chain 不拉低覆盖率；仅 1 个合格 turn 且带 chain
            failed = _turn(chain=[], status="cancelled", outcome="cancelled")
            path = _write(Path(tmp), [_session(failed, failed, _turn())])
            report = precheck(path)
            self.assertTrue(report.ok)
            self.assertEqual(report.eligible_turns, 1)
            self.assertEqual(report.turns_with_chain, 1)

    def test_no_eligible_turns_critical(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as tmp:
            # 整批 run 失败
            path = _write(Path(tmp), [_session(_turn(status="cancelled", outcome="cancelled"))])
            report = precheck(path)
            self.assertFalse(report.ok)
            codes = [i.code for i in report.issues]
            self.assertIn("no_eligible_turns", codes)
            # 缺 question / answer
            path = _write(Path(tmp), [{"thread_id": "t1", "context": [{"status": "completed"}]}])
            report = precheck(path)
            self.assertFalse(report.ok)
            self.assertIn("no_eligible_turns", [i.code for i in report.issues])

    def test_bad_lines_ratio(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as tmp:
            # 1 坏 1 好 → 50% > 1% → critical
            raw = ["not json {{{", json.dumps(_session(_turn()))]
            path = _write(Path(tmp), [], raw_lines=raw)
            report = precheck(path)
            self.assertFalse(report.ok)
            self.assertIn("bad_line_ratio_exceeded", [i.code for i in report.issues])
            # 少量坏行（≤1%）只 warning
            raw = ["bad line"] + [json.dumps(_session(_turn())) for _ in range(200)]
            path = _write(Path(tmp), [], raw_lines=raw)
            report = precheck(path)
            self.assertTrue(report.ok)
            self.assertIn("bad_lines", [i.code for i in report.issues])

    def test_bad_line_ratio_threshold_zero_disabled(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as tmp:
            raw = ["bad line"] + [json.dumps(_session(_turn())) for _ in range(200)]
            path = _write(Path(tmp), [], raw_lines=raw)
            report = precheck(path, max_bad_line_ratio=0.0)
            self.assertFalse(report.ok)

    def test_duplicates_and_empty_context_warnings(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as tmp:
            recs = [
                _session(_turn(), thread_id="t1"),
                _session(_turn(), thread_id="t1"),
                {"thread_id": "t2", "context": []},
            ]
            path = _write(Path(tmp), recs)
            report = precheck(path)
            self.assertTrue(report.ok)
            codes = [i.code for i in report.issues]
            self.assertIn("duplicate_records", codes)
            self.assertIn("empty_context_records", codes)

    def test_chat_ok_and_missing_chain(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as tmp:
            path = _write(Path(tmp), [_chat(), _chat(case_id="c2")])
            report = precheck(path)
            self.assertEqual(report.format, "chat")
            self.assertTrue(report.ok)
            self.assertEqual(report.eligible_turns, 2)
            self.assertEqual(report.chain_coverage, 1.0)

            path = _write(Path(tmp), [_chat(chain=[]), _chat(chain=[])])
            report = precheck(path)
            self.assertFalse(report.ok)
            self.assertIn("missing_chain", [i.code for i in report.issues])

    def test_chat_requires_adapter_eligible_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), [_chat(answer="")])
            report = precheck(path)
            self.assertFalse(report.ok)
            self.assertEqual(report.eligible_turns, 0)
            self.assertIn("no_eligible_turns", [i.code for i in report.issues])

            path = _write(Path(tmp), [_chat(context="not-a-list")])
            report = precheck(path)
            self.assertFalse(report.ok)
            self.assertEqual(report.eligible_turns, 0)
            self.assertIn("no_eligible_turns", [i.code for i in report.issues])

    def test_chat_missing_judge_data(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as tmp:
            path = _write(Path(tmp), [{"judge_data": {"input": {"text": "x"}}}, {"thread_id": "t1"}])
            report = precheck(path, format="chat")
            self.assertEqual(report.empty_context_records, 1)
            self.assertIn("empty_context_records", [i.code for i in report.issues])

    def test_mixed_or_undetectable_format_raises(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as tmp:
            mixed = _write(Path(tmp), [_session(_turn()), _chat()])
            with self.assertRaises(ValueError):
                precheck(mixed)
            undetectable = _write(Path(tmp), [{"foo": "bar"}])
            with self.assertRaises(ValueError):
                precheck(undetectable)
            # 显式 format 绕过嗅探
            report = precheck(mixed, format="session")
            self.assertEqual(report.format, "session")

    def test_explicit_format_skips_sniff(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as tmp:
            path = _write(Path(tmp), [_chat()])
            report = precheck(path, format="chat")
            self.assertEqual(report.format, "chat")


if __name__ == "__main__":
    unittest.main()
