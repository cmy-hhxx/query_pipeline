"""simple_gate: deterministic rejection of high-confidence simple questions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from query_pipeline.steps.simple_gate_stage import simple_gate_reason


class SimpleGateTest(unittest.TestCase):
    def test_short_decision_rejected(self) -> None:
        for q in (
            "Ionq stock is buy or sell right now?",
            "Should I buy xau/usd now?",
            "茅台该不该卖",
            "要不要加仓宁德时代",
            "Is SOFI a good investment?",
            "Aapl buy or hold?",
        ):
            self.assertEqual(simple_gate_reason(q), "short_decision", q)

    def test_single_step_lookup_rejected(self) -> None:
        for q in (
            "贵州茅台今天的股价是多少",
            "what is the stock price of NVDA",
            "宁德时代今天涨了吗",
        ):
            self.assertEqual(simple_gate_reason(q), "single_step_lookup", q)

    def test_pure_screen_rejected(self) -> None:
        for q in (
            "帮我筛选出5只中药龙头股",
            "找出市盈率低于10的股票",
            "给我推荐3只低估值个股",
        ):
            self.assertEqual(simple_gate_reason(q), "pure_condition_screen", q)

    def test_followup_rejected(self) -> None:
        for q in (
            "again, give me the ranking",
            "okay understood now talk about other part",
            "那这些呢",
        ):
            self.assertEqual(simple_gate_reason(q), "context_dependent_followup", q)

    def test_genuinely_complex_kept(self) -> None:
        for q in (
            "帮我分析一下贵州茅台的估值并给出操作建议",
            "回测一个基于5周均线的短线择时策略，比较其与沪深300指数的超额收益",
            "Kindly provide me a High Probability Scalping Setup for XAUUSD",
            "帮我统计沪深主板所有股票第一天涨停第二天收小阳十字星的第三天涨跌情况",
        ):
            self.assertIsNone(simple_gate_reason(q), q)


if __name__ == "__main__":
    unittest.main()
