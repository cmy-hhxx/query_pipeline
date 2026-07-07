from __future__ import annotations

import re

from query_pipeline.rules.normalize import normalize_question


def complexity_reasons(question: str) -> list[str]:
    text = normalize_question(question)
    reasons: list[str] = []
    if len(text) >= 30:
        reasons.append("len_ge_30")
    if len(text) >= 80:
        reasons.append("len_ge_80")
    if re.search(
        r"(为什么|原因|逻辑|怎么看|如何看|分析|比较|对比|区别|影响|风险|机会|前景|趋势|预测|判断|"
        r"评估|复盘|解读|梳理|框架|策略|配置|仓位|止盈|止损|买点|卖点|投资价值|值得|推荐|"
        r"排序|优先级)",
        text,
    ):
        reasons.append("analysis_or_judgement")
    if re.search(
        r"(基本面|技术面|资金面|消息面|估值|财务|政策|宏观|行业|产业链|供需|竞争格局|业绩|"
        r"公告|研报|现金流|利润|营收|PE|PB|ROE|换手|成交量|均线|MACD|KDJ|RSI)",
        text,
        re.I,
    ):
        reasons.append("finance_dimensions")
    if re.search(
        r"(同时|并且|以及|结合|从.*角度|分别|多个|多维|一方面|另一方面|如果|假设|情景|短期|"
        r"中期|长期|近[一二三四五六七八九十0-9]+[天月年]|过去|未来|三年|五年|年度|季度)",
        text,
    ):
        reasons.append("multi_constraint_or_horizon")
    if re.search(
        r"(应该|能不能|是否|还可以|可不可以|适合|怎么操作|怎么办|买|卖|加仓|减仓|持有|建仓|"
        r"清仓|止损|止盈)",
        text,
    ):
        reasons.append("decision_or_action")
    return reasons


def complexity_score(question: str) -> tuple[int, list[str]]:
    reasons = complexity_reasons(question)
    return len(reasons), reasons
