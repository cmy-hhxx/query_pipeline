from __future__ import annotations

CATEGORIES: dict[str, str] = {
    "01": "数据与指标计算",
    "02": "预测与推演",
    "03": "分析研究",
    "04": "机会挖掘",
    "05": "资产配置",
    "06": "投资决策",
    "07": "策略触发与设置",
    "08": "目标执行",
    "09": "动作输出",
}

# English slugs mirror fin_bench's SKILL/fulllink/complex-topic directory names
# so downstream tooling can join on either id or slug.
# NOTE: 07's slug "strategy-evaluation" is the fin_bench directory name (a join key),
# NOT a semantic label — the category's meaning was redefined to 策略触发与设置
# (trigger/alert/monitor), and the old 策略评估/回测审计 meaning moved to 03.
# Do not "fix" the slug to match the CN name; that would break the fin_bench join.
ENGLISH_CATEGORIES: dict[str, str] = {
    "01": "data-metrics-calculation",
    "02": "forecasting-and-projection",
    "03": "analysis-research",
    "04": "opportunity-discovery",
    "05": "asset-allocation",
    "06": "investment-decision",
    "07": "strategy-evaluation",
    "08": "goal-execution",
    "09": "action-output",
}

CATEGORY_KEYS: tuple[str, ...] = tuple(CATEGORIES.keys())
