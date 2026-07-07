from __future__ import annotations

import re

from query_pipeline.rules.normalize import normalize_question

FINANCE_SIGNAL_RE = re.compile(
    r"(股|基金|ETF|指数|债|期货|板块|行业|财报|研报|公告|估值|市盈率|PE|PB|ROE|营收|利润|"
    r"毛利|现金流|资金|北向|南向|大盘|上证|深证|创业板|港股|美股|A股|公司|产业链|概念|"
    r"政策|宏观|利率|汇率|通胀|仓位|止盈|止损|买入|卖出|加仓|减仓|回测|K线|技术面|"
    r"基本面|消息面|资金面|上市|持仓|股东|实控|煤炭|产销量|融资余额|主力|控盘|分红|"
    r"派息|收益率|波动率|成交量|换手|市值|龙头|券商|银行|保险|有色|光伏|半导体|"
    r"算力|机器人)",
    re.I,
)


def finance_reject_reason(question: str) -> str:
    text = normalize_question(question)
    if len(text) < 20 and not FINANCE_SIGNAL_RE.search(text):
        return "short_no_finance_signal"
    return ""
