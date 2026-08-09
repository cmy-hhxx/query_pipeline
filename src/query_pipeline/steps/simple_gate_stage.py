"""Simple-question gate: deterministic rule check on hard rows only.

LLM judgments (complexity_gate / verify) can drift; this gate is the
counterweight — high-confidence simple-question patterns (short decisions,
single-step lookups, pure conditional screens, context-dependent follow-ups)
are rejected by exact rules, independent of the LLM. Only hard rows are
checked; normal rows are untouched. Patterns are deliberately narrow to avoid
false kills on genuinely complex questions.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from typing import Any

from query_pipeline.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

# 短决策：要不要/能不能/该不该 买/卖/持有 + 简短问句；"买还是卖/卖还是持有" 类。
# 注意：单字"买/卖"必须带决策词结构（该不该买/能不能卖）且句尾收束，
# 否则会误杀"给出买卖建议"这类分析请求里的"买/卖"。
_SHORT_DECISION: tuple[re.Pattern[str], ...] = (
    # 决策词必须收束在短句（<=30 字）句尾；长分析请求里的决策词不算短决策
    re.compile(r"^(?=.{0,30}$).{0,18}(能不能|可不可以|是否应该|该不该|要不要).{0,10}(买入|卖出|卖掉|加仓|减仓|补仓|清仓|持有|建仓|止盈|止损)[^。!！?？]{0,8}$"),
    re.compile(r"(该不该|要不要|能不能|可不可以|该不该做|值不值得|划不划算).{0,8}(买|卖|做空|短空|做多|买进|卖出)(?:吗|么|呢|\?|？)?$"),
    re.compile(r"(做空|短空|做多|买进|买入|卖出).{0,6}(划算吗|值不值|值吗|合适吗)(?:[。!！?？]|$)"),
    re.compile(r"^(买|卖|加仓|减仓|补仓|清仓|持有|建仓|止盈|止损|做空|短空|做多).{0,10}(吗|么|呢|还是)?$"),
    re.compile(r"^(buy|sell|hold|trim|add|exit|close|open).{0,24}(or|\?|？| now| today| rn)$", re.I),
    re.compile(r"^(is|are) .{0,40}(a buy|worth buying|a sell|a hold|good investment|a good stock)", re.I),
    re.compile(r"^should i (buy|sell|hold|trim|add|exit|short)", re.I),
    re.compile(r"^.{0,50}(buy or sell|buy or hold|sell or hold|sell or keep|hold or sell)", re.I),
)

# 单步查数：单一标的 + 一个数值问题，且句长短（无分析要求）
_SINGLE_LOOKUP: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(summarize|what is the (wall street |analyst )?(view|consensus|rating|target)|汇总|总结).{0,50}(评级|目标价|观点|consensus|rating|target)", re.I),
    re.compile(r"^.{0,40}(股价|价格|市值|涨跌幅|收益率|市盈率|最新价|收盘价|开盘价|换手率|成交量).{0,24}(是多少|是几|多少|是什么|什么价)$"),
    re.compile(r"^what('s| is) (the )?(price|stock price|market cap|pe|eps) (of|for) .{0,30}\??$", re.I),
    re.compile(r"^.{0,40}(涨了|跌了|涨没涨|跌没跌|涨吗|跌吗|涨到|跌到).{0,12}(吗|呢|\?|？)?$"),
)

# 纯条件筛选：找/筛/选/推 股票名单，无 计算/统计/回测/分析/预测/比较/组合 等任务词
_PURE_SCREEN = re.compile(
    # 整句无任务词（负向前瞻在匹配起点检查全文），再匹配"找/筛/选...+名单"骨架
    r"^(?!.*(回测|统计|计算|分析|预测|比较|组合|策略|评估|排名|资金|筹码|基本面|技术面|风险|机会|走势|支撑|压力))"
    r"(找|选|筛|推|挑|整理|给我|帮我|列出|抓取).{0,60}(只|支|个|批|名单|龙头|个股|股票|标的|etf|etfs)",
    re.I,
)

# 承接前文：承接词开头 + 短句（任务对象大概率来自前文）
_FOLLOWUP_START = re.compile(
    r"^(again|once more|revisa|de nuevo|repite|otra vez|yep|yes|no|okay|ok,|so,|and,|then|also|"
    r"what about|how about|再来|接着|那|这些|上面|刚才|还有|继续|再看看|再分析一下|那这个|那个)"
    r".{0,40}$",
    re.I,
)


def simple_gate_reason(question: str) -> str | None:
    """Return the rejection reason when the question is a high-confidence
    simple pattern, else None."""
    text = question.strip()
    if not text:
        return "blank"
    if _FOLLOWUP_START.match(text) and len(text) <= 60:
        return "context_dependent_followup"
    if any(p.search(text) for p in _SHORT_DECISION):
        return "short_decision"
    if any(p.search(text) for p in _SINGLE_LOOKUP):
        return "single_step_lookup"
    if _PURE_SCREEN.search(text):
        return "pure_condition_screen"
    return None


async def run_simple_gate_stage(
    ctx: PipelineContext,
    client: Any,
    cache: dict[str, dict[str, Any]],
    cache_lock: asyncio.Lock,
) -> PipelineContext:
    """Re-check hard rows with deterministic rules; drop high-confidence
    simple questions that the LLM let through."""
    if not ctx.rows:
        ctx.stats["simple_gate_rejected"] = 0
        return ctx
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in ctx.rows:
        if row.get("difficulty_level") != "hard":
            kept.append(row)
            continue
        inp = row.get("input")
        question = str(inp.get("text") or "") if isinstance(inp, dict) else ""
        reason = simple_gate_reason(question)
        if reason is None:
            kept.append(row)
        else:
            rejected.append(
                {
                    "trace_id": row.get("trace_id", ""),
                    "source_case_id": row.get("source_case_id", ""),
                    "category": row.get("category", ""),
                    "question": question[:200],
                    "reason": reason,
                }
            )
    ctx.rows = kept
    ctx.stats["simple_gate_rejected"] = len(rejected)
    if rejected:
        by_reason = Counter(r["reason"] for r in rejected)
        top = ", ".join(f"{k}={v}" for k, v in by_reason.most_common())
        logger.info("[simple_gate] rejected %d simple-looking hard row(s): %s", len(rejected), top)
    if rejected and ctx.config.debug.dump_intermediates:
        from query_pipeline.io.jsonl import write_jsonl

        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(ctx.path("simple_gate_rejected.jsonl"), rejected)
    return ctx
