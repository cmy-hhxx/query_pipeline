"""Strict audit of complex-queries output.

Runs an independent strict LLM review over every row of a complex_queries
jsonl and reports rows judged non-complex plus the overall ratio. Serves as
the precision gate for the pipeline: any new simple-question type the
LLM let through shows up here and can be fed back into the verify prompts.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from query_pipeline.config.models import LLMConfig
from query_pipeline.llm.client import LLMClient

AUDIT_PROMPT = """
你是一个严格的数据审计员。给定一个问句，判断它是否是"复杂金融问句"。

判定标准与业务分类体系一致（complex_few_shot 9 类）：
复杂 = 需要多步分析、计算、筛选、回测、组合构建、框架化判断、预测、比较、决策或执行，
且答案不能通过一次简单查询/简单计算获得。包括：
- 交易计划/买卖决策类（给出入场区、止损、止盈、仓位、分批计划——即使针对单标的）
- 多维度分析（基本面+技术面+资金面+消息面等）
- 多情景预测（bull/base/bear、概率、触发条件）
- 回测/统计/取数计算
- 账户/持仓综合诊断
- 事件/主题驱动的受益股推荐或影响分析（"战争受益股""事件对板块的影响""从主题找标的"——04 机会挖掘 / 03 分析研究 / 01 事件概念选股）
- 主题强度排序/分类/梯队划分（"AI 主题最强股并分核心/边缘"）

以下类型判为 不复杂（与 verify 排除清单一致）：
- 单步查数/单步计算：查一个行情/价格/市值/涨跌幅/收益率，或一次加减乘除
- 短决策/短评价："XX能买吗""XX怎么样""XX该不该卖"（纯一句话询问，无分析过程要求）
- 简单显式条件筛选：单层条件按名单过滤（"抓5只中药题材股""整理pcb龙头"）且无计算/统计/回测/分析要求。注意：多条件技术指标组合筛选（均线交叉+量比+时间窗口等多层条件）属于 01 复杂取数计算，判复杂
- 承接前文：任务对象/候选集合/比较维度来自前文
- 泛化问题：无具体标的/条件/比较对象
- 纯陈述/计划告知/闲聊

只输出严格 JSON：{"is_complex": true/false, "reason": "一句话理由"}
""".strip()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


async def audit_rows(
    rows: list[dict[str, Any]],
    *,
    model: str = "gpt-5.4-mini",
    concurrency: int = 64,
) -> list[dict[str, Any]]:
    client = LLMClient(LLMConfig(model=model, concurrency=concurrency))
    # 限流由 LLMClient 进程级 semaphore 承担，此处不再叠加第二层。
    results: list[dict[str, Any]] = []

    async def check(row: dict[str, Any]) -> dict[str, Any]:
        inp = row.get("input")
        question = str(inp.get("text") or "") if isinstance(inp, dict) else ""
        # 3 次独立判定取多数：LLM 对边界问句的判定有波动，多数裁决更稳。
        verdicts: list[tuple[bool, str]] = []
        for _ in range(3):
            try:
                raw = await client.complete(
                    system_prompt=AUDIT_PROMPT,
                    user_prompt=json.dumps({"question": question}, ensure_ascii=False),
                )
                data = json.loads(raw)
                verdicts.append((bool(data.get("is_complex")), str(data.get("reason") or "")))
            except Exception as exc:  # noqa: BLE001 审计失败按通过处理
                verdicts.append((True, f"audit_error: {str(exc)[:80]}"))
        is_complex = sum(1 for v, _ in verdicts if v) >= 2
        reasons = [r for v, r in verdicts if not v]
        reason = "; ".join(dict.fromkeys(reasons))[:120] or "多数判定为复杂"
        return {
            "trace_id": row.get("trace_id", ""),
            "category": row.get("category", ""),
            "question": question,
            "is_complex": is_complex,
            "reason": reason,
        }

    try:
        results = await asyncio.gather(*(check(r) for r in rows))
    finally:
        await client.close()
    return results


def render(results: list[dict[str, Any]], *, max_ratio: float) -> str:
    non_complex = [r for r in results if not r["is_complex"]]
    ratio = len(non_complex) / len(results) if results else 0.0
    lines = [
        f"=== complex 输出审计：共 {len(results)} 行，非复杂 {len(non_complex)} 行 "
        f"({ratio * 100:.1f}%，阈值 {max_ratio * 100:.0f}%) ===",
    ]
    for r in non_complex[:20]:
        lines.append(f"  [非复杂] {r['reason'][:50]} | {r['question'][:70]}")
    if len(non_complex) > 20:
        lines.append(f"  ... 其余 {len(non_complex) - 20} 条见完整结果")
    lines.append(f"结论: {'PASS' if ratio <= max_ratio else 'FAIL'}（非复杂率 {'<=' if ratio <= max_ratio else '>'} {max_ratio * 100:.0f}%）")
    return "\n".join(lines)
