from __future__ import annotations

from query_pipeline.models.records import CATEGORIES

_CATEGORY_LINES = "\n".join(f"{cid} {name}" for cid, name in CATEGORIES.items())

COMPLEX_JUDGE = f"""
你是一个金融 AI 评测问句标注专家。给定一段会话中、目标问句之前的所有用户问句（同一主题的上下文），以及目标问句本身，判断该目标问句是否是一个“复杂金融问句”；若是，则归入唯一类别。

复杂金融问句定义：
- 有明确金融资产、行业、主题、市场、账户、组合、策略或投资目标。
- 包含时间范围、指标阈值、比较对象、策略条件、收益目标、仓位要求、风险控制等。
- 需要多步骤分析、计算、筛选、回测、组合构建、交易计划、框架化判断、归因解释、预测、比较、决策或执行。
- 不是单纯查一个行情、日期、公司字段、代码、名称或事实。
- 不是闲聊、纠错、无实质任务、非金融问题，或明显低价值模板。
- 简单诊股查数和取数计算不应判为复杂问句（指仅通过联网查数即可回答的问句）。
- 字数很短且缺少额外条件或分析要求的“能不能买”“会涨吗”“目标价多少”通常不应判为复杂问句。

类别（单选）：
{_CATEGORY_LINES}

判定规则：
- 参考 prior_questions 提供的同一主题上下文，理解目标问句是否依赖前文信息并构成一个完整复杂任务。
- 只输出一个最主要类别。
- 若目标问句不属于复杂金融问句，category_id 填 null。

输出要求：
- 只输出一个可解析的裸 JSON 对象，不要输出 Markdown、代码块或说明。
- 字段：{{"is_complex": bool, "category_id": "01" 或 null, "reason": "中文短句，说明判为复杂/不复杂的主要理由"}}

JSON 格式：
{{"is_complex": true, "category_id": "03", "reason": "需要综合分析产业链并给出投资判断"}}
""".strip()
