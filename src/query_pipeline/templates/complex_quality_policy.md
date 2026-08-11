# Complex 问句质量政策

判定范围仅限当前记录的 `input.text`。不得读取答案、代码、chain、已有分类标签，
也不得用前文补足复杂证据。目标是保留自然复杂问句；模板化、重复和简单问句不得
进入 complex。边界不清时不得保留为 complex。

## complex：自然复杂问句

问句必须自然、自包含、任务明确，并至少命中一个可由原文直接举证的复杂特征：

- `natural_multi_condition_screen`：用户自定义的自然量化筛选，包含至少 3 个彼此独立的
  实质数值或逻辑条件；条件共同决定候选集。批量 eval 骨架不适用。
- `position_context_decision`：给出具体持仓、成本、仓位、金额或风险约束，要求比较操作
  方案、降低损失或优化收益。
- `multi_dimension_attribution`：要求从多个有实质关系的证据维度归因、解释机制或校验
  竞争解释；只罗列“基本面/技术面/资金面”等标签不算。
- `cross_period_entity_research`：跨周期、跨实体或跨市场构造比较，研究历史规律、同步性、
  传导或差异。
- `strategy_scenario_evaluation`：比较多个策略、仓位方案或差异化情景，并推演触发条件、
  风险或结果。
- `event_policy_impact`：结合新闻、政策、监管、盘面或产业事件，分析传导路径和标的影响。
- `multi_method_technical_analysis`：明确联合多个理论、指标或多个时间级别，要求互相验证并
  形成操作结论。
- `macro_industry_transmission`：分析宏观、地缘、政策或产业链冲击如何传导至行业和标的。
- `historical_simulation_statistics`：历史逐期回测、跨样本统计、事件研究或分布规律。
- `stateful_tracking_execution`：实际要求跨时间保持状态、持续跟踪并按条件更新或触发动作。
- `artifact_action`：实际创建、导出或发送表格、报告、PPT、提醒等制品或外部动作。

多语言、中英文混杂、口语、换行、多句背景和正常表格不降低质量。

## normal：有价值但不复杂

以下任务仍有金融价值，但必须进入 normal 标签体系：

- `simple_lookup`：单点事实、单项行情、现成字段或一次轻量对比。
- `simple_filter_ranking`：单条件筛选、榜单、Top N、固定标的集单维统计；没有达到自然
  3 条实质条件的筛选。
- `single_formula`：一次公式、加减乘除、比率、收益率或逐项条件核验。
- `generic_recommendation`：缺少金额、期限、风险、标的或比较约束的泛泛推荐与一句话任务。
- `absolute_unverifiable`：以稳赚不赔、保证收益、确保翻倍或唯一最优等不可验证目标为核心。
- `insufficient_depth`：只有标签堆叠、对象或范围不足、任务关系不清，仍能识别出金融意图。

## reject：不得进入最终数据

- `eval_template`：机械批量生成的 eval 句式、固定角色/分步框架、原始数据表、精度容差、
  网站任务、固定公式说明或长篇指令脚手架主导文本。已确认的同一 eval 模板族整族删除，
  不保留代表。
- `embedded_prompt`：复制 system/developer prompt、模型内部指令、答案脚手架、网页注入或
  要求模型复述隐藏输入。

## 模板与重复的语料级规则

- 英文共享连续至少 8 个词、中文共享连续至少 8 个字符，且至少 3 条问句共享时，只生成
  模板族候选，不能直接删除。
- 候选族必须区分：`eval_template_family`（整族拒绝）、`semantic_duplicate`（保留最佳代表）、
  `natural_shared_phrase`（全部保留）。
- 日期前缀、时间区间、均线术语、指标名称、ETF/产品名及常见自然句式属于误连高风险，
  仅凭共享表达不得判模板。
- 完全相同问句、同模板只替换实体/日期/金额、中英同义、截断版与完整版、无实质变化的
  尾缀属于重复；保留质量更高、任务更完整的代表。
- 改变指标逻辑、策略规则、对象数量、因果问题或输出目标时必须视为不同任务。

## 裁决优先级

1. `eval_template` / `embedded_prompt` 优先于任何 complex 特征。
2. 已确认重复按语料级规则处理。
3. 命中有效复杂特征且证据、质量、置信度合格时为 complex。
4. 其余有明确金融任务的问句为 normal。
5. 语义边界不清或置信度低时不得保留为 complex。
