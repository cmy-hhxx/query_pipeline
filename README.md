# query_pipeline

金融问句清洗标注管线：规则价值门控 → 语义价值门控 → 复杂度判定 → 双体系分类 → 复核 → 回答质量门控 → 去重/翻译，产出标准化的评测问句行（复杂 hard + 有价值普通 normal 两类同文件输出）。

## 运行

```bash
uv run query-pipeline run data/aime/0806.jsonl -o outputs/aime      # 格式自动识别，一行命令
uv run query-pipeline run data/iwencai/chat.jsonl --format chat     # 显式指定
uv run query-pipeline run input.jsonl --no-llm                      # 只跑规则（输出为空属正常）
uv run query-pipeline run input.jsonl --log-dir /var/log/query-pipeline --batch-id upstream-20260810
uv run query-pipeline precheck data/iwencai/opc_cn_expert_0807_0810.jsonl   # 数据预检：跑 LLM 前先查坏行/缺 chain/零合格 turn
uv run query-pipeline suggest data/aime/0806.jsonl                  # 门槛推荐：按候选数展示 10 个参数组合
```

`suggest` 纯规则扫描（不调 LLM），返回全谱 10 个门槛组合（候选数从低到高，
标注当前默认组合 `※默认`），选好后按示例参数跑：

```bash
uv run query-pipeline run data/aime/0806.jsonl --min-tool-calls 5 --min-unique-tools 3
```

`precheck` 纯规则单遍扫描（不调 LLM）：坏行占比 > 1%、合格 turn 的 chain
覆盖率 < 50%、0 个合格 turn 均判严重并中止；重复 / 空 context / 少量坏行只
警告。`run` 默认先跑预检（fail fast，避免浪费 LLM 资源），确认无误再进
LLM 阶段；整体缺 chain 的输入（如上游导出不完整）会被拦下，确属 end2end
输入可 `--allow-no-chain`，或 `--skip-precheck` 整体跳过。

常用旋钮：`--min-tool-calls`、`--min-unique-tools`、`--no-reject-rules`、
`--verify-rounds`、`--dedup-mode`、`--semantic-dedup-threshold`、`--model`、`--concurrency`、
`--api-key/--base-url`（覆盖 .env）、`--skip-precheck`、`--allow-no-chain`。

Python API（最少参数，其余全部默认）：

```python
from query_pipeline import run

summary = run("data/aime/0806.jsonl", output_dir="outputs/aime")
# summary: 各阶段计数 + 输出文件 + success
```

`.env` 提供 `OPENAI_BASE_URL` / `OPENAI_API_KEY`（API/CLI 自动加载）。

## 输入

JSONL，一行一条，格式**自动识别**（`format: auto`，可显式覆盖）：

- **session**：顶层 `thread_id` + `context[]`；每个 turn 含 `question`/`answer`/`trace_id`/`run_id`/`status`/`outcome`/`last_event_type`/`chain`/`tool_names`/`tool_count` 等。所有 turn 参与筛选。
- **chat**：judge_data 包装的单题。`judge_data.context`（前文）+ `judge_data.input.text`（目标问句）+ `chain`/`meta`；只取末轮为候选。chat 记录必然携带 `judge_data.chain`，工具门槛同样生效。

混合/无法识别的文件**报错退出**（不静默处理）。坏行 → `<work_dir>/runtime/diagnostics/bad_lines.jsonl`；输入按 trace/thread id 去重；空 context 会话过滤。

## 流程（stage 可插拔，默认顺序）

```
precheck → preclean → segment → rule_gate → judge → verify → answer_gate → post
```

每个阶段是注册的模块（`register("name")(stage_fn)`），`--stages` / config 可自定义顺序。新增类别 = 改 `templates/` 下的 md；新增阶段 = 实现 `(ctx, client, cache, cache_lock) -> ctx` 并注册。

1. **precheck**（规则）：单遍流式扫描原始输入——坏行占比超阈值 / 合格 turn 整体缺 chain（覆盖率 < 50%）/ 0 个合格 turn 立即中止；重复、空 context、少量坏行只警告。结果写 `summary.precheck`。
2. **preclean**：格式嗅探 + 坏行落盘 + 输入去重 + adapt 成统一 Session。
3. **segment**（session，LLM）：按主题切 2-4 段，失败回退整段。
4. **rule_gate**（规则，双格式生效）：噪声 reject 规则 + 工具门槛（chain 调用 ≥7（session）/ ≥3（chat，分布平坦，≥7 仅覆盖 ~1%）、步骤 ≥1、工具种 ≥2，阈值按数据可调；未显式设置的旋钮按嗅探到的输入格式补默认）。
5. **judge**（LLM，解耦漏斗，每个候选 2-3 次调用；调用失败不发布本批并允许重试）：
   - `value_gate`：输出 `has_executable_task/self_contained/template_severity/contains_embedded_prompt`。只有无任务、依赖不可见上文或严重提示模板才丢弃；泛泛结论、绝对化目标等仍交复杂度门降为 normal
   - `complexity_gate`：只接收当前问句，输出 `route/complex_features/exclusion_reasons/evidence/confidence/question_quality/semantic_signature`。自然、非模板化的 3+ 实质条件筛选进入 complex；单点、单条件、榜单、单公式、泛泛建议和绝对化目标进入 normal；严重 eval 模板和嵌入提示词进入 reject
   - `classify`：复杂 → 9 类（`complex-topic/{id}-{slug}`，difficulty=hard）；非复杂 → 16 类（`{id}-{slug}`，difficulty=normal）
6. **verify**（语料级 + 单问句 LLM 复核）：先用共享长表达、归一化骨架和语义签名生成模板/重复候选；共享 8 词/8 字符只产候选，族级裁决区分整族拒绝、语义重复保留最佳代表、自然共享表达全部保留。随后只把 `input.text` 交给独立复核：complex 保留，normal 重新走 16 类分类，reject 不输出。normal 初判不反向升级。网络、解析或 normal 分类失败会使整批不发布并等待重试，不会固化成业务标签。
7. **answer_gate**（规则）：`meta.last_event_type` 必须为 `runFinished`（runCancelled/runInterrupted/runFailed/runExpired/feedbackUpsert 拒绝）+ 拒绝话术（中英）+ 截断标点 + 回答过短（<50 字）。
8. **post**：Verify 已完成语料级去重时不重复执行；仅在自定义 stage 顺序跳过 Verify 时执行所选去重模式。随后按配置执行翻译。

`llm.enabled=false`：跳过所有 LLM 阶段，只跑规则，输出为空属正常。

## 输出

- `cleaned_queries.jsonl` — 复杂 + 普通问句一行一条（`difficulty_level` 区分）
- `summary.json` — `complex_rows/normal_rows/category_counts` 是初判统计；复核口径使用 `verify_complex_kept`、`verify_to_normal`、`verify_rejected_template`、`template_family_rejected`、`duplicate_removed`、`verify_uncertain`、`verify_failed`；另含最终 complex/normal 数量、复杂特征和类别分布

输出行字段：`trace_id`、`source_case_id`、`category`、`input.text`、`context`（前文）、`chain`、`tools`、`raw_answer`/`text_answer`、`request_time_ms`、`translation`、`meta.reason`/`meta.run_id`/`meta.last_event_type`，以及非破坏性扩展 `meta.complexity_profile` / `meta.semantic_signature`。完整口径见 `templates/filter_out.jsonc`。

## 日志

`run` 同时生成普通日志与三条独立业务流。`--log-dir` 默认是
`<output_dir>/logs`；`--batch-id` 可对齐上游批次，省略时生成
`YYYYMMDDTHHMMSS+0800_<8位随机串>`：

```text
<log_dir>/
├── ordinary/
│   ├── run/<batch_id>.log
│   ├── suggest/<batch_id>.log
│   ├── audit/<batch_id>.log
│   └── qc/<batch_id>.log
└── business/
    ├── cleaned/<batch_id>.log
    ├── complex/<batch_id>.log
    └── normal/<batch_id>.log
```

普通日志是结构化 JSON 行，文件默认 INFO、`-v` 为 DEBUG；终端仍是易读文本。
业务日志扩展名为 `.log`，内容保持一行一个最终输出 JSON 对象：记录完成翻译或确认无需翻译后立即 append + flush，按完成顺序写入。三条流可独立消费，因此 hard/normal 记录会同时出现在 cleaned 与对应类型流中。

相同 `batch_id` 用于续跑：已有业务行按规范化内容指纹跳过，缺失流会补齐；同批次并发运行由排他锁拒绝。日志不自动轮转或删除。`run` 的 API/CLI 摘要通过顶层 `logs` 返回 batch_id 及四类实际路径。

YAML 配置示例：

```yaml
logging:
  dir: outputs/aime/logs
  batch_id: upstream-20260810  # 可省略
  level: INFO                  # INFO | DEBUG
```

`success=false`（退出码 1）：无会话，或 `session_errors`、`llm_failed`、`verify_failed`、`template_family_failed`、`semantic_dedup_failed` 任一非零。输入坏行和翻译失败为 fail-open（记入 summary/诊断文件）；失败批次不发布当前结果，也不覆盖上次产物。

## 断点续跑

杀进程后直接重跑，已完成单元跳过：LLM 缓存在 `<work_dir>/runtime/cache/llm_cache.jsonl`，阶段 checkpoint 在 `<work_dir>/runtime/checkpoints/`（judge/verify/translate）。配置/输入/源码变化自动失效。强制全量重跑：删除这两个位置。

**judge 续跑是 cache-only**：judge checkpoint 只存每会话的完成标记与统计（rows/judged 含 MB 级 chain，不再落盘，由 llm_cache 确定性重建）。删 llm_cache.jsonl 后 judge 会全量重调 LLM（verify/translate 的 checkpoint 仍可复用）；存量旧格式 judge checkpoint（含 rows/judged）在加载时自动迁移为 stats-only。

## 质检（quality）

对输出 jsonl 做质量校验：全量规则 + LLM 抽检。

```bash
uv run query-pipeline-qc run --dataset aime --date 0806                 # 规则 + LLM 抽检（默认 5%）
uv run query-pipeline-qc run --dataset aime --date replay-001 --input /isolated/output/cleaned_queries.jsonl --ratio 1 --gold-gate
uv run query-pipeline-qc run --dataset aime --date 0806 --no-llm
```

QC 产物按日期分目录：`outputs/<数据集>/qc/<date>/`（results.jsonl / overview.json / report.md / sampled.jsonl，同一数据集不同日期互不覆盖）。

逐条规则含：结构、问句长度/乱码、分类（complex-topic/ 前缀校验）、chain、回答非空/截断/**拒绝话术**/**last_event_type**、时间与 token、翻译/理由元信息。LLM 抽检额外给出 `difficulty_ok`；数据集级规则含词面近重复、残余语义签名模板家族、类别偏斜、空值率、未知字段。`overview.quality_gate` 要求 hard 误收率 ≤2%、审计错误率 0、残余模板冗余率 ≤2%；发布回放加 `--ratio 1 --gold-gate`，额外要求人工正例召回 100%、已知负例误入 complex 为 0。

## 模板（可插拔的数据源）

| 文件 | 作用 |
|---|---|
| `templates/categories.md` | 分类体系唯一事实源（复杂 9 类 + 普通 16 类），运行时解析 |
| `templates/complex_few_shot.md` | 复杂 9 类定义与示例，拼装 classify_complex prompt |
| `templates/normal_few_shot.md` | 普通 16 类定义/适用/排除/边界/易混，拼装 classify_normal prompt |
| `templates/complex_quality_policy.md` | complex/normal/reject 的唯一质量口径，供 judge、verify、audit、QC 共用 |
| `templates/stock_names.txt` | 股票名称词典（仅 lexical fallback 使用），一行一个名称 |
| `templates/filter_out.jsonc` | 输出行字段口径参考 |
