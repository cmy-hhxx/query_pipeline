# query_pipeline

金融问句清洗标注管线：规则价值门控 → 语义价值门控 → 复杂度判定 → 双体系分类 → 复核 → 回答质量门控 → 去重/翻译，产出标准化的评测问句行（复杂 hard + 有价值普通 normal 两类同文件输出）。

## 运行

```bash
uv run query-pipeline run data/aime/0806.jsonl -o outputs/aime      # 格式自动识别，一行命令
uv run query-pipeline run data/iwencai/chat.jsonl --format chat     # 显式指定
uv run query-pipeline run input.jsonl --no-llm                      # 只跑规则（输出为空属正常）
uv run query-pipeline suggest data/aime/0806.jsonl                  # 门槛推荐：按候选数展示 10 个参数组合
```

`suggest` 纯规则扫描（不调 LLM），返回全谱 10 个门槛组合（候选数从低到高，
标注当前默认组合 `※默认`），选好后按示例参数跑：

```bash
uv run query-pipeline run data/aime/0806.jsonl --min-tool-calls 5 --min-unique-tools 3
```

常用旋钮：`--min-tool-calls`、`--min-unique-tools`、`--no-reject-rules`、
`--verify-rounds`、`--dedup-threshold`、`--model`、`--concurrency`、
`--api-key/--base-url`（覆盖 .env）。

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

混合/无法识别的文件**报错退出**（不静默处理）。坏行 → `outputs/<数据集>/logs/bad_lines.jsonl`；输入按 trace/thread id 去重；空 context 会话过滤。

## 流程（stage 可插拔，默认顺序）

```
preclean → segment → rule_gate → judge → verify → answer_gate → post
```

每个阶段是注册的模块（`register("name")(stage_fn)`），`--stages` / config 可自定义顺序。新增类别 = 改 `templates/` 下的 md；新增阶段 = 实现 `(ctx, client, cache, cache_lock) -> ctx` 并注册。

1. **preclean**：格式嗅探 + 坏行落盘 + 输入去重 + adapt 成统一 Session。
2. **segment**（session，LLM）：按主题切 2-4 段，失败回退整段。
3. **rule_gate**（规则，双格式生效）：噪声 reject 规则 + 工具门槛（chain 调用 ≥7 / 步骤 ≥1 / 工具种 ≥2，chat 同规则，阈值按数据可调）。
4. **judge**（LLM，解耦漏斗，每个候选 2-3 次调用，失败即弃）：
   - `value_gate`：是否有价值（有任务非闲聊 / 金融相关 / 不依赖不可见上文），无价值丢弃
   - `complexity_gate`：是否复杂（二分类，与打标签解耦）
   - `classify`：复杂 → 9 类（`complex-topic/{id}-{slug}`，difficulty=hard）；非复杂 → 16 类（`{id}-{slug}`，difficulty=normal）
5. **verify**（LLM，带前文问题作指代参考，准入标准不放松）：hard 5 轮 / normal 2 轮（可配），级联从严，任一轮与难度相悖即弃；LLM 失败丢弃（fail-closed）。
6. **answer_gate**（规则）：`meta.last_event_type` 必须为 `runFinished`（runCancelled/runInterrupted/runFailed/runExpired/feedbackUpsert 拒绝）+ 拒绝话术（中英）+ 截断标点 + 回答过短（<50 字）。
7. **post**：`dedup`（股票名词典槽化 + 同模板等价类合并 + token-Jaccard ≥ 0.80，倒排阻塞，10 万行秒级）→ `translate`（中文占比 < 30% 才翻译，写 `translation`，原文保留）。

`llm.enabled=false`：跳过所有 LLM 阶段，只跑规则，输出为空属正常。

## 输出

- `cleaned_queries.jsonl` — 复杂 + 普通问句一行一条（`difficulty_level` 区分）
- `summary.json` — 各阶段计数（input/bad/dup/empty、candidates、value_rejected、complex/normal、verify、answer_gate、dedup、translate、category 分布）+ success

输出行字段：`trace_id`、`source_case_id`、`category`、`input.text`、`context`（前文）、`chain`、`tools`、`raw_answer`/`text_answer`、`request_time_ms`、`translation`、`meta.reason`/`meta.run_id`/`meta.last_event_type`。完整口径见 `templates/filter_out.jsonc`；`meta` 是逃生字段区。

`success=false`（退出码 1）：discover 层出错（session_errors/llm_failed > 0、无会话、输入全为坏行）。失败且零输出时不覆盖上次产物。

## 断点续跑

杀进程后直接重跑，已完成单元跳过：LLM 缓存 + 阶段 checkpoint 都在 `outputs/<数据集>/logs/` 下（judge/verify/translate）。配置/输入/源码变化自动失效。强制全量重跑：删 logs/checkpoints 和 logs/llm_cache.jsonl。

## 质检（quality）

对输出 jsonl 做质量校验：全量规则 + LLM 抽检。

```bash
uv run query-pipeline-qc run --dataset aime --date 0806                 # 规则 + LLM 抽检（默认 5%）
uv run query-pipeline-qc run --dataset aime --date 0806 --no-llm
```

逐条规则含：结构、问句长度/乱码、分类（complex-topic/ 前缀校验）、chain、回答非空/截断/**拒绝话术**/**last_event_type**、时间与 token、翻译/理由元信息。数据集级规则含近重复（槽化 Jaccard ≥ 0.85）、类别偏斜、空值率、未知字段。

## 模板（可插拔的数据源）

| 文件 | 作用 |
|---|---|
| `templates/categories.md` | 分类体系唯一事实源（复杂 9 类 + 普通 16 类），运行时解析 |
| `templates/complex_few_shot.md` | 复杂 9 类定义与示例，拼装 classify_complex prompt |
| `templates/normal_few_shot.md` | 普通 16 类定义/适用/排除/边界/易混，拼装 classify_normal prompt |
| `templates/bad_cases_for_complex.md` | 已确认的复杂误判负例，注入 verify prompt |
| `templates/stock_names.txt` | 股票名称词典（槽化去重用），一行一个名称 |
| `templates/filter_out.jsonc` | 输出行字段口径参考 |
