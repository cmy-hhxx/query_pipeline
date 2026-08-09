# query_pipeline

从多轮金融 agent 会话中抽取复杂金融问句的配置驱动流水线：规则初筛 → LLM 判定/复核 → 去重/翻译，产出标准化的评测问句行。

## 运行

```bash
uv run python run.py                      # 默认 configs/aime/config.yaml
uv run python run.py -c <config.yaml>     # 指定配置
uv run python run.py --dry-run            # 只校验配置，不跑
```

`.env` 提供 `OPENAI_BASE_URL` / `OPENAI_API_KEY`。

| 配置 | 输入 | 输出 |
|---|---|---|
| `configs/aime/config.yaml`（默认） | `data/aime/0807.jsonl` | `outputs/aime/complex_queries_0807.jsonl` |
| `configs/aime/config_0806.yaml` | `data/aime/0806.jsonl` | `outputs/aime/complex_queries_0806.jsonl` |
| `configs/iwencai/config_cn_expert_0805.yaml` | `data/iwencai/cn_expert_daily_2026-08-05.jsonl` | `outputs/iwencai/complex_queries_cn_expert_0805.jsonl` |
| `configs/iwencai/config_cn_expert_0806.yaml` | `data/iwencai/cn_expert_daily_2026-08-06.jsonl` | `outputs/iwencai/complex_queries_cn_expert_0806.jsonl` |
| `configs/iwencai/config_cn_expert_0807.yaml` | `data/iwencai/cn_expert_daily_2026-08-07.jsonl` | `outputs/iwencai/complex_queries_cn_expert_0807.jsonl` |

## 输入

JSONL，一行一条，由 `input.format` 决定解析 dialect（当前配置均为 `session`）：

- **session**：整条多轮会话。顶层 `thread_id` + `context[]`；每个 turn 含 `question`/`answer`/`trace_id`/`run_id`/`request_time`/`status`/`outcome`/`chain`/`tool_names`/`tool_count` 及 token、耗时字段。所有 turn 参与筛选。
- **chat**：judge_data 包装的单题。`judge_data.context`（前文）+ `judge_data.input.text`（目标问句）+ `chain`/`meta`；只取末轮为候选，不切分、不套 step1 门槛。

格式示例见 `data/*/session_sample.jsonl`、`data/*/chat_sample.jsonl`。无法解析的行与 adapt 失败的记录写入 `work/<dataset>/<name>/bad_lines.jsonl` 并跳过。

## 流程

```
输入 → adapt → segment → step1 初筛 → step2 判定 → assemble → verify 复核 → dedup → translate → 输出
```

`session` 走全流程；`chat` 跳过 segment 与 step1。各阶段可开关、阈值可配（见 `configs/*.yaml`），按会话/条目并发执行：

1. **adapt**：按 format 把输入行解析为统一 `Session`/`Turn`，dialect 差异在此收敛。
2. **segment**（仅 session，LLM）：按主题把会话切为 2-4 段；≤1 turn、`segmentation.enabled=false` 或 LLM 失败 → 整会话为一段。
3. **step1 规则初筛**（仅 session）：先 reject 规则（空、过短、低价值、嵌入页面/提示词、纯数字代码等），再 AND 门槛：工具调用数 ≥ `min_chain_tool_calls`（默认 7）、链路步数 ≥ `min_chain_steps`（1）、去重工具数 ≥ `min_unique_tools`（2）；chain 缺失时回退用 `tool_count`/`tool_names`。chat 只要求末轮 `status/outcome/question/answer` 合法（eligible）。
4. **step2 LLM 判定**：对每个候选问句，结合同段上文（段首回退整会话前文），输出 `{is_complex, category_id, reason}`；不复杂则丢弃，复杂则归入 9 类之一（`01`-`09`）。
5. **assemble**：组装输出行——`context`=同段前文、`category`=`{id}-{英文slug}`、`trace_id`/`meta.run_id` 透传、`difficulty_level="hard"`。
6. **verify**：单问句、不带上下文的独立复核，最多 `max_rounds`（默认 3）轮级联且逐轮从严；任一轮判不复杂即剔除；LLM 失败 fail-open 保留。
7. **post**：`dedup`（实体槽化 token-Jaccard ≥ 0.80，并查集聚类，每簇保留最长代表）→ `translate`（中文占比 < 30% 才翻译，写入 `meta.translation`/`translation`）。

`llm.enabled=false`：跳过 segment/step2/verify/translate，只跑规则，输出为空属正常。

## 输出

目录约定 `data/{dataset}/` → `work/{dataset}/<name>/` → `outputs/{dataset}/`：

- `complex_queries_{date}.jsonl` — 复杂问句，一行一条
- `summary_{date}.json` — 计数统计 + `success`

输出行关键字段：`trace_id`、`source_case_id`、`category`、`input.text`、`context`（前文）、`chain`、`tools`、`raw_answer`/`text_answer`、`request_time_ms`、`translation`（非中文问句的译文；原文是中文或翻译失败为 `null`）、`meta.reason`/`meta.run_id`。完整字段口径（含 session/chat 两种输入格式的取值来源）见 `templates/filter_out.jsonc`；`meta` 是逃生字段区，额外键原样保留。

`success=false` 条件（退出码 1）：discover 层出错（`session_errors`/`llm_failed` > 0、无会话、输入全为坏行）；verify/translate 失败 fail-open 不计失败。失败且零输出时不覆盖上一次产物；`debug.dump_intermediates=false` 时不写调试产物并清除上次残留。

## 断点续跑

杀进程后直接重跑，已完成单元跳过：

- **LLM 缓存** `work/<dataset>/<name>/llm_cache.jsonl`：按 `(step, model, prompt, 输入)` 哈希命中
- **单元 checkpoint** `work/<dataset>/<name>/checkpoints/`：`discover` / `verify` / `translate` 三份，按内容 key 命中

配置指纹、输入文件变化或 `src/query_pipeline` 源码改动 → 旧 checkpoint 自动失效。强制全量重跑：删 `checkpoints/`（可连同 `llm_cache.jsonl`）。

## 质检（quality）

对 `outputs/{dataset}/complex_queries_{date}.jsonl` 做质量校验：全量规则 + LLM 抽检（问句质量 / 标签归属）。源文件只读，产物写入 `work/<dataset>/<date>/qc/`。

```bash
uv run query-pipeline-qc run --dataset aime --date 0807                 # 规则 + LLM 抽检（默认 5%）
uv run query-pipeline-qc run --dataset aime --date 0807 --no-llm       # 只跑规则
uv run query-pipeline-qc run --dataset aime --date 0807 --ratio 0.02 --seed 42
```

- 逐条规则（失败 → `fail`）：结构合法、问句长度/乱码、分类 `{id}-{slug}`、chain 结构、回答非空/截断、时间与 token、翻译/理由元信息
- 数据集级规则：恒值字段、近重复问句（槽化 token-Jaccard ≥ 0.85）、长度离群、类别偏斜、空值率、未知字段
- LLM 抽检（默认 5%、seed 42）：低质/标签不符 → 单列 `needs_review`，不计入 fail
- LLM 缓存复用 `work/<dataset>/<date>/llm_cache.jsonl`

产物目录 `work/<dataset>/<date>/qc/`：`results.jsonl`（逐条）、`overview.json`（总览）、`sampled.jsonl`（抽检明细）、`report.md`（人读报告）、`bad_lines.jsonl`。

外部只读接口（Python API）：

```python
from query_pipeline.quality import overview, record_detail

overview("aime", "0807")                        # 总览：三档计数、规则命中、抽样判定、被标记列表
record_detail("aime", "0807", "<trace_id>")     # 单条：原始记录 + 规则明细 + LLM 判定
```
