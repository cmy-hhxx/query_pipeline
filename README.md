# query_pipeline

从多轮 agent 会话中抽取复杂金融 query 的配置驱动流水线。

## 运行

默认读 `configs/aime/config.yaml`，`.env` 提供 `OPENAI_BASE_URL` / `OPENAI_API_KEY`：

```bash
uv run python run.py --dry-run
uv run python run.py
uv run python run.py -c configs/aime/config.yaml --dry-run   # 临时换配置
```

| 配置 | 输入 | 格式 |
|---|---|---|
| `configs/aime/config.yaml`（默认） | `data/aime/0807.jsonl` | `session` |
| `configs/aime/config_0806.yaml` | `data/aime/0806.jsonl` | `session` |
| `configs/iwencai/config_cn_expert_0807.yaml` | `data/iwencai/cn_expert_daily_2026-08-07.jsonl` | `session` |

## 流程

输入 JSONL，由 `input.format` 决定形态：

- **session**：整条会话 → 主题切分 → 规则筛候选 → LLM 判复杂 → 组装 → 独立复核 → 去重/翻译
- **chat**（judge_data 包装）：单题已带上下文，不切分、不套 step1 规则门槛，末轮直接判复杂

| 步骤 | 做什么 |
|---|---|
| segment | LLM 按主题切会话；失败则整会话一段 |
| step1 | 规则：拒低价值/过短，再按工具调用数/链路步数/工具种类筛候选 |
| step2 | LLM `complex_judge`：结合同段上文，输出 `{is_complex, category_id, reason}` |
| step3 | 组装输出行（`context` 为同段上文；`trace_id` = turn 的 `trace_id`，`meta.run_id` = turn 的 `run_id`） |
| step4 | LLM 独立复核（不带上下文），最多 3 轮级联、逐轮从严剔除；失败 fail-open |
| step5 | `dedup`（实体槽化 token-Jaccard≥0.80）→ `translate`（写入 `meta.translation`） |

坏输入行与 `adapt_failed` 记录会追加进 `work/{dataset}/<name>/bad_lines.jsonl`。

## 断点续跑

杀进程后直接重跑即可：

- **LLM 缓存**：`work/.../llm_cache.jsonl`，按 (prompt, question, model) 命中
- **单元 checkpoint**：`work/.../checkpoints/`（discover / verify / translate），已完成单元跳过

配置指纹、输入文件变化，或 `src/query_pipeline` 下任何源码改动，都会自动失效旧 checkpoint。强制全量重跑：删 `checkpoints/`（或连同 `llm_cache.jsonl`）。

## 输出

目录约定：`data/{dataset}/` → `work/{dataset}/<name>/` → `outputs/{dataset}/`

- `outputs/aime/complex_queries_0807.jsonl` — 复杂 query（一行一条）
- `outputs/aime/summary_0807.json` — 计数汇总
- `work/aime/0807/` — 中间产物（`segments` / `candidates` / `judged` / `verified` / `deduped`）

输出行关键字段：`trace_id`、`category`、`input.text`、`context`、`tools`、`meta.reason`；`difficulty_level` 固定 `"hard"`。

运行状态写入 `summary.json` 的 `success` 字段：discover 层出错（`session_errors` / `llm_failed` > 0，或无会话、全部坏行）视为失败；verify / translate 失败是 fail-open，不计失败。失败且零输出时不覆盖上一次产物。`dump_intermediates=false` 时不写调试产物并清除上次运行残留。

## CSV 导出注意

`complex_queries_flat.csv` 由 `clean_script.jq` 从主输出生成，供 WPS/Excel 用。踩过的坑：

1. 字段内换行/Tab → 替换为空格
2. 内嵌 HTML/echarts → 替换为 `[HTML图表代码已省略]`
3. 单格 > 32767 字符会截断并错列 → 截到 ≤32000，完整数据留 jsonl
4. 缺 UTF-8 BOM → 中文乱码，需加 `EF BB BF`

jsonl 是无损源；CSV 仅为表格可读。

## 质检（quality）

对 `outputs/{dataset}/complex_queries_{date}.jsonl` 做质量校验：全量规则 + LLM 抽检（问句质量 / 标签归属）。源文件只读，产物写入 `work/{dataset}/{date}/qc/`。

```bash
uv run query-pipeline-qc run --dataset aime --date 0807              # 规则 + LLM 抽检（默认 5%）
uv run query-pipeline-qc run --dataset aime --date 0807 --no-llm    # 只跑规则，不联网
uv run query-pipeline-qc run --dataset aime --date 0807 --ratio 0.02 --seed 42
```

- 规则失败 → 记录 `fail`（确定）；LLM 判定低质/标签不符 → 单列 `needs_review`（概率性，不并入 fail）
- 逐条规则：结构合法性、问句长度/乱码、分类标签 `{id}-{slug}`、chain 结构、回答非空/截断、时间与 token 顺序、翻译/理由元信息
- 数据集级规则（挂总览）：字段恒值、近重复问句（实体槽化 token-Jaccard≥0.85）、长度离群、类别偏斜、空值率、未知顶层字段
- LLM 缓存复用 `work/{dataset}/{date}/llm_cache.jsonl`，重跑命中不重复调用

产物目录 `work/{dataset}/{date}/qc/`：`results.jsonl`（逐条）、`overview.json`（总览）、`sampled.jsonl`（抽检明细）、`report.md`（人读报告）、`bad_lines.jsonl`（无法解析的行）。

外部 agent 只读接口（Python API）：

```python
from query_pipeline.quality import overview, record_detail

overview("aime", "0807")                        # 总览：三档计数、规则命中、抽样判定、被标记列表
record_detail("aime", "0807", "<trace_id>")     # 单条：原始记录 + 规则明细 + LLM 判定
```
