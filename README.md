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
| `configs/iwencai/config_0807.yaml` | `data/iwencai/0807.jsonl` | `judge_data` |

## 流程

输入 JSONL，由 `input.format` 决定形态：

- **session**：整条会话 → 主题切分 → 规则筛候选 → LLM 判复杂 → 组装 → 独立复核 → 去重/翻译
- **judge_data**：单题已带上下文，跳过切分与规则筛选，直接判复杂

| 步骤 | 做什么 |
|---|---|
| segment | LLM 按主题切会话；失败则整会话一段 |
| step1 | 规则：拒低价值/过短，再按工具调用数/链路步数/工具种类筛候选 |
| step2 | LLM `complex_judge`：结合同段上文，输出 `{is_complex, category_id, reason}` |
| step3 | 组装输出行（`context` 为同段上文；`trace_id` = turn 的 `run_id`） |
| step4 | LLM 独立复核（不带上下文），最多 3 轮级联、逐轮从严剔除；失败 fail-open |
| step5 | `dedup`（实体槽化 token-Jaccard≥0.80）→ `translate`（写入 `meta.translation`） |

## 断点续跑

杀进程后直接重跑即可：

- **LLM 缓存**：`work/.../llm_cache.jsonl`，按 (prompt, question, model) 命中
- **单元 checkpoint**：`work/.../checkpoints/`（session / verify / translate），已完成单元跳过

配置指纹或输入文件变化会自动失效旧 checkpoint。强制全量重跑：删 `checkpoints/`（或连同 `llm_cache.jsonl`）。

## 输出

目录约定：`data/{dataset}/` → `work/{dataset}/<name>/` → `outputs/{dataset}/`

- `outputs/aime/complex_queries_0807.jsonl` — 复杂 query（一行一条）
- `outputs/aime/summary_0807.json` — 计数汇总
- `work/aime/0807/` — 中间产物（`segments` / `candidates` / `judged` / `verified` / `deduped`）

输出行关键字段：`trace_id`、`category`、`input.text`、`context`、`tools`、`meta.reason`；`difficulty_level` 固定 `"hard"`。

## CSV 导出注意

`complex_queries_flat.csv` 由 `clean_script.jq` 从主输出生成，供 WPS/Excel 用。踩过的坑：

1. 字段内换行/Tab → 替换为空格
2. 内嵌 HTML/echarts → 替换为 `[HTML图表代码已省略]`
3. 单格 > 32767 字符会截断并错列 → 截到 ≤32000，完整数据留 jsonl
4. 缺 UTF-8 BOM → 中文乱码，需加 `EF BB BF`

jsonl 是无损源；CSV 仅为表格可读。
