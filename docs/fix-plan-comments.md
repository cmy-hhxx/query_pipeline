# 修复计划 — comments.md 10 个必须修复的点

> 依据：`comments.md` 审查结论 + 全量代码通读复核（每个问题均已在本轮重新确认）。
> 基线：`uv run pytest` 179 passed；`pyright src` 0 error（修复后需保持/回归）。
> 原则：每项修复配回归测试；一个 commit 对应一个 issue；不引入新依赖。

---

## 0. 通用约定

- 每项完成后跑对应测试文件：`uv run pytest tests/<相关文件> -q`
- 全部完成后：`uv run pytest -q` + `pyright src`
- 测试全部用现有 fake LLM client 模式（无真实 LLM 调用）

---

## 1. [高] `api.run` format="auto" 用错门槛（7/2 vs 3/2）+ 门槛参数静默语义缺陷（#1、#2 合并修复）

**根因**：门槛默认值被硬编码在 `api.py` 的分支里，而真实格式要到 preclean 嗅探后才知道；
且 `RuleGateConfig` 的默认值（7/1/2）在 config 文件路径同样存在（configs 里显式写了才没问题）。
**方案（系统级修复，采纳审查意见的首选："门槛决策延迟到嗅探之后"）**：

1. `src/query_pipeline/config/models.py` — `RuleGateConfig` 两个旋钮改为"未设置"哨兵：
   ```python
   min_chain_tool_calls: int | None = None  # None -> 按实际输入格式取默认（session 7 / chat 3）
   min_unique_tools: int | None = None      # None -> 默认 2
   ```
   （`enabled`/`reject_rules`/`min_chain_steps` 不变）
2. `src/query_pipeline/steps/rule_gate_stage.py` — 嗅探后补齐默认值：
   ```python
   _FORMAT_DEFAULTS = {"session": (7, 2), "chat": (3, 2)}

   def _effective_gate(cfg: RuleGateConfig, fmt: str) -> RuleGateConfig:
       calls, tools = _FORMAT_DEFAULTS.get(fmt, _FORMAT_DEFAULTS["session"])
       return cfg.model_copy(update={
           "min_chain_tool_calls": calls if cfg.min_chain_tool_calls is None else cfg.min_chain_tool_calls,
           "min_unique_tools": tools if cfg.min_unique_tools is None else cfg.min_unique_tools,
       })
   ```
   在 `run_rule_gate_stage` 开头：`gate = _effective_gate(cfg.rule_gate, ctx.stats.get("input_format", "session"))`，
   后续 `select_candidates/select_last_only` 用 `gate`。
3. `src/query_pipeline/api.py:111-122` — 删掉格式分支，改为无条件透传（同时修复 #2a/#2b）：
   ```python
   rule_gate = RuleGateConfig(
       min_chain_tool_calls=min_tool_calls,   # None -> 阶段按嗅探格式补默认
       min_unique_tools=min_unique_tools,
       reject_rules=reject_rules,             # 无论是否显式传门槛都生效
   )
   ```
   删除已过时的注释（api.py:107-109、131-132）。

**效果**：#1 auto 不再用错门槛（chat→3/2，session→7/2）；#2a `--no-reject-rules` 永远生效；
#2b 单旋钮不再把另一旋钮重置为 0/1，而是按格式默认补齐；config 文件路径同源修复；
suggest 的 is_default（7/2、3/2）与实际执行门槛重新一致。

**测试**（`tests/test_api.py`）：
- `test_auto_chat_uses_chat_gate`：chat 输入、chain 4 次调用 2 种工具（≥3/2 但 <7/2）→ 候选保留 → `complex_rows == 1`
- `test_auto_session_uses_session_gate`：session 输入、单轮 4 次调用 → 候选 0 → `complex_rows == 0`
- `test_no_reject_rules_effective`：唯一候选问句为"好的"（命中 reject 的 LOW_VALUE_COMMON）→ 默认 `complex_rows == 0`；`reject_rules=False` → `complex_rows == 1`
- `test_single_knob_keeps_format_default`：chat 输入、chain 5 次调用但仅 1 种工具、只传 `min_tool_calls=5` → 旧代码 (5,1) 放行，新代码 min_unique_tools 补 chat 默认 2 → 候选 0

---

## 2. [高] QC `_dataset_category_skew` 零类别误报（#3）

**根因**：`rules.py:311-316` 用 `str(cat).split("-")[0]` 从 category path 反推 id，
complex 路径 `complex-topic/09-…` 得 `"complex"`、normal id 泄漏进集合。

**方案**（`src/query_pipeline/quality/rules.py`，`_dataset_category_skew` 内）：
```python
complex_present: set[str] = set()
for cat in counts:
    if cat.startswith(COMPLEX_PREFIX):                       # 只统计 complex 行
        complex_present.add(cat[len(COMPLEX_PREFIX):].split("-", 1)[0])
zero_ids = [cid for cid in load_taxonomy().complex if cid not in complex_present]
```
（`COMPLEX_PREFIX` 已 import；"other" 等 normal 值天然被前缀过滤）

**测试**（`tests/quality/test_rules.py` `DatasetRuleTest`）：
- 场景 A：仅 complex 09 行 → category_skew 的 detail 不含"零记录类别：09"（且含 01-08）
- 场景 B：仅 normal 01 行 → 仍报 complex 01 零记录

---

## 3. [高] QC `_check_chain` 与 end2end 输出语义冲突（#4）

**根因**：管线合法产出 `capture_mode="end2end"`（输入无 chain → 空 chain，靠 tool_count 过门槛），
`rules.py:115-118` 无条件判空 chain 为 fail。

**方案**（`src/query_pipeline/quality/rules.py`，`_check_chain` 开头）：
```python
if row.get("capture_mode") == "end2end":
    tools = row.get("tools")
    if not isinstance(tools, list) or not tools:
        return False, "end2end 行 tools 缺失或为空"
    return True, "ok"
```
full_link 行校验逻辑不变。

**测试**（`tests/quality/test_rules.py`）：
- end2end 行（`capture_mode="end2end"`、`chain=[]`、`tools=["web_search"]`）→ chain 规则 ok
- end2end 行且 `tools=[]` → chain 规则 fail
- 现有 `test_chain_empty`（full_link 默认行）继续 fail —— 保留，不受影响

---

## 4. [高·缓存一致性] verify checkpoint 陈旧重放（#5）

**根因**：verify 键不含 `prior_questions`（参与 LLM 判定）与 `difficulty_level`（决定轮数与期望）；
`stage_meta` 只为 judge 附加输入 stat。

**方案**：
1. `src/query_pipeline/steps/verify_stage.py` worker 内：先算 `prior = _prior_questions(row)`，
   键改为：
   ```python
   key = content_key(
       str(row.get("source_case_id", "")),
       str(row.get("trace_id", "")),
       question,
       difficulty,                  # 决定 max_rounds 与 expected
       "\n".join(prior),           # 参与判定，前文变化必须换键
   )
   ```
   （`user_prompt` 复用同一个 `prior`，避免重复计算）
2. `src/query_pipeline/io/checkpoint.py` `stage_meta`：judge 分支扩展为
   `if stage in {"judge", "verify", "translate"}:` 都附加 input size/mtime。

**测试**（`tests/io/test_checkpoint.py`）：
- 单元：`stage_meta(cfg, "verify")` / `stage_meta(cfg, "translate")` 含 `input_size/input_mtime_ns`
- 单元：verify 键对 prior_questions、difficulty 敏感（同 (case, trace, question) 不同前文/难度 → 键不同）
- 集成 A（输入变化 → 全量重算）：改写输入文件后重跑 → verify 阶段重新发起 LLM 调用（现 test_input_change_invalidates_session_checkpoint 只覆盖 judge，补 verify 断言）
- 集成 B（difficulty 变化 → 不重放旧裁决）：run1 judge 判 hard、verify 记录 keep；`patch.dict("query_pipeline.prompts.PROMPTS", {"complexity_gate": "新内容"})` 使 judge fingerprint 失效（verify fingerprint 不含该 prompt，不失效）→ run2 judge 重跑判 normal → 断言 verify checkpoint 出现第二个键、`verify_rejected == 1`（旧代码会重放 hard 的 keep 导致误保留）

---

## 5. [高] funnel 布尔字段字符串真值 fail-open（#6）

**根因**：`funnel.py:54,59` 手工 `bool()`：`bool("false") == True`。

**方案**（`src/query_pipeline/session/funnel.py`）：
```python
def parse_value_response(raw):
    return ValueResult.model_validate(_as_dict(raw))

def parse_complexity_response(raw):
    return ComplexityResult.model_validate(_as_dict(raw))
```
pydantic 已实证：`"false"/"no"` → False（正确处理），`"x"/42/None` → ValidationError（⊂ ValueError，
被 funnel 的 `except (ValueError, RuntimeError)` 捕获 → fail-closed 丢弃候选）。缓存路径存的是
`model_dump()` 后的真布尔，回放无影响。

**测试**（新建 `tests/session/test_funnel.py` + `tests/pipeline/test_contract.py`）：
- 单元：`parse_value_response('{"is_valuable": "false"}')` → `is_valuable is False`；`"x"` → 抛 ValueError
- 集成：fake client 的 value gate 返回 `{"is_valuable": "false"}` → `value_rejected == 1`、`complex_rows == 0`（fail-closed）

---

## 6. [中高] preclean 对非 dict `judge_data` 崩溃而非进坏行（#7）

**根因**：`sniff.py:56` key_fn 直接 `.get()` truthy 非 dict → AttributeError 炸整条管线。

**方案**（`src/query_pipeline/io/sniff.py`）：chat key_fn 改为：
```python
def _chat_key(record: dict[str, Any]) -> str:
    judge = record.get("judge_data")
    case_id = judge.get("case_id") if isinstance(judge, dict) else None
    return str(case_id or record.get("trace_id") or "")
```
key 为空 → 行保留 → `adapt_chat` 抛 `ValueError("record missing judge_data object")` →
走既有 adapt-failed 路径进 `bad_lines.jsonl`（与 #7 审查意见"异常行按 bad_lines 处理"一致，零新机制）。

**测试**：
- `tests/io/test_sniff.py`：`preclean_records([{"judge_data": "not-a-dict"}], CHAT)` 不抛、行保留
- `tests/pipeline/test_contract.py`：整管线跑 chat 输入含 `judge_data` 为字符串的行 →
  `input_bad_lines == 1`、bad_lines.jsonl 含 `adapt_failed` 记录、不崩溃（对齐现有 test_adapt_failure_lands_in_bad_lines）

---

## 7. [中·并发/健壮性] 限流职责混乱 + judge 无界 gather + 单行异常弃整批（#8）

**根因**：client 级 semaphore（client.py:24）+ `run_concurrent` 每批 semaphore（runner.py:22）
同值双限；judge 会话层裸 `asyncio.gather`（judge_stage.py:102）无界且无异常兜底。

**方案（保留单一限流点 = client 级；run_concurrent 收敛为纯任务编排 + 单行异常兜底）**：
1. `src/query_pipeline/llm/runner.py`：删除 semaphore 与 `concurrency` 参数，
   `run_concurrent(items, worker, *, description="Processing")`；`wrapped` 内
   `try/except Exception` → `logger.warning` + `results[index] = None`（单行异常不弃整批、
   不遗留孤儿任务），`gather` 不取消其余任务。
2. `src/query_pipeline/llm/client.py`：保留 `self._semaphore`（唯一 choke point），注释更新。
3. `src/query_pipeline/audit.py`：删除 audit 自己的 semaphore（client 级已限流）。
4. `src/query_pipeline/steps/judge_stage.py`：
   - 会话层 `run_concurrent(ctx.sessions, process)` 替代裸 gather；
   - 聚合循环对 `None` 结果按 `session_errors += 1` 处理；
   - `process` 内部 try/except 保留（结构化的 session_error 统计），checkpoint.mark 纳入 try；
   - 候选层 `for j in judged` 对 `j is None`（兜底网产物）按 `llm_failed += 1` 处理。
5. 调用点更新（删除 `concurrency=` 实参）：`segment_stage.py`、`verify_stage.py`、
   `post/translate.py`。
   - `segment_stage` 对 `None` 结果回退 whole_session；
   - `verify_stage` 对 `None` 结果按 `verify_failed += 1`、行丢弃（fail-closed，与现有语义一致）；
   - `translate` 对 `None` 结果按 `translate_failed += 1`。

**测试**：
- `tests/llm/test_runner.py` 重写：无 concurrency 参数；顺序保持；worker 抛异常 → 该位 None、
  其余完成、`run_concurrent` 不抛
- 既有并发相关集成测试（test_contract / test_checkpoint 的 resume 系列）必须全绿 —— 即行为回归护栏

---

## 8. [中] LLMClient 对不可重试 4xx 也指数退避（#9）

**根因**：`client.py:50` 捕获 `APIError` 基类，400/401/404 等永久错误被重试 5 次。

**方案**（`src/query_pipeline/llm/client.py` `_complete_once`）：
```python
from openai import APIConnectionError, APITimeoutError, APIStatusError, RateLimitError
...
try:
    ...
    return content
except (APIConnectionError, APITimeoutError, RateLimitError, ValueError, IndexError) as exc:
    retryable = exc          # 连接/超时/429/解析类：可重试
except APIStatusError as exc:
    if exc.status_code < 500:
        raise                # 4xx 永久错误：不重试，直接抛（fail fast）
    retryable = exc          # 5xx：可重试
last_error = retryable
if attempt == self.config.max_retries:
    break
await asyncio.sleep(...)
```
（注意 except 顺序：RateLimitError ⊂ APIStatusError，必须在前；删除 APIError 基类捕获与重复子类）
调用方契约变化：4xx 现在以原始 openai 异常直接抛出（不再是重试后的 RuntimeError）；
配合 #8 的 run_concurrent 兜底，单行 4xx 会变成该会话 error 而不是整批重试拖慢。

**测试**（新建 `tests/llm/test_client.py`）：
- stub `AsyncOpenAI`（patch `query_pipeline.llm.client.AsyncOpenAI`），create 计数；
  用 `httpx.Response` 构造真实 openai 异常；patch `asyncio.sleep` 为 noop 防慢
- `BadRequestError(400)` → 立即抛出、create 恰好 1 次
- `InternalServerError(500)` → 重试 max_retries 次后 RuntimeError、create 次数 == max_retries
- `RateLimitError(429)` → 同上（可重试路径）

---

## 9. [中·数据安全] 成功但零输出的 run 覆盖上次产物（#10）

**根因**：`pipeline/runner.py:74-81` 守卫是 `if success or ctx.rows`——`success=True, rows=[]`
仍写空文件。审查建议"加 flag 区分"——按 CLAUDE.md"最简单实现"，**不加 flag**：
规则为"零输出且产物已存在 → 跳过并告警"；首次零输出（无历史产物）仍写空文件
（保证 --no-llm 首跑、summary 契约不变）。

**方案**（`src/query_pipeline/pipeline/runner.py`）：
```python
if ctx.rows:
    write_jsonl(cleaned_path, ctx.rows)
    write_jsonl(complex_path, [...hard...])
    write_jsonl(normal_path, [...normal...])
elif success:
    targets = (cleaned_path, complex_path, normal_path)
    existing = [p for p in targets if p.exists()]
    if existing:
        logger.warning("run 成功但零输出：保留上次产物（%s），未用空文件覆盖", ...)
        stats["output_preserved_previous"] = True
    else:
        # 首次空输出：写空文件（--no-llm 等预期空输出场景）
        write_jsonl(cleaned_path, [])
        write_jsonl(complex_path, [])
        write_jsonl(normal_path, [])
```
失败且零输出：不写（现状，不动）；summary 始终写。

**测试**（`tests/pipeline/test_contract.py`）：
- 新用例：run1（LLM on）产出 1 行 → run2（`llm.enabled=False`，success 且空）→
  cleaned_queries.jsonl 仍含 run1 的行、`stats["output_preserved_previous"] is True`
- 现有 `test_llm_disabled_no_rows`（首跑空输出）继续通过（写空文件）

---

## 10. 收尾

1. 全量验证：`uv run pytest -q`（179 + 新增全绿）、`pyright src`（0 error）
2. `comments.md` 顶部加"修复状态"小节，逐条标记 ✅ + commit 引用
3. （可选）README:64/77 措辞微调：空输出保护与 checkpoint 失效语义现已名副其实，无需改动
4. 每个 issue 一个 commit，按上述顺序提交

## 不在本次范围（comments.md 候选未入选的低优先项）

审查明确标注"候选但未入选（供参考）"，本次不修：未使用依赖 rich/jinja2、model/concurrency
默认值不一致、bad_lines.jsonl 混写两种格式、suggest.py 硬编码 /dev/null、cache 中毒不驱逐、
session/judge.py:19 底部 import、cli.py 无意义三元、quality/paths.py 未使用 date 参数、
README 流程缺 simple_gate。如需二期处理可另开计划。
