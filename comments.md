# comments.md — 代码审查：10 个必须修复的点

> **修复状态（2026-08-09）：全部 10 项已修复**。修复前基线 179 passed / pyright 0 error；
> 修复后 207 passed / pyright 0 error（新增 28 个回归测试，均不依赖真实 LLM）。
> 详细修复计划与实现说明见 `docs/fix-plan-comments.md`。

> 审查方式：主代理全量通读 + 4 个子代理并行分工审查（llm/io、steps/session、quality/post、config/api），
> 每个点均经**交叉验证**（≥2 个独立来源 + 代码路径/运行实证）后才记录。维度：健壮性 / 可扩展性 / 优雅 / 简洁。
> 基线：`uv run pytest` 179 passed；`pyright src` 0 error。
主代理全量通读 + 4 个子代理并行分工审查（llm/io、steps/session、quality/post、config/api），
> 每个点均经**交叉验证**（≥2 个独立来源 + 代码路径/运行实证）后才记录。维度：健壮性 / 可扩展性 / 优雅 / 简洁。
> 基线：`uv run pytest` 179 passed；`pyright src` 0 error。

---

## 1. [高·健壮性] `api.run` format="auto" 时 chat 输入用错门槛（7/2 而非 3/2） — ✅ 已修复
**位置**：`src/query_pipeline/api.py:111-119`
**问题**：门槛分支只比较字面 `format == "chat"`，"auto" 落入 else 取 session 默认 `RuleGateConfig()`（7/1/2）；
真实格式由 preclean 阶段嗅探，且不回填门槛。chat 输入 + 默认 `--format auto` 实际被 7/1/2 过滤，
而注释（api.py:107-109）与 suggest（suggest.py:150）都认定 chat 应为 3/1/2（chat 工具调用 ≥7 仅覆盖 ~1%）。
**验证**：运行实证 `rule_gate_for("auto") → (7,2)` vs `("chat") → (3,2)`；suggest 标为"默认"的组合与实际执行门槛分叉。
**修复**：门槛决策延迟到嗅探之后（rule_gate 阶段按 `ctx.stats["input_format"]` 取默认），或 api.run 嗅探后回填。

## 2. [高·健壮性] `api.run` 门槛参数静默语义缺陷：`--no-reject-rules` 无效 + 单参数重置另一门槛 — ✅ 已修复（与 #1 合并系统级修复：None 旋钮由 rule_gate 阶段按嗅探格式补齐，reject_rules 始终透传）
**位置**：`src/query_pipeline/api.py:111-122`
**问题**：(a) 默认门槛分支 `RuleGateConfig()`/`RuleGateConfig(3,2)` 未透传 `reject_rules`，
`--no-reject-rules` 在未同时指定门槛时**完全不生效**；(b) 只传 `--min-tool-calls 5` 时另一旋钮被静默重置为
`min_unique_tools=1`（格式默认 2）、只传 `--min-unique-tools 3` 时 `min_chain_tool_calls=0`（门槛被禁用）。
**验证**：运行实证 `reject_rules=False` 在默认分支返回 True；`(5, None) → (5, 1)`。
**修复**：默认分支透传 `reject_rules`；未显式传入的旋钮按格式默认补齐，而非 0/1。

## 3. [高·健壮性] QC `_dataset_category_skew` 零类别误报（complex id 解析错误） — ✅ 已修复
**位置**：`src/query_pipeline/quality/rules.py:311-316`
**问题**：`str(cat).split("-")[0]` 对 complex 路径 `complex-topic/09-…` 得到 `"complex"` 而非 id `"09"`，
且 normal id（01-16）会泄漏进集合"覆盖"complex id。complex 09 有记录仍报"零记录：09"；
complex 01 无记录但 normal 01 有记录时不报。
**验证**：运行实证——场景 A（仅 complex 09）报零记录 01-09（含 09，误报）；场景 B（仅 normal 01）漏报 01。
**修复**：按 taxonomy path 正确提取 id：`path.split("/",1)[1].split("-",1)[0]`（或直接用 `load_taxonomy().get` 反查）。

## 4. [高·健壮性] QC `_check_chain` 与 end2end 输出语义冲突 — ✅ 已修复
**位置**：`src/query_pipeline/quality/rules.py:115-118`（对照 `models/output.py:51`、`session/assemble.py:50`）
**问题**：管线合法支持 `capture_mode="end2end"`（输入无 chain → 空 chain，靠 tool_count fallback 过 rule_gate），
QC 却无条件判 `chain 缺失/为空` 为 fail → 所有合法 end2end 行必然质检失败，且测试（test_rules.py:81-83）
只断言"空 chain 必 fail"、固化了该误报，无 end2end 用例。
**修复**：`_check_chain` 对 `capture_mode == "end2end"` 跳过（或仅校验 tools 非空），并补 end2end 用例。

## 5. [高·缓存一致性] verify checkpoint 陈旧重放：键不含前文/难度，meta 缺输入指纹 — ✅ 已修复
**位置**：`src/query_pipeline/steps/verify_stage.py:51`、`src/query_pipeline/io/checkpoint.py:86-93`
**问题**：verify 键 = `content_key(source_case_id, trace_id, question)`，不含 `prior_questions`（参与 LLM 判定）
与 `difficulty_level`（决定轮数与期望）；`stage_meta` 只为 judge 附加输入 size/mtime。
输入文件变化后 judge checkpoint 重置重算，verify 却对相同 (case, trace, question) **直接重放旧裁决**
（旧前文、旧难度下的 keep/reject/error），与 README「配置/输入/源码变化自动失效」矛盾。
**验证**：两子代理独立确认；代码路径：judge 重置 → 行重建 → verify 命中旧 checkpoint 短路 LLM cache。
**修复**：verify 键纳入 prior_questions/difficulty 摘要；`stage_meta` 对 verify（及 translate）也附加输入 stat。

## 6. [高·健壮性] funnel 布尔字段字符串真值 fail-open — ✅ 已修复
**位置**：`src/query_pipeline/session/funnel.py:54,59`
**问题**：`ValueResult(is_valuable=bool(data.get("is_valuable")))` / `ComplexityResult(is_complex=bool(...))` 手工 `bool()` 转换，
LLM 若输出 `"is_valuable": "false"`（字符串），`bool("false") == True` → 价值/复杂度门控**静默放行**。
verify 的 `VerifyResult` 走 pydantic 严格校验，此处是漏斗唯一的非严格解析路径。
**验证**：`bool("false")`/`bool("no")` 均返回 True；funnel.py:54,59 无类型校验。
**修复**：ValueResult/ComplexityResult 与 ClassifyResult 一致走 `model_validate` 严格类型校验。

## 7. [中高·健壮性] preclean 对非 dict `judge_data` 崩溃而非进坏行 — ✅ 已修复
**位置**：`src/query_pipeline/io/sniff.py:56`
**问题**：`str(((r.get("judge_data") or {}).get("case_id")) or ...)`——`judge_data` 为 truthy 非 dict（如字符串）时
抛 `AttributeError` 直接炸掉整个 pipeline；而 JSON 语法坏的行反而被容忍进 bad_lines（处理哲学不一致）。
**验证**：运行实证——第 6 行 `judge_data: "not-a-dict"`（前 5 行正常 chat）→ `preclean_records` 抛 AttributeError。
**修复**：key_fn 加 `isinstance` 防护，异常行按 bad_lines 处理（与 adapt 失败一致）。

## 8. [中·并发/健壮性] 限流职责混乱：多层 semaphore 冗余 + judge 会话层无界 gather — ✅ 已修复
**位置**：`src/query_pipeline/llm/client.py:24`、`llm/runner.py:21-30`、`steps/judge_stage.py:102`
**问题**：(a) LLMClient 进程级 semaphore 与 `run_concurrent` 每批 semaphore 同值双限（audit.py 再套一层，共三层），
runner.py 注释所称"single shared choke point"不成立；(b) judge 对全部会话 `asyncio.gather` **无界**
（segment/verify/translate 均用 run_concurrent，唯独 judge 会话层没有），每会话内部再起并发，大输入下任务/内存无上限；
(c) gather 无 `return_exceptions`，worker 未捕获的意外异常（如 cache 磁盘 OSError）中止整个 stage 并遗留孤儿任务。
**验证**：judge_stage.py:102 直接 gather 无 semaphore 包裹；client.py:24 与 runner.py:22 均为 `max(1, concurrency)`。
**修复**：保留单一限流点（client 级即可，run_concurrent 收敛为纯任务编排）；judge 会话层用 `run_concurrent` 包裹；
`gather(return_exceptions=True)` 或 worker 内兜底，避免单行异常弃整批。

## 9. [中·健壮性] LLMClient 对不可重试 4xx 也指数退避重试 — ✅ 已修复
**位置**：`src/query_pipeline/llm/client.py:50`
**问题**：`except (APIConnectionError, APITimeoutError, RateLimitError, APIError, ValueError, IndexError)` 中
`APIError` 基类覆盖 `BadRequestError/AuthenticationError/NotFoundError` 等永久错误；400/401/404 会无意义重试
5 次（退避最长 ~22s+抖动），失败慢且日志误导（RateLimitError 等子类重复列出）。
**验证**：client.py:50 捕获面确认；openai SDK 中 4xx 均继承自 `APIError`。
**修复**：仅对可重试错误重试（连接/超时/429/5xx），其余直接抛；去掉重复子类。

## 10. [中·数据安全] 成功但零输出的 run 覆盖上次产物 — ✅ 已修复
**位置**：`src/query_pipeline/pipeline/runner.py:74-81`
**问题**：注释声称 "avoid clobbering a previous good output with an empty file"，但守卫条件是
`if success or ctx.rows`——`success=True` 且 rows 为空时（全部候选被 value 拒 / 门槛调严后重跑 / 误配）
仍用空文件覆盖上次良好产物，与注释意图相悖（仅失败场景受保护，README:73 也只承诺失败场景）。
**验证**：`success=True, rows=[]` → 写入分支成立（运行实证）；`_run_success` 在无 session_errors 时返回 True。
**修复**：零输出且目标产物已存在时跳过覆盖并告警（显式 `--no-llm` 等预期空输出场景除外，可加 flag 区分）。

---

*候选但未入选的低优先项（供参考）：未使用依赖 rich/jinja2、model/concurrency 默认值多处不一致、
`bad_lines.jsonl` 混写两种格式、`suggest.py` 硬编码 `/dev/null`、cache 中毒不驱逐（funnel/verify 缺 cache.pop）、
`session/judge.py:19` 底部 import、cli.py 无意义三元、`quality/paths.py` 未使用 date 参数、README 流程缺 simple_gate。*
