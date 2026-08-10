# comments.md — 代码审查：10 个必须修复的点

> **修复状态（2026-08-09 第一轮）：全部 10 项已修复**。修复前基线 179 passed / pyright 0 error；
> 修复后 207 passed / pyright 0 error（新增 28 个回归测试，均不依赖真实 LLM）。
> 详细修复计划与实现说明见 `docs/fix-plan-comments.md`。
>
> **第二轮修复状态（2026-08-09）：全部 10 项已修复**（基于提交 d566151 之后的代码）。
> 基线 207 passed → 修复后 **224 passed** / pyright 0 error（新增 17 个回归测试）。
>
> **第三轮修复状态（2026-08-09）：全部 10 项已修复**（基于提交 48f4dc6 之后的代码）。
> 基线 224 passed → 修复后 **244 passed** / pyright 0 error（新增 20 个回归测试，均不依赖真实 LLM）。
> 说明：第三轮 #8 仅涉及 chat 适配器（session 适配器的非 dict turn 按第二轮契约
> 计 empty_sessions，语义不变，未动）。

> 审查方式：主代理全量通读 + 4 个子代理并行分工审查（llm/io、steps/session、quality/post、config/api），
> 每个点均经**交叉验证**（≥2 个独立来源 + 代码路径/运行实证）后才记录。维度：健壮性 / 可扩展性 / 优雅 / 简洁。
> 基线：`uv run pytest` 179 passed；`pyright src` 0 error。
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

---

# 第二轮审查：10 个新问题（基于修复提交 d566151 之后的代码）

> 方法：3 个子代理并行（回归审查 d566151 diff、llm/io/steps 深挖、quality/config/文档深挖）+ 主代理逐项实证。
> 基线：`uv run pytest` 207 passed；`pyright src 0 errors`。
> 前 10 条已全部修复；本条为修复后仍存在 + 修复引入的新问题。

## 1. [高·回归] judge_stage debug 推导对 None 崩溃 → 单个候选异常让整次 run 失败 — ✅ 已修复
**位置**：`src/query_pipeline/steps/judge_stage.py:214-227`（对照 `llm/runner.py:27-34`、`llm/client.py:55-57`）
**问题**：fix 引入的 run_concurrent 兜底网返回 None 后，计数循环（:186-188）已处理 None，但下方 debug 列表推导
`j.get("idx")` 对 None 直接 AttributeError → 被 process() 的 except 捕获 → 整会话变 session_error → `_run_success`=False。
同时 fix #9 让 4xx 立即抛出（APIStatusError 不被 funnel 的 `except (ValueError, RuntimeError)` 捕获）恰好触发此路径：
**修复前 4xx → 重试后包成 RuntimeError → llm_failed 计数、run 继续；修复后单个 400（如 context 超长）→ 整次 run 失败**。
作者写的 None 计数分支实际被 debug 崩溃短路（死代码）。
**验证**：运行实证 `judged=[None]` → `'NoneType' object has no attribute 'get'`；funnel 捕获面与 client 4xx 分支确认。
**修复**：debug 推导对 None 短路（或把 debug 构造移入 None 检查之后）；funnel_candidate 捕获面加入 APIStatusError。

## 2. [高·回归] QC judge 兜底 None 被静默放行（fail-open），与内部异常 fail-closed 语义相反 — ✅ 已修复
**位置**：`src/query_pipeline/quality/judge.py:114-115` + `quality/aggregate.py:49-56`
**问题**：`verdicts = [v for v in verdicts if v is not None]` 直接丢弃兜底网异常行，但 sample_set 仍含该行 →
`build_results` 中 `sampled and judge is not None` 为假 → 状态 **pass**；而 judge_one 内部异常（ValueError/RuntimeError）
→ error 字段 → **needs_review**。同一类失败两种相反结论，质检报告失真（崩溃样本被算作"通过"）。
4xx 立即抛出后该路径更易触发。
**验证**：运行实证——崩溃行 status="pass"（内部异常应为 needs_review）。
**修复**：None 也应记 error 语义（进 needs_review 桶），或从 sample_set 剔除并计数 judge_errors。

## 3. [高·健壮性] audit 系统性故障时输出 PASS + exit 0（精度门失效） — ✅ 已修复
**位置**：`src/query_pipeline/audit.py:76-79`（对照 `cli.py:92-95`）
**问题**：LLM 调用异常被记作 `(True, "audit_error: …")`（判复杂）。API 全挂时所有行 3 票全 True →
非复杂率 0% → render PASS 且退出码 0，故障被完全隐藏（reason 里的 audit_error 不影响结论）。
另：`bool(data.get("is_complex"))` 缺字段判 False（拉低通过率）而畸形 JSON 走异常判 True——同一缺陷两种相反结果。
**验证**：运行实证——mock 全部抛 RuntimeError → 3 行结果全 is_complex=True、ratio=0%、PASS。
**修复**：audit_error 票单独计数，错误率超阈值即 FAIL；缺字段与异常统一按"无法判定"处理。

## 4. [中高·一致性] suggest.py 与 candidates._FORMAT_DEFAULTS 双源漂移（注释谎称"同源"） — ✅ 已修复
**位置**：`src/query_pipeline/suggest.py:150` vs `session/candidates.py:9-10`
**问题**：fix 把格式默认门槛收敛到 `_FORMAT_DEFAULTS`，但 suggest 仍手写 `(7, 2) if fmt == "session" else (3, 2)`；
candidates.py:9 注释声称"suggest.py 的 is_default 标记同源"——实际是两份拷贝。且语义已分叉：
candidates 对未知格式回退 session (7,2)，suggest 对任何非 session 格式取 (3,2)——未知格式时
suggest 标"※默认"的组合与 effective_gate 实际执行门槛不一致。
**验证**：两处代码对照确认；改 _FORMAT_DEFAULTS 不会影响 suggest。
**修复**：suggest 导入 `_FORMAT_DEFAULTS`（或 effective_gate），删手写字面量。

## 5. [中·健壮性] 坏缓存 label 命中后不驱逐，候选被永久丢弃（funnel/verify/translate 无自愈） — ✅ 已修复（含 quality judge_one 同款自愈）
**位置**：`src/query_pipeline/session/funnel.py:86-87`、`steps/verify_stage.py:118-120`、`post/translate.py:88-91`
（对照 `session/segment.py:76-79` 有 `cache.pop`）
**问题**：缓存命中后 parse 失败（如跨版本 schema 变化、手改缓存）→ 候选 fail-closed 丢弃，但条目保留 →
**每次运行重复丢弃**。verify 还会把失败写进 checkpoint，即使缓存修好也重放 drop——双重固化。
segment 有自愈（pop 后重调 LLM），其余三处没有，策略不一致。
**验证**：代码对照确认（funnel/verify 无 cache.pop，segment 有）；子代理注入坏 label 实测复现。
**修复**：parse 失败时 `cache.pop(cache_key)` 并重调，与 segment 一致；verify 失败不落 checkpoint。

## 6. [中·缓存一致性] llm_cache 无源码指纹/版本号，与 checkpoint 失效策略不一致 — ✅ 已修复（src_hash 移入 llm/cache.py 并纳入 cache key）
**位置**：`src/query_pipeline/llm/cache.py:66-73`（对照 `io/checkpoint.py:31-36` `_src_hash`）
**问题**：`make_cache_key` 只含 step+model+prompt+question；checkpoint 却把 src 哈希纳入失效。
任何"改代码不改 prompt"的修复（parse 规则、taxonomy 映射、难度判定）都不会让旧缓存失效，
跨运行静默复用旧 label——本次 bool→pydantic 修复即属此类（旧缓存语义与新旧代码都不同）。
与发现 5 叠加：新代码 + 旧 label → parse 失败 → 候选丢弃且不驱逐。
**验证**：cache.py 与 checkpoint.py 对照确认；本次修复提交即未触碰 cache key。
**修复**：cache key 并入轻量版本号（或复用 `_src_hash` 摘要）。

## 7. [中·可观测性] judge stats 计数缺口：empty_sessions 静默丢弃、session_error 死键、non_complex≡normal_rows 冗余 — ✅ 已修复
**位置**：`src/query_pipeline/steps/judge_stage.py:97,156,195-207`（对照 `pipeline/context.py:59-81`）
**问题**：(a) 空 turns 会话可达（preclean 只滤空 context，`context=[非dict]` 可产出 0 turns），judge 计
`empty_sessions` 但 `ctx.stats` 从不输出该键——空会话静默消失无痕迹；
(b) except 路径的 `"session_error"`（单数）进 counters 后无对应输出键，是死键（实际计数走 "error" 检查）；
(c) `non_complex += 1` 与 `normal_count += 1` 恒同步，`non_complex` 永远等于 `normal_rows`，冗余指标。
**验证**：运行实证——`context=["not-a-dict", 42]` → turns=0、preclean 不拦、judge empty_sessions 无输出键。
**修复**：输出 empty_sessions；删死键；non_complex 与 normal_rows 二选一（建议保留 normal_rows）。

## 8. [中·健壮性] 模板解析静默丢数据/错位：分类 prompt 可被静默污染 — ✅ 已修复（fail-loud + taxonomy↔spec 完整性校验；`## 决策步骤` 降级为文档标题）
**位置**：`src/query_pipeline/prompts/assemble.py:88-128`（parse_normal_few_shot）、`build_normal_classify_prompt:176-188`
**问题**：header 行格式一旦变化（如 `## 02` 缺 slug/name），该类别全部内容**静默并入前一类别并覆盖其同名 section**
（实测：02 的"定义"覆盖 01 的定义，02 消失）——生产分类 prompt 以 02 的定义标注 01，无任何报错；
taxonomy 有类别而 few_shot.md 缺 spec 时同样静默输出空标题行。templates 是 README 声明的"唯一事实源"，
一个 md 笔误即静默污染 LLM 判定。
**验证**：运行实证——malformed header → 01.sections["定义"] 变成 02 的内容。
**修复**：解析时校验 header 匹配与类别完整性（taxonomy ↔ spec 一一对应），不匹配即抛错（fail-loud）。

## 9. [中·一致性] 文档漂移未修：README/CLI 帮助/docstring 与实现仍矛盾 — ✅ 已修复
**位置**：`README.md:48,55`、`src/query_pipeline/api.py:82`、`taxonomy.py:94-103`、`cli.py:70`
**问题**：fix 未触碰文档——(a) README 默认流程仍缺 simple_gate（DEFAULT_STAGES 有）；(b) README:55 仍写
"chain 调用 ≥7 … chat 同规则"，与 chat 默认 3/1/2 矛盾（cli.py:27 帮助文本自证）；(c) api.py:82 work_dir
docstring 声称默认 `work/<input stem>`，实现默认 output_dir；(d) taxonomy.templates_dir docstring 声称
importlib.resources fallback，实现没有；(e) cli `--verify-rounds` 只映射 verify_rounds_hard，normal 轮数无 CLI 旋钮。
**验证**：逐处代码/文档对照确认（README 在 fix 提交中 0 改动）。
**修复**：README 补 simple_gate 与 chat 门槛；docstring 与实现对齐；补 verify_rounds_normal 旋钮或删除参数。

## 10. [中低·一致性] bad_lines.jsonl 混写两种格式（原始文本 + JSON 对象） — ✅ 已修复（统一 JSON 对象行，附 raw 字段）
**位置**：`src/query_pipeline/io/jsonl.py:58-62` + `steps/preclean_stage.py:39-53`
**问题**：语法坏行以原始文本写入，adapt 失败行以 `append_jsonl` 追加 JSON 对象——同一文件两种格式，
下游无法用单一 reader 解析（测试里用 JSON reader 读该文件遇原始行即崩）；与"坏行可审计"的意图相悖。
**验证**：运行实证——bad_lines.jsonl 同时含 `not-json-line` 原始行与 `{"reason":"adapt_failed",...}` JSON 行。
**修复**：统一为 JSON 对象行（附 raw 字段），或按原因分文件（bad_lines.jsonl / adapt_failed.jsonl）。

---

*第二轮候选但未入选（低优先，供参考）：退避 sleep 占用 semaphore permit（队头阻塞）、logging FileHandler 移除不 close、
.env 三套查找逻辑（api/loader/quality）、record_key 无 id 时 `line_?` 撞键、cli 死代码/未使用 import、
model 默认字面量 6 处 + concurrency 256 vs 64、report.md "共 N 条（显示前 N 条）"恒等文案、api.run JSON 往返、
sniff_format 仅采样 5 行、verify 4xx 不落 checkpoint（恢复后重验）。*

---

# 第三轮审查：10 个新问题（基于修复提交 48f4dc6 之后的代码）

> 方法：3 个子代理并行（48f4dc6 回归审查、性能/并发深挖、quality/config/适配器深挖）+ 主代理逐项实证。
> 基线：`uv run pytest` 224 passed；`pyright src 0 errors`。
> 前两轮 20 条已全部修复；本条含**修复不彻底项**（#1）与修复引入的**新问题**（#4）。

## 1. [高·回归不彻底] 模板污染仍在：第二轮 #8 只修了标题行，正文继续混入类别 16 — ✅ 已修复
**位置**：`src/query_pipeline/prompts/assemble.py:69-71,126-128`（对照 `templates/normal_few_shot.md:505`）
**问题**：`parse_normal_few_shot` 对 `# 决策步骤` 只跳过标题行，其后 7 行正文仍被并入**类别 16 的当前 section**。
运行实证：解析真实模板后 `specs["16"].sections["易混类别"]` 末尾含 `7. 对 other、低置信…设置 needs_review=true` 等
指令正文 → `build_normal_classify_prompt()` 生产 prompt 中类别 16 混入无关后处理指令。
同类缺口：`parse_complex_few_shot` 对 `## 02`（类别头少一个 #）不 raise，02 的内容静默并入 01（实证 "特征：a 特征：b"，02 消失）。
根因：只有"匹配的类别头/EOF"能结束类别吸收，文档级标题（`#`）不终止当前 section。
**验证**：运行实证（真实模板解析 + 少 # 场景）。
**修复**：`#`/`##` 级标题行一律终结当前类别（save + current=None），其后正文 fail-loud；并补回归测试。

## 2. [高·可扩展性] judge checkpoint 存全量 rows/judged（含 MB 级 chain），磁盘翻倍 + 全量解析进内存 — ✅ 已修复
**位置**：`src/query_pipeline/io/checkpoint.py:110-152` + `steps/judge_stage.py:87-98`
**问题**：checkpoint 每会话存 rows（含完整 chain，单行实测 3.2MB）+ judged + stats。
实测：500 会话 → judge.jsonl **91MB**，与输出数据量同量级（数据翻倍）；`Checkpoint.load/_read` 每 run 全量解析进内存，
百万行级输入即 GB 级内存。rows/judged 实际可由 llm_cache（funnel 已落盘）确定性重建，checkpoint 只需 stats。
**验证**：子代理实测 outputs/aime/logs/checkpoints/judge.jsonl 91MB / 188KB 均行 / 3.2MB 最大行。
**修复**：checkpoint 只存 stats（rows/judged 由 llm_cache 重建）；或按会话拆分存储避免单行巨对象。

## 3. [高·并发] 退避 sleep 与 5×90s 超时都发生在 semaphore 内，429/5xx 风暴时 permit 被占死 — ✅ 已修复
**位置**：`src/query_pipeline/llm/client.py:28-29,63-64`
**问题**：`complete()` 在 `async with self._semaphore` 内调用 `_complete_once`，而重试循环的
`asyncio.sleep`（累计最长 ~25s）与每次请求的 90s 超时（5 次 ≈ 7.5min）都在锁内。
429/5xx 风暴时全部 permit 被睡觉/挂起请求占死 → 后续健康请求队头阻塞、吞吐归零，风暴后需逐个超时才恢复。
**验证**：代码路径确认（semaphore 包裹完整重试循环）。
**修复**：semaphore 移到单次 API 调用（每次 attempt 重新 acquire），sleep/超时在锁外。

## 4. [中高·可扩展性] llm_cache 无界增长：src_hash 进 key 后代码变更全量废弃旧条目，但永不清理 — ✅ 已修复
**位置**：`src/query_pipeline/llm/cache.py:10-26,86-94`（fix #6 引入）
**问题**：`make_cache_key` 并入 src_hash 后，任何代码改动使**全部**旧 key 永久失效；但 cache 是 append-only，
无裁剪/重置——checkpoint 在 meta 不匹配时 `_seed` 截断重建，cache 文件却只增不减。
实证：仓库现有 outputs/aime/logs/llm_cache.jsonl 2836 行，48f4dc6 后全部成孤儿；每次代码变更重跑即追加一整轮，
`load_cache` 每 run 全量解析（含全部死代条目），活跃开发期内存/启动耗时随迭代线性恶化。
**验证**：三方独立确认（主代理实证 load_cache 全量加载 + 两子代理）。
**修复**：缓存按 src_hash 分片文件，或启动时检测孤儿代并触发一次 rewrite/压缩。

## 5. [中·契约冲突] QC `_check_meta` 与 translate fail-open 语义矛盾：管线接受的行 QC 必判 FAIL — ✅ 已修复
**位置**：`src/query_pipeline/quality/rules.py:219-222` vs `post/translate.py:101-103` + `templates/filter_out.jsonc:69`
**问题**：翻译失败是**故意 fail-open**（put(row, None)，filter_out.jsonc 明示"翻译失败 → null"），
但 `_check_meta` 对非中文行 `translation=null` 判 fail。运行实证：fail-open 行（非中文+translation=null）→ meta 规则 FAIL。
同一状态管线接受、质检判错，且测试只固化了严格语义、无 fail-open 场景。
**验证**：运行实证 + 子代理确认。
**修复**：_check_meta 区分"从未翻译"（null 且无失败记录，判 fail）与"翻译失败"（fail-open 可接受）；
或在 QC 输出中携带 translate_failed 标记供规则区分。

## 6. [中·数据覆盖] quality/paths.py 的 date 是死参数：同一数据集不同日期 QC 产物互相覆盖 — ✅ 已修复
**位置**：`src/query_pipeline/quality/paths.py:15-27`（source_path/qc_dir/llm_cache_path 三函数）
**问题**：三个函数的 date 参数均未使用（date 只进 overview.json 内容字段）。运行实证：
`qc_dir("aime","0806") == qc_dir("aime","0807")` → 同一目录，0806 与 0807 两次 QC 运行
**互相覆盖** results.jsonl/overview.json/report.md，且 overview 内 date 字段与文件名/内容对不上。
**验证**：运行实证 + 子代理确认。
**修复**：qc_dir 路径并入 date（outputs/<dataset>/qc/<date>/），或删掉 date 参数并显式文档化单目录语义。

## 7. [中·行为陷阱] api.run 的 api_key/base_url 用 setdefault：env 已有 key 时显式传参被静默丢弃 — ✅ 已修复
**位置**：`src/query_pipeline/api.py:90-93`（对照 `cli.py:33` help"OPENAI_API_KEY 覆盖"）
**问题**：`os.environ.setdefault("OPENAI_API_KEY", api_key)`——env 已存在 key 时，用户显式
`--api-key new-key` 被静默忽略，实际使用旧 env key。运行实证：env=old-key + 传 new-key → 实际 old-key。
与 CLI 帮助文本"覆盖（默认读 .env）"直接矛盾。
**验证**：运行实证 + 子代理确认。
**修复**：显式传入时用 `os.environ[...] = api_key`（覆盖），或至少 warning 提示被忽略。

## 8. [中·数据丢失] adapters/chat.py 静默丢弃畸形行：非 dict turn 被过滤、畸形 chain 静默置 [] — ✅ 已修复
**位置**：`src/query_pipeline/adapters/chat.py:24-28,48`
**问题**：context 中非 dict turn 被列表推导静默过滤（无计数无日志）；chain 非 list 静默变 []。
chain=[] 时 `chain_tool_calls` 回退 `turn.tool_count`（adapt_chat 未设置 → 0），chat 门槛 3/1/2 下
**整行无声落选**——输入畸形导致输出少行，无任何警告，与 session 适配器/管线其余部分的 fail-loud 哲学不一致。
**验证**：代码路径确认 + 子代理确认。
**修复**：非 dict turn 计数并写 bad_lines；畸形 chain 记 warning（或同样进 bad_lines）。

## 9. [中·可扩展性] run_concurrent 一次性 eager 创建全部协程：verify/translate 上万行 = 上万个 task — ✅ 已修复
**位置**：`src/query_pipeline/llm/runner.py:38`
**问题**：`asyncio.gather(*(wrapped(...) for ... in items))` 一次性创建全部任务；子代理实测
100k 任务峰值 **110MB + 2.1s 纯调度开销**，百万行输入即 >1GB。且每项 checkpoint.mark/put_cache
都 open/close 文件（10k 行 × 5 轮 = 5 万次 open）。
**验证**：子代理实测量化。
**修复**：有界 worker 池（固定 N 个 worker 拉队列），或分块 dispatch（如每 1000 项一批）。

## 10. [中·健壮性] audit 一行坏 JSON 崩溃整个命令，非 dict 行 AttributeError 穿透 — ✅ 已修复
**位置**：`src/query_pipeline/audit.py:45-51,65`（对照 `cli.py:88`）
**问题**：`_load_rows` 直接 `json.loads`，一行坏 JSON 抛异常使整个 audit 命令崩溃（管线端坏行有
bad_lines 落盘容忍，audit 没有——同一仓库两套读取语义）；`check()` 的 `row.get("input")` 在
try/except 之外，非 dict 行的 AttributeError 穿透 asyncio.gather 直接炸掉 audit。
**验证**：代码路径确认 + 子代理确认。
**修复**：_load_rows 复用 read_jsonl_with_bad_lines（或逐行容错）；check() 内对 row 类型兜底。

---

*第三轮候选但未入选（低优先，供参考）：sniff_format 仅采样 5 行（全坏行误拒/混合漏检）、Checkpoint._seed
非原子截断（write_text vs tmp+replace）、PROMPTS 模块导入期 fail-loud（模板漂移 → 全部子命令 import 崩溃）、
build_segment_payload 无超长截断（千 turn 会话爆上下文静默回退）、audit 错误票仍记 is_complex=True（展示误导，
闸门本身正确）、concurrency 默认 256 vs 64 不一致、QC 私有名跨模块导入+shadowing（前两轮已提未修）、
api.run JSON 往返、logging FileHandler 不 close、.env 三套查找、record_key "line_?" 撞键、cli 未用 import、
report.render_markdown results 死参数。*

---

# 第四轮审查：10 个新问题（基于第三轮修复的工作区改动）

> 方法：3 个子代理并行（第三轮修复回归审查、tests/models/quality 契约深挖、cli/config/docs 深挖）+ 主代理逐项实证。
> 基线：`uv run pytest` 244 passed；`pyright src 0 errors`。
> 前 30 条已全部修复；本条含**第三轮修复引入的并发缺陷**（#1）与**修复不完整项**（#3、#5）。
>
> **第四轮修复状态（2026-08-10）：全部 10 项已修复**。
> 基线 244 passed → 修复后 **261 passed** / pyright 0 error（新增 17 个回归测试，均不依赖真实 LLM）。
> 说明：#4 槽位计数约束同时作用于模板层与 Jaccard 层（否则 2 只 vs 3 只股票的示例仍会在
> Jaccard 层以 sim=1.0 合并）；Jaccard 层只约束两侧**共享**槽类型的计数一致，避免股票名词典
> 不完备导致的非对称槽化（nvidia 在词典而 amd 不在）漏并同型查询。`test_representative_selection_and_determinism`
> 原用例（1 个时间段 vs 3 个时间段）在新契约下合法不合并，fixture 已改为同槽位计数。
> #9 错误率默认独立阈值 0（任何一行无法判定即 FAIL），可用 `--max-error-ratio` 放宽。

## 1. [高·回归引入] load_cache 孤儿代 rewrite 并发竞态：双进程同时 rewrite 丢数据 — ✅ 已修复
**位置**：`src/query_pipeline/llm/cache.py:55-71,75-82`
**问题**：第三轮修复让 load_cache 检测到孤儿代时**自动 rewrite 压缩**，但 tmp 名固定（`llm_cache.jsonl.tmp`）且无进程锁。
管线（runner.py:52）与 QC CLI（quality/cli.py:58）共用同一 cache 文件；源码变更后首次运行必然出现孤儿代 →
两进程同时 rewrite：B 的 tmp+replace 截断 A 正在写的内容。子代理实测（模拟双写同一 tmp）：**5000 条仅存 1959 条、
33 行损坏**（损坏行下一轮 load 静默 skip → 大量 LLM 重调）；B rewrite 前 A 的 append 写进旧 inode → 条目静默丢失。
**验证**：子代理并发模拟实测 + 代码路径确认（固定 tmp 名、无锁）。
**修复**：tmp 名带 pid/随机后缀，或 rewrite 加文件锁；或将 rewrite 移到 put_cache 的锁内。

## 2. [高·契约] QC record_key 复合键塌缩：同 (source_case_id|trace_id) 重复行互相覆盖 — ✅ 已修复
**位置**：`src/query_pipeline/quality/aggregate.py:11-17,73-95` + `quality/cli.py:46-49`
**问题**：per_record / sample_set / judge_results 三个 dict 均以 `record_key = source_case_id|trace_id` 为键；
docstring 明言"duplicate trace_ids fold onto their own rows"，但 **dict 无法容纳重复键**——同键第二行覆盖第一行，
规则判定、judge 判定、sampled 标志全部串行错乱。子代理实验：r1 答案过短（应 fail answer 规则）+ r2 同键正常答案 →
**两行共用 r2 的规则结果，r1 显示 pass**；重复行越多漏报越多（上一轮修的 `line_?` 分支未覆盖此分支）。
**验证**：子代理实验复现 + Python dict 语义（确定）。
**修复**：record_key 追加行号/计数消歧（或 per_record 改为 key → list）。

## 3. [中高·行为契约] judge 断点续跑变 cache-only：删 cache 即全量重跑；旧 91MB checkpoint 不迁移 — ✅ 已修复
**位置**：`src/query_pipeline/steps/judge_stage.py:92-100`（对照 `README.md:76-78`）
**问题**：第三轮修复后 judge checkpoint 不再存 rows/judged，resume 完全依赖 llm_cache；
`record` 仅用于 `record is None` 判重。**(a)** 只删 llm_cache（保留 checkpoints）→ 全量重调 LLM
（旧行为：checkpoint 直接复用 rows，零 LLM）——README 仍写"已完成单元跳过：LLM 缓存 + 阶段 checkpoint"，未说明 judge 已变 cache-only；
**(b)** 存量 91MB 旧格式 checkpoint 文件永不被压缩/迁移，每次 run 仍全量解析进内存——本轮声称解决的问题对存量用户仍在。
**验证**：代码路径确认 + 两子代理独立确认。
**修复**：README 同步"judge 续跑依赖 llm_cache，删 cache 即全量重跑"；load 时检测旧格式记录并迁移/截断。

## 4. [中高·数据丢失] dedup 实体集合塌缩：实体数量不同的问句被 template_merge 合并 — ✅ 已修复
**位置**：`src/query_pipeline/post/dedup.py:40,54-101`（对照 `rules/normalize.py:76-87`）
**问题**：tokenize 用**集合**，多个实体槽折叠为单个 `<stock>`；template 层同骨架只看非槽 token 集合相等。
实证："帮我比较一下贵州茅台和宁德时代的走势" vs "帮我比较一下贵州茅台、宁德时代和比亚迪的走势"
（实体均在词典）→ Jaccard=1.0 → **2 实体行被 template_merge 删除，只留 3 实体行**——
两个答案不同的分析请求（2 只 vs 3 只股票的比较）被当作"只换标的的同一模板"合并。
另有边界：两行完全相同的纯槽位文本永不查重（comparable=False 且无测试覆盖）。
**验证**：运行实证（similarity=1.0, method=template_merge）。
**修复**：template 层要求实体槽**数量**一致（比较槽位计数，而非仅集合）；纯槽位行补查重策略或文档化。

## 5. [中·数据丢失被掩盖] translate 失败用 null 覆盖已有译文 + 标记，QC 不再暴露 — ✅ 已修复
**位置**：`src/query_pipeline/post/translate.py:56-68,111-113`
**问题**：第三轮修复让翻译失败落 `meta.translate_failed` 标记、QC 据此放行——但失败分支
`put(row, None, failed=True)` **无条件覆盖已有译文**。实证：已有译文的行（缓存丢失后）重跑失败 →
translation 被覆盖为 None + 标记 → QC 判通过（旧行为 QC fail 会暴露译文丢失）。译文丢失被静默掩盖。
**验证**：运行实证——已有译文行重跑失败 → translation=None、meta.translate_failed=True。
**修复**：失败时**保留已有译文**（仅当 translation 为空才置 null）；或 QC 对"曾有译文后丢失"单独暴露。

## 6. [中·错误处理] 模板 fail-loud 位于模块导入期：模板小改动瘫痪全部 CLI 子命令 — ✅ 已修复
**位置**：`src/query_pipeline/prompts/__init__.py:17-27`（对照 `prompts/assemble.py:80-84,153-157`）
**问题**：PROMPTS 在 import 时构建（build_* 读 templates 三个文件 + fail-loud 校验）；
cli.py:9 import api → runner → steps → prompts 全链。实测 `QUERY_PIPELINE_TEMPLATES` 指向空目录时，
连 `query-pipeline --help` 和 `suggest`（运行时根本不碰 PROMPTS）都抛 FileNotFoundError traceback。
模板缺失/格式错误 = 所有子命令不可用，且报错指向 assemble.py 而非模板行。
**验证**：子代理实测（env 指向空目录 → --help 崩）+ 导入链代码确认（双来源）。
**修复**：PROMPTS 惰性化（resolve_prompt 首次调用时构建），或至少让 --help/suggest 不触发 prompts 构建。

## 7. [中·一致性] answer_gate 与 QC 截断判定分叉 + `_DANGLING_END` shadowing — ✅ 已修复
**位置**：`src/query_pipeline/steps/answer_gate_stage.py:57` vs `quality/rules.py:176-183`
**问题**：gate 判截断有前置条件 `if question and ...`（无问句不判）；QC `_check_truncation` 无条件判。
实验：无 input.text 的行以"，"结尾 → gate 通过（None）、QC truncation fail。且 rules.py 导入
`_DANGLING_END` 后本地重定义（:26，死导入 shadowing），MIN_ANSWER_LEN 两处重复——两侧阈值/条件
后续必然漂移（当前被 question 规则兜住，但契约已分叉）。
**验证**：运行实证（gate None vs QC fail）+ 代码对照。
**修复**：判定逻辑收敛到单一实现（QC 复用 answer_gate 的判定函数），删 shadowing。

## 8. [中·文档-代码脱节] docs/html 与运行时装配脱节：缺 4/9 提示词面板 + 双实现 + 阶段数不符 — ✅ 已修复
**位置**：`docs/html/build.py:105-122`（对照 `docs/html/README.md:4`、`src/query_pipeline/prompts/__init__.py`）
**问题**：(a) build.py 只提取 5/9 个提示词（缺 classify_complex / classify_normal / verify_recheck / complex_judge），
而 docs/html/README.md 声称展示"全部生产提示词"——**最关键的分类提示词无面板**；
(b) build.py:73 `parse_bad_cases` 与 assemble.py 的 `parse_bad_cases` 是同规则**两份实现**（改一处另一处静默漂移）；
(c) 文档称"7 阶段"而运行时 8 阶段（simple_gate 缺失）。
**验证**：build_prompt_sections 代码确认（5 个 section）+ 双实现对照。
**修复**：build.py 补 classify/verify_recheck 面板（或运行时导入 PROMPTS 而非手工提取）；parse_bad_cases 收敛单源。

## 9. [中·健壮性] audit 错误率与非复杂率共用 max_ratio + cli 重复计算 passed — ✅ 已修复
**位置**：`src/query_pipeline/audit.py:132,148`（对照 `cli.py:101-104`）
**问题**：(a) 错误率（审计自身 LLM 失败）与非复杂率共用一个阈值：5% 容错可能系统性掩盖审计失效
（3 次判定全挂也只是 audit_errors=3，占比 ≤5% 即 PASS）；(b) render 内部已计算 passed 但只返回字符串，
cli.py 重新计算一份（errors/total <= max_ratio）——两处判定逻辑易漂移。
**验证**：代码对照确认（render 的 passed 与 cli 的退出码判断各自独立实现）。
**修复**：错误率独立阈值（建议近 0）或独立旋钮；render 返回结构化结论供 cli 复用。

## 10. [中·配置一致性] loader 相对路径基座不一致：显式 cache/checkpoint 解析到 project root，默认解析到 work_dir — ✅ 已修复
**位置**：`src/query_pipeline/config/loader.py:37-44`
**问题**：显式 `checkpoint.dir`/`llm.cache` 相对路径按 project root 解析（`(base / cfg.checkpoint.dir).resolve()`），
而 None 默认落到 `work_dir/logs/…`。实测：config 设 `llm.cache: logs/llm_cache.jsonl` + `work_dir: scratch` →
缓存落在 `<root>/logs` 而非 `scratch/logs`——**--work-dir 覆盖对显式 cache 配置失效**，同一配置两套基座。
**验证**：loader 代码对照 + 子代理实测。
**修复**：显式相对路径统一按 work_dir（或 project root）解析，文档化单一基座。

---

*第四轮候选但未入选（低优先，供参考）：chat fail-loud 整行丢弃 vs session 静默过滤契约不一致（1 个坏 turn 丢 99 个好 turn）、
audit bad_lines 命名三套并存且写输入目录（权限失败换一种崩法）、timing 校验缺口（first_token_time_ms=-5 通过 QC；
chat 的 first_token_time_cost 直通 *_ms 无单位校验）、cli --verify-rounds 负数/0 抛 pydantic 原始 traceback、
load_taxonomy 无视进程内 QUERY_PIPELINE_TEMPLATES 变更、suggest 占比分母含 ineligible 槽位、category_distribution 死字段、
translate 双计数（mark 抛 OSError 时同行走 translated+translate_failed）、mock 分发过宽（prompt 措辞一改测试语义即变）、
QC --model 帮助文本"默认复用管线模型"但实际硬编码、record_detail 对无 trace_id 行不可钻取、LLMConfig 无默认、
cache_path/checkpoint_dir 兜底 logs/logs（第 1 轮已提）、logging FileHandler 不 close。*
