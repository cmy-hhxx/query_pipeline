# comments.md — 代码审查：10 个必须修复的点

> **修复状态（2026-08-09 第一轮）：全部 10 项已修复**。修复前基线 179 passed / pyright 0 error；
> 修复后 207 passed / pyright 0 error（新增 28 个回归测试，均不依赖真实 LLM）。
> 详细修复计划与实现说明见 `docs/fix-plan-comments.md`。
>
> **第二轮修复状态（2026-08-09）：全部 10 项已修复**（基于提交 d566151 之后的代码）。
> 基线 207 passed → 修复后 **224 passed** / pyright 0 error（新增 17 个回归测试）。

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
