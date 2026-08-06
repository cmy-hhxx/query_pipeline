# query_pipeline

Config-driven pipeline that extracts complex financial queries from multi-turn agent sessions.

## Run

The default entrypoint reads the root `config.yaml`:

```bash
uv run python run.py --dry-run
uv run python run.py
```

Use `-c` only for temporary experiments:

```bash
uv run python run.py -c config.yaml --dry-run
```

`.env` supplies `OPENAI_BASE_URL` and `OPENAI_API_KEY`; the loader reads it automatically.

## Flow

Input is JSONL where each line is one session: `{"thread_id": "...", "context": [ {question, answer, run_id, trace_id, tool_names, tool_count, chain, ...}, ... ]}`.

The pipeline processes sessions one at a time (a `Sessions` progress bar counts sessions, with a live `LLM complex judge` bar inside each session; verify and translate each show their own bar):

1. `segment`: one LLM call splits a session's questions into topic-contiguous segments. A topic may not recur — if it does (A, B, A), the whole span is merged into one segment. On LLM failure the whole session is treated as one segment.
2. `step1` (rules): within each segment, pick candidate turns that look complex: reject low-value / blank / too-short text first, then keep a turn only if it clears all of `min_chain_tool_calls` AND `min_chain_steps` AND `min_unique_tools` (default 7 tool calls / 1 chain step / 2 distinct tools).
3. `step2` (LLM `complex_judge`): for each candidate, send the same-segment prior questions plus the current question; the LLM returns `{is_complex, category_id, reason}`.
4. `step3` (assemble): for each turn judged complex, emit one row in the `filter_out.jsonc` schema (`context[]` holds prior turns trimmed to `{question, answer}` — same-segment prior, falling back to every earlier session turn for a segment-leading turn, so only a session's very first turn has empty context; `trace_id` = the turn's `run_id`; `category` = `id-slug`, e.g. `01-data-metrics-calculation`).
5. `step4` (verify, LLM `verify_complex`): pass 1 judges with context, which lets connective short turns ride on rich context — so every exported question is re-judged **standalone** (no context): only questions complex on their own survive. LLM failures keep the row (fail-open) and count as `verify_failed`.
6. `step5` (post): two toggleable modules on the assembled rows. `dedup` (rules, MinHash): character n-gram shingles → 128-perm signature, LSH banding limits candidate pairs; rows with Jaccard ≥ `threshold` (default 0.85) are dropped, keeping the first occurrence; dropped rows with provenance land in `work/deduped.jsonl` (`dedup_removed` in summary). `translate` (LLM): `input.text` is translated to `target` (default `zh`, cached in `work/llm_cache.jsonl`) and written to `meta.translation`; already-CJK text is skipped, LLM failures fall back to the original text (counted as `translate_failed`). Runs dedup before translate so fewer rows hit the LLM.

## Checkpoint / resume

A killed run (network outage, Ctrl-C, OOM) can simply be re-run: completed units are skipped instead of re-done. This works at two layers:

- **LLM call layer**: every successful LLM response is append-cached in `work/llm_cache.jsonl` keyed by (prompt, question, model), so re-runs never re-pay for calls that already succeeded.
- **Unit layer**: each completed unit is marked in an append-only checkpoint under `work/checkpoints/` — `session.jsonl` (one line per processed session, holding its rows/stats/debug), `verify.jsonl` and `translate.jsonl` (one line per row, keyed by content). On resume, units already marked are replayed from the checkpoint and their LLM work is skipped entirely.

Semantics:

- Only units whose LLM work **completed cleanly** are marked. A session or row that hit LLM failures is not marked, so the next run retries exactly those calls — once the network is back they succeed and get marked. (Each run also keeps going past failures: sessions continue, verify keeps rows fail-open, translate falls back to the original text.)
- The session checkpoint is tied to the input file (path, size, mtime) and to a fingerprint of the config + all resolved prompts; the verify/translate checkpoints are tied to the config/prompt fingerprint and are keyed by content, so editing the input only re-processes the changed sessions. Any mismatch is logged (`checkpoint ... starting fresh`) and the file is re-seeded rather than reused, so stale results are never silently served.
- Torn trailing lines from a hard-killed run are dropped on load (that unit just re-runs, with its LLM calls being cache hits).

Progress bars are shown for every LLM-heavy stage. Toggle checkpointing with `checkpoint.enabled`; change the directory with `checkpoint.dir` (both in `config.yaml`). To force a full re-run, delete `work/checkpoints/` (or `work/llm_cache.jsonl` to also drop the call cache).

## Output Contract

One output row per complex query:

```json
{
  "capture_mode": "full_link",
  "user_cohort": "regular",
  "source_case_id": "<thread_id>",
  "answer_key": "",
  "trace_id": "<turn run_id>",
  "category": "01-data-metrics-calculation",
  "input": {"text": "<question>", "image": "", "file": ""},
  "session_round": 3,
  "context": [{"question": "...", "answer": "..."}],
  "chain": [],
  "tools": ["web_search", "finquery"],
  "raw_answer": "...",
  "text_answer": "...",
  "multimodal": [],
  "model_version": "",
  "release_id": "",
  "agent_mode": "",
  "translation": "",
  "user_id": "1885129394",
  "difficulty_level": "hard",
  "first_token_time_ms": 31407,
  "finish_answer_time_ms": 52383,
  "input_tokens": 0,
  "output_tokens": 0,
  "request_time_ms": null,
  "meta": {"reason": "需要多步工具调用与综合判断"}
}
```

Public outputs:

- `outputs/complex_queries.jsonl` — one row per complex query (deduped + translated when `post_stage` is enabled)
- `outputs/summary.json` — per-run counters (sessions, segments, candidates, complex/non-complex/llm-failed rows, verify kept/rejected/failed, dedup removed, translated/skipped/failed, category counts)

When `llm_stage.enabled=false`, rules-based candidate selection still runs but no rows are classified, so `complex_queries.jsonl` stays empty.

Intermediate debug files are written under `work/` (`segments.jsonl`, `candidates.jsonl`, `judged.jsonl`, `verified.jsonl`, `deduped.jsonl`) and LLM responses are cached in `work/llm_cache.jsonl`.

`difficulty_level` is fixed to `"hard"` (rows are already judged complex). Each output row's `meta.reason` carries the judge's rationale and `meta.translation` the translated question; the full per-candidate decision (including non-complex and LLM-failure cases) with `is_complex`/`category_id`/`reason`/`error` is in `work/judged.jsonl` for debugging.
