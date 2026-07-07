# query_pipeline

Config-driven query cleaning and labeling pipeline.

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

## Flow

The pipeline always runs in two stages:

1. `rules_stage`: read `input.text_path`, normalize text, reject invalid rows, apply cleaning rules, deduplicate, then gate low-complexity rows.
2. `llm_stage`: label only rows that passed the rules stage with one unified prompt.

`config.yaml` keeps prompt selection lightweight with `llm_stage.prompt_id`. Prompt text lives in `src/query_pipeline/prompts/`.

## Output Contract

Input JSONL records keep their original top-level structure. The pipeline only adds one top-level field:

```json
{
  "question": "请结合基本面、技术面和资金面分析某只股票未来一个月的风险和机会",
  "query_pipeline_output": {
    "status": "accepted",
    "source_text_path": "question",
    "normalized_text": "请结合基本面、技术面和资金面分析某只股票未来一个月的风险和机会",
    "rule_signals": {
      "complexity_score": 4,
      "complexity_reasons": ["len_ge_30", "analysis_or_judgement", "finance_dimensions", "multi_constraint_or_horizon"]
    },
    "llm_label": {
      "is_complex": true,
      "category_id": "03",
      "category_name": "分析研究类",
      "is_multi_turn": false,
      "difficulty_score": 2.8,
      "difficulty_reason": "需要结合多个分析维度",
      "reason": "该问句要求综合分析金融标的"
    }
  }
}
```

Default public outputs are:

- `outputs/accepted.jsonl`
- `outputs/rejected.jsonl`
- `outputs/skipped.jsonl`
- `outputs/summary.json`

Intermediate files are written under `work/` for debugging.
