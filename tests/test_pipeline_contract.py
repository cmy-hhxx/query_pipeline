from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_pipeline.config.loader import load_pipeline_config
from query_pipeline.models.records import CATEGORIES, parse_core_label_response
from query_pipeline.pipeline.runner import run_pipeline
from query_pipeline.prompts import resolve_prompt


class PipelineContractTest(unittest.TestCase):
    def test_default_config_loads(self) -> None:
        cfg = load_pipeline_config(ROOT / "config.yaml")

        self.assertEqual(cfg.name, "question_pipeline")
        self.assertEqual(cfg.input.text_path, "question")
        self.assertEqual(cfg.llm_stage.prompt_id, "core_label")
        prompt = resolve_prompt(cfg.llm_stage.prompt_id)
        self.assertIn("复杂金融问句", prompt)
        self.assertIn("核心结构化标注", prompt)
        self.assertIn("category_reason", prompt)
        self.assertNotIn("五维独立", prompt)
        self.assertNotIn("intent_labels", prompt)
        self.assertNotIn("demand_labels", prompt)
        self.assertNotIn("domain_label", prompt)
        self.assertNotIn("query_quality", prompt)
        self.assertNotIn("query_difficulty", prompt)
        self.assertNotIn("nlu_reference", prompt)
        self.assertNotIn("rule_signals", prompt)
        self.assertNotIn("旧提示词", prompt)
        self.assertNotIn("旧五维", prompt)
        self.assertNotIn("source_text_path", prompt)
        self.assertNotIn("source_line_number", prompt)

    def test_rules_stage_preserves_input_shape_and_writes_nested_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.jsonl"
            source_text = "请结合基本面、技术面、资金面分析贵州茅台未来一个月的风险和机会"
            rows = [
                {"id": "keep", "payload": {"query": source_text}, "question": "original top-level question"},
                {"id": "dup", "payload": {"query": source_text}},
                {"id": "missing", "payload": {}},
                {"id": "empty", "payload": {"query": "   "}},
                {"id": "invalid", "payload": {"query": {"text": "bad"}}},
                {"id": "skip", "payload": {"query": "贵州茅台走势"}},
            ]
            _write_jsonl(input_path, rows)
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=False))

            summary = run_pipeline(cfg)

            accepted = _read_jsonl(Path(summary.output_files["accepted"]))
            rejected = _read_jsonl(Path(summary.output_files["rejected"]))
            skipped = _read_jsonl(Path(summary.output_files["skipped"]))

            self.assertEqual(len(accepted), 1)
            self.assertEqual(len(skipped), 1)
            self.assertEqual(len(rejected), 4)

            accepted_row = accepted[0]
            self.assertEqual(accepted_row["question"], "original top-level question")
            self.assertEqual(accepted_row["payload"]["query"], source_text)
            self.assertIn("query_pipeline_output", accepted_row)
            self.assertNotIn("normalized_text", accepted_row)
            self.assertNotIn("complexity_score", accepted_row)
            self.assertNotIn("reject_reason", accepted_row)

            output = accepted_row["query_pipeline_output"]
            self.assertEqual(output["status"], "accepted")
            self.assertEqual(output["source_text_path"], "payload.query")
            self.assertEqual(output["normalized_text"], source_text)
            self.assertFalse(any(key.endswith("ver" + "sion") for key in output))
            self.assertGreaterEqual(output["rule_signals"]["complexity_score"], 2)
            self.assertFalse(any(key.endswith("ver" + "sion") for key in output["rule_signals"]))

            reject_reasons = {row["query_pipeline_output"]["reject_reason"] for row in rejected}
            self.assertEqual(
                reject_reasons,
                {"duplicate_exact", "missing_text_path", "blank", "invalid_text_value"},
            )
            self.assertEqual(skipped[0]["query_pipeline_output"]["status"], "skipped")
            self.assertEqual(skipped[0]["query_pipeline_output"]["skip_reason"], "low_complexity_score_or_short")

    def test_core_llm_label_stage_uses_nested_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.jsonl"
            _write_jsonl(
                input_path,
                [{"id": "q1", "payload": {"query": "请分析贵州茅台基本面和技术面风险机会"}}],
            )
            cfg = load_pipeline_config(_write_config(tmp_path, llm_enabled=True))

            with patch("query_pipeline.steps.llm_label.LLMClient", FakeLLMClient):
                summary = run_pipeline(cfg)

            accepted = _read_jsonl(Path(summary.output_files["accepted"]))
            self.assertEqual(len(accepted), 1)
            output = accepted[0]["query_pipeline_output"]
            self.assertEqual(output["status"], "accepted")
            label = output["llm_label"]
            self.assertEqual(label["category_id"], "03")
            self.assertEqual(label["category_name"], "分析研究类")
            self.assertEqual(label["difficulty_score"], 2.8)
            self.assertEqual(label["difficulty_reason"], "需要结合基本面和技术面判断")
            self.assertEqual(label["category_reason"], "该问句要求综合分析金融标的")
            self.assertEqual(
                set(label),
                {
                    "is_complex",
                    "category_id",
                    "category_name",
                    "is_multi_turn",
                    "difficulty_score",
                    "difficulty_reason",
                    "category_reason",
                },
            )
            self.assertNotIn("category_id", accepted[0])
            self.assertEqual(summary.stats["llm_extra_field_rows"], 1)
            self.assertEqual(
                summary.stats["llm_extra_fields"],
                {"category_name": 1, "query_difficulty": 1, "query_quality": 1, "reason": 1},
            )

            cache_path = tmp_path / "work" / "llm_cache.jsonl"
            self.assertTrue(cache_path.exists())
            cache_label = _read_jsonl(cache_path)[0]["label"]
            self.assertNotIn("category_name", cache_label)
            self.assertNotIn("query_quality", cache_label)
            self.assertEqual(cache_label["category_reason"], "该问句要求综合分析金融标的")

    def test_core_label_parser_accepts_all_category_ids(self) -> None:
        for category_id, category_name in CATEGORIES.items():
            raw = json.dumps(
                {
                    "is_complex": True,
                    "category_id": category_id,
                    "category_name": category_name,
                    "is_multi_turn": False,
                    "difficulty_score": 3.0,
                    "difficulty_reason": "需要专业判断",
                    "category_reason": "符合该类别定义",
                },
                ensure_ascii=False,
            )

            parsed = parse_core_label_response(raw)

            self.assertEqual(parsed.category_id, category_id)
            self.assertEqual(parsed.category_name, category_name)
            self.assertEqual(parsed.extra_fields, ("category_name",))

    def test_core_label_parser_nulls_difficulty_for_non_complex_rows(self) -> None:
        raw = json.dumps(
            {
                "is_complex": False,
                "category_id": "03",
                "is_multi_turn": False,
                "difficulty_score": 3.0,
                "difficulty_reason": "需要专业判断",
                "category_reason": "只是简单查行情，不属于复杂金融问句",
            },
            ensure_ascii=False,
        )

        parsed = parse_core_label_response(raw)
        output = parsed.to_output()

        self.assertIsNone(output["category_id"])
        self.assertIsNone(output["category_name"])
        self.assertIsNone(output["difficulty_score"])
        self.assertIsNone(output["difficulty_reason"])
        self.assertEqual(output["category_reason"], "只是简单查行情，不属于复杂金融问句")


class FakeLLMClient:
    def __init__(self, config: object) -> None:
        self.config = config

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        assert system_prompt
        assert "normalized_text" in user_prompt
        assert "rule_signals" not in user_prompt
        assert "source_text_path" not in user_prompt
        assert "source_line_number" not in user_prompt
        return json.dumps(
            {
                "is_complex": True,
                "category_id": "03",
                "category_name": "分析研究类",
                "is_multi_turn": False,
                "difficulty_score": 2.8,
                "difficulty_reason": "需要结合基本面和技术面判断",
                "category_reason": "该问句要求综合分析金融标的",
                "reason": "旧字段应被丢弃",
                "query_quality": "高",
                "query_difficulty": "中",
            },
            ensure_ascii=False,
        )

    async def close(self) -> None:
        return None


def _write_config(tmp_path: Path, *, llm_enabled: bool) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            name: test_pipeline
            input:
              path: input.jsonl
              text_path: payload.query
            output:
              dir: out
              accepted: accepted.jsonl
              rejected: rejected.jsonl
              skipped: skipped.jsonl
              summary: summary.json
            work_dir: work
            rules_stage:
              enabled: true
              clean:
                enabled: true
                min_length: 1
                finance_semantic: false
              exact_dedup:
                enabled: true
              minhash:
                enabled: false
              complexity_gate:
                enabled: true
                min_score: 2
                min_text_length: 1
            llm_stage:
              enabled: {str(llm_enabled).lower()}
              base_url: https://example.invalid
              model: fake-model
              api_key_env: FAKE_API_KEY
              concurrency: 2
              max_retries: 1
              timeout_seconds: 1
              response_format: json_object
              cache: work/llm_cache.jsonl
              prompt_id: core_label
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
