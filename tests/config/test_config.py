"""Config contract: defaults, input-format validation, verify rounds."""

from __future__ import annotations

import unittest
from pathlib import Path

from query_pipeline.config.loader import load_pipeline_config
from query_pipeline.config.models import VerifyConfig

ROOT = Path(__file__).resolve().parents[2]  # repo root (tests/config/ -> project root)

class ConfigContractTest(unittest.TestCase):
    def test_aime_0807_config_loads(self) -> None:
        cfg = load_pipeline_config(ROOT / "configs/aime/0807.yaml")

        self.assertEqual(cfg.name, "aime_0807")
        self.assertEqual(cfg.input.path, (ROOT / "data/aime/0807.jsonl").resolve())
        self.assertEqual(cfg.input.format, "session")
        self.assertEqual(cfg.output.cleaned_queries, "cleaned_queries_0807.jsonl")
        self.assertEqual(cfg.llm.base_url_env, "OPENAI_BASE_URL")
        self.assertEqual(cfg.llm.api_key_env, "OPENAI_API_KEY")
        self.assertEqual(cfg.rule_gate.min_chain_tool_calls, 7)
        self.assertEqual(cfg.rule_gate.min_chain_steps, 1)
        self.assertEqual(cfg.rule_gate.min_unique_tools, 2)
        self.assertEqual(cfg.judge.complexity_prompt, "complexity_gate")

    def test_input_format_validation(self) -> None:
        from query_pipeline.config.models import InputConfig

        self.assertEqual(InputConfig(path=Path("x.jsonl")).format, "auto")
        self.assertEqual(InputConfig(path=Path("x.jsonl"), format="chat").format, "chat")
        self.assertEqual(InputConfig(path=Path("x.jsonl"), format="auto").format, "auto")
        with self.assertRaises(ValueError):
            InputConfig(path=Path("x.jsonl"), format="bogus")

    def test_verify_config_rounds(self) -> None:
        self.assertEqual(VerifyConfig().max_rounds_hard, 5)
        self.assertEqual(VerifyConfig().max_rounds_normal, 2)
        self.assertEqual(VerifyConfig(max_rounds_hard=1).max_rounds_hard, 1)
        with self.assertRaises(ValueError):
            VerifyConfig(max_rounds_hard=0)

