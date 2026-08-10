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

    def test_explicit_relative_cache_checkpoint_resolve_against_work_dir(self) -> None:
        # 第四轮 #10：显式 cache/checkpoint 相对路径与默认值同一基座（work_dir）。
        # 旧实现按 project root 解析：`llm.cache: logs/x.jsonl` + `work_dir: scratch`
        # 会落到 <root>/logs 而非 scratch/logs——--work-dir 覆盖对显式配置失效。
        import tempfile
        import textwrap
        from pathlib import Path

        from query_pipeline.config.loader import load_pipeline_config

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "config.yaml").write_text(
                textwrap.dedent(
                    f"""
                    name: test
                    input:
                      path: input.jsonl
                      format: session
                    output:
                      dir: out
                    work_dir: scratch
                    llm:
                      model: m
                      cache: logs/llm_cache.jsonl
                    checkpoint:
                      dir: logs/checkpoints
                    logging:
                      dir: custom-logs
                      batch_id: upstream-batch
                      level: debug
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            cfg = load_pipeline_config(tmp_path / "config.yaml")
            self.assertEqual(cfg.work_dir, (tmp_path / "scratch").resolve())
            self.assertEqual(cfg.llm.cache, (tmp_path / "scratch" / "logs" / "llm_cache.jsonl").resolve())
            self.assertEqual(cfg.checkpoint.dir, (tmp_path / "scratch" / "logs" / "checkpoints").resolve())
            self.assertEqual(cfg.logging.dir, (tmp_path / "custom-logs").resolve())
            self.assertEqual(cfg.logging.batch_id, "upstream-batch")
            self.assertEqual(cfg.logging.level, "DEBUG")
            # Defaults move to typed runtime directories; explicit overrides above remain untouched.
            (tmp_path / "config2.yaml").write_text(
                textwrap.dedent(
                    """
                    name: test
                    input:
                      path: input.jsonl
                      format: session
                    output:
                      dir: out
                    work_dir: scratch
                    llm:
                      model: m
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            cfg2 = load_pipeline_config(tmp_path / "config2.yaml")
            self.assertEqual(
                cfg2.llm.cache,
                (tmp_path / "scratch" / "runtime" / "cache" / "llm_cache.jsonl").resolve(),
            )
            self.assertEqual(
                cfg2.checkpoint.dir,
                (tmp_path / "scratch" / "runtime" / "checkpoints").resolve(),
            )
            self.assertEqual(cfg2.logging.dir, (tmp_path / "out" / "logs").resolve())
