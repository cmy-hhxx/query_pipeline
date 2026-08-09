"""Stage registry contract: pluggable ordered stages behind a thin runner."""

from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_pipeline.config.loader import load_pipeline_config
from query_pipeline.pipeline.runner import run_pipeline
from query_pipeline.pipeline.stages import DEFAULT_STAGES, REGISTRY, get_stage, register, stage_names


class RegistryTest(unittest.TestCase):
    def test_default_stages_registered(self) -> None:
        self.assertEqual(set(DEFAULT_STAGES), {"preclean", "segment", "rule_gate", "judge", "verify", "post"})
        for name in DEFAULT_STAGES:
            self.assertIn(name, REGISTRY)

    def test_get_unknown_stage_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_stage("nope")

    def test_duplicate_register_raises(self) -> None:
        async def fake(ctx, client, cache, lock):  # type: ignore[no-untyped-def]
            return ctx

        register("dup_test")(fake)
        with self.assertRaises(ValueError):
            register("dup_test")(fake)
        del REGISTRY["dup_test"]

    def test_stage_names_validation(self) -> None:
        self.assertEqual(stage_names(None), list(DEFAULT_STAGES))
        self.assertEqual(stage_names(["preclean"]), ["preclean"])
        with self.assertRaises(ValueError):
            stage_names(["bogus"])


class StageOrderTest(unittest.TestCase):
    """A custom stage list must drive the pipeline; discover-only yields rows
    without verify/post side effects."""

    def test_custom_stage_order_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "thread_id": "t1",
                        "context": [
                            {
                                "question": "帮我构建一个沪深300的增强策略并回测",
                                "answer": "好的，策略如下……",
                                "trace_id": "tr1",
                                "status": "completed",
                                "outcome": "success",
                                "chain": [
                                    {"plan": "p", "tools": [{"name": "web_search", "input": {}, "output": "o"}]}
                                ]
                                * 8,
                                "tool_count": 8,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    name: test_pipeline
                    input:
                      path: {input_path}
                      format: session
                    output:
                      dir: {tmp_path / "out"}
                    work_dir: {tmp_path / "work"}
                    stages: [preclean, rule_gate]
                    segmentation:
                      enabled: false
                    rule_gate:
                      enabled: true
                      min_chain_tool_calls: 7
                    judge:
                      enabled: true
                    llm:
                      enabled: false
                    """
                ),
                encoding="utf-8",
            )

            class FakeClient:  # must not be constructed since llm.enabled=false
                def __init__(self, *a, **k):  # type: ignore[no-untyped-def]
                    raise AssertionError("llm disabled, client must not be created")

            with patch("query_pipeline.pipeline.runner.LLMClient", FakeClient):
                summary = run_pipeline(load_pipeline_config(config_path))
            self.assertTrue(summary.success)
            self.assertEqual(summary.stats["complex_rows"], 0)  # llm off -> no judge
            # preclean + rule_gate only: no output file (runner skips empty writes)


if __name__ == "__main__":
    unittest.main()
