"""Pipeline context: output/log path layout under the work dir."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from query_pipeline.config.models import PipelineConfig
from query_pipeline.pipeline.context import PipelineContext

class RuntimeLayoutTest(unittest.TestCase):
    def test_ctx_path_under_runtime_diagnostics(self) -> None:
        from types import SimpleNamespace

        config = cast(PipelineConfig, SimpleNamespace(work_dir=Path("/x/y")))
        ctx = PipelineContext(config=config)
        self.assertEqual(
            ctx.path("bad_lines.jsonl"),
            Path("/x/y/runtime/diagnostics/bad_lines.jsonl"),
        )

if __name__ == "__main__":
    unittest.main()
