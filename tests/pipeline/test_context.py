"""Pipeline context: output/log path layout under the work dir."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from query_pipeline.pipeline.context import PipelineContext

class LogsLayoutTest(unittest.TestCase):
    def test_ctx_path_under_logs(self) -> None:
        from types import SimpleNamespace

        ctx = PipelineContext(config=SimpleNamespace(work_dir=Path("/x/y")))
        self.assertEqual(ctx.path("bad_lines.jsonl"), Path("/x/y/logs/bad_lines.jsonl"))

if __name__ == "__main__":
    unittest.main()

