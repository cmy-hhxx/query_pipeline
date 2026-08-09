"""Public API + CLI: minimal args, sane defaults, summary contract."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from query_pipeline import run as api_run
from query_pipeline.cli import main as cli_main

def _session_lines() -> str:
    import json as _json

    chain = [
        {"plan": "p", "tools": [{"name": n, "input": {}, "output": "o"}]}
        for n in ("web_search", "finquery", "compute")
    ] * 3
    return "\n".join(
        _json.dumps(
            {
                "thread_id": "t1",
                "context": [
                    {
                        "question": "帮我分析一下贵州茅台的估值并给出买卖建议",
                        "answer": "分析如下，" + "数据" * 30,
                        "trace_id": "tr1",
                        "status": "completed",
                        "outcome": "success",
                        "last_event_type": "runFinished",
                        "chain": chain,
                        "tool_count": 9,
                    }
                ],
            }
        )
        for _ in range(1)
    )

class RecordingClient:
    def __init__(self, config: object) -> None:
        self.config = config

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt.split("\n", 1)[1])
        if "questions" in payload:
            n = len(payload["questions"])
            return json.dumps({"segments": [{"start": 0, "end": n - 1, "topic": "t"}]})
        if "current_question" in payload:
            if "价值判官" in system_prompt:
                return json.dumps({"is_valuable": True})
            if "已判定为复杂金融问句" in system_prompt:
                return json.dumps({"category_id": "01"})
            if "有价值但非复杂" in system_prompt:
                return json.dumps({"category_id": "03"})
            return json.dumps({"is_complex": True})
        if "简单问句识别器" in system_prompt:  # simple_finder 视角
            return json.dumps({"is_simple": False, "reason": "不是简单问句"}, ensure_ascii=False)
        if "question" in payload:
            return json.dumps({"is_complex": True, "reason": "复杂"})
        return json.dumps({"translation": "译"})

    async def close(self) -> None:
        return None

class ApiTest(unittest.TestCase):
    def test_run_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "input.jsonl"
            src.write_text(_session_lines(), encoding="utf-8")
            with patch("query_pipeline.pipeline.runner.LLMClient", RecordingClient):
                summary = api_run(src, output_dir=tmp_path / "out", work_dir=tmp_path / "work")
            self.assertTrue(summary["success"])
            self.assertEqual(summary["stats"]["complex_rows"], 1)
            self.assertEqual(summary["stats"]["output_rows"], 1)
            self.assertEqual(summary["stats"]["input_format"], "session")
            out = Path(summary["output_files"]["cleaned_queries"])
            self.assertTrue(out.exists())
            self.assertEqual(out.name, "cleaned_queries.jsonl")

    def test_missing_input_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            api_run("/nonexistent/input.jsonl")

    def test_default_output_dir(self) -> None:
        # unique dataset dir so the default outputs/<parent> path can never
        # collide with real deliverables under the repo's outputs/; the
        # fixture dir is removed afterwards so tests leave no residue.
        fixture = Path("outputs") / "zz_api_fixture"
        shutil.rmtree(fixture, ignore_errors=True)
        self.addCleanup(shutil.rmtree, fixture, ignore_errors=True)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "zz_api_fixture" / "0806.jsonl"
            src.parent.mkdir()
            src.write_text(_session_lines(), encoding="utf-8")
            with patch("query_pipeline.pipeline.runner.LLMClient", RecordingClient):
                summary = api_run(src, work_dir=tmp_path / "work")
            out = Path(summary["output_files"]["cleaned_queries"])
            self.assertEqual(out.parent, fixture)
            self.assertTrue(out.parent.exists())

class CliTest(unittest.TestCase):
    def test_cli_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "input.jsonl"
            src.write_text(_session_lines(), encoding="utf-8")
            with patch("query_pipeline.pipeline.runner.LLMClient", RecordingClient):
                code = cli_main(
                    ["run", str(src), "-o", str(tmp_path / "out"), "--work-dir", str(tmp_path / "work")]
                )
            self.assertEqual(code, 0)
            self.assertTrue((tmp_path / "out" / "cleaned_queries.jsonl").exists())

    def test_cli_no_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "input.jsonl"
            src.write_text(_session_lines(), encoding="utf-8")
            code = cli_main(["run", str(src), "-o", str(tmp_path / "out"), "--no-llm", "--work-dir", str(tmp_path / "work")])
            self.assertEqual(code, 0)  # rules-only run, empty output is legitimate

if __name__ == "__main__":
    unittest.main()
