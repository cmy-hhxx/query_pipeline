"""Public API + CLI: minimal args, sane defaults, summary contract."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from query_pipeline import run as api_run
from query_pipeline.cli import main as cli_main
from tests._profiles import complexity_label, verify_label

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
                return json.dumps({"category_id": "01", "reason": "复杂归类"})
            if "有价值但非复杂" in system_prompt:
                return json.dumps({"category_id": "03", "reason": "普通归类"})
            return json.dumps(complexity_label(True, goal=payload["current_question"]))
        if "简单问句识别器" in system_prompt:  # simple_finder 视角
            return json.dumps({"is_simple": False, "reason": "不是简单问句"}, ensure_ascii=False)
        if "question" in payload:
            return json.dumps(verify_label(True, reason="复杂", evidence_quote=payload["question"]))
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

    def test_business_logs_match_outputs_and_are_returned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "input.jsonl"
            src.write_text(_session_lines(), encoding="utf-8")
            with patch("query_pipeline.pipeline.runner.LLMClient", RecordingClient):
                summary = api_run(
                    src,
                    output_dir=tmp_path / "out",
                    work_dir=tmp_path / "work",
                    log_dir=tmp_path / "logs",
                    batch_id="api-batch",
                )

            self.assertEqual(summary["logs"]["batch_id"], "api-batch")
            ordinary = Path(summary["logs"]["ordinary"])
            self.assertEqual(
                ordinary,
                (tmp_path / "logs" / "ordinary" / "run" / "api-batch.log").resolve(),
            )
            ordinary_rows = [json.loads(line) for line in ordinary.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(ordinary_rows[0]["command"], "run")
            self.assertEqual(ordinary_rows[-1]["message"], "command_finished")

            output_rows = [
                json.loads(line)
                for line in Path(summary["output_files"]["cleaned_queries"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            business = summary["logs"]["business"]
            cleaned_rows = [json.loads(line) for line in Path(business["cleaned"]).read_text(encoding="utf-8").splitlines()]
            complex_rows = [json.loads(line) for line in Path(business["complex"]).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(cleaned_rows, output_rows)
            self.assertEqual(complex_rows, output_rows)
            self.assertEqual(Path(business["normal"]).read_text(encoding="utf-8"), "")
            artifact_summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(artifact_summary["logs"], summary["logs"])

    def test_business_log_failure_propagates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "input.jsonl"
            src.write_text(_session_lines(), encoding="utf-8")
            with patch("query_pipeline.pipeline.runner.LLMClient", RecordingClient), patch(
                "query_pipeline.io.business_log.BusinessLogWriter.write",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    api_run(
                        src,
                        output_dir=tmp_path / "out",
                        work_dir=tmp_path / "work",
                        log_dir=tmp_path / "logs",
                        batch_id="write-failure",
                    )
            ordinary = tmp_path / "logs" / "ordinary" / "run" / "write-failure.log"
            rows = [json.loads(line) for line in ordinary.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(rows[-1]["message"].startswith("command_failed"))

    def _run_with(self, src: Path, tmp: Path, **kwargs: Any) -> dict:
        with patch("query_pipeline.pipeline.runner.LLMClient", RecordingClient):
            return api_run(src, output_dir=tmp / "out", work_dir=tmp / "work", **kwargs)

    def _chat_line(self, case_id: str, n_calls: int, tool_names: list[str]) -> str:
        import json as _json

        chain = [
            {"plan": "", "tools": [{"name": tool_names[i % len(tool_names)], "input": {}, "output": "o"}]}
            for i in range(n_calls)
        ]
        return _json.dumps(
            {
                "trace_id": f"tr-{case_id}",
                "question": "帮我分析茅台估值并给出买卖建议",
                "judge_data": {
                    "case_id": case_id,
                    "input": {"text": "帮我分析茅台估值并给出买卖建议"},
                    "context": [],
                    "chain": chain,
                    "raw_answer": "分析如下，" + "数据" * 30,
                    "text_answer": "分析如下，" + "数据" * 30,
                },
            }
        )

    def test_auto_chat_uses_chat_gate(self) -> None:
        # chat 输入 + format="auto"：门槛必须是 chat 的 3/2（4 次调用 2 种工具
        # 能过 3/2，但过不了 session 的 7/2）——旧实现 auto 落到 7/2 会过滤掉。
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "input.jsonl"
            src.write_text(self._chat_line("c1", 4, ["web_search", "finquery"]) + "\n", encoding="utf-8")
            summary = self._run_with(src, tmp_path)
        self.assertEqual(summary["stats"]["input_format"], "chat")
        self.assertEqual(summary["stats"]["complex_rows"], 1)

    def test_auto_session_uses_session_gate(self) -> None:
        # session 输入 + format="auto"：门槛必须是 session 的 7/2（4 次调用应被过滤）
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "input.jsonl"
            chain = [
                {"plan": "p", "tools": [{"name": n, "input": {}, "output": "o"}]}
                for n in ("web_search", "finquery")
            ] * 2
            src.write_text(
                json.dumps(
                    {
                        "thread_id": "t1",
                        "context": [
                            {
                                "question": "帮我分析茅台的估值",
                                "answer": "分析如下，" + "数据" * 30,
                                "trace_id": "tr1",
                                "status": "completed",
                                "outcome": "success",
                                "chain": chain,
                                "tool_count": 4,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = self._run_with(src, tmp_path)
        self.assertEqual(summary["stats"]["input_format"], "session")
        self.assertEqual(summary["stats"]["candidates"], 0)

    def test_no_reject_rules_effective(self) -> None:
        # --no-reject-rules 必须生效：问句命中 reject（LOW_VALUE_COMMON）时，
        # 默认被拒；reject_rules=False 时保留（旧实现默认分支吞掉该参数）。
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "input.jsonl"
            chain = [
                {"plan": "p", "tools": [{"name": n, "input": {}, "output": "o"}]}
                for n in ("web_search", "finquery", "compute")
            ] * 3
            src.write_text(
                json.dumps(
                    {
                        "thread_id": "t1",
                        "context": [
                            {
                                "question": "好的",
                                "answer": "好的，" + "数据" * 30,
                                "trace_id": "tr1",
                                "status": "completed",
                                "outcome": "success",
                                "chain": chain,
                                "tool_count": 9,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with patch("query_pipeline.pipeline.runner.LLMClient", RecordingClient):
                default = api_run(src, output_dir=tmp_path / "out1", work_dir=tmp_path / "work1")
                no_reject = api_run(
                    src, output_dir=tmp_path / "out2", work_dir=tmp_path / "work2", reject_rules=False
                )
        self.assertEqual(default["stats"]["candidates"], 0)
        self.assertEqual(no_reject["stats"]["candidates"], 1)
        self.assertEqual(no_reject["stats"]["complex_rows"], 1)

    def test_single_knob_keeps_format_default(self) -> None:
        # 只传 min_tool_calls=5：min_unique_tools 必须补 chat 默认 2（而非旧实现的 1）。
        # 5 次调用但仅 1 种工具 → 被 2 过滤（旧实现 (5,1) 会放行）。
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "input.jsonl"
            src.write_text(self._chat_line("c1", 5, ["web_search"]) + "\n", encoding="utf-8")
            summary = self._run_with(src, tmp_path, min_tool_calls=5)
        self.assertEqual(summary["stats"]["input_format"], "chat")
        self.assertEqual(summary["stats"]["candidates"], 0)

    def test_missing_input_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            work_dir = tmp_path / "work"
            diagnostics = work_dir / "runtime" / "diagnostics"
            diagnostics.mkdir(parents=True)
            previous = diagnostics / "previous-run.jsonl"
            previous.write_text('{"trace_id":"previous"}\n', encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                api_run(
                    tmp_path / "missing.jsonl",
                    output_dir=tmp_path / "out",
                    work_dir=work_dir,
                    log_dir=tmp_path / "logs",
                    batch_id="missing-input",
                )
            self.assertEqual(
                previous.read_text(encoding="utf-8"),
                '{"trace_id":"previous"}\n',
            )
            ordinary = tmp_path / "logs" / "ordinary" / "run" / "missing-input.log"
            rows = [json.loads(line) for line in ordinary.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(rows[-1]["message"].startswith("command_failed"))

    def test_explicit_api_key_overrides_env(self) -> None:
        # env 已有 key 时，显式传参必须覆盖（旧实现 setdefault 静默忽略显式 key）
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "input.jsonl"
            src.write_text(_session_lines(), encoding="utf-8")
            with patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "env-old-key", "OPENAI_BASE_URL": "http://env-old"},
                clear=False,
            ):
                with patch("query_pipeline.pipeline.runner.LLMClient", RecordingClient):
                    api_run(
                        src,
                        output_dir=tmp_path / "out",
                        work_dir=tmp_path / "work",
                        api_key="explicit-new-key",
                        base_url="http://explicit-new",
                    )
                self.assertEqual(os.environ["OPENAI_API_KEY"], "explicit-new-key")
                self.assertEqual(os.environ["OPENAI_BASE_URL"], "http://explicit-new")

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

    def test_suggest_writes_ordinary_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "input.jsonl"
            src.write_text(_session_lines(), encoding="utf-8")
            code = cli_main(
                [
                    "suggest",
                    str(src),
                    "--log-dir",
                    str(tmp_path / "logs"),
                    "--batch-id",
                    "suggest-batch",
                ]
            )
            self.assertEqual(code, 0)
            path = tmp_path / "logs" / "ordinary" / "suggest" / "suggest-batch.log"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["command"], "suggest")
            self.assertEqual(rows[-1]["message"], "command_finished")

if __name__ == "__main__":
    unittest.main()
