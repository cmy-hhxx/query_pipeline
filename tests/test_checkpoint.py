from __future__ import annotations

import asyncio
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
from query_pipeline.io.checkpoint import Checkpoint, content_key
from query_pipeline.pipeline.runner import run_pipeline


def _make_turn(idx: int, question: str) -> dict[str, Any]:
    if idx % 2 == 0:
        # simple turn: too few tool calls to be a candidate
        return {
            "question": question,
            "answer": f"answer{idx}",
            "run_id": f"r{idx}",
            "trace_id": f"trace{idx}",
            "status": "completed",
            "outcome": "success",
            "tool_names": "web_search",
            "tool_count": 1,
            "chain": [{"plan": "", "tools": [{"name": "web_search", "input": {}, "output": "x"}]}],
        }
    # complex turn: 8 tool calls across 3 tools -> passes step1 screening
    names = ("web_search", "finquery", "compute")
    return {
        "question": question,
        "answer": f"answer{idx}",
        "run_id": f"r{idx}",
        "trace_id": f"trace{idx}",
        "status": "completed",
        "outcome": "success",
        "tool_names": "web_search,finquery,compute",
        "tool_count": 8,
        "chain": [{"plan": "", "tools": [{"name": names[i % 3], "input": {}, "output": "x"}]} for i in range(8)],
    }


def _session(thread_id: str, tag: str) -> dict[str, Any]:
    return {
        "thread_id": thread_id,
        "context": [_make_turn(0, f"{tag} simple"), _make_turn(1, f"{tag} complex query")],
    }


class ScriptedClient:
    """Records every call; raises RuntimeError for prompts whose
    current_question / standalone question / text is in fail_on."""

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.fail_on = set(fail_on or ())
        self.calls: list[dict[str, Any]] = []
        self.config: object | None = None

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        assert system_prompt
        payload = json.loads(user_prompt.split("\n", 1)[1])
        self.calls.append(payload)
        if "questions" in payload:  # segmentation
            n = len(payload["questions"])
            return json.dumps({"segments": [{"start": 0, "end": n - 1, "topic": "t"}]}, ensure_ascii=False)
        if "current_question" in payload:  # step2 judge
            if payload["current_question"] in self.fail_on:
                raise RuntimeError("simulated network failure")
            return json.dumps({"is_complex": True, "category_id": "03", "reason": "复杂"}, ensure_ascii=False)
        if "question" in payload:  # verify
            if payload["question"] in self.fail_on:
                raise RuntimeError("simulated network failure")
            return json.dumps({"is_complex": True, "reason": "自身复杂"}, ensure_ascii=False)
        if "text" in payload:  # translate
            if payload["text"] in self.fail_on:
                raise RuntimeError("simulated network failure")
            return json.dumps({"translation": "翻译：" + payload["text"]}, ensure_ascii=False)
        raise AssertionError(f"unexpected payload: {sorted(payload)}")

    async def close(self) -> None:
        return None


def _factory(clients: list[ScriptedClient], fail_on: set[str] | None) -> Any:
    def factory(config: object) -> ScriptedClient:
        client = ScriptedClient(fail_on)
        client.config = config
        clients.append(client)
        return client

    return factory


def run_pipeline_with_fakes(
    cfg: Any, *, session_fail: set[str] | None = None, verify_fail: set[str] | None = None, translate_fail: set[str] | None = None
) -> tuple[Any, list[ScriptedClient], list[ScriptedClient], list[ScriptedClient]]:
    session_clients: list[ScriptedClient] = []
    verify_clients: list[ScriptedClient] = []
    post_clients: list[ScriptedClient] = []
    with patch("query_pipeline.steps.session_stage.LLMClient", _factory(session_clients, session_fail)), patch(
        "query_pipeline.steps.verify_stage.LLMClient", _factory(verify_clients, verify_fail)
    ), patch("query_pipeline.steps.post_stage.LLMClient", _factory(post_clients, translate_fail)):
        summary = run_pipeline(cfg)
    return summary, session_clients, verify_clients, post_clients


class CheckpointUnitTest(unittest.TestCase):
    def test_roundtrip_and_torn_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp.jsonl"

            async def mark_all() -> None:
                cp = Checkpoint(path=path)
                await cp.mark("a", v=1)
                await cp.mark("b", v=2)

            asyncio.run(mark_all())
            # Hard-killed run leaves a torn line: it must be dropped, not crash.
            with path.open("a", encoding="utf-8") as handle:
                handle.write('{"key": "c", "v": 3')

            cp = Checkpoint.load(path)
            self.assertEqual(cp.get("a"), {"key": "a", "v": 1})
            self.assertEqual(cp.get("b"), {"key": "b", "v": 2})
            self.assertIsNone(cp.get("c"))

    def test_meta_mismatch_reseeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp.jsonl"
            cp = Checkpoint.load(path, expected_meta={"pipeline_hash": "h1"})
            asyncio.run(cp.mark("a", v=1))

            # Input/config changed: the file is ignored and re-seeded fresh.
            cp2 = Checkpoint.load(path, expected_meta={"pipeline_hash": "h2"})
            self.assertEqual(cp2.meta, {"pipeline_hash": "h2"})
            self.assertIsNone(cp2.get("a"))
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)  # meta line only

    def test_disabled_checkpoint_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp.jsonl"
            cp = Checkpoint(path=path, enabled=False)
            asyncio.run(cp.mark("a", v=1))
            self.assertIsNone(cp.get("a"))
            self.assertFalse(path.exists())

    def test_content_key(self) -> None:
        self.assertEqual(content_key("a", "b"), content_key("a", "b"))
        self.assertNotEqual(content_key("a", "b"), content_key("a", "c"))


class SessionResumeTest(unittest.TestCase):
    def test_session_stage_resumes_after_network_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sessions = [_session("t0", "S0"), _session("t1", "S1"), _session("t2", "S2")]
            _write_jsonl(tmp_path / "input.jsonl", sessions)
            cfg = load_pipeline_config(_write_config(tmp_path, post_enabled=False))

            # Run 1: session t1's judge call fails (outage). Its session is
            # NOT checkpointed so it retries; t0/t2 are checkpointed.
            summary1, _, _, _ = run_pipeline_with_fakes(cfg, session_fail={"S1 complex query"})
            self.assertEqual(summary1.stats["llm_failed"], 1)
            self.assertEqual(summary1.stats["complex_rows"], 2)
            cp_path = tmp_path / "work/checkpoints/session.jsonl"
            self.assertEqual(_checkpoint_keys(cp_path), {"0", "2"})

            # Run 2: only t1's judge re-runs (cache miss); t0/t2 replay from
            # the checkpoint with zero LLM calls; the previously failed verify
            # row also re-runs once.
            summary2, session2, verify2, _ = run_pipeline_with_fakes(cfg)
            self.assertEqual(summary2.stats["llm_failed"], 0)
            self.assertEqual(summary2.stats["complex_rows"], 3)
            self.assertEqual(summary2.stats["verify_kept"], 3)
            self.assertEqual([c["current_question"] for c in session2[0].calls], ["S1 complex query"])
            self.assertEqual([c["question"] for c in verify2[0].calls], ["S1 complex query"])
            self.assertEqual(_checkpoint_keys(cp_path), {"0", "1", "2"})

            rows = _read_jsonl(tmp_path / "out/complex_queries.jsonl")
            self.assertEqual([r["input"]["text"] for r in rows], ["S0 complex query", "S1 complex query", "S2 complex query"])


class VerifyResumeTest(unittest.TestCase):
    def test_verify_stage_resumes_after_network_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sessions = [_session("t0", "V0"), _session("t1", "V1"), _session("t2", "V2")]
            _write_jsonl(tmp_path / "input.jsonl", sessions)
            cfg = load_pipeline_config(_write_config(tmp_path, post_enabled=False))

            # Run 1: verify for V1 fails (fail-open keeps the row but does not
            # checkpoint it); V0/V2 are checkpointed.
            summary1, _, _, _ = run_pipeline_with_fakes(cfg, verify_fail={"V1 complex query"})
            self.assertEqual(summary1.stats["verify_kept"], 2)
            self.assertEqual(summary1.stats["verify_failed"], 1)
            self.assertEqual(summary1.stats["complex_rows"], 3)

            # Run 2: V0/V2 replay from the verify checkpoint; only V1 re-verifies.
            summary2, session2, verify2, _ = run_pipeline_with_fakes(cfg)
            self.assertEqual(summary2.stats["verify_kept"], 3)
            self.assertEqual(summary2.stats["verify_failed"], 0)
            self.assertEqual(session2[0].calls, [])  # session stage fully replayed
            self.assertEqual([c["question"] for c in verify2[0].calls], ["V1 complex query"])


class TranslateResumeTest(unittest.TestCase):
    def test_translate_stage_resumes_after_network_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sessions = [_session("t0", "T0"), _session("t1", "T1"), _session("t2", "T2")]
            _write_jsonl(tmp_path / "input.jsonl", sessions)
            cfg = load_pipeline_config(_write_config(tmp_path, post_enabled=True))

            # Run 1: translation of T1 fails (falls back to the original text)
            # and is not checkpointed; T0/T2 are.
            summary1, _, _, _ = run_pipeline_with_fakes(cfg, translate_fail={"T1 complex query"})
            self.assertEqual(summary1.stats["translated"], 2)
            self.assertEqual(summary1.stats["translate_failed"], 1)
            self.assertEqual(summary1.stats["complex_rows"], 3)

            # Run 2: T0/T2 reuse checkpointed translations; only T1 re-translates.
            summary2, session2, verify2, post2 = run_pipeline_with_fakes(cfg)
            self.assertEqual(summary2.stats["translated"], 3)
            self.assertEqual(summary2.stats["translate_failed"], 0)
            self.assertEqual(session2[0].calls, [])
            self.assertEqual(verify2[0].calls, [])
            self.assertEqual([c["text"] for c in post2[0].calls], ["T1 complex query"])

            rows = _read_jsonl(tmp_path / "out/complex_queries.jsonl")
            self.assertEqual(rows[1]["meta"]["translation"], "翻译：T1 complex query")


class CheckpointInvalidationTest(unittest.TestCase):
    def test_input_change_invalidates_session_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sessions = [_session("t0", "W0"), _session("t1", "W1")]
            _write_jsonl(tmp_path / "input.jsonl", sessions)
            cfg = load_pipeline_config(_write_config(tmp_path, post_enabled=False))
            run_pipeline_with_fakes(cfg)

            # Rewrite the input with a new question; the session checkpoint no
            # longer matches (size/mtime) and must be re-seeded, while the
            # content-addressed verify checkpoint still serves unchanged rows.
            sessions[0]["context"][1]["question"] = "W0 complex query NEW"
            _write_jsonl(tmp_path / "input.jsonl", sessions)
            summary, session2, verify2, _ = run_pipeline_with_fakes(cfg)

            self.assertEqual(summary.stats["complex_rows"], 2)
            # W0's changed question misses both the session checkpoint and the
            # LLM cache, so its segment and judge calls re-run; W1 stays cached.
            self.assertEqual(
                [c["current_question"] for c in session2[0].calls if "current_question" in c], ["W0 complex query NEW"]
            )
            self.assertEqual([c["question"] for c in verify2[0].calls], ["W0 complex query NEW"])

            rows = _read_jsonl(tmp_path / "out/complex_queries.jsonl")
            texts = [r["input"]["text"] for r in rows]
            self.assertIn("W0 complex query NEW", texts)
            self.assertNotIn("W0 complex query", texts)


def _write_config(tmp_path: Path, *, post_enabled: bool) -> Path:
    config_path = tmp_path / "config.yaml"
    post_block = ""
    if post_enabled:
        post_block = """
            post_stage:
              enabled: true
              dedup:
                enabled: true
                threshold: 0.85
              translate:
                enabled: true
                target: zh
"""
    config_path.write_text(
        textwrap.dedent(
            f"""
            name: test_pipeline
            input:
              path: input.jsonl
            output:
              dir: out
              complex_queries: complex_queries.jsonl
              summary: summary.json
            work_dir: work
            session_stage:
              enabled: true
              segmentation:
                enabled: true
              step1:
                enabled: true
                reject_rules: true
                min_chain_tool_calls: 7
                min_chain_steps: 1
                min_unique_tools: 2
              step2:
                enabled: true
                prompt_id: complex_judge
            {post_block}
            llm_stage:
              enabled: true
              base_url_env: OPENAI_BASE_URL
              model: fake-model
              api_key_env: FAKE_API_KEY
              concurrency: 2
              max_retries: 1
              timeout_seconds: 1
              response_format: json_object
              cache: work/llm_cache.jsonl
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _checkpoint_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") != "meta":
            keys.add(row["key"])
    return keys


if __name__ == "__main__":
    unittest.main()
