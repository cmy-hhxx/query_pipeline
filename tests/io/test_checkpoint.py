from __future__ import annotations

import asyncio
import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from query_pipeline.config.loader import load_pipeline_config
from query_pipeline.io.checkpoint import Checkpoint, _src_hash, content_key, stage_fingerprint
from query_pipeline.pipeline.runner import run_pipeline

def _make_turn(idx: int, question: str) -> dict[str, Any]:
    if idx % 2 == 0:
        # simple turn: too few tool calls to be a candidate
        return {
            "question": question,
            "answer": f"answer{idx} " + "x" * 60,
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
        "answer": f"answer{idx} " + "x" * 60,
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
    current_question / standalone question / text is in the matching fail set."""

    def __init__(
        self,
        *,
        session_fail: set[str] | None = None,
        verify_fail: set[str] | None = None,
        translate_fail: set[str] | None = None,
    ) -> None:
        self.session_fail = set(session_fail or ())
        self.verify_fail = set(verify_fail or ())
        self.translate_fail = set(translate_fail or ())
        self.calls: list[dict[str, Any]] = []
        self.config: object | None = None

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        assert system_prompt
        payload = json.loads(user_prompt.split("\n", 1)[1])
        self.calls.append(payload)
        if "questions" in payload:  # segmentation
            n = len(payload["questions"])
            return json.dumps({"segments": [{"start": 0, "end": n - 1, "topic": "t"}]}, ensure_ascii=False)
        if "价值判官" in system_prompt:  # value gate
            return json.dumps({"is_valuable": True, "reason": "金融相关"}, ensure_ascii=False)
        if "已判定为复杂金融问句" in system_prompt:  # classify complex
            return json.dumps({"category_id": "03", "reason": "复杂归类"}, ensure_ascii=False)
        if "有价值但非复杂" in system_prompt:  # classify normal
            return json.dumps({"category_id": "03", "reason": "普通归类"}, ensure_ascii=False)
        if "current_question" in payload:  # complexity gate
            if payload["current_question"] in self.session_fail:
                raise RuntimeError("simulated network failure")
            return json.dumps({"is_complex": True, "reason": "复杂"}, ensure_ascii=False)
        if "简单问句识别器" in system_prompt:  # simple_finder 视角
            return json.dumps({"is_simple": False, "reason": "不是简单问句"}, ensure_ascii=False)
        if "question" in payload:  # verify
            if payload["question"] in self.verify_fail:
                raise RuntimeError("simulated network failure")
            return json.dumps({"is_complex": True, "reason": "自身复杂"}, ensure_ascii=False)
        if "text" in payload:  # translate
            if payload["text"] in self.translate_fail:
                raise RuntimeError("simulated network failure")
            return json.dumps({"translation": "翻译：" + payload["text"]}, ensure_ascii=False)
        raise AssertionError(f"unexpected payload: {sorted(payload)}")

    async def close(self) -> None:
        return None

def _factory(
    clients: list[ScriptedClient],
    *,
    session_fail: set[str] | None = None,
    verify_fail: set[str] | None = None,
    translate_fail: set[str] | None = None,
) -> Any:
    def factory(config: object) -> ScriptedClient:
        client = ScriptedClient(
            session_fail=session_fail, verify_fail=verify_fail, translate_fail=translate_fail
        )
        client.config = config
        clients.append(client)
        return client

    return factory

def run_pipeline_with_fakes(
    cfg: Any,
    *,
    session_fail: set[str] | None = None,
    verify_fail: set[str] | None = None,
    translate_fail: set[str] | None = None,
) -> tuple[Any, ScriptedClient]:
    """Shared LLMClient for the whole run; fail sets are stage-specific."""
    clients: list[ScriptedClient] = []
    with patch(
        "query_pipeline.pipeline.runner.LLMClient",
        _factory(clients, session_fail=session_fail, verify_fail=verify_fail, translate_fail=translate_fail),
    ):
        summary = run_pipeline(cfg)
    return summary, clients[0] if clients else ScriptedClient()

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
            summary1, _ = run_pipeline_with_fakes(cfg, session_fail={"S1 complex query"})
            self.assertEqual(summary1.stats["llm_failed"], 1)
            self.assertEqual(summary1.stats["complex_rows"], 2)
            cp_path = tmp_path / "work" / "logs" / "checkpoints" / "judge.jsonl"
            self.assertEqual(len(_checkpoint_keys(cp_path)), 2)

            # Run 2: only t1's judge re-runs; t0/t2 replay from checkpoint.
            summary2, client2 = run_pipeline_with_fakes(cfg)
            self.assertEqual(summary2.stats["llm_failed"], 0)
            self.assertEqual(summary2.stats["complex_rows"], 3)
            self.assertEqual(summary2.stats["verify_kept"], 3)
            # t1 re-runs through the funnel: complexity + classify calls
            # (value_gate replays from the LLM cache written in run 1).
            self.assertEqual(
                [c["current_question"] for c in client2.calls if "current_question" in c],
                ["S1 complex query", "S1 complex query"],
            )
            self.assertEqual(
                [c["question"] for c in client2.calls if "question" in c and "current_question" not in c and "questions" not in c and "text" not in c],
                ["S1 complex query"],
            )
            self.assertEqual(len(_checkpoint_keys(cp_path)), 3)

            rows = _read_jsonl(tmp_path / "out/cleaned_queries.jsonl")
            self.assertEqual([r["input"]["text"] for r in rows], ["S0 complex query", "S1 complex query", "S2 complex query"])

class VerifyResumeTest(unittest.TestCase):
    def test_verify_stage_resumes_after_network_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sessions = [_session("t0", "V0"), _session("t1", "V1"), _session("t2", "V2")]
            _write_jsonl(tmp_path / "input.jsonl", sessions)
            cfg = load_pipeline_config(_write_config(tmp_path, post_enabled=False))

            # Run 1: verify for V1 fails (fail-open keeps the row and checkpoints
            # the failure); V0/V2 are checkpointed cleanly.
            summary1, _ = run_pipeline_with_fakes(cfg, verify_fail={"V1 complex query"})
            self.assertEqual(summary1.stats["verify_kept"], 2)
            self.assertEqual(summary1.stats["verify_failed"], 1)
            self.assertEqual(summary1.stats["complex_rows"], 3)

            # Run 2: all three replay from checkpoints; V1's recorded failure
            # replays too (fail-open is sticky), so zero re-verify LLM calls and
            # the same output.
            summary2, client2 = run_pipeline_with_fakes(cfg)
            self.assertEqual(summary2.stats["verify_kept"], 2)
            self.assertEqual(summary2.stats["verify_failed"], 1)
            self.assertEqual(summary2.stats["complex_rows"], 3)
            self.assertEqual(
                [c["question"] for c in client2.calls if "question" in c and "current_question" not in c and "questions" not in c and "text" not in c],
                [],
            )

class TranslateResumeTest(unittest.TestCase):
    def test_translate_stage_resumes_after_network_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sessions = [_session("t0", "T0"), _session("t1", "T1"), _session("t2", "T2")]
            _write_jsonl(tmp_path / "input.jsonl", sessions)
            cfg = load_pipeline_config(_write_config(tmp_path, post_enabled=True))

            # Run 1: translation of T1 fails (leaves translation null) and is
            # not checkpointed; T0/T2 are.
            summary1, _ = run_pipeline_with_fakes(cfg, translate_fail={"T1 complex query"})
            self.assertEqual(summary1.stats["translated"], 2)
            self.assertEqual(summary1.stats["translate_failed"], 1)
            self.assertEqual(summary1.stats["complex_rows"], 3)

            # Run 2: T0/T2 reuse checkpointed translations; only T1 re-translates.
            summary2, client2 = run_pipeline_with_fakes(cfg)
            self.assertEqual(summary2.stats["translated"], 3)
            self.assertEqual(summary2.stats["translate_failed"], 0)
            self.assertEqual([c["text"] for c in client2.calls if "text" in c], ["T1 complex query"])

            rows = _read_jsonl(tmp_path / "out/cleaned_queries.jsonl")
            self.assertEqual(rows[1]["translation"], "翻译：T1 complex query")

class CheckpointInvalidationTest(unittest.TestCase):
    def test_input_change_invalidates_session_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sessions = [_session("t0", "W0"), _session("t1", "W1")]
            _write_jsonl(tmp_path / "input.jsonl", sessions)
            cfg = load_pipeline_config(_write_config(tmp_path, post_enabled=False))
            run_pipeline_with_fakes(cfg)

            # Rewrite the input with a new question; discover checkpoint meta
            # (input size/mtime) no longer matches and is re-seeded.
            sessions[0]["context"][1]["question"] = "W0 complex query NEW"
            _write_jsonl(tmp_path / "input.jsonl", sessions)
            summary, client2 = run_pipeline_with_fakes(cfg)

            self.assertEqual(summary.stats["complex_rows"], 2)
            # W0 re-runs through the full funnel (value+complexity+classify all
            # miss the cache because the question text changed).
            self.assertEqual(
                [c["current_question"] for c in client2.calls if "current_question" in c],
                ["W0 complex query NEW", "W0 complex query NEW", "W0 complex query NEW"],
            )
            self.assertEqual(
                [c["question"] for c in client2.calls if "question" in c and "current_question" not in c and "questions" not in c and "text" not in c],
                ["W0 complex query NEW"],
            )

            rows = _read_jsonl(tmp_path / "out/cleaned_queries.jsonl")
            texts = [r["input"]["text"] for r in rows]
            self.assertIn("W0 complex query NEW", texts)
            self.assertNotIn("W0 complex query", texts)

    def test_stage_meta_attaches_input_stat_for_verify_and_translate(self) -> None:
        from query_pipeline.io.checkpoint import stage_meta

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_jsonl(tmp_path / "input.jsonl", [_session("t0", "S0")])
            cfg = load_pipeline_config(_write_config(tmp_path, post_enabled=False))
            for stage in ("judge", "verify", "translate"):
                meta = stage_meta(cfg, stage)
                self.assertIn("input_size", meta, stage)
                self.assertIn("input_mtime_ns", meta, stage)
                self.assertIn("input_path", meta, stage)

    def test_input_change_reseeds_verify_checkpoint(self) -> None:
        # 输入文件变化 → verify checkpoint 必须整体重播种（README：输入变化自动失效），
        # 不得重放旧前文/旧难度下的裁决。用全新 LLM cache 隔离缓存命中干扰。
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sessions = [_session("t0", "R0")]
            _write_jsonl(tmp_path / "input.jsonl", sessions)
            cfg = load_pipeline_config(_write_config(tmp_path, post_enabled=False))

            summary1, _ = run_pipeline_with_fakes(cfg)
            self.assertEqual(summary1.stats["verify_kept"], 1)

            # 输入追加一个会话（其余不变）+ 全新 cache：verify 仍须重新验证旧会话
            _write_jsonl(tmp_path / "input.jsonl", sessions + [_session("t1", "R1")])
            cfg2 = load_pipeline_config(_write_config(tmp_path, post_enabled=False))
            cfg2.llm.cache = tmp_path / "work" / "llm_cache2.jsonl"
            summary2, client2 = run_pipeline_with_fakes(cfg2)
            self.assertEqual(summary2.stats["verify_kept"], 2)
            verify_calls = [
                c["question"]
                for c in client2.calls
                if "question" in c and "current_question" not in c and "questions" not in c and "text" not in c
            ]
            self.assertIn("R0 complex query", verify_calls)  # 旧行重新验证而非重放

    def test_verify_key_includes_difficulty_no_stale_replay(self) -> None:
        # judge 指纹单独失效（只改 complexity_gate prompt 内容）→ 同一问句 hard→normal；
        # verify checkpoint 不重播种（输入/config/源码/verify prompt 均未变），
        # 必须靠"键含难度"避免重放旧 hard 裁决。
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sessions = [_session("t0", "D0")]
            _write_jsonl(tmp_path / "input.jsonl", sessions)
            cfg = load_pipeline_config(_write_config(tmp_path, post_enabled=False))

            def factory(clients: list, complex_qs: set[str]) -> Any:
                def make(config: object) -> ScriptedClient:
                    client = DifficultyFlipClient(config, complex_qs=complex_qs)
                    clients.append(client)
                    return client
                return make

            clients: list[Any] = []
            with patch("query_pipeline.pipeline.runner.LLMClient", factory(clients, {"D0 complex query"})):
                summary1 = run_pipeline(cfg)
            self.assertEqual(summary1.stats["complex_rows"], 1)
            self.assertEqual(summary1.stats["verify_kept"], 1)

            with patch("query_pipeline.pipeline.runner.LLMClient", factory(clients, set())), patch.dict(
                "query_pipeline.prompts.PROMPTS", {"complexity_gate": "complexity_gate 已更新（内容变化）"}
            ):
                summary2 = run_pipeline(cfg)

            # judge 重跑：复杂度门改为非复杂 → normal 行
            self.assertEqual(summary2.stats["normal_rows"], 1)
            self.assertEqual(summary2.stats["complex_rows"], 0)
            # verify 不得重放 hard 的 keep：normal 期望=非复杂，按缓存 round1 判定应拒绝
            self.assertEqual(summary2.stats["verify_rejected"], 1)
            self.assertEqual(summary2.stats["verify_kept"], 0)
            # verify checkpoint 出现第二个键（normal 难度）
            cp = _read_jsonl(tmp_path / "work" / "logs" / "checkpoints" / "verify.jsonl")
            keys = {r["key"] for r in cp if r.get("type") != "meta"}
            self.assertEqual(len(keys), 2)

    def test_fingerprint_tracks_source(self) -> None:
        # Code changes must invalidate the checkpoint fingerprint (else a behavior fix
        # silently never takes effect on resume).
        self.assertEqual(_src_hash(), _src_hash())  # stable across calls
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_pipeline_config(_write_config(Path(tmp), post_enabled=False))
            with patch("query_pipeline.io.checkpoint._src_hash", return_value="aaaa"):
                fp_a = stage_fingerprint(cfg, "judge")
            with patch("query_pipeline.io.checkpoint._src_hash", return_value="bbbb"):
                fp_b = stage_fingerprint(cfg, "judge")
        self.assertNotEqual(fp_a, fp_b)

def _write_config(tmp_path: Path, *, post_enabled: bool) -> Path:
    config_path = tmp_path / "config.yaml"
    post_block = ""
    if post_enabled:
        post_block = """
            post:
              enabled: true
              dedup:
                enabled: true
                threshold: 0.80
              translate:
                enabled: true
"""
    config_path.write_text(
        textwrap.dedent(
            f"""
            name: test_pipeline
            input:
              path: input.jsonl
              format: session
            output:
              dir: out
              cleaned_queries: cleaned_queries.jsonl
              complex_queries: complex_queries.jsonl
              normal_queries: normal_queries.jsonl
              summary: summary.json
            work_dir: work
            segmentation:
              enabled: true
            rule_gate:
              enabled: true
              reject_rules: true
              min_chain_tool_calls: 7
              min_chain_steps: 1
              min_unique_tools: 2
            judge:
              enabled: true
            verify:
              enabled: true
              prompt_id: verify_complex
              max_rounds_hard: 1
              max_rounds_normal: 1
            {post_block}
            llm:
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

class DifficultyFlipClient(ScriptedClient):
    """ScriptedClient 的变体：复杂度门按 complex_qs 集合判定 hard/normal。

    value/classify/segment/verify 沿用 ScriptedClient 行为（verify 恒判复杂）。
    """

    def __init__(self, config: object, complex_qs: set[str]) -> None:
        super().__init__()
        self.config = config
        self.complex_qs = complex_qs

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt.split("\n", 1)[1])
        self.calls.append(payload)
        if "questions" in payload:
            n = len(payload["questions"])
            return json.dumps({"segments": [{"start": 0, "end": n - 1, "topic": "t"}]}, ensure_ascii=False)
        if "价值判官" in system_prompt:
            return json.dumps({"is_valuable": True, "reason": "金融相关"}, ensure_ascii=False)
        if "已判定为复杂金融问句" in system_prompt:
            return json.dumps({"category_id": "03", "reason": "复杂归类"}, ensure_ascii=False)
        if "有价值但非复杂" in system_prompt:
            return json.dumps({"category_id": "03", "reason": "普通归类"}, ensure_ascii=False)
        if "current_question" in payload:
            is_complex = payload["current_question"] in self.complex_qs
            return json.dumps({"is_complex": is_complex, "reason": "判定"}, ensure_ascii=False)
        if "question" in payload:
            return json.dumps({"is_complex": True, "reason": "自身复杂"}, ensure_ascii=False)
        raise AssertionError(f"unexpected payload: {sorted(payload)}")
