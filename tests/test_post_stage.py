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
from query_pipeline.config.models import DedupConfig, LLMConfig
from query_pipeline.io.jsonl import read_jsonl, write_jsonl
from query_pipeline.llm.cache import load_cache
from query_pipeline.llm.client import LLMClient
from query_pipeline.pipeline.runner import run_pipeline
from query_pipeline.post.dedup import dedup_rows
from query_pipeline.post.translate import needs_translation, translate_rows
from query_pipeline.rules.normalize import exact_token_jaccard, tokenize_question

_BASE = "how does the fed rate hike affect bond prices this year and what should investors watch for"
_NEAR = _BASE.replace("affect", "impact")
_DISTINCT = "compare vanguard and fidelity etf expense ratios"


def _row(text: str, trace_id: str) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "source_case_id": "t1",
        "input": {"text": text, "image": "", "file": ""},
        "meta": {"reason": "r"},
    }


def _llm_cfg(tmp: str) -> LLMConfig:
    return LLMConfig(model="fake-model", cache=Path(tmp) / "cache.jsonl", concurrency=2)


class FakeLLMClient(LLMClient):
    """Unit-test client whose handler may raise to simulate LLM failure."""

    def __init__(self, handler: Any) -> None:
        self._handler = handler

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        return await self._handler(system_prompt, user_prompt)

    async def close(self) -> None:
        return None


class DedupTest(unittest.TestCase):
    def test_exact_and_near_duplicates_dropped(self) -> None:
        rows = [
            _row(_BASE, "r1"),
            _row(_BASE, "r2"),  # exact duplicate
            _row(_NEAR, "r3"),  # near duplicate (word swap)
            _row(_DISTINCT, "r4"),
        ]
        kept, dropped = dedup_rows(rows, DedupConfig())

        self.assertEqual([r["trace_id"] for r in kept], ["r1", "r4"])
        self.assertEqual([d["trace_id"] for d in dropped], ["r2", "r3"])
        self.assertEqual(dropped[0]["dedup_of_trace_id"], "r1")
        self.assertEqual(dropped[1]["dedup_of_trace_id"], "r1")
        self.assertEqual(dropped[0]["similarity"], 1.0)
        self.assertGreaterEqual(dropped[1]["similarity"], 0.8)
        self.assertEqual(dropped[0]["method"], "template_merge")  # identical skeleton group

    def test_near_duplicate_survives_high_threshold(self) -> None:
        kept, dropped = dedup_rows([_row(_BASE, "r1"), _row(_NEAR, "r2")], DedupConfig(threshold=0.99))
        self.assertEqual([r["trace_id"] for r in kept], ["r1", "r2"])
        self.assertEqual(dropped, [])

    def test_distinct_queries_kept(self) -> None:
        rows = [_row(_DISTINCT, "r1"), _row(_BASE, "r2")]
        kept, dropped = dedup_rows(rows, DedupConfig())
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])

    def test_empty_text_never_dropped(self) -> None:
        rows = [_row("", "r1"), _row("", "r2"), _row("   ", "r3")]
        kept, dropped = dedup_rows(rows, DedupConfig())
        self.assertEqual(len(kept), 3)
        self.assertEqual(dropped, [])

    def test_tokenization_deterministic(self) -> None:
        a = tokenize_question(_BASE)
        b = tokenize_question(_BASE)
        self.assertEqual(a, b)
        self.assertEqual(exact_token_jaccard(a, b), 1.0)

    def test_threshold_range_validated(self) -> None:
        with self.assertRaises(ValueError):
            DedupConfig(threshold=1.5)
        with self.assertRaises(ValueError):
            DedupConfig(threshold=-0.1)

    def test_near_duplicate_similarity_above_default_threshold(self) -> None:
        sim = exact_token_jaccard(tokenize_question(_BASE), tokenize_question(_NEAR))
        self.assertGreaterEqual(sim, 0.8)

    def test_template_variants_merged(self) -> None:
        rows = [
            _row("Forecast $EYE for the next 1 day, 1 week, and 1 month — bull / base / bear scenarios", "r1"),
            _row("Forecast $AA for the next 1 day, 1 week, and 1 month — bull / base / bear scenarios", "r2"),
            _row("Forecast $NVDA for the next 1 day, 1 week, and 1 month — bull / base / bear scenarios", "r3"),
        ]
        kept, dropped = dedup_rows(rows, DedupConfig())
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 2)
        self.assertEqual({d["dedup_of_trace_id"] for d in dropped}, {kept[0]["trace_id"]})
        self.assertEqual(dropped[0]["similarity"], 1.0)

    def test_rephrase_merged(self) -> None:
        a = "Check NEARUSDT for the rest of today on 4hr time frame in futures market and recommend as to entry/exit points"
        b = "How is NEARUSDT going to perform for the rest of today on 4hr time frame in futures market and recommend as to entry/exit points"
        kept, dropped = dedup_rows([_row(a, "r1"), _row(b, "r2")], DedupConfig())
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 1)
        # 代表选更长更完整的改写版本(r2),被删的指向它。
        self.assertEqual(kept[0]["trace_id"], "r2")
        self.assertEqual(dropped[0]["dedup_of_trace_id"], "r2")
        self.assertGreaterEqual(dropped[0]["similarity"], 0.8)

    def test_distinct_intent_kept(self) -> None:
        rows = [
            _row("Forecast $AAPL for the next 1 day, 1 week, and 1 month — bull / base / bear scenarios with confidence levels", "r1"),
            _row("Forecast NVDA trend this week", "r2"),
        ]
        kept, dropped = dedup_rows(rows, DedupConfig())
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])

    def test_content_word_not_slotted(self) -> None:
        a = "The price is stalled 1955 now, Do again fundamental analysis [strong/weak], do sentiment/catalyst/outlook analysis [positive/negative] and Do chart analysis on indonesian stock IDX:WIFI chart and give"
        b = "The price is rallying now, Do again fundamental analysis [strong/weak], do sentiment/catalyst/outlook analysis [positive/negative] and Do chart analysis on indonesian stock IDX:CBDK chart and give me "
        # 真实语料中该对 J=0.786<0.8:stalled/rallying 是内容词,不被槽化,保持分开。
        ta = tokenize_question(a)
        tb = tokenize_question(b)
        self.assertIn("stalled", ta)
        self.assertIn("rallying", tb)
        self.assertNotEqual(ta, tb)
        kept, dropped = dedup_rows([_row(a, "r1"), _row(b, "r2")], DedupConfig())
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])

    def test_entity_slotting_switch(self) -> None:
        a = "What is the P/E of $AAPL at 100"
        b = "What is the P/E of $MSFT at 200"
        rows = [_row(a, "r1"), _row(b, "r2")]
        kept, dropped = dedup_rows(rows, DedupConfig(entity_slot=True))
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 1)
        kept2, dropped2 = dedup_rows(rows, DedupConfig(entity_slot=False))
        self.assertEqual(len(kept2), 2)
        self.assertEqual(dropped2, [])

    def test_number_only_difference_merged(self) -> None:
        rows = [
            _row("The price is stalled 1955 now, do again fundamental analysis on this stock", "r1"),
            _row("The price is stalled 875 now, do again fundamental analysis on this stock", "r2"),
        ]
        kept, dropped = dedup_rows(rows, DedupConfig())
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped[0]["similarity"], 1.0)

    def test_representative_selection_and_determinism(self) -> None:
        rows = [
            _row("Forecast  SNDK for the next 1 day, — bull / base / bear scenarios with confidence levels and what would trigger each.", "r1"),
            _row("Forecast $AAPL for the next 1 day, 1 week, and 1 month — bull / base / bear scenarios with confidence levels and what would trigger each.", "r2"),
            _row("Forecast $MSFT for the next 1 day, 1 week, and 1 month — bull / base / bear scenarios with confidence levels and what would trigger each.", "r3"),
        ]
        kept, dropped = dedup_rows(rows, DedupConfig())
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["trace_id"], "r2")
        self.assertEqual({d["dedup_of_trace_id"] for d in dropped}, {"r2"})
        kept2, dropped2 = dedup_rows(rows, DedupConfig())
        self.assertEqual([r["trace_id"] for r in kept2], [r["trace_id"] for r in kept])
        self.assertEqual([d["trace_id"] for d in dropped2], [d["trace_id"] for d in dropped])

    def test_cjk_near_duplicate_merged(self) -> None:
        rows = [
            _row("帮我分析一下nvidia的估值并给出建议", "r1"),
            _row("帮我分析一下amd的估值并给出建议", "r2"),
        ]
        kept, dropped = dedup_rows(rows, DedupConfig())
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 1)

    def test_pure_slot_query_never_dropped(self) -> None:
        rows = [_row("1234", "r1"), _row("5678", "r2")]
        kept, dropped = dedup_rows(rows, DedupConfig())
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])

    def test_chained_pair_merged_transitively(self) -> None:
        # A~B (8/9) and B~C (0.8) both >= threshold; A~C (0.7) below it.
        # Transitive union-find merges all three into one cluster, so the directly
        # >= threshold pair B~C does NOT both survive. One representative stays
        # (longest text: C, 19 chars vs B's 18); the other two are dropped.
        a = "w1 w2 w3 w4 w5 w6 w7 w8"
        b = "w1 w2 w3 w4 w5 w6 w7 w8 w9"
        c = "w1 w2 w3 w4 w5 w6 w7 w9 w10"
        rows = [_row(a, "rA"), _row(b, "rB"), _row(c, "rC")]
        kept, dropped = dedup_rows(rows, DedupConfig())

        self.assertEqual([r["trace_id"] for r in kept], ["rC"])
        self.assertEqual([d["trace_id"] for d in dropped], ["rA", "rB"])
        self.assertEqual({d["dedup_of_trace_id"] for d in dropped}, {"rC"})
        # dropped[1] is B, a direct >= threshold near-dup of representative C.
        self.assertGreaterEqual(dropped[1]["similarity"], 0.8)


class TranslateTest(unittest.TestCase):
    def test_adds_translation_to_row(self) -> None:
        rows = [_row("how to hedge against inflation", "r1")]

        async def handler(system_prompt: str, user_prompt: str) -> str:
            del system_prompt, user_prompt
            return json.dumps({"translation": "如何对冲通胀"}, ensure_ascii=False)

        with tempfile.TemporaryDirectory() as tmp:
            counters = asyncio.run(
                translate_rows(
                    rows,
                    client=FakeLLMClient(handler),
                    llm_cfg=_llm_cfg(tmp),
                    cache={},
                    cache_path=Path(tmp) / "cache.jsonl",
                )
            )

        self.assertEqual(rows[0]["translation"], "如何对冲通胀")
        self.assertEqual(counters, {"translated": 1, "translate_skipped": 0, "translate_failed": 0})

    def test_adds_translation_with_null_meta(self) -> None:
        rows = [
            {
                "trace_id": "r1",
                "source_case_id": "t1",
                "input": {"text": "how to hedge against inflation", "image": "", "file": ""},
                "meta": None,  # input rows may carry meta: null
            }
        ]

        async def handler(system_prompt: str, user_prompt: str) -> str:
            del system_prompt, user_prompt
            return json.dumps({"translation": "如何对冲通胀"}, ensure_ascii=False)

        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(
                translate_rows(
                    rows,
                    client=FakeLLMClient(handler),
                    llm_cfg=_llm_cfg(tmp),
                    cache={},
                    cache_path=Path(tmp) / "cache.jsonl",
                )
            )

        self.assertEqual(rows[0]["translation"], "如何对冲通胀")
        self.assertIsNone(rows[0]["meta"])  # 不再触碰 meta

    def test_chinese_text_skipped_no_llm_call(self) -> None:
        rows = [_row("如何对冲通胀风险", "r1")]

        async def handler(system_prompt: str, user_prompt: str) -> str:  # pragma: no cover - must not be called
            del system_prompt, user_prompt
            raise AssertionError("LLM must not be called for Chinese text")

        with tempfile.TemporaryDirectory() as tmp:
            counters = asyncio.run(
                translate_rows(
                    rows,
                    client=FakeLLMClient(handler),
                    llm_cfg=_llm_cfg(tmp),
                    cache={},
                    cache_path=Path(tmp) / "cache.jsonl",
                )
            )

        self.assertIsNone(rows[0]["translation"])  # 中文原文 → null
        self.assertEqual(counters, {"translated": 0, "translate_skipped": 1, "translate_failed": 0})

    def test_failure_leaves_null(self) -> None:
        rows = [_row("how to hedge against inflation", "r1")]

        async def handler(system_prompt: str, user_prompt: str) -> str:
            del system_prompt, user_prompt
            raise RuntimeError("simulated LLM failure")

        with tempfile.TemporaryDirectory() as tmp:
            counters = asyncio.run(
                translate_rows(
                    rows,
                    client=FakeLLMClient(handler),
                    llm_cfg=_llm_cfg(tmp),
                    cache={},
                    cache_path=Path(tmp) / "cache.jsonl",
                )
            )

        self.assertIsNone(rows[0]["translation"])  # 翻译失败 → null（fail-open 保留行）
        self.assertEqual(counters, {"translated": 0, "translate_skipped": 0, "translate_failed": 1})

    def test_cache_round_trip_reuses_translation(self) -> None:
        rows = [_row("how to hedge against inflation", "r1")]

        async def handler(system_prompt: str, user_prompt: str) -> str:
            del system_prompt, user_prompt
            return json.dumps({"translation": "如何对冲通胀"}, ensure_ascii=False)

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.jsonl"
            cache: dict[str, Any] = {}
            asyncio.run(
                translate_rows(
                    rows,
                    client=FakeLLMClient(handler),
                    llm_cfg=_llm_cfg(tmp),
                    cache=cache,
                    cache_path=cache_path,
                )
            )
            # Second run over a fresh row: the cache file must serve the
            # translation without any LLM call.
            rows2 = [_row("how to hedge against inflation", "r2")]

            async def fail_if_called(system_prompt: str, user_prompt: str) -> str:
                del system_prompt, user_prompt
                raise AssertionError("cached translation must be reused, not re-requested")

            counters2 = asyncio.run(
                translate_rows(
                    rows2,
                    client=FakeLLMClient(fail_if_called),
                    llm_cfg=_llm_cfg(tmp),
                    cache=load_cache(cache_path),
                    cache_path=cache_path,
                )
            )

        self.assertEqual(rows2[0]["translation"], "如何对冲通胀")
        self.assertEqual(counters2, {"translated": 1, "translate_skipped": 0, "translate_failed": 0})

    def test_needs_translation_heuristics(self) -> None:
        self.assertTrue(needs_translation("what is the pe ratio of tesla"))
        self.assertFalse(needs_translation("特斯拉的市盈率是多少"))
        self.assertFalse(needs_translation(""))
        self.assertFalse(needs_translation("   "))
        self.assertFalse(needs_translation("美股最近怎么样，Tesla 涨了吗"))


class PostStagePipelineTest(unittest.TestCase):
    def test_config_defaults_disabled(self) -> None:
        from query_pipeline.config.models import PostConfig

        post = PostConfig()
        self.assertFalse(post.enabled)
        self.assertTrue(post.dedup.enabled)
        self.assertEqual(post.dedup.threshold, 0.80)
        self.assertTrue(post.translate.enabled)

    def test_config_yaml_enables_post_stage(self) -> None:
        cfg = load_pipeline_config(ROOT / "configs/aime/config.yaml")
        self.assertTrue(cfg.post.enabled)
        self.assertEqual(cfg.post.dedup.threshold, 0.80)
        self.assertTrue(cfg.post.translate.enabled)

    def test_end_to_end_dedup_and_translate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            turns = _turns("s1")
            write_jsonl(tmp_path / "input.jsonl", [
                {"thread_id": "t1", "context": turns},
                {"thread_id": "t2", "context": _turns("s2")},
            ])
            cfg = load_pipeline_config(_write_config(tmp_path))

            with patch("query_pipeline.pipeline.runner.LLMClient", FakePipelineLLMClient):
                summary = run_pipeline(cfg)

            self.assertEqual(summary.stats["complex_rows"], 2)  # judged hard rows
            self.assertEqual(summary.stats["output_rows"], 1)  # post-dedup
            self.assertEqual(summary.stats["verify_kept"], 2)
            self.assertEqual(summary.stats["dedup_removed"], 1)
            self.assertEqual(summary.stats["translated"], 1)
            self.assertEqual(summary.stats["translate_skipped"], 0)
            self.assertEqual(summary.stats["translate_failed"], 0)

            rows = list(read_jsonl(tmp_path / "out" / "complex_queries.jsonl"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_case_id"], "t1")  # first occurrence kept
            self.assertEqual(rows[0]["translation"], "利率上升如何影响债券价格？")
            self.assertEqual(rows[0]["meta"]["reason"], "需要预测")

            deduped = list(read_jsonl(tmp_path / "work" / "logs" / "deduped.jsonl"))
            self.assertEqual(len(deduped), 1)
            self.assertEqual(deduped[0]["trace_id"], "s2trace1")
            self.assertEqual(deduped[0]["dedup_of_trace_id"], "s1trace1")
            self.assertEqual(deduped[0]["similarity"], 1.0)

    def test_post_stage_disabled_no_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            write_jsonl(tmp_path / "input.jsonl", [{"thread_id": "t1", "context": _turns("s1")}])
            cfg = load_pipeline_config(_write_config(tmp_path, post_enabled=False))

            with patch("query_pipeline.pipeline.runner.LLMClient", FakePipelineLLMClient):
                summary = run_pipeline(cfg)

            self.assertEqual(summary.stats["complex_rows"], 1)
            self.assertEqual(summary.stats["dedup_removed"], 0)  # post disabled
            rows = list(read_jsonl(tmp_path / "out" / "complex_queries.jsonl"))
            self.assertEqual(
                rows[0]["meta"], {"reason": "需要预测", "request_time": "2026-08-05 04:01:00", "run_id": "s1r1", "last_event_type": None}
            )  # post stage 不再触碰 meta
            self.assertIsNone(rows[0]["translation"])  # 中文原文 → null（post 关闭时亦然）


def _turns(prefix: str) -> list[dict[str, Any]]:
    names = ("web_search", "finquery", "compute")
    return [
        _make_turn(prefix, 0, "hello", tool_names="web_search", tool_count=1, chain=_chain_with_tool_calls(1)),
        _make_turn(
            prefix,
            1,
            "How do rising interest rates affect bond prices?",
            tool_names="web_search,finquery,compute",
            tool_count=8,
            chain=_chain_with_steps(8, names),
        ),
        _make_turn(prefix, 2, "thanks", tool_names="", tool_count=0),
    ]


def _make_turn(
    prefix: str,
    idx: int,
    question: str,
    *,
    tool_names: str = "",
    tool_count: int = 0,
    chain: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "question": question,
        "answer": f"answer{idx} " + "x" * 60,
        "run_id": f"{prefix}r{idx}",
        "trace_id": f"{prefix}trace{idx}",
        "request_time": f"2026-08-05 04:{idx:02d}:00",
        "user_id": f"{prefix}u{idx}",
        "status": "completed",
        "outcome": "success",
        "tool_names": tool_names,
        "tool_count": tool_count,
        "first_token_ms": idx * 100,
        "total_duration_ms": idx * 100 + 200,
        "chain": chain if chain is not None else [],
    }


def _chain_with_tool_calls(n: int, name: str = "t") -> list[dict[str, Any]]:
    return [{"plan": "", "tools": [{"name": name, "input": {}, "output": "x"} for _ in range(n)]}]


def _chain_with_steps(n: int, names: tuple[str, ...]) -> list[dict[str, Any]]:
    return [{"plan": "", "tools": [{"name": names[i % len(names)], "input": {}, "output": "x"}]} for i in range(n)]


class FakePipelineLLMClient:
    """Handles segment / judge / verify / translate payloads for the e2e test."""

    COMPLEX = "How do rising interest rates affect bond prices?"

    def __init__(self, config: object) -> None:
        self.config = config

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        assert system_prompt
        payload = json.loads(user_prompt.split("\n", 1)[1])
        if "questions" in payload:
            n = len(payload["questions"])
            return json.dumps({"segments": [{"start": 0, "end": n - 1, "topic": "topic"}]}, ensure_ascii=False)
        if "价值判官" in system_prompt:
            return json.dumps({"is_valuable": True, "reason": "金融相关"}, ensure_ascii=False)
        if "已判定为复杂金融问句" in system_prompt:
            return json.dumps({"category_id": "02", "reason": "需要预测"}, ensure_ascii=False)
        if "有价值但非复杂" in system_prompt:
            return json.dumps({"category_id": "03", "reason": "普通归类"}, ensure_ascii=False)
        if "current_question" in payload:  # complexity gate
            if payload["current_question"] == self.COMPLEX:
                return json.dumps({"is_complex": True, "reason": "需要预测"}, ensure_ascii=False)
            return json.dumps({"is_complex": False, "reason": "简单查询"}, ensure_ascii=False)
        if "question" in payload:  # pass-2 verify (standalone)
            if payload["question"] == self.COMPLEX:
                return json.dumps({"is_complex": True, "reason": "独立成立"}, ensure_ascii=False)
            return json.dumps({"is_complex": False, "reason": "独立不成立"}, ensure_ascii=False)
        return json.dumps({"translation": "利率上升如何影响债券价格？"}, ensure_ascii=False)

    async def close(self) -> None:
        return None


def _write_config(tmp_path: Path, *, post_enabled: bool = True) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            name: test_pipeline
            input:
              path: input.jsonl
              format: session
            output:
              dir: out
              complex_queries: complex_queries.jsonl
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
            post:
              enabled: {str(post_enabled).lower()}
              dedup:
                enabled: true
                threshold: 0.80
                entity_slot: true
              translate:
                enabled: true
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


if __name__ == "__main__":
    unittest.main()
