from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from query_pipeline.config.models import DedupConfig, LLMConfig
from query_pipeline.post.dedup import semantic_dedup_rows


def _signature(goal: str = "backtest strategy") -> dict[str, Any]:
    return {
        "goal": goal,
        "subject_type": "stock strategy",
        "operations": ["backtest", "compare benchmark"],
        "data_dimensions": ["return", "drawdown"],
        "temporal_shape": "historical periods",
        "output_shape": ["metrics", "conclusion"],
    }


def _row(
    text: str,
    trace_id: str,
    *,
    signature: dict[str, Any] | None = None,
    template_severity: str = "none",
) -> dict[str, Any]:
    profile = {
        "question_quality": "high",
        "semantic_signature": signature or _signature(),
    }
    return {
        "trace_id": trace_id,
        "source_case_id": "case",
        "input": {"text": text},
        "meta": {
            "reason": "r",
            "complexity_profile": profile,
            "semantic_signature": profile["semantic_signature"],
            "value_profile": {"template_severity": template_severity},
        },
    }


class PairClient:
    def __init__(self, labels: dict[frozenset[str], str] | None = None, *, fail: bool = False) -> None:
        self.config = SimpleNamespace(model="fake")
        self.labels = labels or {}
        self.fail = fail
        self.calls = 0

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        if self.fail:
            raise RuntimeError("pair judge unavailable")
        payload = json.loads(user_prompt.split("\n", 1)[1])
        items = []
        for pair in payload["pairs"]:
            questions = frozenset((pair["left"]["question"], pair["right"]["question"]))
            label = self.labels.get(questions, "template_duplicate")
            items.append({"id": pair["id"], "label": label, "reason": f"gold:{label}"})
        return json.dumps({"items": items}, ensure_ascii=False)


def _run(rows: list[dict[str, Any]], client: PairClient):
    with tempfile.TemporaryDirectory() as tmp:
        return asyncio.run(
            semantic_dedup_rows(
                rows,
                DedupConfig(mode="semantic", semantic_candidate_threshold=0.60),
                client=client,
                llm_cfg=LLMConfig(model="fake"),
                cache={},
                cache_path=Path(tmp) / "cache.jsonl",
                cache_lock=asyncio.Lock(),
            )
        )


class SemanticDedupTest(unittest.TestCase):
    def test_template_swap_merges_and_preserves_source_order_on_tie(self) -> None:
        rows = [
            _row("请详细回测贵州茅台的均线策略并输出结果", "long"),
            _row("回测宁德时代均线策略", "short"),
        ]
        kept, dropped, stats = _run(rows, PairClient())
        self.assertEqual([row["trace_id"] for row in kept], ["long"])
        self.assertEqual(dropped[0]["dedup_of_trace_id"], "long")
        self.assertEqual(dropped[0]["decision"], "template_duplicate")
        self.assertIn("direct_edge_similarity", dropped[0])
        self.assertIn("representative_similarity", dropped[0])
        self.assertEqual(stats["semantic_dedup_removed"], 1)

    def test_distinct_strategy_logic_is_kept(self) -> None:
        left = "回测均线突破策略"
        right = "回测MACD金叉策略"
        labels: dict[frozenset[str], str] = {frozenset((left, right)): "distinct"}
        kept, dropped, _ = _run([_row(left, "a"), _row(right, "b")], PairClient(labels))
        self.assertEqual([row["trace_id"] for row in kept], ["a", "b"])
        self.assertEqual(dropped, [])

    def test_representative_prefers_natural_question_over_template_scaffold(self) -> None:
        rows = [
            _row("回测均线策略", "natural"),
            _row("你是资深量化专家。第一步读取数据，第二步回测均线策略。", "template", template_severity="light"),
        ]
        kept, dropped, _ = _run(rows, PairClient())
        self.assertEqual([row["trace_id"] for row in kept], ["natural"])
        self.assertEqual(dropped[0]["dedup_of_trace_id"], "natural")

    def test_star_cluster_does_not_chain_merge(self) -> None:
        a, b, c = "回测A策略", "详细回测B策略", "非常详细地回测C策略"
        labels = {
            frozenset((a, b)): "template_duplicate",
            frozenset((b, c)): "template_duplicate",
            frozenset((a, c)): "distinct",
        }
        kept, dropped, _ = _run(
            [_row(a, "a"), _row(b, "b"), _row(c, "c")], PairClient(labels)
        )
        self.assertEqual([row["trace_id"] for row in kept], ["a", "c"])
        self.assertEqual([row["trace_id"] for row in dropped], ["b"])
        self.assertEqual(dropped[0]["dedup_of_trace_id"], "a")

    def test_pair_judge_failure_keeps_both(self) -> None:
        kept, dropped, stats = _run(
            [_row("回测A策略", "a"), _row("回测B策略", "b")], PairClient(fail=True)
        )
        self.assertEqual([row["trace_id"] for row in kept], ["a", "b"])
        self.assertEqual(dropped, [])
        self.assertEqual(stats["semantic_dedup_failed"], 1)

    def test_exact_text_needs_no_llm(self) -> None:
        client = PairClient(fail=True)
        kept, dropped, stats = _run(
            [_row("同一句问题", "a"), _row("同一句问题", "b")], client
        )
        self.assertEqual([row["trace_id"] for row in kept], ["a"])
        self.assertEqual(dropped[0]["method"], "exact_text")
        self.assertEqual(client.calls, 0)
        self.assertEqual(stats["semantic_dedup_failed"], 0)

    def test_shared_phrase_english_family_uses_semantic_review_only(self) -> None:
        suffix = "use capex = PaymentsToAcquirePropertyPlantAndEquipment from the SEC 10-K."
        rows = [
            _row(f"Did Rambus improve asset productivity in FY2024? Calculate revenue growth and operating margin for FY2023 and FY2024. {suffix}", "r1"),
            _row(f"Did Semtech improve cash generation in FY2024? Calculate OCF margin for FY2023 and FY2024. {suffix}", "r2"),
            _row(f"Did Qorvo improve gross margin in FY2024? Calculate gross margin for FY2023 and FY2024. {suffix}", "r3"),
        ]
        kept, dropped, stats = _run(rows, PairClient())
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 2)
        self.assertTrue(all(item["method"] == "semantic_signature_llm" for item in dropped))

    def test_shared_phrase_chinese_family_uses_semantic_review_only(self) -> None:
        prefix = "请以专业分析师身份，从近一个月的基本面、财务、估值、风险，对比一下"
        rows = [
            _row(prefix + "华海清科 和中芯国际", "c1"),
            _row(prefix + "科森科技 和福瑞医科", "c2"),
            _row(prefix + "太辰光 和巨人网络", "c3"),
        ]
        kept, dropped, stats = _run(rows, PairClient())
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 2)
        self.assertTrue(all(item["method"] == "semantic_signature_llm" for item in dropped))

    def test_template_phrase_below_threshold_kept(self) -> None:
        suffix = "use capex = PaymentsToAcquirePropertyPlantAndEquipment from the SEC 10-K."
        q1 = f"Did Rambus improve asset productivity? {suffix}"
        q2 = f"Did Semtech improve cash generation? {suffix}"
        # 两条问句共享表达只出现 2 次（<3）→ 模板表达层不命中；语义签名共享 token
        # 会生成 LLM 候选对，此处让 LLM 判 distinct，隔离验证 phrase 层行为。
        labels: dict[frozenset[str], str] = {frozenset((q1, q2)): "distinct"}
        rows = [
            _row(q1, "r1", signature=_signature(goal="a")),
            _row(q2, "r2", signature=_signature(goal="b")),
        ]
        kept, dropped, stats = _run(rows, PairClient(labels))
        self.assertEqual([row["trace_id"] for row in kept], ["r1", "r2"])
        self.assertEqual(dropped, [])

    def test_template_phrase_disabled_falls_back_to_llm(self) -> None:
        suffix = "use capex = PaymentsToAcquirePropertyPlantAndEquipment from the SEC 10-K."
        rows = [
            _row(f"Did Rambus improve asset productivity? {suffix}", "r1"),
            _row(f"Did Semtech improve cash generation? {suffix}", "r2"),
            _row(f"Did Qorvo improve gross margin? {suffix}", "r3"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            kept, dropped, stats = asyncio.run(
                semantic_dedup_rows(
                    rows,
                    DedupConfig(mode="semantic", phrase_dedup_enabled=False),
                    client=PairClient(),
                    llm_cfg=LLMConfig(model="fake"),
                    cache={},
                    cache_path=Path(tmp) / "cache.jsonl",
                    cache_lock=asyncio.Lock(),
                )
            )
        self.assertEqual(len(kept), 1)
        self.assertTrue(all(item["method"] != "template_phrase" for item in dropped))

    def test_template_phrase_representative_prefers_natural_question(self) -> None:
        suffix = "use capex = PaymentsToAcquirePropertyPlantAndEquipment from the SEC 10-K."
        rows = [
            _row(f"Did Rambus improve asset productivity? {suffix}", "scaffold1", template_severity="severe"),
            _row(f"Did Semtech improve cash generation? {suffix}", "scaffold2", template_severity="severe"),
            _row(f"Did Qorvo improve gross margin? {suffix}", "natural", template_severity="none"),
        ]
        kept, dropped, _ = _run(rows, PairClient())
        self.assertEqual([row["trace_id"] for row in kept], ["natural"])
        self.assertEqual(dropped[0]["dedup_of_trace_id"], "natural")

    def test_shared_phrase_char_threshold_never_deletes_directly(self) -> None:
        # 默认中文门槛 8 字符只生成族候选；这里的删除只能来自语义复核。
        prefix = "请以专业分析师身份"
        rows = [
            _row(prefix + "分析华海清科的估值", "c1", signature=_signature(goal="a")),
            _row(prefix + "对比科森科技和福瑞医科", "c2", signature=_signature(goal="b")),
            _row(prefix + "分析江淮汽车的基本面", "c3", signature=_signature(goal="c")),
        ]
        kept, dropped, stats = _run(rows, PairClient())
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 2)
        self.assertTrue(all(item["method"] == "semantic_signature_llm" for item in dropped))

    def test_template_phrase_below_char_threshold_kept(self) -> None:
        # 只共享 7 字符（低于默认 8 字符门槛）→ 模板表达层不命中；LLM 判 distinct。
        q1, q2, q3 = "请分析对比一下AAA和BBB", "请分析对比一下CCC和DDD", "请分析对比一下EEE和FFF"
        labels = {
            frozenset((q1, q2)): "distinct",
            frozenset((q1, q3)): "distinct",
            frozenset((q2, q3)): "distinct",
        }
        rows = [
            _row(q1, "c1", signature=_signature(goal="a")),
            _row(q2, "c2", signature=_signature(goal="b")),
            _row(q3, "c3", signature=_signature(goal="c")),
        ]
        kept, dropped, stats = _run(rows, PairClient(labels))
        self.assertEqual([row["trace_id"] for row in kept], ["c1", "c2", "c3"])
        self.assertEqual(dropped, [])
