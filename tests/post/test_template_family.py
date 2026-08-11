from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from query_pipeline.config.models import DedupConfig, LLMConfig
from query_pipeline.post.dedup import review_template_families, semantic_dedup_rows


def _row(text: str, trace_id: str, *, quality: str = "high") -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "source_case_id": "case",
        "difficulty_level": "hard",
        "input": {"text": text},
        "meta": {
            "complexity_profile": {"question_quality": quality},
            "value_profile": {"template_severity": "none"},
        },
    }


class FamilyClient:
    def __init__(self, label: str, *, confidence: str = "high") -> None:
        self.label = label
        self.confidence = confidence

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        assert system_prompt
        payload = json.loads(user_prompt.split("\n", 1)[1])
        return json.dumps(
            {
                "items": [
                    {
                        "family_id": family["family_id"],
                        "label": self.label,
                        "confidence": self.confidence,
                        "reason": "金标族裁决",
                    }
                    for family in payload["families"]
                ]
            },
            ensure_ascii=False,
        )


def _run(rows: list[dict[str, Any]], client: FamilyClient):
    with tempfile.TemporaryDirectory() as tmp:
        return asyncio.run(
            review_template_families(
                rows,
                DedupConfig(),
                client=client,
                llm_cfg=LLMConfig(model="fake"),
                cache={},
                cache_path=Path(tmp) / "cache.jsonl",
                cache_lock=asyncio.Lock(),
            )
        )


class TemplateFamilyReviewTest(unittest.TestCase):
    def setUp(self) -> None:
        prefix = "请以专业分析师身份，从基本面、财务、估值和风险四个维度对比"
        self.rows = [
            _row(prefix + "华海清科与中芯国际", "c1"),
            _row(prefix + "科森科技与福瑞医科", "c2"),
            _row(prefix + "太辰光与巨人网络", "c3"),
        ]

    def test_eval_template_family_rejects_every_member(self) -> None:
        kept, dropped, stats, force_normal, protected = _run(
            self.rows, FamilyClient("eval_template_family")
        )
        self.assertEqual(kept, [])
        self.assertEqual({item["trace_id"] for item in dropped}, {"c1", "c2", "c3"})
        self.assertTrue(all(item["decision"] == "eval_template_family" for item in dropped))
        self.assertEqual(stats["template_family_rejected"], 1)
        self.assertEqual(stats["template_family_rejected_rows"], 3)
        self.assertEqual(force_normal, set())
        self.assertEqual(protected, set())

    def test_semantic_duplicate_keeps_best_representative(self) -> None:
        self.rows[0]["meta"]["complexity_profile"]["question_quality"] = "low"
        self.rows[2]["meta"]["complexity_profile"]["question_quality"] = "medium"
        kept, dropped, stats, _, protected = _run(
            self.rows, FamilyClient("semantic_duplicate")
        )
        self.assertEqual([row["trace_id"] for row in kept], ["c2"])
        self.assertEqual({item["dedup_of_trace_id"] for item in dropped}, {"c2"})
        self.assertEqual(stats["template_family_duplicates"], 2)
        self.assertEqual(protected, set())

    def test_natural_shared_date_phrase_keeps_all(self) -> None:
        rows = [
            _row("截至2026年8月11日，分析沪深300的盈利趋势和估值", "d1"),
            _row("截至2026年8月11日，比较中证500的波动与回撤", "d2"),
            _row("截至2026年8月11日，评估科创50的资金流和情景风险", "d3"),
        ]
        kept, dropped, stats, force_normal, protected = _run(
            rows, FamilyClient("natural_shared_phrase")
        )
        self.assertEqual([row["trace_id"] for row in kept], ["d1", "d2", "d3"])
        self.assertEqual(dropped, [])
        self.assertGreater(stats["template_family_candidates"], 0)
        self.assertEqual(force_normal, set())
        self.assertEqual(protected, {id(row) for row in rows})

        # The family verdict is authoritative: the following semantic pass may
        # not collapse rows that were explicitly judged natural shared language.
        with tempfile.TemporaryDirectory() as tmp:
            final_rows, semantic_dropped, _ = asyncio.run(
                semantic_dedup_rows(
                    kept,
                    DedupConfig(),
                    client=FamilyClient("semantic_duplicate"),
                    llm_cfg=LLMConfig(model="fake"),
                    cache={},
                    cache_path=Path(tmp) / "cache.jsonl",
                    cache_lock=asyncio.Lock(),
                    protected_row_ids=protected,
                )
            )
        self.assertEqual([row["trace_id"] for row in final_rows], ["d1", "d2", "d3"])
        self.assertEqual(semantic_dropped, [])

    def test_natural_shared_phrase_still_removes_exact_duplicates(self) -> None:
        duplicate = "截至2026年8月11日，分析沪深300的盈利趋势和估值"
        rows = [
            _row(duplicate, "d1"),
            _row(duplicate, "d2"),
            _row("截至2026年8月11日，评估科创50的资金流和情景风险", "d3"),
        ]
        kept, _, _, _, protected = _run(rows, FamilyClient("natural_shared_phrase"))

        with tempfile.TemporaryDirectory() as tmp:
            final_rows, dropped, _ = asyncio.run(
                semantic_dedup_rows(
                    kept,
                    DedupConfig(),
                    client=FamilyClient("semantic_duplicate"),
                    llm_cfg=LLMConfig(model="fake"),
                    cache={},
                    cache_path=Path(tmp) / "cache.jsonl",
                    cache_lock=asyncio.Lock(),
                    protected_row_ids=protected,
                )
            )

        self.assertEqual([row["trace_id"] for row in final_rows], ["d1", "d3"])
        self.assertEqual([item["trace_id"] for item in dropped], ["d2"])
        self.assertEqual(dropped[0]["method"], "exact_text")

    def test_low_confidence_family_is_retained_but_forced_normal(self) -> None:
        kept, dropped, _, force_normal, protected = _run(
            self.rows, FamilyClient("natural_shared_phrase", confidence="low")
        )
        self.assertEqual(len(kept), 3)
        self.assertEqual(dropped, [])
        self.assertEqual(force_normal, {id(row) for row in self.rows})
        self.assertEqual(protected, {id(row) for row in self.rows})


if __name__ == "__main__":
    unittest.main()
