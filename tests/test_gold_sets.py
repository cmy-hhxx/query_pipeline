from __future__ import annotations

import json
import unittest
from pathlib import Path

from query_pipeline.quality.gold_gate import load_complex_policy_gold
from query_pipeline.session.funnel import parse_complexity_response
from tests._profiles import complexity_label


FIXTURES = Path(__file__).parent / "fixtures"


def _read_jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines()]


class GoldSetTest(unittest.TestCase):
    def test_manual_complex_policy_corpus_has_fixed_acceptance_counts(self) -> None:
        aime_positive, aime_negative = load_complex_policy_gold("aime")
        iwencai_positive, iwencai_negative = load_complex_policy_gold("iwencai")
        positives = aime_positive + iwencai_positive
        negatives = aime_negative + iwencai_negative

        self.assertEqual(len(positives), 606)  # aime 218 + iwencai 388
        self.assertEqual(len(negatives), 1_254)  # pre_final_review - final
        self.assertEqual((len(aime_positive), len(iwencai_positive)), (218, 388))
        self.assertEqual((len(aime_negative), len(iwencai_negative)), (866, 388))
        self.assertTrue(all(row["expected_route"] == "complex" for row in positives))
        self.assertTrue(all(row["expected_route"] == "not_complex" for row in negatives))

        expected_keys = {"trace_id", "input", "expected_route"}
        for row in positives + negatives:
            self.assertEqual(set(row), expected_keys)
            self.assertEqual(set(row["input"]), {"text"})
            self.assertTrue(str(row["trace_id"]).strip())
            self.assertTrue(str(row["input"]["text"]).strip())

        positive_ids = {row["trace_id"] for row in positives}
        negative_ids = {row["trace_id"] for row in negatives}
        self.assertEqual(len(positive_ids), 606)
        self.assertEqual(len(negative_ids), 1_254)
        self.assertTrue(positive_ids.isdisjoint(negative_ids))

    def test_complexity_gold_obeys_admission_invariant(self) -> None:
        rows = _read_jsonl("complexity_gold.jsonl")
        self.assertGreaterEqual(len(rows), 8)
        for row in rows:
            expected_complex = row["expected_route"] == "complex"
            profile = parse_complexity_response(
                complexity_label(
                    expected_complex,
                    route=row["expected_route"],
                    goal=row["question"],
                    complex_features=row["complex_features"],
                    exclusion_reasons=row["exclusion_reasons"],
                    evidence_quote=row["question"],
                    question_quality=row.get("question_quality", "high"),
                )
            )
            self.assertEqual(profile.route, row["expected_route"], row["id"])
            self.assertEqual(profile.admissible_hard, expected_complex, row["id"])
            self.assertEqual(
                profile.admits_hard_for(row["question"]), expected_complex, row["id"]
            )
            self.assertEqual(
                profile.question_quality,
                row.get("question_quality", "high"),
                row["id"],
            )
        user_example = next(row for row in rows if row["id"] == "filter_many_conditions")
        self.assertEqual(user_example["expected_route"], "complex")
        self.assertEqual(user_example["complex_features"], ["natural_multi_condition_screen"])

    def test_complex_admission_requires_matching_criterion_and_source_evidence(self) -> None:
        question = "回测月末调仓策略的历史收益"
        mismatched = complexity_label(
            True,
            goal=question,
            complex_features=["historical_simulation_statistics"],
            evidence_quote=question,
        )
        mismatched["evidence"][0]["criterion"] = "strategy_scenario_evaluation"
        with self.assertRaises(ValueError):
            parse_complexity_response(mismatched)

        fabricated = parse_complexity_response(
            complexity_label(
                True,
                goal=question,
                complex_features=["historical_simulation_statistics"],
                evidence_quote="问句中不存在的证据",
            )
        )
        self.assertTrue(fabricated.admissible_hard)
        self.assertFalse(fabricated.admits_hard_for(question))

        partially_fabricated = complexity_label(
            True,
            goal=question,
            complex_features=["historical_simulation_statistics"],
            evidence_quote=question,
        )
        partially_fabricated["evidence"].append(
            {
                "criterion": "historical_simulation_statistics",
                "quote": "另一段并不存在的证据",
            }
        )
        profile = parse_complexity_response(partially_fabricated)
        self.assertTrue(profile.admissible_hard)
        self.assertFalse(profile.admits_hard_for(question))

    def test_dedup_pair_gold_covers_merge_and_keep_boundaries(self) -> None:
        rows = _read_jsonl("dedup_pairs_gold.jsonl")
        labels = {row["expected"] for row in rows}
        self.assertIn("template_duplicate", labels)
        self.assertIn("distinct", labels)
        self.assertTrue(any(row["id"] == "count_swap" for row in rows))
        self.assertTrue(any(row["id"] == "strategy_change" for row in rows))
