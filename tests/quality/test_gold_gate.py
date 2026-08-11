from __future__ import annotations

import copy
import unittest

from query_pipeline.quality.gold_gate import (
    evaluate_complex_policy_gold,
    load_complex_policy_gold,
)


class ComplexPolicyGoldGateTest(unittest.TestCase):
    def test_exact_positive_replay_and_no_negative_hard_passes(self) -> None:
        positives, _ = load_complex_policy_gold("aime")
        records = [
            {**copy.deepcopy(row), "difficulty_level": "hard"}
            for row in positives
        ]
        result = evaluate_complex_policy_gold(records, "aime")
        self.assertTrue(result["passed"])
        self.assertEqual(result["positive_recall"], 1.0)
        self.assertEqual(result["negative_false_accepts"], 0)

    def test_missing_positive_fails_recall_gate(self) -> None:
        positives, _ = load_complex_policy_gold("aime")
        records = [
            {**copy.deepcopy(row), "difficulty_level": "hard"}
            for row in positives[1:]
        ]
        result = evaluate_complex_policy_gold(records, "aime")
        self.assertFalse(result["passed"])
        self.assertEqual(result["positive_accepted"], len(positives) - 1)
        self.assertEqual(result["missed_positive_ids"], [positives[0]["trace_id"]])

    def test_known_negative_routed_hard_fails_gate(self) -> None:
        positives, negatives = load_complex_policy_gold("iwencai")
        records = [
            {**copy.deepcopy(row), "difficulty_level": "hard"}
            for row in positives
        ]
        records.append({**copy.deepcopy(negatives[0]), "difficulty_level": "hard"})
        result = evaluate_complex_policy_gold(records, "iwencai")
        self.assertFalse(result["passed"])
        self.assertEqual(result["negative_false_accepts"], 1)
        self.assertEqual(result["negative_false_accept_ids"], [negatives[0]["trace_id"]])

    def test_unknown_dataset_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "不支持"):
            load_complex_policy_gold("unknown")


if __name__ == "__main__":
    unittest.main()
