"""Segment parsing: cache coverage, topic merging, boundary repair, rejection."""

from __future__ import annotations

import json
import unittest

from tests._profiles import complexity_label

from query_pipeline.models.session import Segment, parse_segment_response
from query_pipeline.session.segment import _segments_from_cache

class SegmentParserTest(unittest.TestCase):
    def test_segment_cache_partial_coverage_rejected(self) -> None:
        # A locally-contiguous but partial cache must be rejected as a miss (re-call LLM)
        # rather than pass and later IndexError inside segment_of.
        partial = {"segments": [{"start": 0, "end": 2, "topic": "t"}]}
        with self.assertRaises(ValueError):
            _segments_from_cache(partial, num_turns=5)
        # Full coverage still accepted.
        full = {"segments": [{"start": 0, "end": 2, "topic": "a"}, {"start": 3, "end": 4, "topic": "b"}]}
        self.assertEqual(len(_segments_from_cache(full, num_turns=5)), 2)

    def test_segment_parser_merges_recurring_topics(self) -> None:
        raw = json.dumps(
            {
                "segments": [
                    {"start": 0, "end": 1, "topic": "A"},
                    {"start": 2, "end": 3, "topic": "B"},
                    {"start": 4, "end": 4, "topic": "A"},
                ]
            },
            ensure_ascii=False,
        )

        segments = parse_segment_response(raw, num_turns=5)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].start, 0)
        self.assertEqual(segments[0].end, 4)
        self.assertEqual(segments[0].topic, "A")

    def test_segment_parser_keeps_distinct_topics(self) -> None:
        raw = json.dumps(
            {"segments": [{"start": 0, "end": 2, "topic": "A"}, {"start": 3, "end": 4, "topic": "B"}]},
            ensure_ascii=False,
        )

        segments = parse_segment_response(raw, num_turns=5)

        self.assertEqual([(s.start, s.end, s.topic) for s in segments], [(0, 2, "A"), (3, 4, "B")])

    def test_segment_parser_repairs_small_boundary_slips(self) -> None:
        # LLM dropped index 35 (gap) and ended at 54 instead of 55: both are
        # off-by-one slips that must be snapped into a valid covering, not
        # thrown away (this was silently degrading 56-turn sessions to 1).
        raw = json.dumps(
            {
                "segments": [
                    {"start": 0, "end": 3, "topic": "A"},
                    {"start": 4, "end": 18, "topic": "B"},
                    {"start": 19, "end": 34, "topic": "C"},
                    {"start": 36, "end": 46, "topic": "D"},
                    {"start": 47, "end": 54, "topic": "E"},
                ]
            }
        )
        segments = parse_segment_response(raw, num_turns=56)
        self.assertEqual(
            [(s.start, s.end, s.topic) for s in segments],
            [(0, 3, "A"), (4, 18, "B"), (19, 34, "C"), (35, 46, "D"), (47, 55, "E")],
        )

    def test_segment_parser_rejects_malformed(self) -> None:
        with self.assertRaises(ValueError):
            parse_segment_response(json.dumps({"segments": []}), num_turns=5)
        with self.assertRaises(ValueError):
            parse_segment_response(json.dumps({"segments": [{"start": 2, "end": 3, "topic": "A"}]}), num_turns=5)
        with self.assertRaises(ValueError):
            parse_segment_response(
                json.dumps({"segments": [{"start": 0, "end": 1, "topic": "A"}, {"start": 1, "end": 2, "topic": "B"}]}),
                num_turns=5,
            )

    def test_funnel_parsers(self) -> None:
        from query_pipeline.session.funnel import (
            parse_classify_response,
            parse_complexity_response,
            parse_value_response,
        )

        valuable = parse_value_response(json.dumps({"is_valuable": True, "reason": "金融相关"}))
        self.assertTrue(valuable.is_valuable)

        complex_result = parse_complexity_response(
            json.dumps(complexity_label(True, reason="需要多步分析"), ensure_ascii=False)
        )
        self.assertEqual(complex_result.route, "complex")

        classified = parse_classify_response(json.dumps({"category_id": "03", "reason": "需要多步分析"}, ensure_ascii=False))
        self.assertEqual(classified.category_id, "03")

        with self.assertRaises(ValueError):
            parse_classify_response(json.dumps({"category_id": "99", "reason": "x"}))
