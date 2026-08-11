"""Funnel response parsing: strict boolean validation (fail-closed on garbage)."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from query_pipeline.config.models import LLMConfig
from query_pipeline.models.session import Segment
from query_pipeline.models.turn import Turn
from query_pipeline.session.funnel import (
    funnel_candidate,
    parse_complexity_response,
    parse_value_response,
)
from tests._profiles import complexity_label


class FunnelParseTest(unittest.TestCase):
    def test_string_false_is_false(self) -> None:
        # bool("false") == True —— 手工转换会静默放行；pydantic 正确解析为 False
        result = parse_value_response({"is_valuable": "false"})
        self.assertFalse(result.is_valuable)
        payload = complexity_label(False)
        payload["is_complex"] = "no"
        with self.assertRaises(ValueError):
            parse_complexity_response(payload)

    def test_string_true_is_true(self) -> None:
        self.assertTrue(parse_value_response({"is_valuable": "true"}).is_valuable)

    def test_missing_field_fails_closed(self) -> None:
        # 缺字段 = ValidationError = ValueError 子类 → 候选丢弃（fail-closed）
        with self.assertRaises(ValueError):
            parse_value_response({"reason": "no verdict"})

    def test_garbage_value_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            parse_value_response({"is_valuable": "maybe"})
        with self.assertRaises(ValueError):
            payload = complexity_label(False)
            payload["is_complex"] = 42
            parse_complexity_response(payload)

    def test_reason_optional(self) -> None:
        self.assertIsNone(parse_value_response({"is_valuable": False}).reason)
        self.assertEqual(
            parse_complexity_response(complexity_label(True, reason="r")).reason, "r"
        )

    def test_natural_multi_condition_filter_is_admitted(self) -> None:
        profile = parse_complexity_response(
            complexity_label(
                True,
                complex_features=["natural_multi_condition_screen"],
                evidence_quote="涨幅、市值、量比、换手率",
            )
        )
        self.assertTrue(profile.admissible_hard)

    def test_value_profile_rejects_semantic_non_questions_and_severe_templates(self) -> None:
        no_task = parse_value_response(
            {
                "is_valuable": False,
                "has_executable_task": False,
                "self_contained": True,
                "template_severity": "none",
                "contains_embedded_prompt": False,
            }
        )
        self.assertFalse(no_task.admissible)
        self.assertEqual(no_task.rejection_kind, "no_task")

        severe = parse_value_response(
            {
                "is_valuable": False,
                "has_executable_task": True,
                "self_contained": True,
                "template_severity": "severe",
                "contains_embedded_prompt": False,
            }
        )
        self.assertFalse(severe.admissible)
        self.assertEqual(severe.rejection_kind, "template")

        natural_generic = parse_value_response(
            {
                "is_valuable": True,
                "has_executable_task": True,
                "self_contained": True,
                "template_severity": "light",
                "contains_embedded_prompt": False,
            }
        )
        self.assertTrue(natural_generic.admissible)

    def test_only_normal_classification_can_receive_prior_questions(self) -> None:
        class PayloadClient:
            def __init__(self, *, complex_route: bool) -> None:
                self.complex_route = complex_route
                self.calls: list[tuple[str, dict]] = []

            async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
                payload = json.loads(user_prompt.split("\n", 1)[1])
                self.calls.append((system_prompt, payload))
                if "价值判官" in system_prompt:
                    return json.dumps({"is_valuable": True})
                if "complex_features" in system_prompt:
                    return json.dumps(
                        complexity_label(
                            self.complex_route,
                            goal=payload["current_question"],
                            evidence_quote=payload["current_question"],
                        )
                    )
                return json.dumps(
                    {
                        "category_id": "01" if self.complex_route else "03",
                        "reason": "测试归类",
                    }
                )

        turns = [
            Turn(question="前文问题", answer="前文答案", trace_id="t0"),
            Turn(question="当前金融问题", answer="答案", trace_id="t1"),
        ]

        async def run_case(complex_route: bool) -> PayloadClient:
            client = PayloadClient(complex_route=complex_route)
            with tempfile.TemporaryDirectory() as tmp:
                await funnel_candidate(
                    client=client,  # type: ignore[arg-type]
                    turns=turns,
                    segments=[Segment(0, 1, "topic")],
                    idx=1,
                    llm_cfg=LLMConfig(model="fake"),
                    cache={},
                    cache_path=Path(tmp) / "cache.jsonl",
                    cache_lock=asyncio.Lock(),
                )
            return client

        complex_client = asyncio.run(run_case(True))
        self.assertTrue(
            all(payload == {"current_question": "当前金融问题"} for _, payload in complex_client.calls)
        )

        normal_client = asyncio.run(run_case(False))
        normal_payloads = [payload for _, payload in normal_client.calls]
        self.assertEqual(normal_payloads[:2], [{"current_question": "当前金融问题"}] * 2)
        self.assertEqual(
            normal_payloads[-1],
            {"prior_questions": ["前文问题"], "current_question": "当前金融问题"},
        )

    def test_ungrounded_evidence_fails_all_routes(self) -> None:
        question = "当前金融问题"

        class UngroundedClient:
            def __init__(self, route: str) -> None:
                self.route = route

            async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
                payload = json.loads(user_prompt.split("\n", 1)[1])
                if "价值判官" in system_prompt:
                    return json.dumps({"is_valuable": True})
                return json.dumps(
                    complexity_label(
                        self.route == "complex",
                        route=self.route,
                        goal=payload["current_question"],
                        evidence_quote="问句中不存在的证据",
                    )
                )

        async def run_case(route: str) -> dict:
            with tempfile.TemporaryDirectory() as tmp:
                return await funnel_candidate(
                    client=UngroundedClient(route),  # type: ignore[arg-type]
                    turns=[Turn(question=question, answer="答案", trace_id="t1")],
                    segments=[Segment(0, 0, "topic")],
                    idx=0,
                    llm_cfg=LLMConfig(model="fake"),
                    cache={},
                    cache_path=Path(tmp) / "cache.jsonl",
                    cache_lock=asyncio.Lock(),
                )

        for route in ("complex", "normal", "reject"):
            with self.subTest(route=route):
                result = asyncio.run(run_case(route))
                self.assertIn("evidence quote must be copied", result["error"])
                self.assertNotIn("dropped", result)


if __name__ == "__main__":
    unittest.main()
