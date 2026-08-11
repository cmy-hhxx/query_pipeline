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
    PARSE_MAX_ATTEMPTS,
    _call,
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

    def test_ungrounded_evidence_fails_only_complex_route(self) -> None:
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

        # complex：逐字证据缺失 → 仍 fail-closed（质量闸门只对正向声明生效）
        result = asyncio.run(run_case("complex"))
        self.assertIn("evidence quote must be copied", result["error"])
        self.assertNotIn("dropped", result)

        # reject：负向声明不强制证据 → 正常按模板排除 dropped，不再报错
        result = asyncio.run(run_case("reject"))
        self.assertIsNone(result["error"])
        self.assertEqual(result["dropped"], "complexity_reject")

    def test_normal_reject_routes_tolerate_missing_evidence(self) -> None:
        # ③ normal/reject 不再要求每条 exclusion 都有证据：空/乱证据也能通过
        for route in ("normal", "reject"):
            with self.subTest(route=route):
                profile = parse_complexity_response(
                    complexity_label(
                        False,
                        route=route,
                        evidence_quote="问句中不存在的证据",
                    )
                )
                self.assertTrue(profile.evidence_is_grounded_for("当前金融问题"))

    def test_complex_route_requires_evidence_for_each_feature(self) -> None:
        # ② 覆盖匹配：声明两个 feature 但只给一条证据 → 必须失败
        label = complexity_label(
            True,
            complex_features=["multi_dimension_attribution", "cross_period_entity_research"],
        )
        label["evidence"] = [
            {"criterion": "multi_dimension_attribution", "quote": label["evidence"][0]["quote"]}
        ]
        with self.assertRaises(ValueError):
            parse_complexity_response(label)

    def test_stray_evidence_is_tolerated(self) -> None:
        # ② 多余的 evidence（不在声明的 feature 里）不再导致失败
        label = complexity_label(True, complex_features=["multi_dimension_attribution"])
        label["evidence"].append(
            {"criterion": "cross_period_entity_research", "quote": label["evidence"][0]["quote"]}
        )
        profile = parse_complexity_response(label)
        self.assertTrue(profile.admissible_hard)

    def test_invalid_enum_values_are_filtered(self) -> None:
        # 模型偶发输出非法枚举：清洗后过滤，路由自洽即可解析（不再 fail-closed 丢候选）
        label = complexity_label(True, complex_features=["multi_dimension_attribution"])
        label["complex_features"].append("不存在的特征")
        label["evidence"].append({"criterion": "不存在的特征", "quote": "问句"})
        profile = parse_complexity_response(label)
        self.assertEqual(profile.complex_features, ["multi_dimension_attribution"])

    def test_grounding_tolerates_whitespace_and_fullwidth(self) -> None:
        # 逐字证据的空白/全半角差异不算失配（LLM 常见行为）
        profile = parse_complexity_response(
            complexity_label(True, evidence_quote="8月7股价跌")
        )
        self.assertTrue(profile.evidence_is_grounded_for("8月7 股价跌 且放量"))  # 空白差异
        profile2 = parse_complexity_response(
            complexity_label(True, evidence_quote="8月7，股价跌")
        )
        self.assertTrue(profile2.evidence_is_grounded_for("8月7,股价跌且放量"))  # 全角→半角

    def test_parse_failure_retries_llm(self) -> None:
        # ① 解析/校验失败重调：首次坏输出 → 重调返回合法值
        class FlakyClient:
            def __init__(self) -> None:
                self.calls = 0

            async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
                self.calls += 1
                if self.calls == 1:
                    return json.dumps({"is_valuable": "garbage"})
                return json.dumps({"is_valuable": True})

        async def run() -> tuple:
            client = FlakyClient()
            with tempfile.TemporaryDirectory() as tmp:
                parsed = await _call(
                    client=client,  # type: ignore[arg-type]
                    llm_cfg=LLMConfig(model="fake"),
                    cache={},
                    cache_path=Path(tmp) / "cache.jsonl",
                    cache_lock=asyncio.Lock(),
                    step="value_gate",
                    prompt_id="value_gate",
                    payload={"current_question": "q"},
                    parse=parse_value_response,
                )
            return parsed, client.calls

        parsed, calls = asyncio.run(run())
        self.assertTrue(parsed.is_valuable)
        self.assertEqual(calls, 2)
        self.assertLessEqual(calls, PARSE_MAX_ATTEMPTS)


if __name__ == "__main__":
    unittest.main()
