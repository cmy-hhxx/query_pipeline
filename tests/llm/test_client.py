"""LLMClient retry policy: only connection/timeout/429/5xx/parse errors retry.

4xx permanent errors (400/401/404/…) must fail fast instead of burning
max_retries backoffs.
"""

from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import patch

import httpx
from openai import APIConnectionError, BadRequestError, InternalServerError, RateLimitError

from query_pipeline.config.models import LLMConfig
from query_pipeline.llm.client import LLMClient

_REQ = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _exc(cls: type, status: int) -> Exception:
    return cls("boom", response=httpx.Response(status, request=_REQ), body=None)


class _StubCompletions:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0

    async def create(self, **kwargs: object) -> object:
        self.calls += 1
        raise self.exc


class _StubClient:
    def __init__(self, exc: Exception) -> None:
        self.chat = type("Chat", (), {"completions": _StubCompletions(exc)})()


def _make_client(exc: Exception, *, max_retries: int = 3) -> tuple[LLMClient, _StubCompletions]:
    completions = _StubCompletions(exc)
    stub = type("Stub", (), {"chat": type("Chat", (), {"completions": completions})()})()

    async def fake_init(self_: object) -> None:  # noqa: ARG001
        return None

    with patch("query_pipeline.llm.client.AsyncOpenAI", return_value=stub):
        with patch.dict(os.environ, {"FAKE_KEY": "k", "FAKE_URL": "http://x"}):
            client = LLMClient(
                LLMConfig(
                    model="m",
                    api_key_env="FAKE_KEY",
                    base_url_env="FAKE_URL",
                    concurrency=2,
                    max_retries=max_retries,
                    timeout_seconds=1,
                )
            )
            # 不真实发请求：仅借用构造路径
            client.close = fake_init  # type: ignore[method-assign]
    return client, completions


class RetryPolicyTest(unittest.TestCase):
    async def _run(self, client: LLMClient, exc: Exception) -> Exception:
        async def noop_sleep(_: float) -> None:
            return None

        with patch("query_pipeline.llm.client.asyncio.sleep", new=noop_sleep):
            try:
                await client.complete(system_prompt="s", user_prompt="u")
            except Exception as raised:  # noqa: BLE001
                return raised
        raise AssertionError("expected an exception")

    def test_bad_request_400_no_retry(self) -> None:
        exc = _exc(BadRequestError, 400)
        client, completions = _make_client(exc)
        raised = asyncio.run(self._run(client, exc))
        self.assertIsInstance(raised, BadRequestError)
        self.assertEqual(completions.calls, 1)  # 不重试

    def test_internal_server_error_500_retries(self) -> None:
        exc = _exc(InternalServerError, 500)
        client, completions = _make_client(exc)
        raised = asyncio.run(self._run(client, exc))
        self.assertIsInstance(raised, RuntimeError)
        self.assertEqual(completions.calls, 3)  # max_retries 次

    def test_rate_limit_429_retries(self) -> None:
        exc = _exc(RateLimitError, 429)
        client, completions = _make_client(exc)
        raised = asyncio.run(self._run(client, exc))
        self.assertIsInstance(raised, RuntimeError)
        self.assertEqual(completions.calls, 3)

    def test_connection_error_retries(self) -> None:
        exc = APIConnectionError(request=_REQ)
        client, completions = _make_client(exc)
        raised = asyncio.run(self._run(client, exc))
        self.assertIsInstance(raised, RuntimeError)
        self.assertEqual(completions.calls, 3)


if __name__ == "__main__":
    unittest.main()
