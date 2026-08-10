from __future__ import annotations

import asyncio
import os
import random
from typing import Any, cast

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

from query_pipeline.config.models import LLMConfig


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"{config.api_key_env} is not set")
        base_url = os.environ.get(config.base_url_env)
        if not base_url:
            raise RuntimeError(f"{config.base_url_env} is not set")
        self.config = config
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        # Cap in-flight HTTP calls process-wide across all pipeline stages.
        # 这是全管线唯一的限流点（run_concurrent/audit 不再叠加 semaphore）。
        self._semaphore = asyncio.Semaphore(max(1, config.concurrency))

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                # semaphore 只包单次 API 调用：退避 sleep 与 90s 超时若都在锁内
                # （5 次 ≈ 7.5min），429/5xx 风暴时全部 permit 被睡觉/挂起请求占死，
                # 健康请求队头阻塞、吞吐归零，风暴后逐个超时才恢复。
                async with self._semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format=cast(Any, {"type": self.config.response_format}),
                        temperature=0,
                        timeout=self.config.timeout_seconds,
                    )
                if not response.choices:
                    raise ValueError("empty response choices")
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("empty response content")
                return content
            except (APIConnectionError, APITimeoutError, RateLimitError, ValueError, IndexError) as exc:
                # 连接/超时/429/响应解析类：可重试（RateLimitError 必须先于 APIStatusError 捕获）
                retryable: Exception = exc
            except APIStatusError as exc:
                # 4xx 是永久错误（参数/鉴权/不存在等），重试无意义且慢（最长 ~22s+抖动），直接抛；
                # 5xx 服务端错误可重试。
                if exc.status_code < 500:
                    raise
                retryable = exc
            last_error = retryable
            if attempt == self.config.max_retries:
                break
            sleep_seconds = min(30.0, 1.5 * (2 ** (attempt - 1))) + random.uniform(0, 0.5)
            await asyncio.sleep(sleep_seconds)
        raise RuntimeError(f"LLM request failed after retries: {last_error}")

    async def close(self) -> None:
        await self.client.close()
