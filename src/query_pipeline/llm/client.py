from __future__ import annotations

import asyncio
import os
import random
from typing import Any, cast

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from query_pipeline.config.models import LLMStageConfig


class LLMClient:
    def __init__(self, config: LLMStageConfig) -> None:
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"{config.api_key_env} is not set")
        base_url = os.environ.get(config.base_url_env)
        if not base_url:
            raise RuntimeError(f"{config.base_url_env} is not set")
        self.config = config
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        # Cap in-flight HTTP calls process-wide. Session concurrency × per-session
        # judge concurrency would otherwise multiply into thousands of open requests.
        self._semaphore = asyncio.Semaphore(max(1, config.concurrency))

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        async with self._semaphore:
            return await self._complete_once(system_prompt=system_prompt, user_prompt=user_prompt)

    async def _complete_once(self, *, system_prompt: str, user_prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
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
            except (APIConnectionError, APITimeoutError, RateLimitError, APIError, ValueError, IndexError) as exc:
                last_error = exc
                if attempt == self.config.max_retries:
                    break
                sleep_seconds = min(30.0, 1.5 * (2 ** (attempt - 1))) + random.uniform(0, 0.5)
                await asyncio.sleep(sleep_seconds)
        raise RuntimeError(f"LLM request failed after retries: {last_error}")

    async def close(self) -> None:
        await self.client.close()
