from __future__ import annotations

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
        self.config = config
        self.client = AsyncOpenAI(api_key=api_key, base_url=config.base_url)

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
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
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("empty response content")
                return content
            except (APIConnectionError, APITimeoutError, RateLimitError, APIError, ValueError) as exc:
                last_error = exc
                if attempt == self.config.max_retries:
                    break
                sleep_seconds = min(30.0, 1.5 * (2 ** (attempt - 1))) + random.uniform(0, 0.5)
                import asyncio

                await asyncio.sleep(sleep_seconds)
        raise RuntimeError(f"LLM request failed after retries: {last_error}")

    async def close(self) -> None:
        await self.client.close()
