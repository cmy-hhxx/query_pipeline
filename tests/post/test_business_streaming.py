"""Business rows become readable in completion order during translation."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

from query_pipeline.config.models import LLMConfig
from query_pipeline.io.business_log import BusinessLogWriter
from query_pipeline.llm.client import LLMClient
from query_pipeline.post.translate import translate_rows


class ControlledTranslationClient:
    def __init__(self, release_slow: asyncio.Event) -> None:
        self.release_slow = release_slow

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        text = json.loads(user_prompt.split("\n", 1)[1])["text"]
        if text == "slow question":
            await self.release_slow.wait()
        return json.dumps({"translation": f"译:{text}"}, ensure_ascii=False)


class BusinessStreamingTest(unittest.TestCase):
    def test_flushes_in_completion_order_before_batch_finishes(self) -> None:
        async def scenario(tmp_path: Path) -> None:
            release_slow = asyncio.Event()
            fast_logged = asyncio.Event()
            rows = [
                {
                    "trace_id": "slow",
                    "difficulty_level": "hard",
                    "input": {"text": "slow question"},
                },
                {
                    "trace_id": "fast",
                    "difficulty_level": "hard",
                    "input": {"text": "fast question"},
                },
            ]
            with BusinessLogWriter(tmp_path / "logs", "batch") as writer:
                def emit(row: dict) -> None:
                    writer.write(row)
                    if row["trace_id"] == "fast":
                        fast_logged.set()

                task = asyncio.create_task(
                    translate_rows(
                        rows,
                        client=cast(LLMClient, ControlledTranslationClient(release_slow)),
                        llm_cfg=LLMConfig(model="fake", cache=tmp_path / "cache.jsonl"),
                        cache={},
                        cache_path=tmp_path / "cache.jsonl",
                        on_complete=emit,
                    )
                )
                await asyncio.wait_for(fast_logged.wait(), timeout=1)
                visible = [
                    json.loads(line)
                    for line in writer.paths["cleaned"].read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual([row["trace_id"] for row in visible], ["fast"])
                self.assertFalse(task.done())
                release_slow.set()
                await task

                complete = [
                    json.loads(line)
                    for line in writer.paths["cleaned"].read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual([row["trace_id"] for row in complete], ["fast", "slow"])

        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(scenario(Path(tmp)))


if __name__ == "__main__":
    unittest.main()
