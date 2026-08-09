from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_pipeline.llm.runner import run_concurrent


class RunConcurrentTest(unittest.TestCase):
    async def _run(self, concurrency: int) -> list[int]:
        return await run_concurrent([1, 2, 3], lambda x: _sq(x), concurrency=concurrency)

    def test_clamps_nonpositive_concurrency(self) -> None:
        # concurrency=0 previously deadlocked on Semaphore(0); -1 raised ValueError.
        self.assertEqual(asyncio.run(self._run(0)), [1, 4, 9])
        self.assertEqual(asyncio.run(self._run(-1)), [1, 4, 9])

    def test_positive_concurrency_unchanged(self) -> None:
        self.assertEqual(asyncio.run(self._run(2)), [1, 4, 9])


async def _sq(x: int) -> int:
    await asyncio.sleep(0)
    return x * x


if __name__ == "__main__":
    unittest.main()
