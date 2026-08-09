from __future__ import annotations

import asyncio
import unittest

from query_pipeline.llm.runner import run_concurrent


class RunConcurrentTest(unittest.TestCase):
    async def _run(self) -> list[int]:
        return await run_concurrent([1, 2, 3], lambda x: _sq(x))

    def test_order_preserved(self) -> None:
        self.assertEqual(asyncio.run(self._run()), [1, 4, 9])

    def test_worker_exception_does_not_abort_batch(self) -> None:
        # 单行异常：该位返回 None 并告警，其余项照常完成，run_concurrent 不抛。
        async def worker(x: int) -> int:
            if x == 2:
                raise RuntimeError("boom")
            return await _sq(x)

        results = asyncio.run(run_concurrent([1, 2, 3], worker))
        self.assertEqual(results, [1, None, 9])

    def test_all_fail_returns_nones(self) -> None:
        async def worker(_: int) -> int:
            raise RuntimeError("boom")

        results = asyncio.run(run_concurrent([1, 2], worker))
        self.assertEqual(results, [None, None])


async def _sq(x: int) -> int:
    await asyncio.sleep(0)
    return x * x


if __name__ == "__main__":
    unittest.main()
