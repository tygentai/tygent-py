import asyncio
import unittest

from examples.dag_example import run_scheduler, run_sequential


class TestExamplesBenchmarks(unittest.TestCase):
    def test_scheduler_faster(self):
        seq = asyncio.run(run_sequential())
        par = asyncio.run(run_scheduler())
        self.assertLess(par, seq)


if __name__ == "__main__":
    unittest.main()
