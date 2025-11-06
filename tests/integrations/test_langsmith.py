"""Tests for LangSmith tracker."""

import unittest

import pytest

from tygent.integrations.langsmith import LangSmithTracker


class MockLangSmithClient:
    def __init__(self):
        self.logged = []

    async def log_run(self, dag_name, inputs, outputs, tags=None):
        self.logged.append((dag_name, inputs, outputs, tags))


class TestLangSmithTracker(unittest.TestCase):
    def setUp(self):
        self.client = MockLangSmithClient()
        self.tracker = LangSmithTracker(self.client)

    @pytest.mark.asyncio
    async def test_log_run(self):
        await self.tracker.log_run(
            dag_name="demo",
            inputs={"a": 1},
            outputs={"b": 2},
            tags=["t1"],
        )
        self.assertEqual(len(self.client.logged), 1)
        self.assertEqual(self.client.logged[0][0], "demo")


if __name__ == "__main__":
    unittest.main()
