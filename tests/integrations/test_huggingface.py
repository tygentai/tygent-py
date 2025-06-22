"""Tests for HuggingFace integration."""

import unittest
import pytest
from tygent.integrations.huggingface import (
    HuggingFaceNode,
    HuggingFaceIntegration,
    HuggingFaceBatchProcessor,
)


class MockHFModel:
    def __call__(self, prompt, **kwargs):
        return f"out:{prompt}"


class TestHuggingFaceNode(unittest.TestCase):
    def setUp(self):
        self.model = MockHFModel()
        self.node = HuggingFaceNode(
            name="test_node",
            model=self.model,
            prompt_template="Hello {name}",
        )

    def test_initialization(self):
        self.assertEqual(self.node.name, "test_node")

    @pytest.mark.asyncio
    async def test_execute(self):
        result = await self.node.execute({"name": "Bob"})
        self.assertIn("Bob", result)


class TestHuggingFaceIntegration(unittest.TestCase):
    def setUp(self):
        self.model = MockHFModel()
        self.integration = HuggingFaceIntegration(self.model)

    def test_add_node(self):
        node = self.integration.add_node(
            name="greet", prompt_template="Hi {who}", dependencies=[]
        )
        self.assertEqual(node.name, "greet")

    @pytest.mark.asyncio
    async def test_execute(self):
        self.integration.add_node(name="greet", prompt_template="Hi {who}")
        self.integration.add_node(
            name="echo", prompt_template="{greet}", dependencies=["greet"]
        )
        results = await self.integration.execute({"who": "Sam"})
        self.assertIn("greet", results)
        self.assertIn("echo", results)


class TestHuggingFaceBatchProcessor(unittest.TestCase):
    def setUp(self):
        self.model = MockHFModel()
        self.processor = HuggingFaceBatchProcessor(
            self.model, batch_size=2, max_concurrent_batches=1
        )

    @pytest.mark.asyncio
    async def test_process(self):
        async def process_fn(prompt, model):
            return model(prompt)

        prompts = ["a", "b", "c"]
        results = await self.processor.process(prompts, process_fn)
        self.assertEqual(results, ["out:a", "out:b", "out:c"])


if __name__ == "__main__":
    unittest.main()
