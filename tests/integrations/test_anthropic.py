"""Tests for Anthropic integration."""

import unittest
import pytest
from tygent.integrations.anthropic import (
    AnthropicNode,
    AnthropicIntegration,
    AnthropicBatchProcessor,
)


class MockClaudeMessages:
    async def create(self, model, messages, max_tokens=256, temperature=0.7, **kwargs):
        # Echo back the prompt content
        content = messages[0]["content"]
        return type("Resp", (), {"content": content})()


class MockClaudeClient:
    def __init__(self):
        self.messages = MockClaudeMessages()


class TestAnthropicNode(unittest.TestCase):
    def setUp(self):
        self.client = MockClaudeClient()
        self.node = AnthropicNode(
            name="test_node",
            client=self.client,
            prompt_template="Hello {name}",
        )

    def test_initialization(self):
        self.assertEqual(self.node.name, "test_node")
        self.assertEqual(self.node.model_name, "claude-3-opus-20240229")

    @pytest.mark.asyncio
    async def test_execute(self):
        result = await self.node.execute({"name": "Alice"})
        self.assertIn("Alice", result)


class TestAnthropicIntegration(unittest.TestCase):
    def setUp(self):
        self.client = MockClaudeClient()
        self.integration = AnthropicIntegration(self.client)

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
        results = await self.integration.execute({"who": "Bob"})
        self.assertIn("greet", results)
        self.assertIn("echo", results)


class TestAnthropicBatchProcessor(unittest.TestCase):
    def setUp(self):
        self.client = MockClaudeClient()
        self.processor = AnthropicBatchProcessor(
            self.client, batch_size=2, max_concurrent_batches=1
        )

    @pytest.mark.asyncio
    async def test_process(self):
        async def process_fn(prompt, client):
            return prompt.upper()

        prompts = ["a", "b", "c"]
        results = await self.processor.process(prompts, process_fn)
        self.assertEqual(results, ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
