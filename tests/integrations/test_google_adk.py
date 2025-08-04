"""Tests for Google ADK integration."""

import asyncio
import unittest
from typing import Any

import pytest
from google.adk.agents.base_agent import BaseAgent
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.genai import types

from tygent.integrations.google_adk import GoogleADKIntegration, GoogleADKNode


class EchoAgent(BaseAgent):
    name: str = "echo"

    async def _run_async_impl(self, ctx) -> Any:  # type: ignore[override]
        text = ""
        if ctx.user_content and ctx.user_content.parts:
            text = ctx.user_content.parts[0].text
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=text)]),
        )


class TestGoogleADKNode(unittest.TestCase):
    def setUp(self):
        self.runner = InMemoryRunner(EchoAgent())
        self.runner.session_service.create_session_sync(
            app_name=self.runner.app_name, user_id="user", session_id="session"
        )
        self.node = GoogleADKNode(
            name="test",
            runner=self.runner,
            prompt_template="Hello {who}",
        )

    def test_init(self):
        self.assertEqual(self.node.name, "test")
        self.assertEqual(self.node.prompt_template, "Hello {who}")

    @pytest.mark.asyncio
    async def test_execute(self):
        result = await self.node.execute({"who": "Ada"})
        self.assertEqual(
            result[0].content.parts[0].text,
            "Hello Ada",
        )


class TestGoogleADKIntegration(unittest.TestCase):
    def setUp(self):
        self.runner = InMemoryRunner(EchoAgent())
        self.runner.session_service.create_session_sync(
            app_name=self.runner.app_name, user_id="user", session_id="session"
        )
        self.integration = GoogleADKIntegration(self.runner)

    def test_add_node(self):
        node = self.integration.add_node("greet", "Hi {name}")
        self.assertEqual(node.name, "greet")

    @pytest.mark.asyncio
    async def test_execute(self):
        self.integration.add_node("greet", "Hi {name}")
        results = await self.integration.execute({"name": "Bob"})
        self.assertEqual(
            results["greet"][0].content.parts[0].text,
            "Hi Bob",
        )

    @pytest.mark.asyncio
    async def test_dependency_prompt(self):
        self.integration.add_node("first", "hello")
        self.integration.add_node(
            "second",
            "second uses {first}",
            dependencies=["first"],
        )
        results = await self.integration.execute({})
        self.assertEqual(
            results["second"][0].content.parts[0].text,
            "second uses hello",
        )


if __name__ == "__main__":
    unittest.main()
