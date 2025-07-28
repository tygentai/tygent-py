import asyncio
import unittest
from typing import Any

from tygent import accelerate


async def inc_fn(inputs):
    return {"a": inputs.get("x", 0) + 1}


async def double_fn(inputs):
    return {"b": inputs.get("a", 0) * 2}


class DummyFramework:
    def get_plan(self):
        return {
            "name": "dummy",
            "steps": [
                {"name": "inc", "func": inc_fn},
                {"name": "double", "func": double_fn, "dependencies": ["inc"]},
            ],
        }


class TestAccelerate(unittest.TestCase):
    def test_accelerate_plan(self):
        plan = DummyFramework().get_plan()
        accel = accelerate(plan)

        async def run():
            result = await accel.execute({"x": 2})
            return result["results"]["double"]["b"]

        value = asyncio.run(run())
        self.assertEqual(value, 6)

    def test_accelerate_framework(self):
        accel = accelerate(DummyFramework())

        async def run():
            result = await accel.execute({"x": 3})
            return result["results"]["double"]["b"]

        value = asyncio.run(run())
        self.assertEqual(value, 8)

    def test_accelerate_decorator(self):
        @accelerate
        async def add_one(x):
            return x + 1

        result = asyncio.run(add_one(4))
        self.assertEqual(result, 5)

    def test_accelerate_in_running_loop(self):
        @accelerate
        async def add_one(x):
            await asyncio.sleep(0)
            return x + 1

        async def run():
            coro = add_one(5)
            self.assertTrue(asyncio.iscoroutine(coro))
            return await coro

        value = asyncio.run(run())
        self.assertEqual(value, 6)

    def test_accelerate_google_adk_runner(self):
        from google.adk.agents.base_agent import BaseAgent
        from google.adk.events import Event
        from google.adk.runners import InMemoryRunner
        from google.genai import types

        class EchoAgent(BaseAgent):
            name: str = "echo"

            async def _run_async_impl(self, ctx) -> Any:  # type: ignore[override]
                text = ctx.user_content.parts[0].text if ctx.user_content else ""
                yield Event(
                    invocation_id=ctx.invocation_id,
                    author=self.name,
                    content=types.Content(role="model", parts=[types.Part(text=text)]),
                )

        from tygent.integrations.google_adk import GoogleADKIntegration

        runner = InMemoryRunner(EchoAgent())
        runner.session_service.create_session_sync(
            app_name=runner.app_name, user_id="user", session_id="session"
        )
        accel = accelerate(runner)
        self.assertIsInstance(accel, GoogleADKIntegration)

        accel.add_node("greet", "Hi {name}")

        async def run():
            result = await accel.execute({"name": "Foo"})
            return result["results"]["greet"][0].content.parts[0].text

        value = asyncio.run(run())
        self.assertEqual(value, "Hi Foo")


if __name__ == "__main__":
    unittest.main()
