import asyncio
import time
import unittest

import pytest

from tygent.integrations.google_ai import GoogleAIIntegration, GoogleAINode
from tygent.integrations.huggingface import HuggingFaceIntegration, HuggingFaceNode


class DelayModel:
    async def __call__(self, prompt, **kwargs):
        await asyncio.sleep(0.1)
        return f"out:{prompt}"

    async def generateContent(self, prompt, **kwargs):
        await asyncio.sleep(0.1)

        class Resp:
            def __init__(self, text):
                self.text_content = text

            @property
            def response(self):
                return self

            def text(self):
                return self.text_content

        return Resp(f"resp:{prompt}")


class TestIntegrationPerformance(unittest.TestCase):
    @pytest.mark.asyncio
    async def test_huggingface_parallel(self):
        model = DelayModel()
        integ = HuggingFaceIntegration(model)
        integ.add_node("n1", "a {x}")
        integ.add_node("n2", "b {x}")

        start = time.perf_counter()
        await integ.execute({"x": "y"})
        parallel_time = time.perf_counter() - start
        self.assertLess(parallel_time, 0.2)

    @pytest.mark.asyncio
    async def test_google_ai_parallel(self):
        model = DelayModel()
        integ = GoogleAIIntegration(model)
        integ.addNode("g1", "hi {x}")
        integ.addNode("g2", "bye {x}")

        start = time.perf_counter()
        await integ.execute({"x": "z"})
        parallel_time = time.perf_counter() - start
        self.assertLess(parallel_time, 0.2)
