import asyncio
import unittest

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


if __name__ == "__main__":
    unittest.main()
