import asyncio
import unittest

from tygent.plan_parser import parse_plan
from tygent.scheduler import Scheduler


async def add_fn(inputs):
    return {"sum": inputs.get("a", 0) + inputs.get("b", 0)}


async def mult_fn(inputs):
    return {"product": inputs.get("sum", 0) * inputs.get("factor", 1)}


class TestPlanParser(unittest.TestCase):
    def test_parse_and_execute(self):
        plan = {
            "name": "math",
            "steps": [
                {"name": "add", "func": add_fn, "dependencies": [], "critical": True},
                {"name": "mult", "func": mult_fn, "dependencies": ["add"]},
            ],
        }
        dag, critical = parse_plan(plan)
        scheduler = Scheduler(dag)
        scheduler.priority_nodes = critical

        async def run():
            result = await scheduler.execute({"a": 2, "b": 3, "factor": 5})
            return result["results"]["mult"]["product"]

        value = asyncio.run(run())
        self.assertEqual(value, 25)
