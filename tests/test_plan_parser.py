import asyncio
import unittest

from tygent.plan_parser import parse_plan, parse_plans
from tygent.scheduler import Scheduler


async def add_fn(inputs):
    return {"sum": inputs.get("a", 0) + inputs.get("b", 0)}


async def mult_fn(inputs):
    return {"product": inputs.get("sum", 0) * inputs.get("factor", 1)}


async def inc_fn(inputs):
    return {"num": inputs.get("x", 0) + 1}


async def double_fn(inputs):
    return {"double": inputs.get("num", 0) * 2}


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

    def test_parse_multiple_plans(self):
        plan1 = {
            "name": "math",
            "steps": [
                {"name": "add", "func": add_fn},
                {"name": "mult", "func": mult_fn, "dependencies": ["add"]},
            ],
        }

        plan2 = {
            "name": "proc",
            "steps": [
                {"name": "inc", "func": inc_fn},
                {"name": "double", "func": double_fn, "dependencies": ["inc"]},
            ],
        }

        dag, critical = parse_plans([plan1, plan2])
        scheduler = Scheduler(dag)
        scheduler.priority_nodes = critical

        async def run():
            result = await scheduler.execute({"a": 2, "b": 3, "factor": 5, "x": 4})
            return (
                result["results"]["mult"]["product"],
                result["results"]["double"]["double"],
            )

        mult_val, double_val = asyncio.run(run())
        self.assertEqual(mult_val, 25)
        self.assertEqual(double_val, 10)
