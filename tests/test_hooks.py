import asyncio
import unittest

from tygent import DAG, Scheduler, ToolNode


class TestHooks(unittest.TestCase):
    def test_stop_via_hook(self):
        events = []

        async def hook(stage, node, inputs, output, scheduler):
            events.append((stage, node.name))
            if stage == "after_execute" and node.name == "a":
                return False

        async def a_fn(inputs):
            return {"val": 1}

        async def b_fn(inputs):
            return {"done": True}

        dag = DAG("hooks")
        dag.add_node(ToolNode("a", a_fn))
        dag.add_node(ToolNode("b", b_fn))
        dag.add_edge("a", "b", {"val": "val"})

        scheduler = Scheduler(dag, hooks=[hook])

        result = asyncio.run(scheduler.execute({}))
        self.assertIn("a", result["results"])
        self.assertNotIn("b", result["results"])
        self.assertIn(("audit", "a"), events)


if __name__ == "__main__":
    unittest.main()
