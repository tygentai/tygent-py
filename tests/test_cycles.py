import asyncio
import unittest

from tygent.dag import DAG
from tygent.nodes import Node
from tygent.scheduler import FixedPointTermination, Scheduler


class StatefulLoopNode(Node):
    def __init__(self, name: str, events: list) -> None:
        super().__init__(name)
        self.events = events

    async def prepare(self, context):
        self.events.append(("prepare", self.name, context.iteration))

    async def finalize(self, context, result):
        self.events.append(("finalize", self.name, context.iteration, result))

    async def run(self, inputs, context):
        state = context.load_state({"value": 0})
        value = int(state.get("value", 0))
        target = int(inputs.get("target", 3))
        if value < target:
            value += 1
        new_state = {"value": value}
        context.save_state(new_state)
        return new_state


class MirrorNode(Node):
    def __init__(self, name: str, events: list) -> None:
        super().__init__(name)
        self.events = events

    async def prepare(self, context):
        self.events.append(("prepare", self.name, context.iteration))

    async def finalize(self, context, result):
        self.events.append(("finalize", self.name, context.iteration, result))

    async def run(self, inputs, context):
        payload = inputs.get("A", {"value": 0})
        context.save_state(payload)
        return payload


class TestCyclicExecution(unittest.TestCase):
    def test_fixed_point_loop_execution(self):
        events = []
        dag = DAG("loop")
        node_a = StatefulLoopNode("A", events)
        node_b = MirrorNode("B", events)
        dag.add_node(node_a)
        dag.add_node(node_b)
        dag.add_edge("A", "B")
        dag.add_edge("B", "A")

        scheduler = Scheduler(dag)
        scheduler.register_termination_policy(
            ["A", "B"], FixedPointTermination(max_iterations=5)
        )

        async def run():
            return await scheduler.execute({"target": 3})

        result = asyncio.run(run())
        self.assertEqual(result["results"]["A"]["value"], 3)
        self.assertEqual(result["results"]["B"]["value"], 3)

        # Session state should persist for both nodes
        self.assertEqual(scheduler.session_store.get_node_state("A"), {"value": 3})
        self.assertEqual(scheduler.session_store.get_node_state("B"), {"value": 3})

        # Lifecycle hooks invoked for each iteration
        prepare_events = [evt for evt in events if evt[0] == "prepare"]
        finalize_events = [evt for evt in events if evt[0] == "finalize"]
        self.assertTrue(prepare_events)
        self.assertEqual(len(prepare_events), len(finalize_events))
        # Final iteration should report iteration >= 1 for cyclic nodes
        self.assertTrue(any(evt[2] >= 1 for evt in prepare_events if evt[1] == "A"))

        # Re-running should reuse session state and stabilise in one iteration
        events.clear()
        result_again = asyncio.run(run())
        self.assertEqual(result_again["results"]["A"]["value"], 3)
        self.assertLessEqual(
            len([evt for evt in events if evt[0] == "prepare" and evt[1] == "A"]),
            len(prepare_events),
        )


if __name__ == "__main__":
    unittest.main()
