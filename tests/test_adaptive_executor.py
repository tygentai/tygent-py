import asyncio
import unittest

from tygent.adaptive_executor import AdaptiveExecutor, RewriteRule
from tygent.dag import DAG
from tygent.nodes import ToolNode


class TestAdaptiveExecutor(unittest.TestCase):
    def test_dag_copy_not_modified(self):
        async def base_fn(inputs):
            return {"x": 1}

        async def new_fn(inputs):
            return {"y": inputs.get("x", 0) + 1}

        dag = DAG("base")
        dag.add_node(ToolNode("base", base_fn))

        def trigger(state):
            return True

        def action(current_dag, state):
            new_dag = current_dag.copy()
            new_dag.add_node(ToolNode("new", new_fn))
            new_dag.add_edge("base", "new")
            return new_dag

        rule = RewriteRule(trigger, action, name="add_new")
        executor = AdaptiveExecutor(dag, [rule], max_modifications=1)

        async def run_test():
            result = await executor.execute({})
            self.assertEqual(len(dag.nodes), 1)  # original DAG unchanged
            self.assertIn("new", result["final_dag"].nodes)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
