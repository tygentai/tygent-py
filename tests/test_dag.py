"""
Tests for the DAG module.
"""

import asyncio
import os
import sys
import unittest
from typing import List

# Add the parent directory to the path so we can import tygent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tygent import Scheduler
from tygent.dag import DAG
from tygent.nodes import LLMNode, ToolNode


class TestDAG(unittest.TestCase):
    """Test the DAG implementation."""

    def test_dag_creation(self):
        """Test that a DAG can be created."""
        dag = DAG("test_dag")
        self.assertEqual(dag.name, "test_dag")
        self.assertEqual(len(dag.nodes), 0)
        self.assertEqual(len(dag.edges), 0)

    def test_add_node(self):
        """Test that nodes can be added to a DAG."""
        dag = DAG("test_dag")

        async def test_tool(inputs):
            return {"result": "test"}

        tool_node = ToolNode("test_tool", test_tool)
        dag.add_node(tool_node)

        self.assertEqual(len(dag.nodes), 1)
        self.assertIn("test_tool", dag.nodes)
        self.assertEqual(dag.nodes["test_tool"], tool_node)

    def test_add_edge(self):
        """Test that edges can be added between nodes."""
        dag = DAG("test_dag")

        async def tool1(inputs):
            return {"data": "from_tool1"}

        async def tool2(inputs):
            return {"data": f"processed_{inputs.get('data', '')}"}

        dag.add_node(ToolNode("tool1", tool1))
        dag.add_node(ToolNode("tool2", tool2))

        dag.add_edge("tool1", "tool2", {"data": "data"})

        self.assertEqual(len(dag.edges), 1)
        self.assertIn("tool1", dag.edges)
        self.assertEqual(dag.edges["tool1"], ["tool2"])
        self.assertEqual(dag.edge_mappings["tool1"]["tool2"], {"data": "data"})

    def test_topological_order(self):
        """Test that topological ordering works properly."""
        dag = DAG("test_dag")

        async def dummy_tool(inputs):
            return {}

        # Create a simple diamond-shaped DAG
        #    A
        #   / \
        #  B   C
        #   \ /
        #    D

        for node_id in ["A", "B", "C", "D"]:
            dag.add_node(ToolNode(node_id, dummy_tool))

        dag.add_edge("A", "B")
        dag.add_edge("A", "C")
        dag.add_edge("B", "D")
        dag.add_edge("C", "D")

        order = dag.get_topological_order()

        # Verify that A comes before B and C, and B and C come before D
        self.assertTrue(order.index("A") < order.index("B"))
        self.assertTrue(order.index("A") < order.index("C"))
        self.assertTrue(order.index("B") < order.index("D"))
        self.assertTrue(order.index("C") < order.index("D"))


class TestExecution(unittest.TestCase):
    """Test the execution of DAGs."""

    def test_execution(self):
        """Test that a simple DAG can be executed successfully."""
        from tygent import Scheduler

        async def simple_add(inputs):
            a = inputs.get("a", 0)
            b = inputs.get("b", 0)
            return {"sum": a + b}

        async def simple_multiply(inputs):
            sum_value = inputs.get("sum", 0)
            factor = inputs.get("factor", 1)
            return {"product": sum_value * factor}

        async def run_test():
            dag = DAG("math_dag")

            dag.add_node(ToolNode("add", simple_add))
            dag.add_node(ToolNode("multiply", simple_multiply))

            dag.add_edge("add", "multiply", {"sum": "sum"})

            scheduler = Scheduler(dag)

            # Execute with inputs a=2, b=3, factor=5, expecting product=25
            result = await scheduler.execute({"a": 2, "b": 3, "factor": 5})

            self.assertIn("results", result)
            self.assertIn("multiply", result["results"])
            self.assertEqual(result["results"]["multiply"]["product"], 25)

        asyncio.run(run_test())

    def test_dag_execute_helper(self):
        """Test the DAG.execute convenience method."""

        async def foo(inputs):
            return {"out": inputs.get("val", 0) + 1}

        async def run_test():
            dag = DAG("helper")
            dag.add_node(ToolNode("foo", foo))
            result = await dag.execute({"val": 1})
            self.assertEqual(result["results"]["foo"]["out"], 2)

        asyncio.run(run_test())

    def test_mixed_llm_and_tool_nodes(self):
        """Ensure DAG executes mixed LLM and tool nodes in dependency order."""

        class DummyLLMNode(LLMNode):
            async def execute(self, inputs):
                topic = inputs.get("topic", "")
                return {"text": f"Info about {topic}"}

        async def process(inputs):
            return {"result": inputs.get("text", "").upper()}

        async def run_test():
            dag = DAG("mixed")
            dag.add_node(DummyLLMNode("llm"))
            dag.add_node(ToolNode("tool", process))
            dag.add_edge("llm", "tool", {"text": "text"})

            scheduler = Scheduler(dag)
            result = await scheduler.execute({"topic": "AI"})

            self.assertEqual(result["results"]["tool"]["result"], "INFO ABOUT AI")

        asyncio.run(run_test())

    def test_compute_critical_path_and_scheduler(self):
        """Nodes with longer downstream latency should be prioritized."""

        executed: List[str] = []

        async def a_fn(inputs):
            executed.append("A")
            return {"A": True}

        async def b_fn(inputs):
            executed.append("B")
            return {"B": True}

        async def c_fn(inputs):
            executed.append("C")
            return {"C": True}

        async def d_fn(inputs):
            executed.append("D")
            return {"D": True}

        dag = DAG("cp")
        dag.add_node(ToolNode("A", a_fn, latency_estimate=1))
        dag.add_node(ToolNode("B", b_fn, latency_estimate=3))
        dag.add_node(ToolNode("C", c_fn, latency_estimate=2))
        dag.add_node(ToolNode("D", d_fn, latency_estimate=1))
        dag.add_edge("A", "B")
        dag.add_edge("A", "C")
        dag.add_edge("B", "D")
        dag.add_edge("C", "D")

        cp = dag.compute_critical_path()
        self.assertEqual(cp["D"], 1)
        self.assertEqual(cp["B"], 4)
        self.assertEqual(cp["C"], 3)
        self.assertEqual(cp["A"], 5)

        async def run_sched():
            scheduler = Scheduler(dag)
            scheduler.configure(max_parallel_nodes=1)
            await scheduler.execute({})

        asyncio.run(run_sched())
        self.assertEqual(executed, ["A", "B", "C", "D"])

    def test_latency_model_influences_schedule(self):
        """Scheduler should use latency models when no fixed estimate is set."""

        executed: List[str] = []

        async def a_fn(inputs):
            executed.append("A")
            return {"A": True}

        async def b_fn(inputs):
            executed.append("B")
            return {"B": True}

        async def c_fn(inputs):
            executed.append("C")
            return {"C": True}

        async def d_fn(inputs):
            executed.append("D")
            return {"D": True}

        dag = DAG("model")

        dag.add_node(ToolNode("A", a_fn, latency_model=lambda n: 1.0))
        dag.add_node(ToolNode("B", b_fn, latency_model=lambda n: 3.0))
        dag.add_node(ToolNode("C", c_fn, latency_model=lambda n: 2.0))
        dag.add_node(ToolNode("D", d_fn, latency_model=lambda n: 1.0))

        dag.add_edge("A", "B")
        dag.add_edge("A", "C")
        dag.add_edge("B", "D")
        dag.add_edge("C", "D")

        cp = dag.compute_critical_path()
        self.assertEqual(cp["B"], 4.0)
        self.assertEqual(cp["C"], 3.0)

        async def run_sched():
            scheduler = Scheduler(dag)
            scheduler.configure(max_parallel_nodes=1)
            await scheduler.execute({})

        asyncio.run(run_sched())
        self.assertEqual(executed, ["A", "B", "C", "D"])


class TestSchedulerConstraints(unittest.TestCase):
    """Tests for token budgets and rate limiting."""

    def test_token_budget_exhaustion(self):
        async def fn(inputs):
            return {}

        dag = DAG("budget")
        dag.add_node(ToolNode("a", fn, token_cost=3))
        dag.add_node(ToolNode("b", fn, token_cost=3))
        dag.add_edge("a", "b")

        scheduler = Scheduler(dag)
        scheduler.configure(token_budget=5, max_parallel_nodes=1)

        async def run():
            await scheduler.execute({})

        with self.assertRaises(RuntimeError):
            asyncio.run(run())

    def test_rate_limiting(self):
        async def fn(inputs):
            return {}

        dag = DAG("rate")
        dag.add_node(ToolNode("a", fn))
        dag.add_node(ToolNode("b", fn))
        dag.add_node(ToolNode("c", fn))
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")

        scheduler = Scheduler(dag)
        scheduler.configure(max_parallel_nodes=1, requests_per_minute=1)

        sleeps: List[float] = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        async def run():
            original_sleep = asyncio.sleep
            asyncio.sleep = fake_sleep
            try:
                await scheduler.execute({})
            finally:
                asyncio.sleep = original_sleep

        asyncio.run(run())
        self.assertTrue(len(sleeps) >= 1)


if __name__ == "__main__":
    unittest.main()
