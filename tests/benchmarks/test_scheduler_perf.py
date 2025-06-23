import asyncio
import time
import unittest

from tygent.dag import DAG
from tygent.nodes import ToolNode
from tygent.scheduler import Scheduler


async def sleep_fn(delay: float, inputs):
    await asyncio.sleep(delay)
    return {"delay": delay}


async def sequential_execute(dag: DAG) -> float:
    order = dag.get_topological_order()
    start = time.perf_counter()
    outputs = {}
    for name in order:
        node = dag.getNode(name)
        deps = {dep: outputs[dep] for dep in node.dependencies}
        mapped = {}
        for dep, out in deps.items():
            if dep in dag.edge_mappings and name in dag.edge_mappings[dep]:
                for src, tgt in dag.edge_mappings[dep][name].items():
                    if src in out:
                        mapped[tgt] = out[src]
        inputs = mapped
        res = await node.execute(inputs)
        outputs[name] = res
    time_taken = time.perf_counter() - start
    return time_taken


async def scheduler_execute(dag: DAG, **cfg) -> float:
    scheduler = Scheduler(dag)
    scheduler.configure(**cfg)
    start = time.perf_counter()
    await scheduler.execute({})
    return time.perf_counter() - start


class TestSchedulerBenchmarks(unittest.TestCase):
    def _build_dag(self) -> DAG:
        dag = DAG("bench")
        dag.add_node(ToolNode("A", lambda i: sleep_fn(0.1, i), latency_estimate=0.1))
        dag.add_node(ToolNode("B", lambda i: sleep_fn(0.2, i), latency_estimate=0.2))
        dag.add_node(ToolNode("C", lambda i: sleep_fn(0.3, i), latency_estimate=0.3))
        dag.add_node(ToolNode("D", lambda i: sleep_fn(0.1, i), latency_estimate=0.1))
        dag.add_edge("A", "B")
        dag.add_edge("A", "C")
        dag.add_edge("B", "D")
        dag.add_edge("C", "D")
        return dag

    def test_scheduler_vs_sequential(self):
        dag = self._build_dag()
        seq = asyncio.run(sequential_execute(dag))
        par = asyncio.run(scheduler_execute(dag, max_parallel_nodes=2))
        self.assertLess(par, seq)

    def test_latency_aware_scheduler(self):
        dag = self._build_dag()
        normal = asyncio.run(scheduler_execute(dag, max_parallel_nodes=1))
        latency = asyncio.run(
            scheduler_execute(
                dag,
                max_parallel_nodes=2,
                token_budget=10,
                requests_per_minute=100,
            )
        )
        self.assertLess(latency, normal)


if __name__ == "__main__":
    unittest.main()
