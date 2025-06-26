import asyncio
import time

from tygent.dag import DAG
from tygent.nodes import ToolNode
from tygent.scheduler import Scheduler


async def sleep_node(delay: float, inputs: dict) -> dict:
    await asyncio.sleep(delay)
    return {"delay": delay}


def build_dag() -> DAG:
    dag = DAG("example")
    dag.add_node(ToolNode("A", lambda i: sleep_node(0.1, i), latency_estimate=0.1))
    dag.add_node(ToolNode("B", lambda i: sleep_node(0.2, i), latency_estimate=0.2))
    dag.add_node(ToolNode("C", lambda i: sleep_node(0.3, i), latency_estimate=0.3))
    dag.add_node(ToolNode("D", lambda i: sleep_node(0.1, i), latency_estimate=0.1))
    dag.add_edge("A", "B")
    dag.add_edge("A", "C")
    dag.add_edge("B", "D")
    dag.add_edge("C", "D")
    return dag


async def run_sequential() -> float:
    dag = build_dag()
    order = dag.get_topological_order()
    outputs = {}
    start = time.perf_counter()
    for name in order:
        node = dag.getNode(name)
        deps = {d: outputs[d] for d in node.dependencies}
        result = await node.execute(deps)
        outputs[name] = result
    return time.perf_counter() - start


async def run_scheduler(max_parallel_nodes: int = 2) -> float:
    dag = build_dag()
    scheduler = Scheduler(dag)
    scheduler.configure(max_parallel_nodes=max_parallel_nodes)
    start = time.perf_counter()
    await scheduler.execute({})
    return time.perf_counter() - start


if __name__ == "__main__":
    seq = asyncio.run(run_sequential())
    par = asyncio.run(run_scheduler())
    print(f"Sequential time: {seq:.3f}s")
    print(f"Scheduler time: {par:.3f}s")
