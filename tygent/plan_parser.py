"""Plan parser for generating DAGs from structured plans."""

from typing import Any, Dict, List, Tuple

from .dag import DAG
from .nodes import ToolNode


def parse_plan(plan: Dict[str, Any]) -> Tuple[DAG, List[str]]:
    """Parse a plan dictionary into a :class:`~tygent.dag.DAG`.

    The plan format is expected to be::

        {
            "name": "dag_name",
            "steps": [
                {
                    "name": "step1",
                    "func": callable,
                    "dependencies": ["other_step"],
                    "critical": True
                },
                ...
            ]
        }

    ``critical`` steps are returned so the scheduler can prioritise them.

    Parameters
    ----------
    plan:
        Dictionary describing the plan.

    Returns
    -------
    Tuple[DAG, List[str]]
        The constructed DAG and list of critical node names.
    """

    dag_name = plan.get("name", "plan_dag")
    dag = DAG(dag_name)
    critical: List[str] = []

    steps: List[Dict[str, Any]] = plan.get("steps", [])

    # First create nodes
    for step in steps:
        node_name = step["name"]
        func = step.get("func", lambda _inputs: None)
        node = ToolNode(node_name, func)
        dag.add_node(node)
        if step.get("critical"):
            critical.append(node_name)

    # Then add edges
    for step in steps:
        node_name = step["name"]
        for dep in step.get("dependencies", []):
            dag.add_edge(dep, node_name)

    return dag, critical
