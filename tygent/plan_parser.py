"""Plan parser for generating DAGs from structured plans."""

from typing import Any, Dict, Iterable, List, Tuple

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


def parse_plans(plans: Iterable[Dict[str, Any]]) -> Tuple[DAG, List[str]]:
    """Combine multiple plan dictionaries into a single DAG.

    Parameters
    ----------
    plans:
        Iterable of plan dictionaries in the same format accepted by
        :func:`parse_plan`.

    Returns
    -------
    Tuple[DAG, List[str]]
        Merged DAG and combined list of critical node names.
    """

    merged = DAG("combined_plan")
    critical_all: List[str] = []

    for plan in plans:
        dag, critical = parse_plan(plan)

        # Add nodes
        for node in dag.nodes.values():
            if merged.hasNode(node.name):
                raise ValueError(f"Duplicate node name {node.name} in plans")
            merged.add_node(node)

        # Add edges with metadata
        for src, targets in dag.edges.items():
            for tgt in targets:
                meta = dag.edge_mappings.get(src, {}).get(tgt)
                merged.add_edge(src, tgt, meta)

        critical_all.extend(critical)

    return merged, critical_all
