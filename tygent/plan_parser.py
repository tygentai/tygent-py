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
    cycle_groups: Dict[str, Dict[str, Any]] = {}

    # First create nodes
    for step in steps:
        node_name = step["name"]
        func = step.get("func", lambda _inputs: None)
        token_cost = (
            int(step.get("token_cost", 0)) if step.get("token_cost") is not None else 0
        )
        latency_estimate = step.get("latency_estimate")
        metadata = step.get("metadata")
        interactive = bool(step.get("interactive"))
        session_cfg = step.get("session") or {}
        if metadata is None:
            metadata = {}
        else:
            metadata = dict(metadata)
        if interactive:
            metadata["interactive"] = True
        if session_cfg:
            metadata["session"] = dict(session_cfg)
        node = ToolNode(
            node_name,
            func,
            token_cost=token_cost,
            latency_estimate=latency_estimate,
            metadata=metadata,
        )
        dag.add_node(node)
        if step.get("critical"):
            critical.append(node_name)

        loop_info = step.get("cycle") or step.get("loop")
        if loop_info:
            group = loop_info.get("group") or node_name
            spec = cycle_groups.setdefault(
                group,
                {"nodes": [], "termination": loop_info.get("termination")},
            )
            spec["nodes"].append(node_name)
            if loop_info.get("termination"):
                spec["termination"] = loop_info["termination"]

    # Then add edges
    for step in steps:
        node_name = step["name"]
        for dep in step.get("dependencies", []):
            dag.add_edge(dep, node_name)

    if cycle_groups:
        cycle_policies: Dict[str, Dict[str, Any]] = {}
        for spec in cycle_groups.values():
            nodes = tuple(sorted(spec["nodes"]))
            termination = spec.get("termination") or {}
            cycle_policies[nodes] = termination
        dag.metadata["cycle_policies"] = cycle_policies

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

        if getattr(dag, "metadata", None):
            source_cycle = dag.metadata.get("cycle_policies")
            if source_cycle:
                merged_cycle = merged.metadata.setdefault("cycle_policies", {})
                merged_cycle.update(source_cycle)

        critical_all.extend(critical)

    return merged, critical_all
