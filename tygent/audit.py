from __future__ import annotations

"""Utilities for auditing plans and DAG execution."""

import json
from typing import Any, Dict, Iterable, List

from .dag import DAG
from .plan_parser import parse_plan, parse_plans


def audit_dag(dag: DAG) -> str:
    """Return a human readable summary of a :class:`~tygent.dag.DAG`."""

    lines = [f"DAG: {dag.name}"]
    for name, node in dag.nodes.items():
        deps = ", ".join(node.dependencies) if node.dependencies else "none"
        lines.append(f"- {name}: depends on {deps}")
    return "\n".join(lines)


def audit_plan(plan: Dict[str, Any]) -> str:
    """Generate an audit summary for a single plan dictionary."""

    dag, _ = parse_plan(plan)
    return audit_dag(dag)


def audit_plans(plans: Iterable[Dict[str, Any]]) -> str:
    """Generate an audit summary for multiple plan dictionaries."""

    dag, _ = parse_plans(plans)
    return audit_dag(dag)
