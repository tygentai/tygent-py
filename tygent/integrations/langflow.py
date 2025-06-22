"""Langflow Integration for Tygent.

This module mirrors the other integration files by providing a small
wrapper around a Langflow workflow.  The workflow's nodes are converted
into :class:`~tygent.nodes.BaseNode` objects which are then executed via
Tygent's :class:`~tygent.scheduler.Scheduler`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

try:
    import requests  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    requests = None

from ..dag import DAG
from ..nodes import BaseNode
from ..scheduler import Scheduler


class LangflowNode(BaseNode):
    """A node that forwards execution to a Langflow server."""

    def __init__(self, name: str, node_id: str, flow_id: str, base_url: str):
        super().__init__(name)
        self.node_id = node_id
        self.flow_id = flow_id
        self.base_url = base_url.rstrip("/")

    async def execute(self, inputs: Dict[str, Any]) -> Any:  # noqa: D401
        if requests is None:
            raise RuntimeError("requests module not available")
        resp = requests.post(
            f"{self.base_url}/api/v1/process/{self.flow_id}/node/{self.node_id}",
            json={"inputs": inputs},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


class LangflowIntegration:
    """Execute Langflow workflows using a DAG and scheduler."""

    def __init__(self, flow_data: Dict[str, Any], base_url: str = "http://localhost:7860"):
        self.flow_data = flow_data
        self.flow_id = flow_data.get("id", "flow")
        self.base_url = base_url.rstrip("/")
        self.dag = DAG(f"langflow_{self.flow_id}")
        self.scheduler = Scheduler(self.dag)
        self._build_dag()

    # ------------------------------------------------------------------
    def _build_dag(self) -> None:
        nodes = self.flow_data.get("nodes", [])
        edges = self.flow_data.get("edges", [])

        for node in nodes:
            node_id = node.get("id")
            dag_node = LangflowNode(node_id, node_id, self.flow_id, self.base_url)
            self.dag.add_node(dag_node)

        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            if src and tgt:
                self.dag.add_edge(src, tgt)

    # ------------------------------------------------------------------
    def optimize(self, options: Optional[Dict[str, Any]] = None) -> "LangflowIntegration":
        options = options or {}
        if "maxParallelCalls" in options:
            self.scheduler.max_parallel_nodes = options["maxParallelCalls"]
        if "maxExecutionTime" in options:
            self.scheduler.max_execution_time = options["maxExecutionTime"]
        if "priorityNodes" in options:
            self.scheduler.priority_nodes = options["priorityNodes"]
        return self

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return await self.scheduler.execute(self.dag, inputs)


# ---------------------------------------------------------------------------
# Convenience helpers


def accelerate_langflow_flow(flow_data: Dict[str, Any], base_url: str = "http://localhost:7860") -> LangflowIntegration:
    """Return a :class:`LangflowIntegration` for the supplied flow."""
    return LangflowIntegration(flow_data, base_url)


def optimize_langflow_workflow(workflow_path: str) -> Dict[str, Any]:
    """Static analysis of a Langflow JSON workflow file."""
    with open(workflow_path, "r") as f:
        flow = json.load(f)

    nodes = flow.get("nodes", [])
    edges = flow.get("edges", [])

    analysis = {
        "total_nodes": len(nodes),
        "total_connections": len(edges),
        "parallel_opportunities": 0,
        "sequential_bottlenecks": 0,
        "optimization_recommendations": [],
        "estimated_speedup": "1x",
    }

    deps: Dict[str, List[str]] = {}
    for edge in edges:
        deps.setdefault(edge.get("target"), []).append(edge.get("source"))

    independent = [n["id"] for n in nodes if n.get("id") not in deps]
    analysis["parallel_opportunities"] = len(independent)
    if analysis["parallel_opportunities"] > 1:
        analysis["estimated_speedup"] = f"{min(len(independent),4)}x"
        analysis["optimization_recommendations"].append(
            "Enable parallel execution for independent nodes"
        )

    if deps:
        max_deps = max(len(d) for d in deps.values())
        if max_deps > 3:
            analysis["sequential_bottlenecks"] = max_deps
            analysis["optimization_recommendations"].append(
                "Consider breaking down complex dependency chains"
            )

    return analysis


def tygent_langflow(flow_data: Dict[str, Any], base_url: str = "http://localhost:7860"):
    """Decorator that executes the wrapped function via ``LangflowIntegration``."""

    integration = LangflowIntegration(flow_data, base_url)

    def decorator(func):
        async def wrapper(*args, **kwargs):
            inputs = args[0] if args else kwargs
            return await integration.execute(inputs)

        return wrapper

    return decorator

