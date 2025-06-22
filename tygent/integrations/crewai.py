"""CrewAI Integration for Tygent.

This module exposes a lightweight integration with `crewai` that maps
`Task` objects to :class:`~tygent.nodes.BaseNode` instances so they can be
executed via Tygent's :class:`~tygent.scheduler.Scheduler`.

The previous implementation relied on a deprecated ``TygentAgent`` base
class.  This version mirrors the structure of the other integration
modules by using ``DAG`` and ``Scheduler`` directly.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

try:
    from crewai import Crew  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    Crew = None  # type: ignore

from ..dag import DAG
from ..nodes import BaseNode
from ..scheduler import Scheduler


class CrewAITaskNode(BaseNode):
    """A DAG node wrapping a CrewAI task."""

    def __init__(self, name: str, task: Any, agent: Any):
        super().__init__(name)
        self.task = task
        self.agent = agent

    async def execute(self, inputs: Dict[str, Any]) -> Any:  # noqa: D401
        """Execute the wrapped CrewAI task."""
        try:
            if hasattr(self.agent, "execute_task"):
                result = self.agent.execute_task(self.task, context=inputs)
            elif hasattr(self.task, "execute"):
                result = self.task.execute(inputs)
            else:
                result = None
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except Exception as exc:  # pragma: no cover - passthrough errors
            raise RuntimeError(f"Error running task {self.name}: {exc}") from exc


class CrewAIIntegration:
    """Build a DAG from a CrewAI ``Crew`` for optimised execution."""

    def __init__(self, crew: Any):
        if Crew is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "CrewAI is not installed. Install with: pip install crewai"
            )
        self.crew = crew
        self.dag = DAG("crewai_workflow")
        self.scheduler = Scheduler(self.dag)
        self._build_dag()

    # ------------------------------------------------------------------
    def _build_dag(self) -> None:
        tasks: List[Any] = getattr(self.crew, "tasks", [])
        id_map: Dict[int, str] = {}
        for idx, task in enumerate(tasks):
            name = getattr(task, "id", f"task_{idx}")
            agent = getattr(task, "agent", None)
            node = CrewAITaskNode(name, task, agent)
            self.dag.add_node(node)
            id_map[id(task)] = name
        for task in tasks:
            deps = getattr(task, "dependencies", [])
            for dep in deps or []:
                src = id_map.get(id(dep))
                tgt = id_map.get(id(task))
                if src and tgt:
                    self.dag.add_edge(src, tgt)

    # ------------------------------------------------------------------
    def optimize(self, options: Optional[Dict[str, Any]] = None) -> "CrewAIIntegration":
        """Apply simple optimisation options to the scheduler."""
        options = options or {}
        if "maxParallelCalls" in options:
            self.scheduler.max_parallel_nodes = options["maxParallelCalls"]
        if "maxExecutionTime" in options:
            self.scheduler.max_execution_time = options["maxExecutionTime"]
        if "priorityNodes" in options:
            self.scheduler.priority_nodes = options["priorityNodes"]
        return self

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the Crew via the scheduler."""
        return await self.scheduler.execute(self.dag, inputs)


# ---------------------------------------------------------------------------
# Convenience helpers


def accelerate_crew(crew: Any) -> CrewAIIntegration:
    """Return a :class:`CrewAIIntegration` for the given crew."""
    return CrewAIIntegration(crew)


def optimize_crew_workflow(crew: Any) -> Dict[str, Any]:
    """Simple static analysis of a CrewAI crew to highlight optimisation hints."""
    agents = getattr(crew, "agents", [])
    tasks = getattr(crew, "tasks", [])

    analysis = {
        "total_agents": len(agents),
        "total_tasks": len(tasks),
        "parallel_opportunities": 0,
        "sequential_bottlenecks": 0,
        "optimization_recommendations": [],
        "estimated_speedup": "1x",
    }

    independent = [t for t in tasks if not getattr(t, "dependencies", [])]
    analysis["parallel_opportunities"] = len(independent)
    if analysis["parallel_opportunities"] > 1:
        analysis["estimated_speedup"] = f"{min(len(independent), len(agents), 4)}x"
        analysis["optimization_recommendations"].append(
            f"Execute {len(independent)} independent tasks in parallel"
        )

    dependent = [t for t in tasks if getattr(t, "dependencies", [])]
    analysis["sequential_bottlenecks"] = len(dependent)
    if len(dependent) > 2:
        analysis["optimization_recommendations"].append(
            "Consider reducing task dependencies to enable more parallelism"
        )

    return analysis


def tygent_crew(crew: Any):
    """Decorator that executes the wrapped function via ``CrewAIIntegration``."""

    integration = CrewAIIntegration(crew)

    def decorator(func):
        async def wrapper(*args, **kwargs):
            inputs = args[0] if args else kwargs
            return await integration.execute(inputs)

        return wrapper

    return decorator

