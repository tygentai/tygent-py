"""Google ADK integration for Tygent.

This module provides a minimal integration with Google's
`google-adk` Agent Development Kit. It exposes a node that wraps
``google.adk.Runner`` so it can be executed as part of a Tygent DAG.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

try:
    from google.adk import Runner  # type: ignore
    from google.genai import types as genai_types  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Runner = None  # type: ignore
    genai_types = None  # type: ignore

from ..dag import DAG
from ..nodes import LLMNode
from ..scheduler import Scheduler


class GoogleADKNode(LLMNode):
    """Node for executing ``google.adk`` runners."""

    def __init__(
        self,
        name: str,
        runner: Any,
        prompt_template: str = "",
        dependencies: Optional[List[str]] = None,
        user_id: str = "user",
        session_id: str = "session",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, model=runner, prompt_template=prompt_template)
        self.runner = runner
        self.user_id = user_id
        self.session_id = session_id
        if dependencies:
            self.dependencies = dependencies
        self.kwargs = kwargs

    async def execute(self, inputs: Dict[str, Any]) -> Any:  # noqa: D401
        """Execute the wrapped runner."""
        prompt = self._format_prompt(inputs, {})
        if genai_types is not None:
            content = genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=prompt)],
            )
        else:
            content = prompt

        events = []
        async for event in self.runner.run_async(
            user_id=self.user_id,
            session_id=self.session_id,
            new_message=content,
            **self.kwargs,
        ):
            events.append(event)
        return events

    def _format_prompt(
        self, inputs: Dict[str, Any], node_outputs: Dict[str, Any]
    ) -> str:
        def _to_text(value: Any) -> Any:
            if isinstance(value, list) and value:
                evt = value[0]
                content = getattr(evt, "content", None)
                parts = getattr(content, "parts", None)
                if parts:
                    part = parts[0]
                    return getattr(part, "text", str(value))
            return value

        variables = {**inputs}
        for key, val in node_outputs.items():
            variables[key] = _to_text(val)

        try:
            return self.prompt_template.format(**variables)
        except KeyError as e:
            missing = str(e).strip("'")
            return self.prompt_template.replace(
                f"{{{missing}}}", f"[Missing: {missing}]"
            )


class GoogleADKIntegration:
    """Integration for Google ADK runners."""

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self.dag = DAG(name="google_adk_dag")
        self.scheduler = Scheduler(self.dag)

    def add_node(
        self,
        name: str,
        prompt_template: str,
        dependencies: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> GoogleADKNode:
        node = GoogleADKNode(
            name=name,
            runner=self.runner,
            prompt_template=prompt_template,
            dependencies=dependencies,
            **kwargs,
        )
        self.dag.add_node(node)
        return node

    def optimize(self, options: Dict[str, Any]) -> None:
        if "maxParallelCalls" in options:
            self.scheduler.max_parallel_nodes = options["maxParallelCalls"]
        if "maxExecutionTime" in options:
            self.scheduler.max_execution_time = options["maxExecutionTime"]
        if "priorityNodes" in options:
            self.scheduler.priority_nodes = options["priorityNodes"]

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return await self.scheduler.execute(inputs)


def patch() -> None:
    """Patch ``google.adk.Runner`` to run through Tygent's scheduler."""
    if Runner is None:
        return

    original = Runner.run_async

    async def patched(self: Any, *args: Any, **kwargs: Any):
        dag = DAG("google_adk_run")
        node = LLMNode("call", model=self)
        node.execute = lambda _: original(self, *args, **kwargs)
        dag.add_node(node)
        scheduler = Scheduler(dag)
        result = await scheduler.execute({})
        return result["results"]["call"]

    setattr(Runner, "_tygent_run_async", original)
    Runner.run_async = patched
