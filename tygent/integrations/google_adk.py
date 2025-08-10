"""Google ADK integration for Tygent.

This module provides a minimal integration with Google's
`google-adk` Agent Development Kit. It exposes a node that wraps
``google.adk.Runner`` so it can be executed as part of a Tygent DAG.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

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
        log_usage: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, model=runner, prompt_template=prompt_template)
        self.runner = runner
        self.user_id = user_id
        self.session_id = session_id
        self.log_usage = log_usage
        if dependencies:
            self.dependencies = dependencies
        self.kwargs = kwargs

    async def execute(self, inputs: Dict[str, Any]) -> Any:  # noqa: D401
        """Execute the wrapped runner."""
        # All inputs come from dependency outputs, so format them as variables
        prompt = self._format_prompt({}, inputs)
        if genai_types is not None:
            content = genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=prompt)],
            )
        else:
            content = prompt

        events = []
        usage = None
        start = time.time()
        async for event in self.runner.run_async(
            user_id=self.user_id,
            session_id=self.session_id,
            new_message=content,
            **self.kwargs,
        ):
            events.append(event)
            if usage is None:
                usage = getattr(event, "usage_metadata", None)
        duration = time.time() - start
        if self.log_usage:
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_token_count", None)
                response_tokens = getattr(usage, "candidates_token_count", None)
                logger.info(
                    "%s executed in %.2fs: %s input tokens, %s output tokens",
                    self.name,
                    duration,
                    prompt_tokens,
                    response_tokens,
                )
            else:
                logger.info(
                    "%s executed in %.2fs: token counts unavailable",
                    self.name,
                    duration,
                )
        # Wrap the result so dependency names map to unique keys
        return {self.name: events}

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
        log_usage: bool = False,
        **kwargs: Any,
    ) -> GoogleADKNode:
        node = GoogleADKNode(
            name=name,
            runner=self.runner,
            prompt_template=prompt_template,
            dependencies=dependencies,
            log_usage=log_usage,
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
        """Execute the DAG and return flattened node outputs."""
        raw = await self.scheduler.execute(inputs)
        outputs: Dict[str, Any] = {}
        for name, value in raw.get("results", {}).items():
            if isinstance(value, dict) and name in value:
                outputs[name] = value[name]
            else:
                outputs[name] = value
        return outputs


def patch() -> None:
    """Patch ``google.adk.Runner`` to run through Tygent's scheduler."""
    if Runner is None:
        return

    original = Runner.run_async

    async def patched(self: Any, *args: Any, **kwargs: Any):
        dag = DAG("google_adk_run")
        node = LLMNode("call", model=self)

        async def run(_):
            events = []
            async for evt in original(self, *args, **kwargs):
                events.append(evt)
            return events

        node.execute = run
        dag.add_node(node)
        scheduler = Scheduler(dag)
        result = await scheduler.execute({})
        for event in result["results"]["call"]:
            yield event

    setattr(Runner, "_tygent_run_async", original)
    Runner.run_async = patched
