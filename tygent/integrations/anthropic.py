"""
Anthropic Claude Integration for Tygent.

This module provides integration with Anthropic's Claude models for
optimized DAG execution and batch processing.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional

try:
    from anthropic import AsyncAnthropic
except Exception:  # pragma: no cover - anthropic may not be installed
    AsyncAnthropic = None  # type: ignore

from ..dag import DAG
from ..nodes import LLMNode
from ..scheduler import Scheduler


class AnthropicNode(LLMNode):
    """Node for executing Anthropic Claude API calls."""

    def __init__(
        self,
        name: str,
        client: Any,
        model: str = "claude-3-opus-20240229",
        prompt_template: str = "",
        dependencies: Optional[List[str]] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, model=client, prompt_template=prompt_template)
        self.client = client
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        if dependencies:
            self.dependencies = dependencies
        self.kwargs = kwargs

    async def execute(self, inputs: Dict[str, Any]) -> Any:
        prompt = self._format_prompt(inputs, {})
        try:
            if hasattr(self.client, "messages"):
                response = await self.client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    **self.kwargs,
                )
                if hasattr(response, "content"):
                    content = response.content
                    if isinstance(content, list):
                        return "".join(
                            part.text if hasattr(part, "text") else str(part)
                            for part in content
                        )
                    return content
                return response
            else:
                response = await self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    **self.kwargs,
                )
                return getattr(response, "completion", response)
        except Exception as e:  # pragma: no cover - runtime errors
            print(f"Error executing Anthropic node {self.name}: {e}")
            raise


class AnthropicIntegration:
    """Integration for Anthropic Claude models with Tygent's DAG system."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.dag = DAG(name="anthropic_dag")
        self.scheduler = Scheduler(self.dag)

    def add_node(
        self,
        name: str,
        prompt_template: str,
        dependencies: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AnthropicNode:
        node = AnthropicNode(
            name=name,
            client=self.client,
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


class AnthropicBatchProcessor:
    """Process lists of prompts with Anthropic's API in batches."""

    def __init__(
        self, client: Any, batch_size: int = 5, max_concurrent_batches: int = 2
    ) -> None:
        self.client = client
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches

    async def process(
        self, prompts: List[str], process_fn: Callable[[str, Any], Any]
    ) -> List[Any]:
        batches = [
            prompts[i : i + self.batch_size]
            for i in range(0, len(prompts), self.batch_size)
        ]
        results: List[Any] = []
        for i in range(0, len(batches), self.max_concurrent_batches):
            current = batches[i : i + self.max_concurrent_batches]
            tasks = [
                asyncio.create_task(process_fn(p, self.client))
                for batch in current
                for p in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(r for r in batch_results if not isinstance(r, Exception))
        return results


def patch() -> None:
    """Patch anthropic.AsyncAnthropic to run through Tygent's scheduler."""
    if AsyncAnthropic is None:
        return

    original_create = getattr(AsyncAnthropic, "messages", None)
    if original_create is None:
        return
    original_create = getattr(AsyncAnthropic.messages, "create", None)
    if original_create is None:
        return

    async def _node_fn(self, *args, **kwargs):
        return await original_create(self, *args, **kwargs)

    async def patched(self, *args, **kwargs):
        dag = DAG("anthropic_generate")
        node = LLMNode("call", model=self)
        node.execute = lambda inputs: _node_fn(self, *args, **kwargs)
        dag.add_node(node)
        scheduler = Scheduler(dag)
        result = await scheduler.execute({})
        return result["results"]["call"]

    setattr(AsyncAnthropic.messages, "_tygent_create", original_create)
    setattr(AsyncAnthropic.messages, "create", patched)
