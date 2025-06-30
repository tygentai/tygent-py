"""
HuggingFace Hub Integration for Tygent.

This module allows Tygent DAGs to use models hosted on HuggingFace Hub,
with optional asynchronous generation and streaming support.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional

from ..dag import DAG
from ..nodes import LLMNode
from ..scheduler import Scheduler


class HuggingFaceNode(LLMNode):
    """Node for executing HuggingFace Hub models."""

    def __init__(
        self,
        name: str,
        model: Any,
        prompt_template: str = "",
        dependencies: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, model=model, prompt_template=prompt_template)
        self.model = model
        self.stream = stream
        if dependencies:
            self.dependencies = dependencies
        self.kwargs = kwargs

    async def execute(self, inputs: Dict[str, Any]) -> Any:
        prompt = self._format_prompt(inputs, {})
        try:
            if self.stream and hasattr(self.model, "astream"):
                chunks: List[str] = []
                async for token in self.model.astream(prompt, **self.kwargs):
                    chunks.append(str(token))
                return "".join(chunks)
            if hasattr(self.model, "__call__"):
                if asyncio.iscoroutinefunction(self.model.__call__):
                    return await self.model(prompt, **self.kwargs)
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, self.model, prompt, **self.kwargs
                )
            if hasattr(self.model, "generate"):
                if asyncio.iscoroutinefunction(self.model.generate):
                    return await self.model.generate(prompt, **self.kwargs)
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, self.model.generate, prompt, **self.kwargs
                )
            raise ValueError("Unsupported HuggingFace model")
        except Exception as e:  # pragma: no cover - runtime errors
            print(f"Error executing HuggingFace node {self.name}: {e}")
            raise


class HuggingFaceIntegration:
    """Integration for HuggingFace Hub models with Tygent's DAG system."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.dag = DAG(name="huggingface_dag")
        self.scheduler = Scheduler(self.dag)

    def add_node(
        self,
        name: str,
        prompt_template: str,
        dependencies: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> HuggingFaceNode:
        node = HuggingFaceNode(
            name=name,
            model=self.model,
            prompt_template=prompt_template,
            dependencies=dependencies,
            stream=stream,
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


class HuggingFaceBatchProcessor:
    """Process prompts with HuggingFace models in batches."""

    def __init__(
        self, model: Any, batch_size: int = 8, max_concurrent_batches: int = 2
    ) -> None:
        self.model = model
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
                asyncio.create_task(process_fn(p, self.model))
                for batch in current
                for p in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(r for r in batch_results if not isinstance(r, Exception))
        return results


def patch() -> None:
    """Patch transformers.Pipeline to run through Tygent's scheduler."""
    try:
        from transformers import Pipeline  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return

    original_call = getattr(Pipeline, "__call__", None)
    if original_call is None:
        return

    def patched(self, *args, **kwargs):
        async def _node_fn(_):
            if asyncio.iscoroutinefunction(original_call):
                return await original_call(self, *args, **kwargs)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, original_call, self, *args, **kwargs
            )

        async def run():
            dag = DAG("hf_pipeline")
            node = LLMNode("call", model=self)
            node.execute = _node_fn
            dag.add_node(node)
            scheduler = Scheduler(dag)
            result = await scheduler.execute({})
            return result["results"]["call"]

        return asyncio.run(run())

    setattr(Pipeline, "_tygent_call", original_call)
    setattr(Pipeline, "__call__", patched)
