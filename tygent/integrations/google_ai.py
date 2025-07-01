"""
Google AI integration for Tygent.

This module provides integration with Google's Gemini models
for optimized execution of multi-step workflows.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional

from tygent.dag import DAG
from tygent.nodes import LLMNode
from tygent.scheduler import Scheduler


class GoogleAINode(LLMNode):
    """Node for executing Google AI model calls."""

    def __init__(
        self,
        name: str,
        model: Any,
        prompt_template: str = "",
        dependencies: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize a Google AI node.

        Args:
            name: The name of the node
            model: Google AI model instance
            prompt_template: Template string for the prompt
            dependencies: List of node names this node depends on
        """
        super().__init__(name)
        self.model = model
        self.prompt_template = prompt_template
        # Set dependencies if provided
        if dependencies:
            self.dependencies = dependencies
        self.kwargs = kwargs

    async def execute(self, inputs: Dict[str, Any]) -> Any:
        """
        Execute the node with the given inputs.

        Args:
            inputs: Dictionary of input values

        Returns:
            The result of the model call
        """
        prompt = self._format_prompt(inputs, {})
        response = await self.model.generateContent(prompt, **self.kwargs)
        return response.response.text()

    def _format_prompt(
        self, inputs: Dict[str, Any], node_outputs: Dict[str, Any]
    ) -> str:
        """
        Format the prompt template with input variables and node outputs.

        Args:
            inputs: Dictionary of input values
            node_outputs: Dictionary of outputs from dependency nodes

        Returns:
            Formatted prompt string
        """
        # Combine inputs and node_outputs
        variables = {**inputs, **node_outputs}

        # Format the prompt template
        try:
            return self.prompt_template.format(**variables)
        except KeyError as e:
            # Handle missing variables gracefully
            missing_key = str(e).strip("'")
            return self.prompt_template.replace(
                f"{{{missing_key}}}", f"[Missing: {missing_key}]"
            )


class GoogleAIIntegration:
    """
    Integration for Google AI models with Tygent's DAG-based optimization.
    """

    def __init__(self, model: Any):
        """
        Initialize the Google AI integration.

        Args:
            model: Google AI model instance
        """
        self.model = model
        self.dag = DAG(name="google_ai_dag")
        self.scheduler = Scheduler(self.dag)

    def addNode(
        self, name: str, prompt_template: str, dependencies: List[str] = [], **kwargs
    ) -> GoogleAINode:
        """
        Add a node to the execution DAG.

        Args:
            name: The name of the node
            prompt_template: Template string for the prompt
            dependencies: List of node names this node depends on

        Returns:
            The created node
        """
        node = GoogleAINode(
            name=name,
            model=self.model,
            prompt_template=prompt_template,
            dependencies=dependencies,
            **kwargs,
        )
        self.dag.addNode(node)
        return node

    def optimize(self, options: Dict[str, Any]) -> None:
        """
        Set optimization parameters for the execution.

        Args:
            options: Dictionary of optimization parameters
                - maxParallelCalls: Maximum number of parallel calls
                - maxExecutionTime: Maximum execution time in milliseconds
                - priorityNodes: List of node names to prioritize
        """
        if "maxParallelCalls" in options:
            self.scheduler.max_parallel_nodes = options["maxParallelCalls"]
        if "maxExecutionTime" in options:
            self.scheduler.max_execution_time = options["maxExecutionTime"]
        if "priorityNodes" in options:
            self.scheduler.priority_nodes = options["priorityNodes"]

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the DAG with the given inputs.

        Args:
            inputs: Dictionary of input values

        Returns:
            Dictionary mapping node names to their outputs
        """
        return await self.scheduler.execute(inputs)


class GoogleAIBatchProcessor:
    """
    Batch processor for Google AI operations.
    """

    def __init__(
        self, model: Any, batch_size: int = 10, max_concurrent_batches: int = 2
    ):
        """
        Initialize the batch processor.

        Args:
            model: Google AI model instance
            batch_size: Number of items in each batch
            max_concurrent_batches: Maximum number of batches to process concurrently
        """
        self.model = model
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches

    async def process(
        self, items: List[Any], process_fn: Callable[[Any, Any], Any]
    ) -> List[Any]:
        """
        Process a list of items in optimized batches.

        Args:
            items: List of items to process
            process_fn: Function that processes a single item with signature:
                        async def process_fn(item, model) -> result

        Returns:
            List of results
        """
        # Split items into batches
        batches = [
            items[i : i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]

        results = []

        # Process batches with concurrency limit
        for i in range(0, len(batches), self.max_concurrent_batches):
            current_batches = batches[i : i + self.max_concurrent_batches]
            batch_tasks = []

            for batch in current_batches:
                # Process each item in the batch concurrently
                batch_tasks.extend(
                    [
                        asyncio.create_task(process_fn(item, self.model))
                        for item in batch
                    ]
                )

            # Wait for all tasks in the current set of batches to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Filter out exceptions
            valid_results = [
                result for result in batch_results if not isinstance(result, Exception)
            ]

            results.extend(valid_results)

        return results


def patch() -> None:
    """Patch google.generativeai to run through Tygent's scheduler."""
    try:
        from google.generativeai import GenerativeModel  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return

    def _wrap(method_name: str) -> None:
        original = getattr(GenerativeModel, method_name, None)
        if original is None:
            return

        async def _node_fn(self, *args, **kwargs):
            if asyncio.iscoroutinefunction(original):
                return await original(self, *args, **kwargs)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, original, self, *args, **kwargs)

        if asyncio.iscoroutinefunction(original):

            async def patched(self, *args, **kwargs):
                dag = DAG("google_ai_generate")
                dag.add_node(LLMNode("call", model=self, prompt_template=""))
                dag.nodes["call"].execute = lambda inputs: _node_fn(
                    self, *args, **kwargs
                )
                scheduler = Scheduler(dag)
                result = await scheduler.execute({})
                return result["results"]["call"]

        else:

            def patched(self, *args, **kwargs):
                async def run():
                    dag = DAG("google_ai_generate")
                    dag.add_node(LLMNode("call", model=self, prompt_template=""))
                    dag.nodes["call"].execute = lambda inputs: _node_fn(
                        self, *args, **kwargs
                    )
                    scheduler = Scheduler(dag)
                    result = await scheduler.execute({})
                    return result["results"]["call"]

                return asyncio.run(run())

        setattr(GenerativeModel, f"_tygent_{method_name}", original)
        setattr(GenerativeModel, method_name, patched)

    for name in ("generate_content", "generateContent"):
        _wrap(name)
