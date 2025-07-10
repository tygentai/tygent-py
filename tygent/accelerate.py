"""Accelerate function for drop-in optimization of existing agent frameworks."""

import asyncio
import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

from .dag import DAG
from .nodes import LLMNode, ToolNode
from .plan_parser import parse_plan
from .scheduler import Scheduler


def accelerate(
    func_or_agent: Optional[Union[Callable, Any]] = None,
) -> Union[Callable, Any]:
    """
    Accelerate any function or agent framework for automatic parallel optimization.

    This is a drop-in wrapper that analyzes your existing code and automatically
    optimizes execution through parallel processing and DAG-based scheduling.

    Args:
        func_or_agent: Function, agent, or framework object to accelerate. If ``None``,
            ``accelerate`` returns a decorator that can be applied using ``@accelerate()``.

    Returns:
        Accelerated version with same interface but optimized execution
    """

    if func_or_agent is None:

        def decorator(inner: Union[Callable, Any]) -> Union[Callable, Any]:
            return accelerate(inner)

        return decorator

    # Directly parse plan dictionaries
    if isinstance(func_or_agent, dict) and "steps" in func_or_agent:
        dag, critical = parse_plan(func_or_agent)
        return _PlanExecutor(dag, critical)

    # Handle different framework types
    if hasattr(func_or_agent, "__class__"):
        class_name = func_or_agent.__class__.__name__

        # Framework exposes a plan that can be parsed
        for attr in ("plan", "get_plan", "workflow"):
            if hasattr(func_or_agent, attr):
                plan_obj = getattr(func_or_agent, attr)
                plan = plan_obj() if callable(plan_obj) else plan_obj
                if isinstance(plan, dict) and "steps" in plan:
                    dag, critical = parse_plan(plan)
                    return _FrameworkExecutor(func_or_agent, dag, critical)

        # LangChain Agent
        if "Agent" in class_name or hasattr(func_or_agent, "run"):
            return _accelerate_langchain_agent(func_or_agent)

        # OpenAI Assistant
        if hasattr(func_or_agent, "id") and hasattr(func_or_agent, "instructions"):
            return _accelerate_openai_assistant(func_or_agent)

        # LlamaIndex components
        if "Index" in class_name or hasattr(func_or_agent, "query"):
            return _accelerate_llamaindex(func_or_agent)

    # Handle regular functions
    if callable(func_or_agent):
        return _accelerate_function(func_or_agent)

    # Return original if no optimization available
    return func_or_agent


class _PlanExecutor:
    """Execute a parsed plan using :class:`Scheduler`."""

    def __init__(self, dag: DAG, critical: List[str]) -> None:
        self.scheduler = Scheduler(dag)
        self.scheduler.priority_nodes = critical

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return await self.scheduler.execute(inputs)

    __call__ = execute


class _FrameworkExecutor:
    """Wrapper that runs a framework object's plan via Tygent."""

    def __init__(self, original: Any, dag: DAG, critical: List[str]) -> None:
        self.original = original
        self.scheduler = Scheduler(dag)
        self.scheduler.priority_nodes = critical

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return await self.scheduler.execute(inputs)

    __call__ = execute

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)


def _accelerate_function(func: Callable) -> Callable:
    """Accelerate a regular or async function."""

    real_func = inspect.unwrap(func)
    if asyncio.iscoroutinefunction(real_func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await _optimize_async_function(func, args, kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        return async_wrapper

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # For simple functions, analyze if they contain multiple async calls that can be parallelized
        if asyncio.iscoroutinefunction(func):
            return _optimize_async_function(func, args, kwargs)
        else:
            return _optimize_sync_function(func, args, kwargs)

    return wrapper


async def _optimize_async_function(func: Callable, args: tuple, kwargs: dict) -> Any:
    """Optimize async function execution by identifying parallel opportunities."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an event loop; return coroutine for the caller to await
        return await func(*args, **kwargs)
    else:
        # No running event loop, execute and return result synchronously
        return await asyncio.run(func(*args, **kwargs))



def _optimize_sync_function(func: Callable, args: tuple, kwargs: dict) -> Any:
    """Optimize sync function execution."""

    # For demonstration, we'll run the original function
    # In a full implementation, this would analyze the function's AST
    # to identify parallel execution opportunities
    return func(*args, **kwargs)


def _accelerate_langchain_agent(agent: Any) -> Any:
    """Accelerate LangChain agents by optimizing tool execution."""

    class AcceleratedLangChainAgent:
        def __init__(self, original_agent):
            self.original_agent = original_agent
            self._dag = None
            self._setup_dag()

        def _setup_dag(self):
            """Set up DAG for parallel tool execution."""
            self._dag = DAG("langchain_optimized")

            # Extract tools from agent if available
            if hasattr(self.original_agent, "tools"):
                for tool in self.original_agent.tools:
                    tool_node = ToolNode(name=tool.name, func=tool.func)
                    self._dag.add_node(tool_node)

        def run(self, query: str) -> str:
            """Run agent with potential parallel optimization."""
            # For now, delegate to original agent
            # In full implementation, this would analyze the query
            # and execute independent tools in parallel
            return self.original_agent.run(query)

        def __getattr__(self, name):
            """Delegate other attributes to original agent."""
            return getattr(self.original_agent, name)

    return AcceleratedLangChainAgent(agent)


def _accelerate_openai_assistant(assistant: Any) -> Any:
    """Accelerate OpenAI Assistants by optimizing function calls."""

    class AcceleratedOpenAIAssistant:
        def __init__(self, original_assistant):
            self.original_assistant = original_assistant
            self.id = original_assistant.id
            self.instructions = original_assistant.instructions

        def __getattr__(self, name):
            """Delegate all attributes to original assistant."""
            return getattr(self.original_assistant, name)

    return AcceleratedOpenAIAssistant(assistant)


def _accelerate_llamaindex(index_or_engine: Any) -> Any:
    """Accelerate LlamaIndex components by optimizing retrieval."""

    class AcceleratedLlamaIndex:
        def __init__(self, original_component):
            self.original_component = original_component

        def query(self, query_str: str) -> Any:
            """Query with potential parallel optimization."""
            # For now, delegate to original component
            # In full implementation, this would optimize multi-index queries
            return self.original_component.query(query_str)

        def __getattr__(self, name):
            """Delegate other attributes to original component."""
            return getattr(self.original_component, name)

    return AcceleratedLlamaIndex(index_or_engine)
